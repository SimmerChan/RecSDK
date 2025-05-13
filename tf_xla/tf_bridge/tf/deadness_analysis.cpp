/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tf_bridge/tf/deadness_analysis.h"

#include <deque>

#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tf_bridge/tf/errors.h"
#include "tf_bridge/tf/xla_cluster_util.h"
#include "tf_bridge/tf_compatible.h"

// ALGORITHM OVERVIEW
// ==================
//
// We map every output produced by each node in the TensorFlow graph (including
// control dependence) into an instance of the Predicate class.  Instances of
// Predicate denote logical formulas and mapping a node `n` to a predicate
// `pred` implies that `n` is live whenever `pred` is true.  Then we can deduce
// mismatching liveness in the inputs to node by comparing the predicate those
// inputs are mapped to.  The core logic of this pass resides in creating the
// map from TensorFlow nodes to predicates.
//
//
// MAPPING NODES TO PREDICATES, MODULO CYCLES
// ------------------------------------------
//
// If we ignore cycles for a moment, computing predicates is fairly
// straightforward.  We traverse the graph in a topological order, mapping each
// node to a predicate based on the predicates its inputs are mapped to.  For
// instance a Merge(X, Y) node will be mapped to OR(PredicateFor(X),
// PredicateFor(Y)).  Roughtly speaking, we abstractly interpret each node on
// the "liveness" domain, where values in the domain represent if a tensor
// carries a dead signal or not.
//
//
// DEALING WITH CYCLES
// -------------------
//
// We map Merge nodes that are the target of a backedge to AndRecurrence
// instances.  An AndRecurrence with Start() = S and Step() = X, printed as
// {S,&,X}, *roughly* represents the infinite list of predicates
// [S,S&X,S&X&X,S&X&X, ...].  So {S,&,X} can be used to represent the predicate
// for Merge in a graph like:
//
//     Init
//       |
//       v
//     Merge <-----------+
//       |               |
//       v               |
//      Incr             |
//       |               |
//       v               |
//      Switch <- Cond   |
//       |               |
//       v (oidx: 1)     |
//       |               |
//       +---------------+
//
// Where S is the predicate for Init and X is the predicate that asserts that
// Cond is true.  {S,&,X} states that Merge is live on the first "iteration" iff
// S is true, live on the second iteration iff "S&X" is true, live on the third
// iteration iff "S&X&X" is true etc.  There is a subtlety here, S&X&X would
// normally be equivalent to S&X which isn't quite what we want to represent.
// Instead we want {S,&,X} to denote the infinite list [S, S&X,
// S&X&X',S&X&X'&X'', ...] where X, X', X'' are predicates that assert Cond is
// true on iteration 0, 1, 2 respectively.  This is made more precise in the
// comment on the AndRecurrence class.
//
// The general algorithm that deals with cycles does two topological-order
// iterations over the graph.  On the first iteration it assigns a symbolic
// predicate to merge nodes with backedges.  On the second iteration it tries
// to pattern match the predicates for the backedges of these merges and infer
// an AndRecurrence for the merge.  In other words, we do a data flow analysis
// where the data-flow lattice has two elements, Symbolic and NonSymbolic with
// Symbolic > NonSymbolic.  The lattice has height = 2 so two iterations are
// sufficient to converge.
//
// We first do an optimistic analysis and, if it does not converge, we then fall
// back to a pessimistic analysis.  The optimistic analysis assigns the same
// symbolic predicate to all the merge nodes whose preceding enter nodes have
// the same frame name on the first iteration.  On the second iteration, if all
// the merge nodes are pattern matched into the same AndRecurrence predicate
// instance, the optimistic assignment of the same symbolic predicate is correct
// and the analyzed result is taken.
//
// Otherwise, if the optimistic analysis fails to converge, we then obtain the
// result by falling back to the pessimistic analysis which assigns a unique
// symbolic predicate to each merge on the first iteration.  We still use
// symbolic predicates for merges for which we can't pattern match on the
// backedge predicate.  This is conservatively correct.

namespace tensorflow {
namespace npu_xla {

namespace {

using absl::StatusOr;

// Represents a logical predicate, used as described in the algorithm overview
// above.
class Predicate {
public:
    enum class Kind {
        K_AND,
        K_OR,
        K_NOT,
        K_AND_RECURRENCE,
        K_SYMBOL,
        K_INIT_SYMBOL
    };

    virtual string ToString() const = 0;

    // An ID assigned to the Predicate at construction time.  Conceptually like a
    // pointer, except that it is stable across runs.
    int64 id() const
    {
        return id_;
    }

    virtual absl::Span<Predicate* const> GetOperands() const = 0;

    virtual Kind Kind() const = 0;
    virtual ~Predicate() {}

    // Invokes func on p and on all of its operands recursively.  Does not invoke
    // `func` on the same Predicate instance twice.  Aborts the search if `func`
    // returns true.
    template <typename FunctionTy>
    static void Visit(Predicate* p, const FunctionTy& func);

protected:
    explicit Predicate(int64 id) : id_(id) {}

private:
    const int64 id_;

    TF_DISALLOW_COPY_AND_ASSIGN(Predicate);
};

// Represents a logical conjunction of a set of predicates.
class AndPredicate : public Predicate {
public:
    explicit AndPredicate(int64 id, std::vector<Predicate*> operands) : Predicate(id), operands_(std::move(operands)) {}

    string ToString() const override
    {
        if (operands().empty()) {
            return "#true";
        }

        std::vector<string> operands_str;
        std::transform(operands().begin(), operands().end(), std::back_inserter(operands_str),
                       [](Predicate* pred) { return pred->ToString(); });

        return absl::StrCat("(", absl::StrJoin(operands_str, " & "), ")");
    }

    Kind Kind() const override
    {
        return Kind::K_AND;
    }

    absl::Span<Predicate* const> GetOperands() const override
    {
        return operands_;
    }
    absl::Span<Predicate* const> operands() const
    {
        return operands_;
    }

private:
    std::vector<Predicate*> operands_;
};

// Represents a logical disjunction of a set of predicates.
class OrPredicate : public Predicate {
public:
    explicit OrPredicate(int64 id, std::vector<Predicate*> operands) : Predicate(id), operands_(std::move(operands)) {}

    string ToString() const override
    {
        if (operands().empty()) {
            return "#false";
        }

        std::vector<string> operands_str;
        std::transform(operands().begin(), operands().end(), std::back_inserter(operands_str),
                       [](Predicate* pred) { return pred->ToString(); });

        return absl::StrCat("(", absl::StrJoin(operands_str, " | "), ")");
    }

    Kind Kind() const override
    {
        return Kind::K_OR;
    }
    absl::Span<Predicate* const> GetOperands() const override
    {
        return operands_;
    }
    absl::Span<Predicate* const> operands() const
    {
        return operands_;
    }

private:
    std::vector<Predicate*> operands_;
};

// Represents a logical negation of a set of predicates.
class NotPredicate : public Predicate {
public:
    explicit NotPredicate(int64 id, Predicate* operand) : Predicate(id), operands_({operand}) {}

    string ToString() const override
    {
        return absl::StrCat("~", Operand()->ToString());
    }

    Kind Kind() const override
    {
        return Kind::K_NOT;
    }
    Predicate* Operand() const
    {
        return operands_[0];
    }
    absl::Span<Predicate* const> GetOperands() const override
    {
        return operands_;
    }

private:
    std::array<Predicate*, 1> operands_;
};

// Represents the liveness of an induction variable.  For users inside the loop
// this represents the "current" liveness of the induction variable.  For users
// outside the loop it represents the "last" liveness of the induction variable.
//
// More concretely, an and recurrence {S,&,X}<loop> represents the liveness of V
// in the following graph:
//
//   V = Merge(S', V_NextIt)
//   V = Op(V, X')
//   V_NextIt = NextIteration(V)
//
// where Predicate(S') = S and Predicate(X') = X.
//
// `X` may contain symbolic predicates and the operations corresponding to these
// symbolic predicates are either in frame `loop` or outside it.  The symbols
// that are inside frame `loop` are loop variant (i.e. can have different
// liveness in each loop iteration) and the symbols that are outside frame
// `loop` are loop invariant (i.e. have the same liveness across all
// iterations).
class AndRecurrencePredicate : public Predicate {
public:
    explicit AndRecurrencePredicate(int64 id, Predicate* start, Predicate* step, std::vector<string> frame)
        : Predicate(id),
          operands_({start, step}),
          frame_(std::move(frame))
    {
    }

    Predicate* Start() const
    {
        return operands_[0];
    }
    Predicate* Step() const
    {
        return operands_[1];
    }
    absl::Span<const string> frame() const
    {
        return frame_;
    }

    string ToString() const override
    {
        return absl::StrCat("{", Start()->ToString(), ",&,", Step()->ToString(), "}<", absl::StrJoin(frame(), ";"),
                            ">");
    }

    Kind Kind() const override
    {
        return Kind::K_AND_RECURRENCE;
    }

    absl::Span<Predicate* const> GetOperands() const override
    {
        return operands_;
    }

private:
    std::array<Predicate*, 2> operands_; // 创建2个元素的数组
    std::vector<string> frame_;
};

// Represents an uninterpreted symbol in a logical predicate.
//
// Two predicates are equivalent iff they are equivalent for all assignments to
// the symbols contained in them, i.e. predicates are forall qualified over
// symbols.
class SymbolPredicate : public Predicate {
public:
    explicit SymbolPredicate(int64 id, TensorId tensor_id, bool mustBeTrue)
        : Predicate(id),
          tensor_id_(std::move(tensor_id)),
          mustBeTrue_(mustBeTrue)
    {
    }

    string ToString() const override
    {
        return MustBeTrue() ? absl::StrCat("*", tensor_id_.ToString()) : tensor_id_.ToString();
    }

    Kind Kind() const override
    {
        return Kind::K_SYMBOL;
    }
    absl::Span<Predicate* const> GetOperands() const override
    {
        return {};
    }

    // If `MustBeTrue()` is true this SymbolPredicate represents the proposition
    // "tensor_id() is live and evaluates to true".
    // If `MustBeTrue()` is false then this SymbolPredicate represents the
    // proposition "tensor_id() is live (and may evaluate to any value)"
    TensorId tensor_id() const
    {
        return tensor_id_;
    }
    bool MustBeTrue() const
    {
        return mustBeTrue_;
    }

private:
    TensorId tensor_id_;
    bool mustBeTrue_;
};

// Represents an uninterpreted symbol in a logical predicate.
//
// Two predicates are equivalent iff they are equivalent for all assignments to
// the symbols contained in them, i.e. predicates are forall qualified over
// symbols.
class IntSymbolPredicate : public Predicate {
public:
    explicit IntSymbolPredicate(int64 id, TensorId tensor_id, absl::optional<int> must_have_value)
        : Predicate(id),
          tensor_id_(std::move(tensor_id)),
          must_have_value_(must_have_value)
    {
    }

    string ToString() const override
    {
        return must_have_value().has_value() ? absl::StrCat(tensor_id_.ToString(), "=", *must_have_value_)
                                             : tensor_id_.ToString();
    }

    Kind Kind() const override
    {
        return Kind::K_INIT_SYMBOL;
    }
    absl::Span<Predicate* const> GetOperands() const override
    {
        return {};
    }

    // If `must_have_value().has_value()` is true, then this IntSymbolPredicate
    // represents the proposition "tensor_id() is live and evaluates to
    // `*must_have_value()`".
    // If `must_have_value().has_value()` is false, then this IntSymbolPredicate
    // represents the proposition "tensor_id() is live (and may evaluate to any
    // value)".
    TensorId tensor_id() const
    {
        return tensor_id_;
    }
    const absl::optional<int>& must_have_value() const
    {
        return must_have_value_;
    }

private:
    TensorId tensor_id_;
    absl::optional<int> must_have_value_;
};

template <typename FunctionTy>
void Predicate::Visit(Predicate* p, const FunctionTy& func)
{
    std::unordered_set<Predicate*> visited;
    std::vector<Predicate*> stack;

    stack.push_back(p);
    visited.insert(p);

    while (!stack.empty()) {
        Predicate* current = stack.back();
        stack.pop_back();
        bool done = func(current);
        if (done) {
            return;
        }
        for (Predicate* op : current->GetOperands()) {
            if (visited.insert(op).second) {
                stack.push_back(op);
            }
        }
    }
}

// Creates and owns Predicate instances.  Simplifies predicates as it creates
// them.
class PredicateFactory {
public:
    Predicate* MakeAndPredicate(absl::Span<Predicate* const> operands)
    {
        return MakeAndOrImpl(operands, true);
    }

    Predicate* MakeOrPredicate(absl::Span<Predicate* const> operands)
    {
        return MakeAndOrImpl(operands, false);
    }

    Predicate* MakeNotPredicate(Predicate* pred)
    {
        auto it = make_not_predicate_cache_.find(pred);
        if (it != make_not_predicate_cache_.end()) {
            return it->second;
        }

        Predicate* result = MakeNotPredicateImpl(pred);

        bool insertSuccessful = make_not_predicate_cache_.insert({pred, result}).second;
        (void)insertSuccessful;
        DCHECK(insertSuccessful);

        return result;
    }

    Predicate* MakeAndRecurrencePredicate(Predicate* start, Predicate* step, std::vector<string> frame)
    {
        SignatureForAndRec signature(start, step, std::move(frame));
        auto it = interned_and_rec_instances_.find(signature);
        if (it != interned_and_rec_instances_.end()) {
            return it->second.get();
        }

        std::unique_ptr<Predicate> new_pred =
            Make<AndRecurrencePredicate>(std::get<0>(signature), std::get<1>(signature), std::get<2>(signature));
        Predicate* newPredPtr = new_pred.get();
        bool inserted = interned_and_rec_instances_.emplace(signature, std::move(new_pred)).second;
        (void)inserted;
        DCHECK(inserted);
        return newPredPtr;
    }

    Status MakeSymbolPredicate(Node* node, int output_idx, bool mustBeTrue, Predicate** predicate)
    {
        TensorId tensor_id(node->name(), output_idx);

        // bool is_boolean_tensor =
        //     BaseType(node->output_type(tensor_id.index())) == DT_BOOL;
        // TF_RET_CHECK(!mustBeTrue || is_boolean_tensor);

        if (node->type_string() == "Const" && mustBeTrue) {
            const TensorProto* proto = nullptr;
            TF_RETURN_IF_ERROR(GetNodeAttr(node->def(), "value", &proto));

            Tensor tensor(proto->dtype());
            TF_RET_CHECK(tensor.FromProto(*proto));

            *predicate = tensor.scalar<bool>()() ? MakeTrue() : MakeFalse();
            return absl::OkStatus();
        }

        SignatureForSymbol signature = {tensor_id, mustBeTrue};
        auto it = interned_symbol_instances_.find(signature);
        if (it == interned_symbol_instances_.end()) {
            std::unique_ptr<Predicate> new_pred = Make<SymbolPredicate>(tensor_id, mustBeTrue);
            Predicate* newPredPtr = new_pred.get();
            interned_symbol_instances_.emplace(std::move(signature), std::move(new_pred));
            *predicate = newPredPtr;
        } else {
            *predicate = it->second.get();
        }

        return absl::OkStatus();
    }

    Status MakeSymbolPredicate(Node* node, int output_idx, absl::optional<int> must_have_value, Predicate** predicate)
    {
        TensorId tensor_id(node->name(), output_idx);

        if (must_have_value.has_value() && node->type_string() == "Const") {
            const TensorProto* proto = nullptr;
            TF_RETURN_IF_ERROR(GetNodeAttr(node->def(), "value", &proto));

            Tensor tensor(proto->dtype());
            TF_RET_CHECK(tensor.FromProto(*proto));

            *predicate = tensor.scalar<int32>()() == *must_have_value ? MakeTrue() : MakeFalse();
            return absl::OkStatus();
        }
        SignatureForIntSymbol signature = {tensor_id, must_have_value};
        auto it = interned_int_symbol_instances_.find(signature);
        if (it == interned_int_symbol_instances_.end()) {
            std::unique_ptr<Predicate> new_pred = Make<IntSymbolPredicate>(tensor_id, must_have_value);
            Predicate* newPredPtr = new_pred.get();
            interned_int_symbol_instances_.emplace(std::move(signature), std::move(new_pred));
            *predicate = newPredPtr;
        } else {
            *predicate = it->second.get();
        }

        return absl::OkStatus();
    }

    Predicate* MakeTrue()
    {
        return MakeAndPredicate({});
    }
    Predicate* MakeFalse()
    {
        return MakeOrPredicate({});
    }

    ~PredicateFactory()
    {
        DCHECK_EQ(stackDepth_, 0) << "Unnested IncrementStackDepth?";
    }

private:
    Predicate* MakeNotPredicateImpl(Predicate* pred)
    {
        IncrementStackDepth stackFrame(this);
        if (!stackFrame.HasOverflowed()) {
            if (Predicate* simplified = SimplifyUsingDeMorgan(pred)) {
                return simplified;
            }

            // ~~A => A
            if (auto* notPred = dynamic_cast<NotPredicate*>(pred)) {
                return notPred->Operand();
            }
        }

        SignatureForNot signature = pred;
        auto it = interned_not_instances_.find(signature);
        if (it == interned_not_instances_.end()) {
            std::unique_ptr<Predicate> new_pred = Make<NotPredicate>(pred);
            Predicate* newPredPtr = new_pred.get();
            interned_not_instances_.emplace(signature, std::move(new_pred));
            return newPredPtr;
        } else {
            return it->second.get();
        }
    }

    Predicate* SimplifyUsingDeMorgan(Predicate* pred)
    {
        // ~(A & B & C & ...) => ~A | ~B | ~C | ~...
        // ~(A | B | C | ...) -> ~A & ~B & ~C & ~...
        Predicate::Kind kind = pred->Kind();
        if (kind == Predicate::Kind::K_AND || kind == Predicate::Kind::K_OR) {
            std::vector<Predicate*> new_operands;
            std::transform(pred->GetOperands().begin(), pred->GetOperands().end(), std::back_inserter(new_operands),
                           [&](Predicate* p) { return MakeNotPredicate(p); });
            return kind == Predicate::Kind::K_OR ? MakeAndPredicate(new_operands) : MakeOrPredicate(new_operands);
        }

        return nullptr;
    }

    template <typename PredicateT, typename... Args>
    std::unique_ptr<Predicate> Make(Args&&... args)
    {
        // If we ever expose the Predicate class outside this .cc file then we may
        // want to make this hard to misuse (by accidentally passing in an arbitrary
        // integer to the Predicate constructor for instance).
        return std::make_unique<PredicateT>(id_counter_++, std::forward<Args>(args)...);
    }

    Predicate* MakeAndOrImpl(absl::Span<Predicate* const> operands, bool isAnd);
    Predicate* MakeInternedAndOr(std::vector<Predicate*> simplified_ops, Predicate::Kind predKind);

    // Predicate instances are interned, meaning that there is only a single
    // instance of a Predicate object with a given content.  This makes checking
    // for structural equality super-cheap -- we can just compare pointers.
    //
    // We intern predicates by maintaining a map from the content of a Predicate
    // to the only instance of said predicate we allow to exist in the
    // interned_and_or_instances_, interned_not_instances_ and
    // interned_symbol_instances_ fields.  These maps also double up as storage
    // for the owning pointers to predicate instances.
    using SignatureForAndOr = std::pair<Predicate::Kind, absl::Span<Predicate* const>>;
    using SignatureForNot = Predicate*;
    using SignatureForAndRec = std::tuple<Predicate*, Predicate*, std::vector<string>>;
    using SignatureForSymbol = std::pair<SafeTensorId, bool>;
    using SignatureForIntSymbol = std::pair<SafeTensorId, absl::optional<int32>>;

    struct HashSignatureForAndOr {
        size_t operator()(const SignatureForAndOr& signature) const
        {
            size_t hash = ::tensorflow::hash<Predicate::Kind>()(signature.first);
            for (Predicate* p : signature.second) {
                hash = Hash64Combine(hash, ::tensorflow::hash<Predicate*>()(p));
            }
            return hash;
        }
    };

    struct HashSignatureForSymbol {
        size_t operator()(const SignatureForSymbol& signature) const
        {
            return Hash64Combine(SafeTensorId::Hasher()(signature.first), ::tensorflow::hash<bool>()(signature.second));
        }
    };

    struct HashSignatureForIntSymbol {
        size_t operator()(const SignatureForIntSymbol& signature) const
        {
            return Hash64Combine(
                SafeTensorId::Hasher()(signature.first),
                Hash64Combine(::tensorflow::hash<bool>()(signature.second.has_value()),
                              ::tensorflow::hash<int32>()(signature.second.has_value() ? *signature.second : 0)));
        }
    };

    // Used to limit recursion to avoid blowing up the stack and cap compile time.
    class IncrementStackDepth {
    public:
        explicit IncrementStackDepth(PredicateFactory* parent) : parent_(parent)
        {
            parent_->stackDepth_++;
        }

        bool HasOverflowed() const
        {
            const int kMaxStackDepth = 8;
            return parent_->stackDepth_ >= kMaxStackDepth;
        }

        ~IncrementStackDepth()
        {
            parent_->stackDepth_--;
        }

    private:
        PredicateFactory* parent_;
    };

    // A cache for the MakeNotPredicate function.
    //
    // NB! This is *not* the same as `interned_not_instances_`.
    // `interned_not_instances_` maps ensures pointer identity for `NotPredicate`
    // instances, i.e., it ensures there at most one instance of Not(predicate)
    // for any given predicate whereas `make_not_predicate_cache_` simply caches
    // the result of the `MakeNotPredicate` function.  The values in
    // `interned_not_instances_` are always instance of `NotPredicate` whereas the
    // values in `make_not_predicate_cache_` may not be (for instance it will map
    // Not(Not(A)) to A).
    std::unordered_map<Predicate*, Predicate*> make_not_predicate_cache_;

    std::unordered_map<SignatureForAndOr, std::unique_ptr<Predicate>, HashSignatureForAndOr> interned_and_or_instances_;
    std::unordered_map<SignatureForNot, std::unique_ptr<Predicate>> interned_not_instances_;
    struct HashSignatureForAndRec {
        std::size_t operator()(const SignatureForAndRec& k) const
        {
            auto result = std::hash<Predicate*>{}(std::get<0>(k)) ^ std::hash<Predicate*>{}(std::get<1>(k));
            for (auto s : std::get<2>(k)) { // 获取第2个元素
                result = result ^ std::hash<string>{}(s);
            }
            return result;
        }
    };
    std::unordered_map<SignatureForAndRec, std::unique_ptr<Predicate>, HashSignatureForAndRec>
        interned_and_rec_instances_;
    std::unordered_map<SignatureForSymbol, std::unique_ptr<Predicate>, HashSignatureForSymbol>
        interned_symbol_instances_;
    std::unordered_map<SignatureForIntSymbol, std::unique_ptr<Predicate>, HashSignatureForIntSymbol>
        interned_int_symbol_instances_;
    int64 id_counter_ = 0;
    int stackDepth_ = 0;
};

Predicate* PredicateFactory::MakeInternedAndOr(std::vector<Predicate*> simplified_ops, Predicate::Kind predKind)
{
    std::stable_sort(simplified_ops.begin(), simplified_ops.end(),
                     [](Predicate* a, Predicate* b) { return a->id() < b->id(); });

    auto it = interned_and_or_instances_.find({predKind, simplified_ops});
    if (it != interned_and_or_instances_.end()) {
        return it->second.get();
    }

    simplified_ops.shrink_to_fit();
    // NB!  Because we'll use a non-owning reference to simplified_ops in the
    // key for interned_and_or_instances_ we need to be careful to std::move()
    // it all the way through.
    absl::Span<Predicate* const> operands_slice = simplified_ops;
    std::unique_ptr<Predicate> new_pred = predKind == Predicate::Kind::K_AND
                                              ? Make<AndPredicate>(std::move(simplified_ops))
                                              : Make<OrPredicate>(std::move(simplified_ops));

    Predicate* newPredPtr = new_pred.get();
    interned_and_or_instances_.emplace(SignatureForAndOr(predKind, operands_slice), std::move(new_pred));
    return newPredPtr;
}

}  // namespace npu_xla
}  // namespace tensorflow
