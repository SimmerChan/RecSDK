import pytest
import torch
from torchrec import PoolingType
from torchrec.distributed.types import ShardingEnv, ParameterSharding
from hybrid_torchrec import HashEmbeddingBagCollection, HashEmbeddingBagConfig
from hybrid_torchrec.distributed.embeddingbag import device_is_in, _pin_and_move, HybridShardedEmbeddingBagCollection

FEAT_NAMES = [["phone", "clothes"], ["user"]]
TABLE_NAMES = ["product", "user"]
EMBEBD_DIMS = [1024, 1024]
NUM_EMBEBDS = [10240, 10240]


class TestModel(torch.nn.Module):
    def __init__(self, table_names, feat_names, embed_dims, num_embeds):
        super().__init__()
        table_configs = []

        for table_name, feat_name, dim, num_embed in zip(
                table_names, feat_names, embed_dims, num_embeds
        ):
            config = HashEmbeddingBagConfig(
                name=table_name,
                embedding_dim=dim,
                num_embeddings=num_embed,
                feature_names=feat_name,
                pooling=PoolingType.SUM,
            )
            table_configs.append(config)

        self.ebc = HashEmbeddingBagCollection(device="cpu", tables=table_configs)
        self.input_dim = sum([len(f) * d for f, d in zip(feat_names, embed_dims)])
        self.linear_net = torch.nn.Linear(self.input_dim, self.input_dim)


@pytest.mark.parametrize("device", [torch.device("cuda:0"), torch.device("cpu"), "npu:0", "cpu"])
@pytest.mark.parametrize("check_device", [["meta", "cpu"]])
def test_device_check_func(device, check_device: list[str]):
    device_is_in(device, check_device)


def test_pin_and_move_cpu():
    device = torch.device("cpu")
    tensor = torch.tensor([1, 2, 3])

    result = _pin_and_move(tensor, device)

    assert result.device.type == "cpu"
    assert torch.equal(result, tensor)
    assert not result.is_pinned()  # CPU上不应被pin


def test_hybrid_sharded_ebc_init():
    model = TestModel(TABLE_NAMES, FEAT_NAMES, EMBEBD_DIMS, NUM_EMBEBDS)
    parameter_sharding = {
        table_name: ParameterSharding(
            sharding_types=["row_wise"], compute_kernels=["fused"]
        )
        for table_name in TABLE_NAMES
    }
    env = ShardingEnv(world_size=1, rank=0)
    HybridShardedEmbeddingBagCollection(model, parameter_sharding, env, env)
