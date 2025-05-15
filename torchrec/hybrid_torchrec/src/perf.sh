perf record -F 99 --call-graph dwarf ./build/Main -- sleep 60 
# perf report -n --stdio
# perf script -i perf.data &> perf.unfold
