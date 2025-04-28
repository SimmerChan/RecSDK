
set -x

python3 prof_matmul.py clean

python3 prof_matmul.py gen_test_data test_case.csv 0
