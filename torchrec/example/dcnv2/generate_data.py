import numpy as np
from scipy.stats import pareto
from concurrent.futures import ProcessPoolExecutor

# 帕累托参数
shape_param = 5  # 帕累托分布的形状参数 alpha（必须大于0）,越小尾越大
scale_param = 1.0  # 帕累托分布的尺度参数 x_m（通常设置为一个较小的正数，例如1.0）

length = 195841983
max_rows = 40000000
days = 24
cats = 26
dense_dim = 13
labels_num = 2
max_num_lst = [
    40000000,
    39060,
    17295,
    7424,
    20265,
    3,
    7122,
    1543,
    63,
    40000000,
    3067956,
    405282,
    10,
    2209,
    11938,
    155,
    4,
    976,
    14,
    40000000,
    40000000,
    40000000,
    590152,
    12973,
    108,
    36,
]


def generate(i):
    # labels生成
    day_0_labels = np.random.randint(0, labels_num, size=(length, 1)).astype(np.int32)
    # denses生成
    day_0_dense = np.random.uniform(0, 1, size=(length, dense_dim)).astype(np.float32)
    # sparse生成
    day_0_sparse = np.empty((length, cats), dtype=np.int32)

    for index, max_num in enumerate(max_num_lst):
        if max_num >= max_rows:
            random_pareto_floats = pareto.rvs(
                shape_param, scale=scale_param, size=length
            )
            random_pareto_ints = np.clip(
                np.round(
                    (max_num - 1)
                    * (random_pareto_floats / (random_pareto_floats.max()))
                ),
                0,
                max_num,
            ).astype(np.int32)
            day_0_sparse[:, index] = random_pareto_ints
        else:
            day_0_sparse[:, index] = np.random.randint(0, max_num, size=length)

    np.save(f"day_{i}_labels.npy", day_0_labels)
    np.save(f"day_{i}_dense.npy", day_0_dense)
    np.save(f"day_{i}_sparse.npy", day_0_sparse)


a = ProcessPoolExecutor(days)
for i in range(days):
    a.submit(generate, i)
