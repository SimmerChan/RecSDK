import numpy as np
from scipy.stats import pareto
from concurrent.futures import ProcessPoolExecutor

# 帕累托参数
SHAPE_PARAM = 5  # 帕累托分布的形状参数 alpha（必须大于0）,越小尾越大
SCALE_PARAM = 1.0  # 帕累托分布的尺度参数 x_m（通常设置为一个较小的正数，例如1.0）

LENGTH = 195841983
MAX_ROWS = 40000000
DAYS = 24
CATS = 26
DENSE_DIM = 13
LABELS_NUM = 2
MAX_NUM_LIST = [
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
    day_0_labels = np.random.randint(0, LABELS_NUM, size=(LENGTH, 1)).astype(np.int32)
    # denses生成
    day_0_dense = np.random.uniform(0, 1, size=(LENGTH, DENSE_DIM)).astype(np.float32)
    # sparse生成
    day_0_sparse = np.empty((LENGTH, CATS), dtype=np.int32)

    for index, max_num in enumerate(MAX_NUM_LIST):
        if max_num >= MAX_ROWS:
            random_pareto_floats = pareto.rvs(
                SHAPE_PARAM, scale=SCALE_PARAM, size=LENGTH
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
            day_0_sparse[:, index] = np.random.randint(0, max_num, size=LENGTH)

    np.save(f"day_{i}_labels.npy", day_0_labels)
    np.save(f"day_{i}_dense.npy", day_0_dense)
    np.save(f"day_{i}_sparse.npy", day_0_sparse)


thrad_pool = ProcessPoolExecutor(DAYS)
for i in range(DAYS):
    thrad_pool.submit(generate, i)
