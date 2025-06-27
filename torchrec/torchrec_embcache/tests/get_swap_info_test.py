import embedding_cache
import torch
import random


class Gen:
    def __init__(self, max_key):
        self.max_key = max_key

    def gen_keys(self, batch_size):
        return [random.randint(0, self.max_key - 1) for _ in range(batch_size)]


def swap_manager_test(batch_size, cache_size, test_num):
    gen = Gen(cache_size * 10)
    emb_bag_config = embedding_cache.EmbConfig()
    emb_bag_config.table_name = "test_table"
    emb_bag_config.emb_dim = 128  # Example dimension
    emb_bag_config.optim_num = 2  # Example optim number
    emb_bag_config.cache_size = cache_size

    manager = embedding_cache.EmbcacheManager([emb_bag_config])  # Assuming single-hot

    cache = [0] * cache_size
    key2off = {}

    for t in range(test_num):
        print(f"\n===================== Testing batch {t} =====================")
        keys = gen.gen_keys(batch_size)
        keys_bak = keys.copy()
        key_set = set(keys)

        batch_keys_tensor = torch.tensor(keys, dtype=torch.int64)
        jagged_offs_tensor = [0, batch_size]

        # Get SwapInfo
        swap_info_future = manager.compute_swap_info_async(batch_keys_tensor, jagged_offs_tensor)
        swap_info = swap_info_future.get()

        swapout_keys = swap_info.swapout_keys[0]
        swapout_offs = swap_info.swapout_offs
        swapin_keys = swap_info.swapin_keys[0]
        swapin_offs = swap_info.swapin_offs
        batch_offs = swap_info.batch_offs

        print("swapout size:", len(swapout_keys))
        print("swapin size:", len(swapin_keys))
        print()

        # 1. key 和 off size 相同
        assert len(swapout_keys) == len(swapout_offs)
        assert len(swapin_keys) == len(swapin_offs)

        # 2. 执行 swapout，验证 swapoutKeys 和 swapoutOffs 都在 cache 中，
        # 且 swapoutKeys 不在 keys 中，且 off 都在范围内
        for i in range(len(swapout_keys)):
            key = swapout_keys[i]
            off = swapout_offs[i]

            assert off < cache_size
            assert cache[off] == key
            assert key in key2off
            assert key not in key_set

            cache[off] = 0
            del key2off[key]

        # 3. 执行 swapin，验证 swapinKeys 都在 keys 中，
        # 且 swapinKeys 和 swapoutOffs 都不在 cache 中，且 off 都在范围内
        for i in range(len(swapin_keys)):
            key = swapin_keys[i]
            off = swapin_offs[i]

            assert off < cache_size
            assert key in key_set
            assert key not in key2off

            cache[off] = key
            key2off[key] = off

        # 4. 验证 keys 都在 cache 中，且 key 都转化成 off
        for i in range(len(batch_offs)):
            key = keys_bak[i]
            off = batch_offs[i]
            cache[off] = key
            key2off[key] = off
            assert key in key2off
            assert cache[off] == key
            assert key2off[key] == off

        print(f"Test {t} passed!")


if __name__ == "__main__":
    batch_size = 0
    cache_size = 0
    test_num = 100
    swap_manager_test(batch_size, cache_size, test_num)

    batch_size = 1
    cache_size = 1
    test_num = 100
    swap_manager_test(batch_size, cache_size, test_num)

    batch_size = 10000
    cache_size = 10000
    test_num = 100
    swap_manager_test(batch_size, cache_size, test_num)
