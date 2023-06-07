from general_test import CacheProfiler
import logging


def test_livejournel_hit():
    print("begin")
    cache = CacheProfiler()
    cache.init_config(
        dataset_name="livejournal",
        gpu_list=[0],
        sample_gpu=0,
        static_cache_policy="degree",
        static_cache_ratio=0.2,
        dynamic_cache_policy="FIFO",
        dynamic_cache_ratio=0.000001,
        batch_size=6000,
    )
    cache.cache_nids_to_gpu()
    print("begin")

    (
        static_hit_ratio,
        dynamic_hit_ratio,
        relevant_dynamic_hit_ratio,
        global_hit_tatio,
    ) = cache.degree_and_fifo_mixed_analysis_on_single()
    print("static hit ration:{},cache ration:{:.2f}".format(static_hit_ratio, 0.2))


def test_livejournel_hit_static():
    cache = CacheProfiler()
    cache.init_config(
        dataset_name="livejournal",
        gpu_list=[0],
        sample_gpu=0,
        static_cache_policy="degree",
        static_cache_ratio=0.2,
        batch_size=6000,
    )
    cache.cache_nids_to_gpu()
    logging.info("begin to compute")
    (
        hit_ratio,
        access_count,
        hit_count,
    ) = cache.static_cache_analysis_on_single()
    print(
        "hit ratio:{} \taccess count:{} \thit count:{}".format(
            hit_ratio, access_count, hit_count
        )
    )


if __name__ == "__main__":
    test_livejournel_hit_static()
