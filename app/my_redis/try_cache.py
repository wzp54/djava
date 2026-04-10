# try_cache.py
import time
from cache import get_cache, set_cache


def run_cache_test():
    print("=" * 50)
    print("🚀 开始测试 Redis 缓存模块 (cache.py)")
    print("=" * 50)

    test_query = "地球上最高的山峰是哪座？"
    test_answer = "珠穆朗玛峰（Mount Everest）。"

    # ==================================
    # 场景 1：第一次查询，应该查不到（未命中）
    # ==================================
    print(f"\n[测试 1] 尝试查询一个还没有被问过的问题：")
    print(f"问题 -> '{test_query}'")

    result1 = get_cache(test_query)
    if result1 is None:
        print("✅ 测试通过：缓存中没有该问题，正确返回 None。")
    else:
        print(f"❌ 测试失败：竟然查到了奇怪的数据 -> {result1}")
        # 如果你之前跑过这个脚本，可能会查出数据，这也很正常

    # ==================================
    # 场景 2：大模型生成了答案，把它存进 Redis
    # ==================================
    print(f"\n[测试 2] 模拟大模型生成了答案，正在写入 Redis 缓存...")
    try:
        set_cache(test_query, test_answer)
        print("✅ 测试通过：数据写入 Redis 成功！")
    except Exception as e:
        print(f"❌ 写入失败，请检查 Redis 是否启动、密码是否正确。错误信息: {e}")
        return

    # 稍微等个0.5秒，确保数据落盘（虽然 Redis 很快，但稍微等一下结果更稳）
    time.sleep(0.5)

    # ==================================
    # 场景 3：第二次查询同样的问题，应该直接秒回
    # ==================================
    print(f"\n[测试 3] 模拟用户第二次问了完全一样的问题：")

    start_time = time.time()
    result2 = get_cache(test_query)
    end_time = time.time()

    if result2 == test_answer:
        print(f"✅ 测试通过：成功命中缓存！")
        print(f"提取出的答案 -> {result2}")
        print(f"⏱️ 缓存查询耗时 -> {(end_time - start_time) * 1000:.2f} 毫秒 (极速!)")
    else:
        print("❌ 测试失败：没有从缓存中拿到正确的答案。")

    print("\n" + "=" * 50)
    print("🎉 所有测试流程执行完毕！")


if __name__ == "__main__":
    run_cache_test()