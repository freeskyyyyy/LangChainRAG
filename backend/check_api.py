r"""
阿里云百炼 API 连通性测试

【为什么必须测这个】
我在 .env 里写的模型名（qwen-plus / text-embedding-v4 / gte-rerank-v2）
是**根据经验填的默认值**，不保证在你账号下一定可用：
  · 阿里会不时更新模型版本、下线旧版本
  · 不同账号开通的服务范围可能不同
  · 免费额度和计费策略因模型而异

所以这里做一次实测：把候选模型挨个试一遍，报告哪些能用、哪些不能用。
测试完就知道该往 .env 里填什么了。

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe check_api.py
"""

import sys

sys.path.insert(0, ".")

from app.core.config import settings  # noqa: E402

# 候选模型清单：按「推荐优先级」从高到低排列
CHAT_CANDIDATES = ["qwen-plus", "qwen-flash", "qwen-turbo", "qwen-max", "qwen-long"]
EMBED_CANDIDATES = [
    "text-embedding-v4",
    "text-embedding-v3",
    "text-embedding-v2",
    "text-embedding-v1",
]
RERANK_CANDIDATES = ["gte-rerank-v2", "gte-rerank"]

LINE = "=" * 68


def header(text: str) -> None:
    print(f"\n{LINE}\n  {text}\n{LINE}")


def brief(err: Exception) -> str:
    """把冗长的报错压缩成一行关键信息"""
    msg = str(err).replace("\n", " ")
    return msg[:200] + ("..." if len(msg) > 200 else "")


def test_chat() -> str | None:
    """测试对话模型：让模型回一句话，看能不能通"""
    header("一、对话模型测试（负责生成回答）")
    from langchain_community.chat_models import ChatTongyi

    for name in CHAT_CANDIDATES:
        try:
            # 【关键】必须显式传 dashscope_api_key。
            # ChatTongyi 默认只会去读「操作系统的环境变量 DASHSCOPE_API_KEY」，
            # 而我们的密钥只写在 .env 文件里，没设成系统环境变量，所以必须手动传进去。
            # （另外两个组件 DashScopeEmbeddings / TextReRank 同理，都要显式传。）
            llm = ChatTongyi(
                model=name,
                temperature=0.3,
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
            )
            resp = llm.invoke("请只回复两个字：正常")
            text = resp.content if hasattr(resp, "content") else str(resp)
            print(f"  ✅ {name:<16} 可用 —— 模型回复: {text.strip()[:40]}")
            return name
        except Exception as e:
            print(f"  ❌ {name:<16} 不可用 —— {brief(e)}")
    return None


def test_embedding() -> str | None:
    """测试向量化模型：把一句话变成数字坐标，看能不能通"""
    header("二、向量化模型测试（把文字变成可检索的数字坐标）")
    from langchain_community.embeddings import DashScopeEmbeddings

    for name in EMBED_CANDIDATES:
        try:
            emb = DashScopeEmbeddings(model=name, dashscope_api_key=settings.DASHSCOPE_API_KEY)
            vec = emb.embed_query("这款手机的电池容量是多少")
            print(f"  ✅ {name:<20} 可用 —— 生成了 {len(vec)} 维向量")
            return name
        except Exception as e:
            print(f"  ❌ {name:<20} 不可用 —— {brief(e)}")
    return None


def test_rerank() -> str | None:
    """测试重排序模型：给几段文字打分排序，看能不能通"""
    header("三、重排序模型测试（对检索结果精排，提升准确率）")

    # 直接调 dashscope 原生 SDK，避免 LangChain 包装层带来的额外变量
    try:
        import dashscope
        from dashscope import TextReRank
    except Exception as e:
        print(f"  ❌ 无法导入 dashscope 重排序模块 —— {brief(e)}")
        return None

    dashscope.api_key = settings.DASHSCOPE_API_KEY

    documents = [
        "iPhone 16 的电池容量是 3561mAh。",
        "这款笔记本电脑重 1.8 公斤，适合商务出行。",
        "本店支持 7 天无理由退货。",
    ]
    query = "iPhone 16 电池多大"

    for name in RERANK_CANDIDATES:
        try:
            resp = TextReRank.call(
                model=name,
                query=query,
                documents=documents,
                top_n=3,
                return_documents=False,
            )
            if resp.status_code == 200:
                results = resp.output.results
                order = [(r.index, round(r.relevance_score, 4)) for r in results]
                print(f"  ✅ {name:<16} 可用")
                print(f"     排序结果 (原文序号, 相关度): {order}")
                print(f"     → 最相关的是第 {order[0][0]} 条: 「{documents[order[0][0]][:30]}...」")
                return name
            else:
                print(f"  ❌ {name:<16} 返回错误码 {resp.status_code} —— {str(resp.message)[:120]}")
        except Exception as e:
            print(f"  ❌ {name:<16} 不可用 —— {brief(e)}")
    return None


if __name__ == "__main__":
    print(f"\n{LINE}")
    print("  阿里云百炼 API 连通性测试".center(60))
    print(f"{LINE}")

    key = settings.DASHSCOPE_API_KEY
    print(f"\n  API Key : {key[:10]}...{key[-6:]}")
    print(f"  长度    : {len(key)} 个字符")

    chat_ok = test_chat()
    embed_ok = test_embedding()
    rerank_ok = test_rerank()

    # ---------------- 汇总 ----------------
    header("测试结果汇总")
    print(f"  对话模型  : {'✅ ' + chat_ok if chat_ok else '❌ 没有可用的'}")
    print(f"  向量化模型: {'✅ ' + embed_ok if embed_ok else '❌ 没有可用的'}")
    print(f"  重排序模型: {'✅ ' + rerank_ok if rerank_ok else '❌ 没有可用的'}")

    print(f"\n{LINE}")
    if chat_ok and embed_ok:
        print("  结论：核心能力可用 ✅")
        print("\n  请把下面三行填进 backend/.env 的对应位置：")
        print(f"      LLM_MODEL={chat_ok}")
        print(f"      EMBEDDING_MODEL={embed_ok}")
        print(f"      RERANK_MODEL={rerank_ok or '(重排序不可用，可暂时跳过这项优化)'}")
    else:
        print("  结论：核心能力不可用 ❌，请把上面的报错信息发给我。")
    print(f"{LINE}\n")
