r"""
RAG 组件 API 探测脚本

【为什么先写这个】
LangChain 1.4.0 是个大版本，模块结构和网上多数教程（0.x）差别很大。
如果直接照着自己的记忆写业务代码，很可能写完 500 行才发现
"这个类根本不在这个模块里"或者"参数名变了"，返工代价很高。

所以先用这个脚本把要用到的每个组件"摸一遍家底"：
  · 类在不在？
  · 构造参数叫什么？
  · 有哪些方法？

确认之后再写业务代码，一次就能跑通。

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe check_rag_api.py
"""

import inspect
import sys
import traceback

sys.path.insert(0, ".")

LINE = "=" * 70
OK = 0
BAD = 0


def probe(title: str, importer):
    """探测一项：导入成功就打印它有什么方法/参数"""
    global OK, BAD
    print(f"\n{'-' * 70}\n  {title}\n{'-' * 70}")
    try:
        obj = importer()
        OK += 1
        print("  ✅ 导入成功")

        if inspect.isclass(obj):
            # 打印构造参数
            try:
                sig = inspect.signature(obj.__init__)
                params = [
                    p.name
                    for p in sig.parameters.values()
                    if p.name not in ("self", "args", "kwargs")
                ]
                if params:
                    print(f"     构造参数: {', '.join(params[:14])}")
            except (ValueError, TypeError):
                pass

            # 打印主要方法（过滤掉下划线开头的内部方法）
            methods = [
                m for m in dir(obj)
                if not m.startswith("_") and callable(getattr(obj, m, None))
            ]
            key_methods = [
                m for m in methods
                if any(k in m.lower() for k in (
                    "split", "search", "add", "delete", "invoke", "stream",
                    "embed", "similarity", "retrieve", "compress", "rerank",
                    "batch", "get", "update", "reset", "from_",
                ))
            ]
            if key_methods:
                print(f"     关键方法: {', '.join(key_methods[:14])}")
        return obj

    except Exception as e:
        BAD += 1
        print(f"  ❌ 失败: {type(e).__name__}: {e}")
        return None


def main() -> int:
    print(f"\n{LINE}")
    print("  RAG 组件 API 探测".center(60))
    print(f"{LINE}")

    # ---------------- 1. 核心数据结构 ----------------
    probe(
        "1. Document（文档对象：文本 + 元数据的容器）",
        lambda: __import__("langchain_core.documents", fromlist=["Document"]).Document,
    )

    probe(
        "2. 消息类型（HumanMessage / AIMessage / SystemMessage）",
        lambda: __import__("langchain_core.messages", fromlist=["HumanMessage"]).HumanMessage,
    )

    # ---------------- 2. 文本切分 ----------------
    splitter_cls = probe(
        "3. RecursiveCharacterTextSplitter（递归字符切分器）",
        lambda: __import__(
            "langchain_text_splitters", fromlist=["RecursiveCharacterTextSplitter"]
        ).RecursiveCharacterTextSplitter,
    )
    if splitter_cls:
        try:
            s = splitter_cls(chunk_size=100, chunk_overlap=20)
            pieces = s.split_text("这是一个测试。" * 50)
            print(f"     实测: 把 350 字切成 {len(pieces)} 段，首段长度 {len(pieces[0])}")
        except Exception as e:
            print(f"     ⚠️ 实测失败: {e}")

    # ---------------- 3. 向量库 ----------------
    probe(
        "4. Chroma（向量数据库封装）",
        lambda: __import__("langchain_chroma", fromlist=["Chroma"]).Chroma,
    )

    probe(
        "5. chromadb 原生客户端",
        lambda: __import__("chromadb", fromlist=["PersistentClient"]).PersistentClient,
    )

    # ---------------- 4. 模型接入 ----------------
    chat_cls = probe(
        "6. ChatTongyi（对话模型）",
        lambda: __import__(
            "langchain_community.chat_models", fromlist=["ChatTongyi"]
        ).ChatTongyi,
    )
    if chat_cls:
        try:
            sig = inspect.signature(chat_cls.__init__)
            for name in ("dashscope_api_key", "model", "temperature", "streaming"):
                mark = "✅" if name in sig.parameters else "❌"
                print(f"     {mark} 参数 '{name}'")
        except (ValueError, TypeError):
            pass

    probe(
        "7. DashScopeEmbeddings（向量化）",
        lambda: __import__(
            "langchain_community.embeddings", fromlist=["DashScopeEmbeddings"]
        ).DashScopeEmbeddings,
    )

    rerank_cls = probe(
        "8. DashScopeRerank（重排序）",
        lambda: __import__(
            "langchain_community.document_compressors", fromlist=["DashScopeRerank"]
        ).DashScopeRerank,
    )
    if rerank_cls:
        try:
            sig = inspect.signature(rerank_cls.__init__)
            for name in ("model", "top_n", "dashscope_api_key"):
                mark = "✅" if name in sig.parameters else "❌"
                print(f"     {mark} 参数 '{name}'")
        except (ValueError, TypeError):
            pass

    # ---------------- 5. 提示词与链 ----------------
    probe(
        "9. ChatPromptTemplate（提示词模板）",
        lambda: __import__(
            "langchain_core.prompts", fromlist=["ChatPromptTemplate"]
        ).ChatPromptTemplate,
    )

    probe(
        "10. StrOutputParser（把模型输出转成纯字符串）",
        lambda: __import__(
            "langchain_core.output_parsers", fromlist=["StrOutputParser"]
        ).StrOutputParser,
    )

    probe(
        "11. RunnableLambda（把普通函数接进链里）",
        lambda: __import__(
            "langchain_core.runnables", fromlist=["RunnableLambda"]
        ).RunnableLambda,
    )

    # ---------------- 6. 文档解析 ----------------
    probe("12. pypdf（PDF 解析）", lambda: __import__("pypdf", fromlist=["PdfReader"]).PdfReader)
    probe("13. python-docx（Word 解析）", lambda: __import__("docx", fromlist=["Document"]).Document)
    probe("14. openpyxl（Excel 解析）", lambda: __import__("openpyxl", fromlist=["load_workbook"]).load_workbook)

    # ---------------- 7. BM25 ----------------
    probe("15. BM25（关键词检索）", lambda: __import__("rank_bm25", fromlist=["BM25Okapi"]).BM25Okapi)

    # ---------------- 8. 中文分词（BM25 需要）----------------
    print(f"\n{'-' * 70}\n  16. 中文分词（BM25 检索中文必须用）\n{'-' * 70}")
    found_tokenizer = False
    for mod_name, attr in (("jieba", "lcut"), ("jieba", "cut")):
        try:
            mod = __import__(mod_name, fromlist=[attr])
            fn = getattr(mod, attr)
            words = fn("iPhone 16 的电池容量是多少")
            print(f"  ✅ {mod_name}.{attr} 可用")
            print(f"     实测分词: {list(words)}")
            found_tokenizer = True
            break
        except ImportError:
            continue
        except Exception as e:
            print(f"  ⚠️ {mod_name} 装了但报错: {e}")
            break
    if not found_tokenizer:
        print("  ❌ 没有找到中文分词库（jieba 未安装）")
        print("     → 影响：BM25 检索中文时只能按单字切分，准确率会下降")
        print("     → 解决：pip install jieba")

    # ---------------- 汇总 ----------------
    print(f"\n{LINE}")
    print(f"  探测完成：{OK} 项成功 / {BAD} 项失败")
    print(f"{LINE}")

    if BAD:
        print("\n  ⚠️ 有组件导入失败，说明我的代码计划需要调整。")
        print("     请把上面的 ❌ 部分发给我。\n")
    else:
        print("\n  ✅ 所有组件都可用，可以开始写 RAG 代码了。\n")

    return 0 if BAD == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
