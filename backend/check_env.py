r"""
环境自检脚本 —— 确认依赖装好了、并摸清各组件的真实导入路径

【为什么需要这个脚本】
LangChain 已经升级到 1.x 大版本，模块结构相比老教程（0.x）变化很大。
网上大量文章还在用 `from langchain_community.chat_models import ChatTongyi`
这类老写法，在 1.x 上可能已经失效。

这个脚本不猜、不照抄教程，而是**直接扫描你本地实际装好的包**，
把"阿里云百炼的对话/向量化/重排序到底该从哪里导入"这个问题查清楚。

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe check_env.py
"""

import importlib
import pkgutil
import sys
import traceback

# ==================================================================
# 想找的目标：类名 → 我们期望它们在哪
# ==================================================================
TARGETS = {
    "ChatTongyi": "阿里云百炼的对话模型（负责生成回答）",
    "DashScopeEmbeddings": "阿里云百炼的向量化模型（把文字变成数字坐标）",
    "DashScopeRerank": "阿里云百炼的重排序模型（对检索结果精排）",
}

# 优先尝试的候选路径（从最可能到最不可能）
CANDIDATE_PATHS = [
    "langchain_community.chat_models",
    "langchain_community.chat_models.tongyi",
    "langchain_community.embeddings",
    "langchain_community.embeddings.dashscope",
    "langchain_community.document_compressors",
    "langchain_community.document_compressors.dashscope_rerank",
    "langchain_dashscope",
    "langchain_aliyun",
    "langchain_community.rerank",
]

# 需要检查的基础包
BASE_PACKAGES = [
    "fastapi", "uvicorn", "sqlalchemy", "aiosqlite", "pydantic",
    "pydantic_settings", "bcrypt", "jwt", "loguru", "cachetools",
    "sse_starlette", "httpx", "multipart",
    "langchain", "langchain_core", "langchain_community",
    "langchain_text_splitters", "langchain_chroma", "chromadb",
    "dashscope", "rank_bm25", "pypdf", "docx", "openpyxl",
    "markdown", "bs4", "pandas",
]


def line(char: str = "-", n: int = 68) -> None:
    print(char * n)


def header(text: str) -> None:
    print()
    line("=")
    print(f"  {text}")
    line("=")


def check_python() -> None:
    header("一、Python 环境")
    print(f"  版本    : {sys.version.split()[0]}")
    print(f"  解释器  : {sys.executable}")
    print(f"  虚拟环境: {'是（正确）' if 'venv' in sys.executable.lower() else '否（警告！应使用 .venv 里的解释器）'}")


def check_base_packages() -> dict:
    header("二、基础依赖包检查")
    ok, missing = [], []
    for name in BASE_PACKAGES:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "")
            ok.append((name, ver))
        except Exception as e:
            missing.append((name, f"{type(e).__name__}: {e}"))

    print(f"  ✅ 已装好 {len(ok)} 个：")
    for name, ver in ok:
        print(f"       {name:<28} {ver}")

    if missing:
        print(f"\n  ❌ 缺失 {len(missing)} 个：")
        for name, err in missing:
            print(f"       {name:<28} {err}")
    else:
        print("\n  🎉 全部依赖都已就位！")

    return {"ok": ok, "missing": missing}


def find_targets() -> None:
    header("三、查找百炼三大组件的真实导入路径")
    print("  （这是关键——LangChain 1.x 的模块结构可能和老教程不同）\n")

    found: dict[str, str] = {}

    # ---- 第一步：试已知候选路径 ----
    for path in CANDIDATE_PATHS:
        try:
            mod = importlib.import_module(path)
        except Exception:
            continue
        for cls_name in TARGETS:
            if cls_name in found:
                continue
            if hasattr(mod, cls_name):
                found[cls_name] = f"{path}.{cls_name}"
                print(f"  ✅ 在候选路径找到: {path}.{cls_name}")

    # ---- 第二步：还没找到的，就全盘扫描所有 langchain* / dashscope* 包 ----
    still_missing = [c for c in TARGETS if c not in found]
    if still_missing:
        print(f"\n  候选路径没找全（还差 {len(still_missing)} 个），开始全盘扫描已安装的包……")
        for finder, name, _ispkg in pkgutil.iter_modules():
            if not (name.startswith("langchain") or name.startswith("dashscope")):
                continue
            try:
                mod = importlib.import_module(name)
            except Exception:
                continue
            for cls_name in still_missing:
                if cls_name in found:
                    continue
                if hasattr(mod, cls_name):
                    found[cls_name] = f"{name}.{cls_name}"
                    print(f"  ✅ 扫描找到: {name}.{cls_name}")
            still_missing = [c for c in TARGETS if c not in found]
            if not still_missing:
                break

    # ---- 汇总 ----
    print()
    line()
    for cls_name, desc in TARGETS.items():
        if cls_name in found:
            print(f"  ✅ {cls_name}")
            print(f"     用途: {desc}")
            print(f"     导入: from {found[cls_name].rsplit('.', 1)[0]} import {cls_name}")
        else:
            print(f"  ❌ {cls_name} —— 未找到！")
            print(f"     用途: {desc}")
            print(f"     说明: 可能需要额外安装包，或已改名为别的组件")
        print()


def check_dashscope_key() -> None:
    header("四、检查阿里云百炼配置")
    try:
        from app.core.config import settings
    except Exception as e:
        print(f"  ⚠️ 无法加载配置文件: {type(e).__name__}: {e}")
        return

    key = settings.DASHSCOPE_API_KEY
    if not key or key.startswith("sk-请填入"):
        print("  ❌ 未配置 DASHSCOPE_API_KEY，请检查 backend/.env")
        return

    masked = f"{key[:8]}...{key[-6:]}" if len(key) > 20 else "(过短)"
    print(f"  ✅ API Key 已配置: {masked}")
    print(f"     对话模型  : {settings.LLM_MODEL}")
    print(f"     向量化模型: {settings.EMBEDDING_MODEL}")
    print(f"     重排序模型: {settings.RERANK_MODEL}")


if __name__ == "__main__":
    print()
    print("╔" + "═" * 66 + "╗")
    print("║" + "  环境自检 —— 确认依赖与组件导入路径".center(56) + "║")
    print("╚" + "═" * 66 + "╝")

    check_python()

    result = check_base_packages()

    find_targets()

    check_dashscope_key()

    print()
    line("=")
    if result["missing"]:
        print("  结论：有依赖没装好，请把上面的 ❌ 部分发给我。")
    else:
        print("  结论：环境检查通过 ✅  可以进入下一步开发。")
    line("=")
    print()
