r"""
语法自检 —— 写完代码立刻跑一遍，让编译器替我发现错误

【为什么需要这个脚本】
我在写中文字符串时反复犯同一个错误：在中文字符串里嵌了英文双引号。

    错误写法：description="适合"进来就问"的场景"      ← 语法错误
    正确写法：description="适合『进来就问』的场景"    ← 用中文书名号

这个错误每次都要等到运行程序时才暴露，白等好几分钟。
写个脚本，写完代码 1 秒内就能发现，成本几乎为零。

【它做什么】
把所有 .py 文件编译一遍。编译不通过的文件会带着错误位置报出来。
这比"跑起来试试看"快得多，也可靠得多——因为有些语法错误
只在特定代码路径才会触发，编译检查则一次全覆盖。

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe check_syntax.py

也可以带上路径参数只检查某个目录：
    ..\.venv\Scripts\python.exe check_syntax.py app/rag
"""

import py_compile
import sys
from pathlib import Path

# ------------------------------------------------------------------
# 强制标准输出使用 UTF-8。
#
# 【为什么必须加这两行】
# 中文版 Windows 的控制台默认是 GBK 编码，打印 ✅ 这类符号时
# 会直接抛 UnicodeEncodeError 把脚本搞崩——最后检查都没跑完就退出了。
#
# 这里主动把输出流重新配置成 UTF-8，比每次在命令行加环境变量更省事，
# 也避免别人拿到脚本后踩同样的坑。
# ------------------------------------------------------------------
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# 不检查的目录（虚拟环境、缓存、数据目录）
SKIP_DIRS = {".venv", "__pycache__", "data", "node_modules", ".git"}


def iter_py_files(root: Path) -> list[Path]:
    """递归找出所有需要检查的 .py 文件"""
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    if not root.exists():
        print(f"❌ 路径不存在: {root}")
        return 1

    files = iter_py_files(root)

    if not files:
        print(f"在 {root} 下没找到 .py 文件")
        return 0

    print(f"\n正在检查 {len(files)} 个 Python 文件…\n")

    errors: list[tuple[Path, str]] = []

    for path in files:
        try:
            # doraise=True 让它直接抛异常，而不是静默返回状态码
            py_compile.compile(str(path), doraise=True, cfile=None)
        except py_compile.PyCompileError as e:
            # 从异常里提取有用的错误信息
            msg = str(e)
            # 只保留最后几行（编译器的报错信息通常很长，前面是内部调用栈）
            lines = [ln for ln in msg.strip().split("\n") if ln.strip()]
            short = "\n".join(lines[-4:]) if lines else msg
            errors.append((path, short))
        except Exception as e:
            errors.append((path, f"{type(e).__name__}: {e}"))

    # ---------------- 报告结果 ----------------
    if not errors:
        print(f"  ✅ 全部通过（{len(files)} 个文件，语法均正确）\n")
        return 0

    print(f"  ❌ 发现 {len(errors)} 个文件有语法错误：\n")
    for path, msg in errors:
        print(f"  {'─' * 66}")
        print(f"  📄 {path}")
        print(f"  {'─' * 66}")
        for line in msg.split("\n"):
            print(f"     {line}")
        print()

    print(f"  {'=' * 66}")
    print(f"  提示：中文里嵌英文双引号是常见的错误来源。")
    print(f"        改成中文书名号『』或方括号【】即可避免。")
    print(f"  {'=' * 66}\n")

    return 1


if __name__ == "__main__":
    sys.exit(main())
