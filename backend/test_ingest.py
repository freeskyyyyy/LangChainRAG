r"""
入库流水线测试：解析 → 清洗 → 切分

【测什么】
验证前三个 RAG 模块能否正确处理真实文件，并输出质量报告。
如果切分质量不好（比如碎片太多），后面的检索效果一定差，
所以这一步必须先看数据。

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe test_ingest.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from app.rag.cleaner import TextCleaner  # noqa: E402
from app.rag.loader import DocumentParser, get_parser_info  # noqa: E402
from app.rag.splitter import ParentChildSplitter  # noqa: E402

LINE = "=" * 70

# ------------------------------------------------------------------
# 造一份仿真商品资料（Markdown 格式）
# ------------------------------------------------------------------
SAMPLE_MD = """# iPhone 16 Pro 商品资料

## 一、产品概述

iPhone 16 Pro 是苹果公司于 2024 年 9 月发布的旗舰智能手机，
采用钛金属边框设计，搭载 A18 Pro 芯片，定位高端商务与影像创作人群。
本机型提供原色钛金属、沙漠色钛金属、白色钛金属、黑色钛金属四种配色。

## 二、核心参数

| 参数项 | 规格 |
| --- | --- |
| 屏幕尺寸 | 6.3 英寸 Super Retina XDR 显示屏 |
| 分辨率 | 2622 x 1206 像素，460 ppi |
| 处理器 | A18 Pro 芯片，6 核 CPU + 6 核 GPU |
| 运行内存 | 8GB |
| 存储容量 | 128GB / 256GB / 512GB / 1TB |
| 后置摄像头 | 4800 万像素主摄 + 4800 万超广角 + 1200 万长焦 |
| 前置摄像头 | 1200 万像素原深感摄像头 |
| 电池容量 | 3561 mAh |
| 充电功率 | 有线 20W，MagSafe 无线 25W |
| 防水等级 | IP68（可在 6 米深水中停留 30 分钟） |
| 机身重量 | 199 克 |
| 操作系统 | iOS 18 |

## 三、电池与续航说明

iPhone 16 Pro 内置 3561mAh 锂离子电池。在标准测试条件下，
视频播放最长可达 27 小时，音频播放最长可达 85 小时。

使用 20W 或更高功率的电源适配器，约 30 分钟可充至 50% 电量。
需要注意的是，包装盒内**不附带充电器**，只有一根 USB-C 转 USB-C 的线缆。
用户需要自行准备支持 USB-PD 协议的充电头。

关于电池健康度：正常使用情况下，电池在 500 次完整充放电循环后，
容量会保持在原始容量的 80% 左右。若电池健康度低于 80%，
系统会提示"电池需要维修"。

## 四、常见问题

**问：iPhone 16 Pro 支持双卡双待吗？**
答：支持。国行版本支持双 nano-SIM 卡，同时也支持 eSIM。

**问：这款手机防水吗？可以带去游泳吗？**
答：具备 IP68 级防水防尘能力，可防日常泼溅、雨水和短暂浸泡。
但苹果官方明确说明：**防水性能并非永久有效**，且不建议带入泳池或海水，
因为水中的化学物质和高速水流会破坏密封结构。进水损坏不在保修范围内。

**问：包装盒里都有什么？**
答：包装内含：iPhone 16 Pro 手机一台、USB-C 转 USB-C 编织线缆一根、
说明书和保修卡。不含充电器、不含耳机。

**问：支持无线充电吗？**
答：支持。支持 Qi2 标准和 MagSafe 磁吸无线充电，
使用 MagSafe 充电器时最高功率 25W。

## 五、售后与保修政策

本商品享受国家三包政策。自签收之日起 7 日内，
商品出现性能故障可申请退货或换货；15 日内可申请换货或维修。

整机保修期为 1 年，主要部件保修期为 2 年。
保修范围不包括：人为损坏、进液、私自拆机、使用非原装配件造成的损坏。

如需申请售后服务，请通过官方客服热线或线下授权服务网点办理，
办理时需提供购买凭证（订单截图或发票）。

## 六、价格与促销

iPhone 16 Pro 官方指导价：
- 128GB 版本：7999 元
- 256GB 版本：8999 元
- 512GB 版本：10999 元
- 1TB 版本：12999 元

促销活动：以旧换新最高可抵扣 3000 元；
学生认证用户可享 9.5 折优惠。活动详情请咨询在线客服。
"""

# 造一份带"页眉页脚"的仿真多页文档，用来测试页眉页脚识别功能
SAMPLE_PAGES = [
    f"""XX公司内部资料 第{i}页 共3页
产品型号：华为 MateBook X Pro
屏幕：14.2 英寸 3.1K 触控屏
重量：1.38 公斤
{i}
XX公司内部资料 第{i}页 共3页"""
    for i in range(1, 4)
]


def main() -> int:
    print(f"\n{LINE}")
    print("  入库流水线测试：解析 → 清洗 → 切分".center(60))
    print(f"{LINE}")

    tmpdir = Path(tempfile.mkdtemp(prefix="rag_test_"))
    ok = True

    # ==============================================================
    # 第一步：解析
    # ==============================================================
    print(f"\n{'-' * 70}\n  一、文档解析\n{'-' * 70}")

    md_file = tmpdir / "iPhone16Pro商品资料.md"
    md_file.write_text(SAMPLE_MD, encoding="utf-8")

    try:
        docs = DocumentParser.parse(md_file)
        print(f"  ✅ Markdown 解析成功：产出 {len(docs)} 个文档块")
        for d in docs:
            sec = d.metadata.get("section", "")
            print(f"     · 「{sec}」 {len(d.page_content)} 字")
    except Exception as e:
        print(f"  ❌ Markdown 解析失败: {type(e).__name__}: {e}")
        return 1

    # 测试不支持格式的报错
    bad_file = tmpdir / "test.xyz"
    bad_file.write_text("hello")
    try:
        DocumentParser.parse(bad_file)
        print("  ❌ 不支持的格式竟然没报错")
        ok = False
    except Exception as e:
        print(f"  ✅ 不支持的格式被正确拒绝（{type(e).__name__}）")

    print(f"\n  支持的格式：{', '.join(get_parser_info()['supported_extensions'])}")

    # ==============================================================
    # 第二步：清洗
    # ==============================================================
    print(f"\n{'-' * 70}\n  二、文本清洗\n{'-' * 70}")

    cleaner = TextCleaner()

    # 测试单块清洗
    dirty = "iPhone​16 的电池容量为 3561mAh。   \n- 3 -\n=======\n第二行内容。"
    cleaned = cleaner.clean_text(dirty)
    print("  原始文本（含零宽空格、页码、分隔线）:")
    print(f"     {dirty!r}")
    print("  清洗后:")
    print(f"     {cleaned!r}")
    checks = [
        ("删除零宽空格", "​" not in cleaned),
        ("删除页码行", "- 3 -" not in cleaned),
        ("删除分隔线", "=====" not in cleaned),
        ("保留正文", "3561mAh" in cleaned and "第二行内容" in cleaned),
    ]
    for name, cond in checks:
        print(f"     {'✅' if cond else '❌'} {name}")
        ok = ok and cond

    # 测试页眉页脚识别
    #
    # 【注意这里为什么要手动构造 Document 列表】
    # 页眉页脚识别的前提是"有多页可比较"。
    # 如果写成一个 .txt 文件，解析出来是**一个**文档块，就没有"多页"可言了。
    # 所以这里直接手工构造 3 个 Document，模拟 PDF 解析出来的一页一个块。
    from langchain_core.documents import Document

    page_docs = [
        Document(page_content=text, metadata={"page": i, "source": "华为笔记本资料.pdf"})
        for i, text in enumerate(SAMPLE_PAGES, start=1)
    ]

    cleaned_docs = cleaner.clean_documents(page_docs)
    combined = " ".join(d.page_content for d in cleaned_docs)

    hf_removed = "XX公司内部资料" not in combined
    content_kept = "MateBook X Pro" in combined and "1.38 公斤" in combined
    print(f"\n  页眉页脚识别测试（模拟 3 页 PDF）:")
    print(f"     {'✅' if hf_removed else '❌'} 重复的页眉已被删除")
    print(f"     {'✅' if content_kept else '❌'} 正文内容被保留")
    print(f"     清洗后内容: {combined[:80]}...")
    ok = ok and hf_removed and content_kept

    # ==============================================================
    # 第三步：切分
    # ==============================================================
    print(f"\n{'-' * 70}\n  三、父子文档切分\n{'-' * 70}")

    splitter = ParentChildSplitter()
    chunks = splitter.split(docs, doc_id=1)

    if not chunks:
        print("  ❌ 切分没有产出任何内容")
        return 1

    stats = splitter.analyze_chunks(chunks)
    print(f"  ✅ 切分完成")
    print(f"     子块总数    : {stats['total_chunks']}")
    print(f"     父块总数    : {stats['total_parents']}")
    print(f"     子块平均字数: {stats['child_avg_chars']}（配置目标 {splitter.child_size}）")
    print(f"     子块字数范围: {stats['child_min_chars']} ~ {stats['child_max_chars']}")
    print(f"     父块平均字数: {stats['parent_avg_chars']}（配置目标 {splitter.parent_size}）")
    print(f"     平均每父块含: {stats['avg_children_per_parent']} 个子块")
    print(f"     过小碎片比例: {stats['tiny_ratio']}")

    # 质量判断
    if stats["child_avg_chars"] < splitter.child_size * 0.5:
        print("  ⚠️  子块平均字数偏小，可能有切分参数需要调整")
    else:
        print("  ✅ 切分质量正常")

    # 展示几个真实片段，让用户直观看到切分效果
    print(f"\n  切分效果抽样（前 3 个子块）:")
    for c in chunks[:3]:
        print(f"\n     ┌─ 子块 #{c.child_index}（属于父块 #{c.parent_index}）")
        print(f"     │  向量ID: {c.vector_id[:16]}...")
        print(f"     │  字数: {len(c.child_content)}")
        body = c.child_content.replace("\n", " ")[:110]
        print(f"     └─ 内容: {body}{'...' if len(c.child_content) > 110 else ''}")

    # 验证父子关系是否正确
    print(f"\n  父子关系验证:")
    parent_indexes = {c.parent_index for c in chunks}
    multi_child_parents = [
        p for p in parent_indexes
        if sum(1 for c in chunks if c.parent_index == p) > 1
    ]
    print(f"     {'✅' if multi_child_parents else '⚠️ '} "
          f"{len(multi_child_parents)} 个父块包含多个子块（父子策略生效）")

    # 验证同一个父块下的子块共享相同的父块内容
    if multi_child_parents:
        sample_parent = multi_child_parents[0]
        same_parent_children = [c for c in chunks if c.parent_index == sample_parent]
        all_same = len({c.parent_content for c in same_parent_children}) == 1
        print(f"     {'✅' if all_same else '❌'} 同一父块下的子块共享相同的父块文本")
        ok = ok and all_same

    # 验证向量 ID 唯一性
    ids = [c.vector_id for c in chunks]
    unique_ok = len(ids) == len(set(ids))
    print(f"     {'✅' if unique_ok else '❌'} 向量 ID 唯一（不会互相覆盖）")
    ok = ok and unique_ok

    # ==============================================================
    # 质量分析
    # ==============================================================
    print(f"\n{'-' * 70}\n  四、文本质量分析\n{'-' * 70}")
    full_text = "\n".join(d.page_content for d in docs)
    analysis = cleaner.analyze_quality(full_text)
    for k, v in analysis.items():
        print(f"     {k:20} {v}")

    # ==============================================================
    # 汇总
    # ==============================================================
    print(f"\n{LINE}")
    if ok:
        print("  ✅ 入库流水线测试通过")
        print("     解析 → 清洗 → 切分 三个环节都工作正常。")
    else:
        print("  ⚠️  有检查项未通过，请看上面的 ❌")
    print(f"{LINE}\n")

    # 清理临时文件
    for f in tmpdir.glob("*"):
        f.unlink()
    tmpdir.rmdir()

    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n❌ 测试异常: {type(exc).__name__}: {exc}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
