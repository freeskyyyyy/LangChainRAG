r"""
检索质量评估脚本 —— 本项目唯一的"考试成绩单"

【为什么要专门做一个评估脚本】
因为"感觉回答得还不错"是靠不住的。每次改动切片、换模型、调阈值之后，
必须用同一套题目重新考一遍，才知道到底是变好了还是变差了。
没有这把尺子，所有优化都是瞎猜。

【核心指标：为什么要分两层看】
假设你问"雪纺怎么洗"，系统返回的结果可能有两种"对法"：

  第一层 · 文件级命中：返回的是《洗涤养护.txt》里的内容
      → 太容易了。整个知识库只有3个文件，蒙也能蒙对三分之一。

  第二层 · 片段级命中：返回的那一小段**真的是讲雪纺的**，不是讲冰丝或羊毛的
      → 这才是真正有用的。因为引用卡片上显示的就是这一段，
        显示错了，用户看到的就是"答非所问"。

一份文档往往被切成几十片，所以第二层的难度高得多。
**只看第一层会严重高估系统能力** —— 我第一版就犯了这个错。

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe eval_retrieval.py
（需要后端服务已启动，知识库里有数据）

【输出】
    终端打印 + 写入 data/logs/检索评估报告.txt
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

BASE = "http://127.0.0.1:8000"
REPORT = Path(__file__).parent / "data" / "logs" / "检索评估报告.txt"

LINES: list[str] = []


def out(text: str = "") -> None:
    LINES.append(text)
    print(text)


# ==================================================================
# 题库
# ==================================================================
# 每条 = (问题, 期望文件名, 答案里必须出现的那句原话)
#
# 【为什么判据是"答案原话"而不是"关键词"】
#
# 我第一版用的是关键词（问真丝就检查片段里有没有"真丝"两个字），
# 结果测出 15/15 满分 —— 但那是**假的**。
#
# 因为旧切片切得太粗，有一片的内容长这样：
#     「4. 轻薄化纤材质（春季风衣、防晒衣）……
#       二、夏季服装（真丝、麻麻、冰丝、短袖纯棉、雪纺）
#       1. 真丝材质（夏季连衣裙、衬衫）……」
# 这一片**提到了**"真丝"（在夏季那节的标题里顺带列了个名字），
# 但它实际讲的是**化纤**。关键词检查一看有"真丝"就判命中，
# 于是把一个错的答案记成了对的。
#
# 后来改成检查"答案原话"：要答上"真丝怎么洗"，
# 片段里必须出现「真丝专用中性洗涤剂」这句**真正的答案**。
# 光被提名不算数。这样一来就骗不过去了。
#
# 这件事的教训：**判据太松，会把失败测成成功**，
# 比判据太严危险得多 —— 严了只是虚惊一场，松了会让你以为做完了。
#
# ==================================================================
# 【第二次修正：又有三条期望值本身写错了】
# ==================================================================
# 改完判据再跑，还是有三条"没命中"。我逐条去核对原文，发现**全是我的错**：
#
#   问题：夏天穿什么颜色比较凉快
#     我期望：「金属色配饰」          ← 这是第 6 节"配饰搭配"里的词，跟夏天无关
#     系统返回：第 4 节"季节与服装颜色选择…夏季：清爽降温"  ← 这才是对的
#
#   问题：参加宴会派对穿什么颜色
#     我期望：「超过三种颜色」        ← 这是第 5 节"搭配禁忌"里的词
#     系统返回：第 2 节"场合与服装颜色选择…宴会/派对：正红、酒红"  ← 这才是对的
#
#   问题：羊毛大衣能用洗衣机洗吗
#     我期望：「羊毛专用洗涤剂」      ← 这是"羊毛衫/羊绒大衣"那节的词
#     系统返回："厚羊毛大衣材质…必须干洗，禁止水洗、机洗"  ← 问"羊毛大衣"，
#                                       返回"厚羊毛大衣"这节才是最佳答案
#
# 三条都是**我把答案的出处张冠李戴了**，系统答得没错。
#
# 这已经是第二次栽在同一个坑里了：第一次是判据太松（把失败测成成功），
# 这次是题库标错了（把成功测成失败）。两次都不是代码的问题，是**尺子的问题**。
# 所以以后看到"没命中"，第一反应应该是去核对原文，而不是去改检索代码。
CASES: list[tuple[str, str, str]] = [
    # ---- 尺码对照表 ----
    ("我身高168 体重120斤，穿什么尺码", "尺码推荐", "165-175"),
    ("190cm 210斤以上穿什么码", "尺码推荐", "190"),
    ("体重100斤身高160，选哪个码", "尺码推荐", "90-115"),

    # ---- 颜色搭配（每条都对着原文核过出处）----
    ("黄皮适合穿什么颜色", "颜色选择", "豆沙色"),            # 第1节 肤色
    ("面试的时候穿什么颜色合适", "颜色选择", "商务会议"),      # 第2节 场合
    ("我腿粗想显瘦穿什么颜色", "颜色选择", "显瘦搭配"),        # 第3节 体型
    ("冬天适合穿什么颜色的衣服", "颜色选择", "樱花粉"),        # 第4节 季节
    ("夏天穿什么颜色比较凉快", "颜色选择", "清爽降温"),        # 第4节 季节
    ("参加宴会派对穿什么颜色", "颜色选择", "正红、酒红"),      # 第2节 场合

    # ---- 面料洗涤养护 ----
    ("真丝连衣裙怎么洗", "洗涤养护", "真丝专用中性洗涤剂"),
    ("羽绒服可以机洗吗", "洗涤养护", "羽绒服专用洗涤剂"),
    ("雪纺洗的时候能拧干吗", "洗涤养护", "禁止用力拧绞"),
    ("羊毛衫能用洗衣机洗吗", "洗涤养护", "羊毛专用洗涤剂"),    # 改问"羊毛衫"以对准羊毛那节
    ("灯芯绒外套怎么洗", "洗涤养护", "避免绒面起毛"),
    ("针织衫洗完怎么晾", "洗涤养护", "针织专用模式"),
    ("冰丝衣服能泡多久", "洗涤养护", "禁止长时间浸泡"),
    ("冬天的加绒牛仔裤怎么洗", "洗涤养护", "绒层脱落"),
]

# 知识库里根本没有的问题，正确答案是"明确拒答"，不能编
UNANSWERABLE: list[str] = [
    "公司食堂几点开饭",
    "这款手机电池多大",
    "你们发什么快递",
]


def login() -> str:
    req = urllib.request.Request(
        BASE + "/api/auth/login",
        data=json.dumps({"username": "admin", "password": "123456"}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())["access_token"]


# ------------------------------------------------------------------
# 限流保护
# ------------------------------------------------------------------
# 【为什么必须加这个】
# 后端有个限流器：**每个用户每分钟最多 20 次请求**（防止接口被刷爆）。
# 我第一版评估脚本一口气打了 33 次请求，超过 20 次之后，
# 后面的请求全被拒了（HTTP 429），返回空结果。
# 而脚本看到"空结果"就当成"没有召回"，**静默地**把分数统计成了空，
# 最后报出一份看起来正常、实际残缺的报告。
#
# 这种错最危险 —— 它不报错、不崩溃，就是悄悄给你错的数字。
#
# 【解决办法】
# 每次请求之间强制间隔，把速度压在限流线以下。
# 顺便在收到 429 时等一下重试，双保险。
_last_call = [0.0]
MIN_INTERVAL = 3.2        # 秒。60 / 3.2 ≈ 18 次每分钟，留一点余量


def _throttle() -> None:
    """让两次请求之间至少隔 MIN_INTERVAL 秒"""
    elapsed = time.monotonic() - _last_call[0]
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last_call[0] = time.monotonic()


def search(query: str, token: str, top_k: int = 5) -> dict:
    boundary = "----EvalBoundary5566"
    body = b""
    for k, v in (("query", query), ("top_k", str(top_k)), ("compare", "false")):
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
        body += f"{v}\r\n".encode()
    body += f"--{boundary}--\r\n".encode()

    for attempt in (1, 2):
        _throttle()
        req = urllib.request.Request(
            BASE + "/api/kb/search",
            data=body,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Authorization": f"Bearer {token}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt == 1:
                # 被限流了 → 等一分钟再试一次
                print("     （触发限流，等 62 秒后重试）")
                time.sleep(62)
                continue
            return {"error": f"HTTP {e.code}"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    return {"error": "重试后仍然失败"}


def count_sections(text: str) -> int:
    """
    数一数这一段里包含几个"编号小节标题"（1. xxx / 2. xxx / 一、xxx）。

    【这个数字代表什么】
    代表引用卡片会给用户看多少"不相关的信息"。

        1  = 干干净净，这段就讲一件事          ← 理想
        3  = 三种面料混在一起，答案埋在中间     ← 用户得自己找

    【为什么需要这个指标】
    "答案在不在这一段里"这个判据，区分不出好坏 —— 因为
    500 字的大片段里**恰好也能包含正确答案那句话**（它把整个真丝小节都包进去了，
    只是同时还包了冰丝和麂皮绒）。所以大小两版都能过这一关，测出来数字一模一样。

    但用户实际的感受差别很大：
        旧版引用卡片：一大段，里面三种面料，你得自己找哪句是真丝
        新版引用卡片：就一小节，标题写着"4. 雪纺材质"

    这个指标量的就是这个差别。数字越小越好，理想值是 1。
    """
    import re as _re
    return len(_re.findall(r"^\s*(?:\d+[.．]|[一二三四五六七八九十]+、|#{1,6}\s)", text, _re.M))


def main() -> int:
    token = login()
    out("=" * 76)
    out("  检索质量评估报告")
    out("=" * 76)

    # ---------------- 一、文件级命中 ----------------
    out("\n【第一层】文件级命中：返回的内容来自正确的文件吗？")
    out("-" * 76)
    out("  这一层很简单（知识库只有3个文件），但它是第二层的前提。\n")

    file_hit = 0
    sec_hit = 0
    total = len(CASES)
    details: list[str] = []

    # 顺手把每条问题的最高分记下来，后面画分数分布要用。
    # （第一版是单独再循环一遍重新请求，白白多打一轮请求，
    #   正好撞上限流，把分数统计打残了。一次循环干两件事就好。）
    answ_scores: list[float] = []
    focus_counts: list[int] = []      # 第一名片段里含几个小节（越少越聚焦）
    top1_lengths: list[int] = []      # 第一名片段的字数

    for q, want_file, want_kw in CASES:
        res = search(q, token, top_k=5)
        results = res.get("results", [])
        if not results:
            details.append(f"  ❌ {q}\n     └─ 完全没有召回任何内容")
            continue

        top = results[0]
        name = top.get("file_name", "")

        if top.get("rerank_score") is not None:
            answ_scores.append(top["rerank_score"])

        _top_text = top.get("content") or ""
        focus_counts.append(count_sections(_top_text))
        top1_lengths.append(len(_top_text))

        ok_file = want_file in name
        if ok_file:
            file_hit += 1

        # 第二层：正确的关键词有没有出现在**第一名**那一小段里
        top_text = top.get("content") or ""
        ok_sec_top1 = want_kw in top_text

        # 放宽版：前5名里有没有任何一段包含关键词（只要能捞回来就还有救）
        any_text = " ".join((r.get("content") or "") for r in results)
        ok_sec_any = want_kw in any_text

        if ok_sec_top1:
            sec_hit += 1

        mark = "✅" if ok_sec_top1 else ("🟡" if ok_sec_any else "❌")
        details.append(
            f"  {mark} {q}\n"
            f"     └─ 期望：{want_file} 且含「{want_kw}」\n"
            f"     └─ 实际第一名：{name} | 重排 {top.get('rerank_score') or '-'}\n"
            f"     └─ 文件{'对' if ok_file else '错'} | "
            f"片段{'对' if ok_sec_top1 else '错'}"
            f"{'（但前5名里有正确段落）' if ok_sec_any and not ok_sec_top1 else ''}"
        )
        if not ok_sec_top1:
            head = top_text[:70].replace("\n", " ")
            details.append(f"     └─ 第一名实际内容：{head}…")

    for d in details:
        out(d)

    # ---------------- 二、拒答能力 ----------------
    out("\n【第二层】拒答能力：问知识库里没有的东西，会不会硬编？")
    out("-" * 76)
    out("  判断依据：第一名的重排序分数够不够低。分数低 → 系统会拒答，正确。\n")

    refused = 0
    refuse_scores: list[float] = []
    for q in UNANSWERABLE:
        res = search(q, token, top_k=1)
        results = res.get("results", [])
        if not results:
            out(f"  ✅ {q}  → 零召回（会被拒答）")
            refused += 1
            continue
        sc = results[0].get("rerank_score")
        refuse_scores.append(sc or 0.0)
        # 阈值从配置读，这里先用 0.15（.env 里的值）
        will_refuse = (sc or 0) < 0.15
        if will_refuse:
            refused += 1
        out(f"  {'✅' if will_refuse else '❌'} {q}\n"
            f"     └─ 最高分 {sc:.4f} → "
            f"{'低于阈值，会拒答' if will_refuse else '高于阈值，会硬答（有编造风险）'}")

    # ---------------- 三、分数分布 ----------------
    out("\n【第三层】分数分布：有答案的和没答案的，分得开吗？")
    out("-" * 76)

    def rng(xs: list[float]) -> str:
        return f"{min(xs):.4f} ~ {max(xs):.4f}" if xs else "（无数据）"

    out(f"\n  有答案的问题  最高分范围  {rng(answ_scores)}   （{len(answ_scores)} 个）")
    out(f"  无答案的问题  最高分范围  {rng(refuse_scores)}   （{len(refuse_scores)} 个）")

    if answ_scores and refuse_scores:
        gap = min(answ_scores) - max(refuse_scores)
        out(f"\n  有答案最低 {min(answ_scores):.4f} − 无答案最高 {max(refuse_scores):.4f} "
            f"= {gap:+.4f}")
        if gap > 0.05:
            mid = (min(answ_scores) + max(refuse_scores)) / 2
            out(f"  ✅ 分得开。阈值取中间值 {mid:.3f} 最稳；")
            out(f"     当前 .env 阈值是 0.15，"
                f"{'偏高，建议下调' if mid < 0.13 else '基本合适'}")
        else:
            out("  ⚠️  挨得太近，阈值不好定，需要改善检索质量")

    # ---------------- 四、总成绩 ----------------
    out("\n" + "=" * 76)
    out("  总成绩")
    out("=" * 76)
    out(f"\n  ① 文件级命中（找对文件了吗）      {file_hit}/{total}  "
        f"（{file_hit / total * 100:.0f}%）")
    out(f"  ② 答案级命中（找到答案原话了吗）  {sec_hit}/{total}  "
        f"（{sec_hit / total * 100:.0f}%）  ← 这个才是真实水平")
    out(f"  ③ 拒答正确（不知道的会说不知道）  {refused}/{len(UNANSWERABLE)}  "
        f"（{refused / len(UNANSWERABLE) * 100:.0f}%）")

    if focus_counts:
        avg_focus = sum(focus_counts) / len(focus_counts)
        avg_len = sum(top1_lengths) / len(top1_lengths)
        clean = sum(1 for n in focus_counts if n <= 1)
        out(f"  ④ 引用聚焦度（越接近 1 越好）     平均 {avg_focus:.2f} 个小节"
            f"（理想值 1.0）")
        out(f"     第一名片段平均 {avg_len:.0f} 字 | "
            f"干干净净只有一个小节的：{clean}/{len(focus_counts)}")

    out("\n  【怎么读这四个数字】")
    out("    ① 高、② 低 = 找对了文件，但返回的那一段不是讲这个问题的。")
    out("       用户看到的表现就是：引用卡片上的原文跟回答对不上。")
    out("    ③ 低 = 问知识库里没有的东西，它硬编了一个答案。")
    out("\n" + "=" * 76 + "\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(LINES), encoding="utf-8")
    print(f"\n报告已写入：{REPORT}\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n中断。\n")
        sys.exit(130)
    except Exception as exc:
        print(f"\n❌ 异常: {type(exc).__name__}: {exc}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
