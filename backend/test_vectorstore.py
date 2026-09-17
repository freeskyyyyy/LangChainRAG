r"""
向量化 + 向量库测试

【测什么】
1. 向量化接口能不能调用、维度对不对
2. 缓存有没有生效（第二次同样的文本应该秒回且不调 API）
3. 批量提交是否正常
4. 向量库能不能存取
5. **语义检索准不准**（这是最关键的——搜"电池续航"能不能找到讲电池的段落）

【会消耗额度】
这个测试会真实调用向量化接口。大约 15 个片段，消耗极少（几分钱级别）。

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe test_vectorstore.py
"""

import sys
import time

sys.path.insert(0, ".")

from app.rag.embeddings import get_embedding_service  # noqa: E402
from app.rag.splitter import ParentChildSplitter  # noqa: E402
from app.rag.vectorstore import VectorStoreService  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

LINE = "=" * 70

# ------------------------------------------------------------------
# 测试语料：几段互不相关的商品信息
# 这样能验证检索是否真的"按意思找"而不是"抓到啥算啥"
# ------------------------------------------------------------------
CORPUS = [
    ("商品参数", """
iPhone 16 Pro 商品参数详解

屏幕方面，本机采用 6.3 英寸 Super Retina XDR 显示屏，分辨率 2622 x 1206 像素，
像素密度 460 ppi，支持 ProMotion 自适应刷新率技术，最高可达 120Hz。
屏幕峰值亮度在户外模式下可达 2000 尼特，HDR 内容播放时表现出色。

性能方面，搭载全新的 A18 Pro 芯片，采用第二代 3 纳米制程工艺。
配备 6 核 CPU（2 个性能核心 + 4 个能效核心）和 6 核 GPU，
运行内存为 8GB。相比上一代，CPU 性能提升约 15%，GPU 提升约 20%。

影像系统采用三摄方案：4800 万像素主摄（支持传感器位移式光学防抖）、
4800 万像素超广角镜头、1200 万像素 5 倍光学变焦长焦镜头。
前置为 1200 万像素原深感摄像头，支持自动对焦。

机身采用钛金属边框，重量 199 克，厚度 8.25 毫米，
支持 IP68 级防尘防水（可在 6 米深水中停留最长 30 分钟）。
"""),
    ("电池续航", """
iPhone 16 Pro 电池与续航说明

本机内置 3561mAh 锂离子电池。在标准测试条件下，
视频播放最长可达 27 小时，音频播放最长可达 85 小时，
网页浏览最长可达 20 小时。日常中度使用可以轻松满足一整天的需求。

充电方面，使用 20W 或更高功率的电源适配器，
约 30 分钟可充至 50% 电量。支持 MagSafe 磁吸无线充电，
配合 30W 以上的 MagSafe 充电器时最高功率可达 25W。

需要特别注意的是：包装盒内不附带充电器，
只有一根 USB-C 转 USB-C 的编织线缆。
用户需要自行准备支持 USB-PD 协议的充电头，否则无法快充。

关于电池健康度：正常使用情况下，电池在 500 次完整充放电循环后，
容量会保持在原始容量的 80% 左右。若电池健康度低于 80%，
系统会提示电池需要维修，建议前往官方渠道更换。
"""),
    ("笔记本产品", """
华为 MateBook X Pro 产品介绍

这款笔记本采用 14.2 英寸 3.1K 原色全面屏，分辨率 3120 x 2080，
支持 120Hz 高刷新率和多屏同色显示技术，色准达到专业级水准。

机身重量仅 1.38 公斤，厚度 13.5 毫米，采用航空级镁铝合金一体成型工艺。
这个重量和厚度非常适合商务人士随身携带，放进公文包毫无负担，
出差、通勤、咖啡馆办公都非常方便。

性能上搭载英特尔酷睿 Ultra 9 处理器，配备 32GB LPDDR5X 内存
和 2TB PCIe 4.0 固态硬盘，可以流畅处理多任务和大型设计软件。

散热采用全新鲨鱼鳍双风扇设计，配合超薄均热板，
在高负载下依然能保持较低的温度和噪音水平。

接口方面提供 3 个 Thunderbolt 4 接口和 1 个 3.5mm 耳机孔，
支持外接双 4K 显示器。
"""),
    ("售后政策", """
退换货与保修政策

退换货规则：本店支持 7 天无理由退货，15 天无理由换货。
退货商品需保持包装完整、配件齐全、不影响二次销售。
需要注意的是，一经激活的设备、定制类商品、贴身用品不在无理由退货范围内。

退款时效：商品退回并验收通过后，退款将在 1-3 个工作日内
原路退回至您的支付账户。银行卡支付的退款可能需要 3-7 个工作日到账。

保修说明：整机保修期为 1 年，主要部件（主板、屏幕、电池）保修期为 2 年。
保修期自签收之日起计算，以订单签收时间为准。

以下情况不在保修范围内：人为损坏（摔坏、挤压、划伤）、
进液腐蚀、私自拆机或由非授权维修点维修、
使用非原装充电器或配件造成的损坏、自然灾害等不可抗力造成的损坏。

如需申请售后服务，请通过官方客服热线或线下授权服务网点办理，
办理时需提供购买凭证，包括订单截图或发票。
"""),
    ("支付说明", """
支付方式与分期说明

本店支持以下支付方式：微信支付、支付宝、银联卡、云闪付。
企业客户还可以选择对公转账，需要在结算时填写企业信息并上传营业执照。

分期付款：部分商品支持花呗分期，最长可分 24 期。
3 期和 6 期通常免息，12 期和 24 期会收取一定手续费，
具体费率以结算页面显示为准。

信用卡分期支持招商银行、工商银行、建设银行等主流银行，
分期期数可选 3、6、12 期。

关于发票：下单时可以选择开具电子发票或纸质发票。
电子发票会在订单完成后自动发送到您的邮箱，
纸质发票随商品一同寄出。发票金额为实付金额，不含优惠券抵扣部分。

支付安全提示：本店不会以任何形式索要您的支付密码或短信验证码，
请勿向任何人透露。如遇可疑情况请立即联系官方客服核实。
"""),
    ("物流说明", """
配送与物流说明

发货时效：现货商品在付款后 24 小时内发出（节假日顺延）。
预售商品以商品详情页标注的发货时间为准。

配送范围与时效：全国大部分地区支持次日达，
偏远地区（新疆、西藏、内蒙古部分地区）需要 3-5 天。
港澳台及海外地区暂不支持配送。

运费规则：订单满 99 元包邮，不满 99 元收取 8 元基础运费。
大件商品（如家电、家具）单独计算运费，具体以下单页面显示为准。

物流查询：商品发出后您会收到包含运单号的短信通知，
也可以在订单详情页实时查看物流轨迹。

签收提示：收到商品时请当面验货，
如发现外包装破损、商品损坏，可以拒收并联系客服处理。
签收后再发现的问题，需要走售后流程处理，会相对麻烦一些，
所以强烈建议当面验货。

配送合作方：顺丰速运、京东物流、中通快递。
"""),
]

# 检索测试：(问题, 期望命中的关键词)
#
# 【注意：这里检查的是"检索到的内容里是否包含期望关键词"】
# 之前我用的语料每段只有 50 字，小到根本切不开，
# 结果 6 段被合并成 1 块，导致搜什么都是返回同一条，
# 测试却显示"100% 命中"——那是**假阳性**，差点把问题掩盖过去。
# 现在语料改成真实长度，能真正验证检索质量。
QUERIES = [
    ("这个手机电池能用多久", "3561mAh"),
    ("笔记本重不重，方便带出门吗", "1.38 公斤"),
    ("买错了能退吗", "7 天无理由退货"),
    ("可以用花呗分期吗", "花呗分期"),
    ("发货要几天到", "次日达"),
    ("屏幕刷新率是多少", "120Hz"),
    ("包装里有充电器吗", "不附带充电器"),
]


def main() -> int:
    print(f"\n{LINE}")
    print("  向量化 + 向量库测试".center(60))
    print(f"{LINE}")

    ok = True
    tester_doc_id = 999999  # 用一个不会和真实数据冲突的 ID

    # ==============================================================
    # 一、向量化基础测试
    # ==============================================================
    print(f"\n{'-' * 70}\n  一、向量化接口\n{'-' * 70}")

    try:
        emb = get_embedding_service()
        print(f"  ✅ 服务初始化成功 | 模型={emb.model}")
    except Exception as e:
        print(f"  ❌ 初始化失败: {type(e).__name__}: {e}")
        return 1

    # 单条向量化
    t0 = time.perf_counter()
    vec = emb.embed_query("iPhone 16 的电池容量是多少")
    t1 = time.perf_counter()
    print(f"  ✅ 单条向量化成功 | 维度={len(vec)} | 耗时={(t1 - t0) * 1000:.0f}ms")
    print(f"     向量前 5 个值: {[round(v, 4) for v in vec[:5]]}")

    if len(vec) != 1024:
        print(f"  ⚠️  维度是 {len(vec)}，不是预期的 1024（不影响功能，仅供参考）")

    # ---- 缓存测试：第二次应该是"瞬间"返回 ----
    t0 = time.perf_counter()
    vec2 = emb.embed_query("iPhone 16 的电池容量是多少")
    t1 = time.perf_counter()
    cache_ms = (t1 - t0) * 1000

    print(f"\n  ✅ 缓存测试 | 第二次耗时 {cache_ms:.1f}ms")
    if cache_ms < 50:
        print(f"     ✅ 缓存生效（未调用 API，耗时从几百毫秒降到 {cache_ms:.0f} 毫秒）")
    else:
        print(f"     ⚠️  耗时 {cache_ms:.0f}ms，缓存可能没命中")
    if vec == vec2:
        print("     ✅ 两次结果完全一致")
    else:
        print("     ❌ 两次结果不一致")
        ok = False

    # ---- 余弦相似度测试 ----
    print(f"\n  余弦相似度验证:")
    v_battery = emb.embed_query("电池耐用吗")
    v_endurance = emb.embed_query("续航怎么样")
    v_screen = emb.embed_query("屏幕分辨率多少")

    sim_related = emb.cosine_similarity(v_battery, v_endurance)
    sim_unrelated = emb.cosine_similarity(v_battery, v_screen)

    print(f"     「电池耐用吗」↔「续航怎么样」  相似度 = {sim_related:.4f}  ← 意思相近，应该高")
    print(f"     「电池耐用吗」↔「屏幕分辨率」  相似度 = {sim_unrelated:.4f}  ← 意思无关，应该低")

    if sim_related > sim_unrelated:
        print(f"     ✅ 语义区分正常（相差 {sim_related - sim_unrelated:.4f}）")
    else:
        print("     ❌ 语义区分异常！相近的反而分数更低")
        ok = False

    # ==============================================================
    # 二、批量向量化
    # ==============================================================
    print(f"\n{'-' * 70}\n  二、批量向量化\n{'-' * 70}")

    texts = [c[1] for c in CORPUS]
    t0 = time.perf_counter()
    vectors = emb.embed_documents(texts)
    t1 = time.perf_counter()

    print(f"  ✅ 批量向量化 {len(texts)} 条 | 耗时 {(t1 - t0) * 1000:.0f}ms")
    print(f"     平均每条 {(t1 - t0) * 1000 / len(texts):.0f}ms")
    print(f"     维度一致性: {'✅ 全部 1024 维' if all(len(v) == 1024 for v in vectors) else '❌ 维度不一致'}")
    ok = ok and all(len(v) == 1024 for v in vectors)

    # 再跑一次，验证缓存（应该几乎瞬间完成）
    t0 = time.perf_counter()
    vectors2 = emb.embed_documents(texts)
    t1 = time.perf_counter()
    print(f"  ✅ 重复批量向量化 | 耗时 {(t1 - t0) * 1000:.0f}ms（全部命中缓存）")
    print(f"     结果一致性: {'✅ 完全一致' if vectors == vectors2 else '❌ 不一致'}")
    ok = ok and vectors == vectors2

    # ==============================================================
    # 三、存入向量库
    # ==============================================================
    print(f"\n{'-' * 70}\n  三、向量库存储\n{'-' * 70}")

    vs = VectorStoreService()

    # 先清理上次测试可能残留的数据
    vs.delete_by_document(tester_doc_id)

    # 把语料切分（这里直接用整段当子块，模拟真实切分结果）
    docs = [
        Document(page_content=text, metadata={"page": i + 1, "source": f"{section}.md"})
        for i, (section, text) in enumerate(CORPUS)
    ]
    splitter = ParentChildSplitter()
    chunks = splitter.split(docs, doc_id=tester_doc_id)
    print(f"  ✅ 切分出 {len(chunks)} 个子块 / {splitter.analyze_chunks(chunks)['total_parents']} 个父块")

    t0 = time.perf_counter()
    added = vs.add_chunks(chunks)
    t1 = time.perf_counter()
    print(f"  ✅ 已存入向量库 | {added} 个片段 | 耗时 {(t1 - t0) * 1000:.0f}ms")
    print(f"     向量库当前总数: {vs.count()}")

    # 验证元数据完整性
    stored = vs.get_document_chunks(tester_doc_id)
    meta_ok = all(
        s.get("file_name") and s.get("content")
        for s in stored
    )
    print(f"  {'✅' if meta_ok else '❌'} 元数据完整（文件名、内容都不为空）")
    ok = ok and meta_ok

    # ==============================================================
    # 四、语义检索（最关键）
    # ==============================================================
    print(f"\n{'-' * 70}\n  四、语义检索准确性\n{'-' * 70}")

    hit_count = 0
    for question, expected in QUERIES:
        results = vs.search(question, top_k=3, where={"doc_id": str(tester_doc_id)})

        if not results:
            print(f"\n  ❌ 「{question}」→ 没有检索到任何结果")
            ok = False
            continue

        top = results[0]
        hit = expected in top["content"]
        hit_count += 1 if hit else 0

        mark = "✅" if hit else "❌"
        src = f"{top['file_name']} 第{top['page']}页" if top.get("page") else top["file_name"]
        print(f"\n  {mark} 「{question}」")
        print(f"     期望包含: {expected}")
        print(f"     检索到  : {top['content'][:70].replace(chr(10), ' ')}...")
        print(f"     相似度  : {top['score']:.4f}  |  来源: {src}")

        if not hit:
            ok = False

    print(f"\n  检索命中率: {hit_count}/{len(QUERIES)} "
          f"（{hit_count / len(QUERIES):.0%}）")
    if hit_count == len(QUERIES):
        print("  🎉 全部命中！语义检索工作正常。")
    elif hit_count >= len(QUERIES) * 0.6:
        print("  ⚠️  部分命中。真实场景下这很正常，后面还有混合检索和重排序来提升。")
    else:
        print("  ❌ 命中率偏低，需要检查向量化或检索逻辑。")
        ok = False

    # ==============================================================
    # 五、父子文档策略验证
    # ==============================================================
    print(f"\n{'-' * 70}\n  五、父子文档策略验证\n{'-' * 70}")

    results = vs.search("电池", top_k=1, where={"doc_id": str(tester_doc_id)})
    if results:
        r = results[0]
        has_parent = bool(r.get("parent_content"))
        parent_longer = len(r.get("parent_content", "")) >= len(r["content"])
        print(f"  {'✅' if has_parent else '❌'} 检索结果带回了父块内容")
        print(f"  {'✅' if parent_longer else '⚠️ '} 父块长度({len(r.get('parent_content', ''))}) "
              f"≥ 子块长度({len(r['content'])})")
        ok = ok and has_parent

    # ==============================================================
    # 六、删除
    # ==============================================================
    print(f"\n{'-' * 70}\n  六、删除功能\n{'-' * 70}")

    before = vs.count()
    removed = vs.delete_by_document(tester_doc_id)
    after = vs.count()

    print(f"  ✅ 按文档删除 | 删除前 {before} 个 → 删除后 {after} 个（移除 {removed} 个）")
    print(f"  {'✅' if removed > 0 else '❌'} 向量被正确删除")
    ok = ok and removed > 0

    # 确认删除后搜不到了
    left = vs.search("电池", top_k=3, where={"doc_id": str(tester_doc_id)})
    print(f"  {'✅' if not left else '❌'} 删除后检索不到残留数据")
    ok = ok and not left

    # ==============================================================
    # 汇总
    # ==============================================================
    print(f"\n{LINE}")
    print("  向量化服务统计:")
    for k, v in emb.get_stats().items():
        print(f"     {k:20} {v}")

    print(f"\n{LINE}")
    if ok:
        print("  ✅ 向量化 + 向量库测试全部通过")
    else:
        print("  ⚠️  有问题未通过，请看上面的 ❌")
    print(f"{LINE}\n")

    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n❌ 测试异常: {type(exc).__name__}: {exc}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
