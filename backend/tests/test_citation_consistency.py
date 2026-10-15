"""引用输出回归：跨 token 校验、Markdown 保留、缓存一致性，不调用模型。"""

import unittest
from unittest.mock import Mock, patch

from app.core.config import settings
from app.rag.citations import CitationFilter, validate_answer
from app.rag.chain import RAGChain
from app.rag.prompt import build_context
from tests.test_rag_regression import make_chain, chunk, stream


def tokens(events):
    return "".join(e["data"]["text"] for e in events if e["event"] == "token")


class CitationConsistencyTests(unittest.TestCase):
    def test_general_removes_only_prose_citation_markers(self):
        source = "说明[1][999]。代码 `[1]`，数组 [1, 2]。\n```python\nx = [1]\n```\n~~~txt\n[2]\n~~~\n[1](https://example.com/a(b)) [1][link]\n[1]: /path\n\\[1]"
        expected = source.replace("说明[1][999]", "说明")
        for size in (1, 2, 5, len(source)):
            with self.subTest(size=size):
                filter_ = CitationFilter([])
                output = "".join(filter_.feed(source[i:i + size]) for i in range(0, len(source), size))
                output += filter_.feed("", final=True)
                self.assertEqual(output, expected)
                self.assertEqual(filter_.used, set())

    def test_keeps_adjacent_valid_markers_and_only_used_cards(self):
        answer, cards, source = validate_answer("依据[2][1][99]，示例 `[3]`。", [{"index": 1}, {"index": 2}, {"index": 3}], "knowledge_base")
        self.assertEqual(answer, "依据[2][1]，示例 `[3]`。")
        self.assertEqual([c["index"] for c in cards], [1, 2])
        self.assertEqual(source, "knowledge_base")

    def test_every_split_boundary_has_the_same_result(self):
        text = "正常文字[2][99]，`[1]`和[3](https://example.com)，结尾[2]"
        for boundary in range(len(text) + 1):
            filter_ = CitationFilter([2])
            answer = filter_.feed(text[:boundary]) + filter_.feed(text[boundary:]) + filter_.feed("", final=True)
            self.assertEqual(answer, text.replace("[99]", ""))
            self.assertEqual(filter_.used, {2})

    def test_code_or_links_do_not_claim_knowledge_base_source(self):
        for answer in ("示例 `[1]`。", "```\n[1]\n```", "[1](https://example.com)"):
            cleaned, cards, source = validate_answer(answer, [{"index": 1}], "knowledge_base")
            self.assertEqual((cleaned, cards, source), (answer, [], "general"))

    def test_huge_marker_does_not_require_integer_conversion(self):
        answer, cards, source = validate_answer("说明[" + "9" * 5000 + "]。", [], "general")
        self.assertEqual((answer, cards, source), ("说明。", [], "general"))

    def test_sync_stream_and_cache_write_are_identical(self):
        original = "按通用知识回答：Python。虚构[1][9]，代码 `[1]`。"
        chain, _ = make_chain(answer=original)
        chain.enable_cache = True
        cache = Mock()
        cache.lookup.return_value = None
        with patch("app.rag.chain.get_semantic_cache", return_value=cache):
            normal = chain.ask("解释Python")
            events = stream(chain, "解释Python")
        expected = original.replace("[1][9]", "")
        self.assertEqual(normal["answer"], expected)
        self.assertEqual(tokens(events), expected)
        self.assertEqual(normal["citations"], [])
        for call in cache.store.call_args_list:
            self.assertEqual(call.args[1], expected)
            self.assertEqual(call.args[2], [])

    def test_existing_cache_is_validated_before_sync_and_stream_return(self):
        chain, _ = make_chain()
        chain.enable_cache = True
        cache = Mock()
        cache.lookup.return_value = {"answer": "旧答案[2][99]，`[1]`", "citations": [{"index": 1}, {"index": 2}],
                                     "knowledge_source": "knowledge_base", "cache_similarity": 1.0}
        with patch("app.rag.chain.get_semantic_cache", return_value=cache):
            result = chain.ask("测试")
            events = stream(chain, "测试")
        self.assertEqual(result["answer"], "旧答案[2]，`[1]`")
        self.assertEqual(tokens(events), result["answer"])
        self.assertEqual(result["citations"], [{"index": 2}])
        self.assertEqual(next(e["data"] for e in events if e["event"] == "citations"), result["citations"])

    def test_general_cache_cannot_attach_old_cards(self):
        chain, _ = make_chain()
        chain.enable_cache = True
        cache = Mock()
        cache.lookup.return_value = {"answer": "通用[1]。", "citations": [{"index": 1}], "knowledge_source": "general"}
        with patch("app.rag.chain.get_semantic_cache", return_value=cache):
            result = chain.ask("测试")
        self.assertEqual((result["answer"], result["citations"], result["no_answer"]), ("通用。", [], True))

    def test_validates_only_context_given_to_model(self):
        chain, _ = make_chain([chunk(), chunk()], "水温[1][2]。")
        with patch.object(settings, "MAX_CONTEXT_CHUNKS", 1):
            result = chain.ask("纯棉怎么洗？")
            events = stream(chain, "纯棉怎么洗？")
        self.assertEqual(result["answer"], "水温[1]。")
        self.assertEqual(tokens(events), result["answer"])
        self.assertEqual(len(result["citations"]), 1)

    def test_zero_score_and_parent_only_context_are_preserved(self):
        cards = RAGChain.build_citations([{**chunk(0.0), "score": .9, "content": "", "parent_content": "只有父片段"}])
        self.assertEqual((cards[0]["score"], cards[0]["content"]), (0.0, "只有父片段"))
        self.assertIn("相关度0.00", build_context([{**chunk(0.0), "score": .9}]))

    def test_filter_emits_plain_text_without_waiting_for_full_answer(self):
        filter_ = CitationFilter([1])
        self.assertEqual(filter_.feed("立即输出这句话"), "立即输出这句话")
        self.assertEqual(filter_.feed("["), "")
        self.assertEqual(filter_.feed("1]"), "")
        self.assertEqual(filter_.feed("。"), "[1]。")


if __name__ == "__main__":
    unittest.main()
