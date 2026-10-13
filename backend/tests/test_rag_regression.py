"""Offline RAG behavior checks: python -m unittest tests.test_rag_regression -v.

Retrieval and generation are mocked; no database or provider calls are made.
Failures describe behavioral requirements, including previously existing defects.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.core.config import settings
from app.core.logging import logger
from app.rag.chain import RAGChain
from app.rag.prompt import build_rag_prompt
from app.rag.semantic_cache import SemanticCache


class RewriteResult(SimpleNamespace):
    def to_dict(self):
        return vars(self)


class Generator:
    def __init__(self, answer):
        self.answer = answer
        self.inputs = []

    def invoke(self, prompt):
        self.inputs.append(prompt)
        return self.answer

    async def astream(self, prompt):
        self.inputs.append(prompt)
        for text in [self.answer[:5], self.answer[5:], ""]:
            yield SimpleNamespace(content=text)


def make_chain(chunks=None, answer="按通用知识回答：Python 是一种编程语言。", rewritten=None):
    retriever = Mock()
    retriever.retrieve.return_value = chunks or []
    rewriter = Mock()
    rewriter.rewrite.side_effect = lambda question, history: RewriteResult(
        original=question, rewritten=rewritten or question,
        changed=bool(rewritten and rewritten != question), reason="test",
    )
    chain = RAGChain(retriever=retriever, rewriter=rewriter, enable_cache=False)
    generator = Generator(answer)
    chain._get_prompt_chain = Mock(return_value=generator)
    return chain, generator


def chunk(score=0.8):
    return {"content": "纯棉衣物水温不超过30℃。", "parent_content": "纯棉衣物水温不超过30℃。",
            "file_name": "洗涤养护.txt", "doc_id": 1, "rerank_score": score}


def stream(chain, question="Python 是什么？", **kwargs):
    async def collect():
        return [event async for event in chain.astream(question, **kwargs)]
    return asyncio.run(collect())


class RAGRegressionTests(unittest.TestCase):
    def test_empty_retrieval_generates_general_answer(self):
        chain, generator = make_chain()
        result = chain.ask("Python 是什么？")
        self.assertTrue(generator.inputs)
        self.assertEqual(result["knowledge_source"], "general")
        self.assertEqual(result["citations"], [])
        self.assertIn("Python", result["answer"])

    def test_low_relevance_excludes_unrelated_context(self):
        chain, generator = make_chain([chunk(settings.RERANK_SCORE_THRESHOLD / 2)])
        result = chain.ask("Python 是什么？")
        self.assertEqual(result["knowledge_source"], "general")
        self.assertEqual(result["citations"], [])
        self.assertNotIn("纯棉", generator.inputs[0]["context"])

    def test_relevant_context_keeps_citations(self):
        chain, generator = make_chain([chunk()], "水温不超过30℃。[1]")
        result = chain.ask("纯棉怎么洗？")
        self.assertEqual(result["knowledge_source"], "knowledge_base")
        self.assertEqual(result["citations"][0]["document_id"], 1)
        self.assertIn("纯棉", generator.inputs[0]["context"])

    def test_rerank_threshold_boundary_is_inclusive(self):
        chain, _ = make_chain([chunk(settings.RERANK_SCORE_THRESHOLD)])
        self.assertEqual(chain.ask("纯棉怎么洗？")["knowledge_source"], "knowledge_base")

    def test_vector_threshold_is_independent(self):
        low = {"content": "低相关", "score": settings.SCORE_THRESHOLD / 2, "source": "vector"}
        chain, _ = make_chain([low])
        self.assertEqual(chain.ask("Python 是什么？")["knowledge_source"], "general")

    def test_stream_general_events_complete_without_false_citations(self):
        chain, generator = make_chain()
        events = stream(chain)
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(events[-1]["data"]["knowledge_source"], "general")
        self.assertEqual(next(e["data"] for e in events if e["event"] == "citations"), [])
        self.assertEqual("".join(e["data"]["text"] for e in events if e["event"] == "token"), generator.answer)
        self.assertTrue(any(e["data"].get("stage") == "general_knowledge" for e in events))

    def test_stream_knowledge_base_has_retrieval_and_citations(self):
        chain, _ = make_chain([chunk()], "水温不超过30℃。[1]")
        events = stream(chain, "纯棉怎么洗？")
        self.assertIn("retrieval", [e["event"] for e in events])
        self.assertEqual(events[-1]["data"]["knowledge_source"], "knowledge_base")
        self.assertEqual(len(next(e["data"] for e in events if e["event"] == "citations")), 1)

    def test_generation_error_is_an_error_event(self):
        chain, _ = make_chain()
        chain._get_prompt_chain.side_effect = RuntimeError("simulated provider failure")
        events = stream(chain)
        self.assertEqual(events[-1]["event"], "error")
        self.assertNotIn("done", [e["event"] for e in events])

    def test_prompt_can_be_formatted(self):
        messages = build_rag_prompt().format_messages(context="无相关资料", question="Python 是什么？")
        self.assertEqual(messages[-1].content, "Python 是什么？")
        self.assertIn("按通用知识回答", messages[0].content)

    def test_cache_preserves_source_in_sync_and_stream_paths(self):
        chain, _ = make_chain()
        chain.enable_cache = True
        cache = Mock()
        cache.lookup.return_value = {"answer": "按通用知识回答：Python。", "citations": [],
                                     "knowledge_source": "general", "cache_similarity": 1.0}
        with patch("app.rag.chain.get_semantic_cache", return_value=cache):
            result = chain.ask("Python 是什么？")
            events = stream(chain)
        self.assertTrue(result["cache_hit"])
        self.assertEqual(result["knowledge_source"], "general")
        self.assertEqual(events[-1]["data"]["knowledge_source"], "general")
        chain._get_prompt_chain.assert_not_called()

    def test_general_question_records_knowledge_base_miss(self):
        chain, _ = make_chain()
        self.assertTrue(chain.ask("Python 是什么？")["no_answer"],
                        "QALog.no_answer measures knowledge-base coverage, not overall answer availability.")

    def test_stream_general_question_records_knowledge_base_miss(self):
        chain, _ = make_chain()
        self.assertTrue(stream(chain)[-1]["data"]["no_answer"],
                        "This flag is persisted to QALog.no_answer as a knowledge-base miss.")

    def test_followup_generation_receives_resolved_subject(self):
        chain, generator = make_chain(rewritten="Python 有哪些常见的数据类型？")
        chain.ask("它有哪些常见的数据类型？", history=[{"role": "user", "content": "我在学习Python"}])
        self.assertIn("Python", str(generator.inputs[0]),
                      "Generation receives neither the resolved question nor dialogue history.")

    def test_same_followup_in_different_histories_does_not_share_answer(self):
        chain, _ = make_chain(rewritten="Java 有哪些常见的数据类型？")
        chain.enable_cache = True
        cache = SemanticCache()
        embedder = Mock()
        embedder.embed_query.return_value = [1.0, 0.0]
        embedder.cosine_similarity.return_value = 1.0
        with patch("app.rag.semantic_cache.get_embedding_service", return_value=embedder), \
             patch("app.rag.chain.get_semantic_cache", return_value=cache):
            cache.store("它有哪些常见的数据类型？", "Python 有列表和字典。", extra={"knowledge_source": "general"})
            result = chain.ask("它有哪些常见的数据类型？", history=[{"role": "user", "content": "我在学习Java"}])
        self.assertFalse(result["cache_hit"], "The cache ignores history and bypasses query rewriting.")


if __name__ == "__main__":
    logger.remove()
    unittest.main(verbosity=2)
