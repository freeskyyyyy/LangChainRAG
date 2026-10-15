import unittest
from unittest.mock import Mock, patch
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile

from benchmark_metrics import classify, summarize, percentile
from app.rag.retriever import HybridRetriever
from tests.benchmark_live import snapshot


class BenchmarkMetricsTests(unittest.TestCase):
    def test_answer_phrase_in_wrong_file_is_not_a_hit(self):
        result = classify([{'file_name': '错误文件', 'content': '答案原话'}, {'file_name': '正确文件', 'content': '答案原话'}], '正确', '答案原话')
        self.assertFalse(result['answer_top1'])
        self.assertTrue(result['answer_top5'])
        self.assertEqual(result['reciprocal_rank'], .5)

    def test_transport_error_cannot_be_reported_as_complete_quality_run(self):
        result = summarize([{'error': 'network'}], expected=2)
        self.assertFalse(result['complete'])
        self.assertEqual((result['valid'], result['errors'], result['attempted']), (0, 1, 1))
        self.assertIsNone(result['mrr'])
        self.assertIsNone(result['p95_ms'])

    def test_latency_percentiles_and_missing_case_are_explicit(self):
        records = [{'elapsed_ms': value, 'quality': classify([], 'file', 'answer')} for value in (10, 20, 100)]
        result = summarize(records, expected=4)
        self.assertEqual((result['p50_ms'], result['p95_ms']), (20, 100))
        self.assertFalse(result['complete'])
        self.assertEqual(percentile([], .5), None)

    def test_answer_outside_top_five_is_not_recall_at_five(self):
        result = classify([{'file_name': 'wrong', 'content': ''}] * 5 + [{'file_name': 'correct', 'content': 'answer'}], 'correct', 'answer')
        self.assertFalse(result['answer_top5'])
        self.assertEqual(result['reciprocal_rank'], 1 / 6)

    def test_snapshot_copies_database_and_binary_without_changing_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'
            source.mkdir()
            with closing(sqlite3.connect(source / 'chroma.sqlite3')) as conn:
                conn.execute('CREATE TABLE marker (id INTEGER)')
                conn.execute('INSERT INTO marker VALUES (7)')
                conn.commit()
            (source / 'index.bin').write_bytes(b'unchanged-index')
            before = (source / 'chroma.sqlite3').read_bytes()
            target = Path(folder) / 'target'
            snapshot(source.resolve(), target)
            self.assertEqual((source / 'chroma.sqlite3').read_bytes(), before)
            self.assertEqual((target / 'index.bin').read_bytes(), b'unchanged-index')
            with closing(sqlite3.connect(target / 'chroma.sqlite3')) as conn:
                self.assertEqual(conn.execute('SELECT id FROM marker').fetchall(), [(7,)])

    def test_vector_failure_is_visible_in_debug(self):
        with patch('app.rag.vectorstore.get_vector_store', return_value=Mock(search=Mock(side_effect=ConnectionError('offline')))):
            result, debug = HybridRetriever(enable_bm25=False, enable_rerank=False).retrieve('test', return_debug=True)
        self.assertEqual(result, [])
        self.assertEqual(debug['vector_error'], 'ConnectionError')

    def test_rerank_fallback_is_not_a_successful_rerank_measurement(self):
        candidate = {'file_name': 'file', 'content': 'content', 'score': .9}
        with patch('app.rag.vectorstore.get_vector_store', return_value=Mock(search=lambda *args, **kwargs: [candidate])):
            retriever = HybridRetriever(enable_bm25=False)
            with patch.object(retriever, '_rerank', return_value=[candidate]):
                _, debug = retriever.retrieve('test', return_debug=True)
        self.assertFalse(debug['rerank_applied'])


if __name__ == '__main__':
    unittest.main()
