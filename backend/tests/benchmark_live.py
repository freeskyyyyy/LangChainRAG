"""真实资料复测：向量库快照 + 独立进程缓存，不写实际数据库或会话。

python -m tests.benchmark_live --run-live
需要已配置的模型联网权限；调用真实向量化、重排序及生成服务。
"""

import argparse
from contextlib import closing
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

from benchmark_metrics import classify, summarize
from eval_retrieval import CASES, UNANSWERABLE


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


def fingerprint(root):
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def snapshot(source, target):
    before = fingerprint(source)
    if not (source / 'chroma.sqlite3').is_file():
        raise RuntimeError('没有找到现有向量库，不创建空库进行评估')
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('chroma.sqlite3*'))
    with closing(sqlite3.connect((source / 'chroma.sqlite3').as_uri() + '?mode=ro', uri=True)) as src:
        with closing(sqlite3.connect(target / 'chroma.sqlite3')) as dst:
            src.backup(dst)
    if fingerprint(source) != before:
        raise RuntimeError('复制期间向量库发生变化，请待上传或重建完成后重试')


def local_worker(args):
    """只读 SQLite 和本地 BM25；不构造向量化客户端，不调用外部服务。"""
    from app.core.database import DATABASE_URL
    from app.rag.bm25 import BM25Index
    from app.core.logging import logger
    logger.remove()
    prefix = 'sqlite+aiosqlite:///'
    if not DATABASE_URL.startswith(prefix):
        raise RuntimeError('本地只读评估当前支持 SQLite 数据库')
    database_path = Path(DATABASE_URL[len(prefix):]).resolve()
    with closing(sqlite3.connect(database_path.as_uri() + '?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        corpus = [dict(row) for row in conn.execute('''
            SELECT c.vector_id AS id, c.content, c.parent_content, c.document_id AS doc_id,
                   c.chunk_index AS child_index, c.page, c.sheet_name, d.file_name
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE d.status = 'indexed' ORDER BY c.id
        ''')]
    if not corpus:
        raise RuntimeError('没有可评估的已索引片段')
    index = BM25Index()
    start = time.perf_counter()
    index.build(corpus)
    build_ms = round((time.perf_counter() - start) * 1000, 2)
    records = []
    for query, file, phrase in CASES:
        available = any(file in c['file_name'] and phrase in c['content'] for c in corpus)
        start = time.perf_counter()
        results = index.search(query, top_k=5)
        row = {'query': query, 'expected_file': file, 'expected_phrase': phrase,
               'results': results, 'elapsed_ms': round((time.perf_counter() - start) * 1000, 2),
               'quality': classify(results, file, phrase),
               'error': None if available else 'expected_evidence_absent_from_database'}
        records.append(row)
        print(f"本地 BM25 {len(records)}/{len(CASES)}: {row['quality']['answer_top1']}", flush=True)
    summary = summarize(records, len(CASES))
    report = {'created_at': datetime.now().astimezone().isoformat(), 'mode': 'local_bm25_only',
              'status': 'completed' if summary['complete'] else 'incomplete',
              'live_status': 'not_run_pending_authorization',
              'corpus': {'chunks': len(corpus), 'files': sorted({c['file_name'] for c in corpus})},
              'method': '只读 SQLite 已索引片段，本地 BM25；不代表向量 / 重排序 / 模型问答结果；无并发、无外部 API',
              'index_build_ms': build_ms, 'summary': summary, 'retrieval': records}
    write_report(Path(args.output), report)
    print(json.dumps({'status': report['status'], 'summary': summary}, ensure_ascii=False), flush=True)
    return 0 if summary['complete'] else 1


def worker(args):
    from app.core.config import settings
    from app.core.logging import logger
    from app.core.cache import MemoryCacheBackend
    import app.core.cache as cache_module
    import app.rag.vectorstore as vector_module
    from app.rag.vectorstore import VectorStoreService
    from app.rag.embeddings import get_embedding_service
    from app.rag.bm25 import ensure_ready
    from app.rag.retriever import HybridRetriever
    from app.rag.chain import RAGChain
    from app.rag.citations import CitationFilter
    import asyncio

    logger.remove()
    # 即便用户配置 Redis，本轮也只使用当前进程的独立缓存。
    cache_module._cache_instance = MemoryCacheBackend()
    vector_module._store = VectorStoreService(persist_dir=str(Path(args.worker).resolve()))
    vector = vector_module._store
    raw = vector._store._collection.get(include=['documents', 'metadatas'])
    corpus = [{'file_name': (meta or {}).get('file_name', ''), 'content': text}
              for text, meta in zip(raw['documents'], raw['metadatas'])]
    digest = hashlib.sha256(json.dumps(sorted(corpus, key=lambda row: (row['file_name'], row['content'])), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    report = {'created_at': datetime.now().astimezone().isoformat(), 'status': 'running',
              'models': {'embedding': settings.EMBEDDING_MODEL, 'rerank': settings.RERANK_MODEL, 'llm': settings.LLM_MODEL},
              'corpus': {'chunks': len(corpus), 'files': sorted({r['file_name'] for r in corpus}), 'sha256': digest},
              'method': '17题、三方案轮换执行顺序；独立缓存，逐项记录向量缓存状态；p95为最近秩估计，小样本描述性统计',
              'thresholds': {'rerank': settings.RERANK_SCORE_THRESHOLD, 'vector': settings.SCORE_THRESHOLD},
              'retrieval': [], 'coverage': [], 'generation': [], 'summary': {}}
    output = Path(args.output)
    write_report(output, report)
    try:
        if not corpus:
            raise RuntimeError('向量快照中没有资料')
        index = ensure_ready()
        if not index.is_ready:
            raise RuntimeError('快照 BM25 索引未就绪')
        variants = [('vector_only', False, False), ('hybrid', False, True), ('hybrid_rerank', True, True)]
        retrievers = {name: HybridRetriever(enable_rerank=rerank, enable_bm25=bm25) for name, rerank, bm25 in variants}
        embedder = get_embedding_service()
        last_rerank = 0.0

        def retrieve(query, name):
            nonlocal last_rerank
            if name == 'hybrid_rerank':
                delay = args.interval - (time.monotonic() - last_rerank)
                if delay > 0:
                    time.sleep(delay)
                last_rerank = time.monotonic()
            before = dict(embedder.stats)
            start = time.perf_counter()
            results, debug = retrievers[name].retrieve(query, top_k=5, return_debug=True)
            errors = [key for key in ('vector_error', 'bm25_error') if debug.get(key)]
            if name == 'hybrid_rerank' and results and not debug.get('rerank_applied'):
                errors.append('rerank_fallback')
            return {'results': results, 'timings': debug, 'elapsed_ms': round((time.perf_counter() - start) * 1000, 2),
                    'embedding_api_calls': embedder.stats['api_calls'] - before['api_calls'],
                    'embedding_cache_hits': embedder.stats['cache_hits'] - before['cache_hits'],
                    'error': ','.join(errors) or None}

        for number, (query, file, phrase) in enumerate(CASES):
            available = any(file in item['file_name'] and phrase in item['content'] for item in corpus)
            order = variants[number % 3:] + variants[:number % 3]
            for name, _, _ in order:
                row = {'query': query, 'variant': name, 'expected_file': file, 'expected_phrase': phrase}
                if available:
                    row.update(retrieve(query, name))
                    row['quality'] = classify(row['results'], file, phrase)
                else:
                    row['error'] = 'expected_evidence_absent_from_snapshot'
                report['retrieval'].append(row)
                print(f"检索 {number + 1}/{len(CASES)} {name}: {'ERROR ' + row['error'] if row['error'] else row['quality']['answer_top1']}", flush=True)
                write_report(output, report)
                # 网络降级不能继续跑成看似成功的全量报告。
                if row.get('error') and row['error'] != 'expected_evidence_absent_from_snapshot':
                    raise RuntimeError('检索链路降级，评估中止：' + row['error'])

        for query in UNANSWERABLE:
            row = {'query': query, **retrieve(query, 'hybrid_rerank')}
            if not row['error'] and row['results']:
                first = row['results'][0]
                score, threshold = RAGChain._normalize_score(first), RAGChain._get_threshold(first)
                row.update({'score': score, 'threshold': threshold, 'knowledge_base_eligible': score >= threshold})
            else:
                row['knowledge_base_eligible'] = False if not row['error'] else None
            report['coverage'].append(row)
            write_report(output, report)

        chain = RAGChain(retriever=retrievers['hybrid_rerank'], enable_cache=True)

        async def ask(query, history=None, use_cache=False):
            start = time.perf_counter()
            first_token = None
            answer, cards, done, error = '', [], {}, None
            async for event in chain.astream(query, history=history, use_cache=use_cache):
                data = event['data']
                if event['event'] == 'token':
                    if first_token is None:
                        first_token = round((time.perf_counter() - start) * 1000, 2)
                    answer += data['text']
                elif event['event'] == 'citations':
                    cards = data
                elif event['event'] == 'done':
                    done = data
                elif event['event'] == 'error':
                    error = data.get('message', 'generation_error')
            check = CitationFilter(c['index'] for c in cards)
            valid_answer = check.feed(answer, final=True)
            return {'query': query, 'answer': answer, 'citations': cards, 'done': done, 'error': error,
                    'first_token_ms': first_token, 'elapsed_ms': round((time.perf_counter() - start) * 1000, 2),
                    'citation_consistent': answer == valid_answer and all(c['index'] in check.used for c in cards)}

        chat_cases = [('你的模型是什么？', 'system', settings.LLM_MODEL),
                      ('地球为什么会有昼夜交替？请用两句话解释。', 'general', '自转'),
                      ('雪纺衣服应该怎么洗？请简短回答。', 'knowledge_base', '拧'),
                      ('真丝连衣裙怎么洗？请简短回答。', 'knowledge_base', '中性')]
        history = []
        for query, source, phrase in chat_cases:
            time.sleep(args.interval)
            row = asyncio.run(ask(query, use_cache=True))
            row['answer_probe_present'] = phrase in row['answer']
            row['behavior_passed'] = bool(row['answer'] and row['done'] and not row['error'] and row['citation_consistent']
                                          and row['done'].get('knowledge_source') == source)
            report['generation'].append(row)
            if source == 'general':
                history = [{'role': 'user', 'content': query}, {'role': 'assistant', 'content': row['answer']}]
            print('问答: ' + query + ' => ' + str(row['behavior_passed']), flush=True)
            write_report(output, report)
        time.sleep(args.interval)
        followup = asyncio.run(ask('它绕着什么公转？请简短回答。', history=history))
        followup['behavior_passed'] = bool(not followup['error'] and followup['done'] and followup['citation_consistent']
                                            and '太阳' in followup['answer'] and '地球' in followup['done'].get('search_query', ''))
        report['generation'].append(followup)
        cached = asyncio.run(ask(chat_cases[1][0], use_cache=True))
        cached['behavior_passed'] = bool(cached['done'].get('cache_hit') and cached['citation_consistent']
                                          and cached['answer'] == report['generation'][1]['answer'])
        report['generation'].append(cached)
        for name, _, _ in variants:
            report['summary'][name] = summarize([r for r in report['retrieval'] if r['variant'] == name], len(CASES))
        report['embedding_stats'] = embedder.get_stats()
        complete = all(item['complete'] for item in report['summary'].values())
        passed = all(row['behavior_passed'] for row in report['generation']) and all(not row['error'] for row in report['coverage'])
        report['status'] = 'completed' if complete else 'incomplete'
        report['behavior_passed'] = passed
        return 0 if complete and passed else 1
    except Exception as error:
        report['status'] = 'incomplete'
        report['fatal_error'] = str(error)
        return 1
    finally:
        for name in ('vector_only', 'hybrid', 'hybrid_rerank'):
            report['summary'][name] = summarize([r for r in report['retrieval'] if r['variant'] == name], len(CASES))
        write_report(output, report)
        print(json.dumps({'status': report['status'], 'summary': report['summary'], 'behavior_passed': report.get('behavior_passed')}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-live', action='store_true', help='运行真实模型复测')
    parser.add_argument('--local-only', action='store_true', help='只读现有 SQLite，运行不联网的 BM25 基线')
    parser.add_argument('--output', default='data/logs/benchmark_live_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
    parser.add_argument('--interval', type=float, default=3.2)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.interval < 0:
        parser.error('interval 不能为负数')
    if args.worker:
        if not args.run_live:
            parser.error('worker 必须显式使用 run-live')
        try:
            return worker(args)
        except Exception as error:
            write_report(Path(args.output), {'status': 'incomplete', 'fatal_error': type(error).__name__, 'summary': {}})
            print('模型评估初始化失败：' + type(error).__name__, flush=True)
            return 1
    if args.local_only:
        if args.run_live:
            parser.error('local-only 与 run-live 不能同时使用')
        return local_worker(args)
    if not args.run_live:
        parser.print_help()
        return 0
    from app.core.config import settings
    base = Path('data/logs').resolve()
    base.mkdir(parents=True, exist_ok=True)
    output = Path(args.output).resolve()
    # 父进程清理快照前，子进程已退出，Windows 上不会遗留 Chroma 文件句柄。
    try:
        with tempfile.TemporaryDirectory(prefix='benchmark-snapshot-', dir=base) as folder:
            target = Path(folder).resolve()
            if not target.is_relative_to(base):
                raise RuntimeError('快照路径不在测试目录中')
            snapshot(settings.chroma_path.resolve(), target / 'chroma')
            result = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'tests.benchmark_live', '--worker', str(target / 'chroma'),
                                     '--run-live', '--output', str(output), '--interval', str(args.interval)])
            return result.returncode
    except Exception as error:
        write_report(output, {'status': 'incomplete', 'fatal_error': type(error).__name__, 'summary': {}})
        print('快照准备失败：' + type(error).__name__, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
