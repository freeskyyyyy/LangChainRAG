"""评估统计：传输 / 降级与检索质量分开，文件和答案原话必须同时匹配。"""

import math


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def classify(results, expected_file, phrase):
    matches = [i + 1 for i, item in enumerate(results)
               if expected_file in (item.get('file_name') or '') and phrase in (item.get('content') or '')]
    return {'file_top1': bool(results and expected_file in (results[0].get('file_name') or '')),
            'answer_top1': bool(matches and matches[0] == 1), 'answer_top5': bool(matches and matches[0] <= 5),
            'reciprocal_rank': 1 / matches[0] if matches else 0}


def summarize(records, expected):
    valid = [r for r in records if not r.get('error')]
    n = len(valid)
    latency = [r['elapsed_ms'] for r in valid]
    return {'expected': expected, 'attempted': len(records), 'valid': n,
            'errors': len(records) - n, 'complete': len(records) == expected and n == expected,
            'file_top1': sum(r['quality']['file_top1'] for r in valid),
            'answer_top1': sum(r['quality']['answer_top1'] for r in valid),
            'answer_top5': sum(r['quality']['answer_top5'] for r in valid),
            'mrr': sum(r['quality']['reciprocal_rank'] for r in valid) / n if n else None,
            'p50_ms': percentile(latency, .5), 'p95_ms': percentile(latency, .95)}
