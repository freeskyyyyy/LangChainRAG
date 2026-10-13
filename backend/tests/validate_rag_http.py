"""Live provider validation against an already running, isolated test server.

Run from backend/: python -m tests.validate_rag_http
Creates and deletes only its own sessions. Writes a detailed JSON report.
"""

import json
import re
from pathlib import Path

import httpx


def main():
    records = []
    checks = []
    sessions = set()

    def check(name, condition, detail=None):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=120) as client:
        login = client.post("/api/auth/login", json={"username": "admin", "password": "123456"})
        login.raise_for_status()
        headers = {"Authorization": "Bearer " + login.json()["access_token"]}

        def ask(question, session_id=None, use_cache=True, regenerate=False):
            events = []
            event_type = None
            with client.stream("POST", "/api/chat/stream", headers=headers,
                               json={"question": question, "session_id": session_id,
                                     "use_cache": use_cache, "regenerate": regenerate}) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data = json.loads(line[5:].strip())
                        events.append({"event": event_type, "data": data})
                        if event_type in {"session", "saved"} and data.get("session_id"):
                            sessions.add(data["session_id"])
            result = {"question": question, "events": events,
                      "answer": "".join(e["data"]["text"] for e in events if e["event"] == "token"),
                      "done": next((e["data"] for e in events if e["event"] == "done"), {}),
                      "citations": next((e["data"] for e in events if e["event"] == "citations"), []),
                      "saved": next((e["data"] for e in events if e["event"] == "saved"), {})}
            records.append(result)
            print(json.dumps({"question": question, "answer_preview": result["answer"][:180],
                              "done": result["done"], "citations": len(result["citations"])}, ensure_ascii=False), flush=True)
            return result

        try:
            general = ask("请用两句话解释Python中的列表与元组有什么区别。")
            check("general answer generated without error", bool(general["answer"]) and not any(e["event"] == "error" for e in general["events"]))
            check("general source identified", general["done"].get("knowledge_source") == "general")
            check("general answer announces source", "按通用知识回答" in general["answer"])
            check("general answer has no fabricated citations", not general["citations"] and not re.search(r"\[\d+\]", general["answer"]))
            check("general request records knowledge-base miss", general["done"].get("no_answer") is True, general["done"])

            saved = general["saved"]
            if saved.get("session_id"):
                response = client.get(f"/api/sessions/{saved['session_id']}/messages", headers=headers)
                response.raise_for_status()
                history = response.json()
                general["history"] = history
                message = next((m for m in history if m.get("id") == saved.get("assistant_message_id")), {})
                check("general answer saved", message.get("content") == general["answer"])
                check("source survives history reload", message.get("knowledge_source") == "general", sorted(message))

            cached = ask(general["question"])
            check("repeat hits semantic cache", cached["done"].get("cache_hit") is True)
            check("cached source remains general", cached["done"].get("knowledge_source") == "general")
            check("cached answer unchanged", cached["answer"] == general["answer"])

            regenerated = ask(general["question"], session_id=saved.get("session_id"), regenerate=True)
            check("regenerate produces fresh response rather than cached answer", regenerated["done"].get("cache_hit") is False,
                  regenerated["done"])

            kb = ask("雪纺衣服应该怎么洗？请简短回答。", use_cache=False)
            check("knowledge-base answer generated", bool(kb["answer"]))
            check("knowledge-base source identified", kb["done"].get("knowledge_source") == "knowledge_base")
            check("knowledge-base citations retained", any("洗涤养护" in c["file_name"] for c in kb["citations"]))
            check("knowledge-base answer cites numbered source", bool(re.search(r"\[\d+\]", kb["answer"])))

            intro = ask("我正在学习Python，请用一句话介绍它。", use_cache=False)
            check("general introduction has no fabricated citations", not intro["citations"] and
                  not re.search(r"\[\d+\]", intro["answer"]), intro["answer"])
            followup = ask("它有哪些常见的数据类型？请简短回答。", session_id=intro["saved"].get("session_id"), use_cache=False)
            check("followup rewritten with subject", "Python" in followup["done"].get("search_query", ""))
            check("followup generated an answer", bool(followup["answer"]))
            check("general followup has no fabricated citations", not followup["citations"] and
                  not re.search(r"\[\d+\]", followup["answer"]), followup["answer"])
        finally:
            for session_id in sessions:
                response = client.delete(f"/api/sessions/{session_id}", headers=headers)
                check(f"cleanup session {session_id}", response.status_code == 200)
            report = {"checks": checks, "requests": records,
                      "passed": sum(c["passed"] for c in checks),
                      "failed": sum(not c["passed"] for c in checks)}
            output = Path("data/logs/validation_rag_http_20261013.json")
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"passed": report["passed"], "failed": report["failed"],
                              "failures": [c for c in checks if not c["passed"]]}, ensure_ascii=False), flush=True)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
