"""Live validation: python -m tests.validate_general_knowledge --url http://127.0.0.1:8768.

Connects to the configured provider through a running backend. Use a test database.
"""

import argparse
import json
import re
from pathlib import Path

import httpx

from app.core.config import settings


def main(url):
    checks, records, sessions = [], [], set()

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    with httpx.Client(base_url=url, timeout=120) as client:
        login = client.post("/api/auth/login", json={"username": settings.ADMIN_USERNAME,
                                                    "password": settings.ADMIN_PASSWORD})
        login.raise_for_status()
        headers = {"Authorization": "Bearer " + login.json()["access_token"]}

        def ask(question, session_id=None):
            events, event_type = [], None
            with client.stream("POST", "/api/chat/stream", headers=headers,
                               json={"question": question, "session_id": session_id, "use_cache": False}) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data = json.loads(line[5:].strip())
                        events.append({"event": event_type, "data": data})
                        if event_type in {"session", "saved"} and data.get("session_id"):
                            sessions.add(data["session_id"])
            result = {
                "question": question, "events": events,
                "answer": "".join(e["data"]["text"] for e in events if e["event"] == "token"),
                "done": next((e["data"] for e in events if e["event"] == "done"), {}),
                "citations": next((e["data"] for e in events if e["event"] == "citations"), []),
                "saved": next((e["data"] for e in events if e["event"] == "saved"), {}),
            }
            records.append(result)
            check(question + ": completed and saved", bool(result["answer"]) and bool(result["saved"]) and
                  bool(result["done"]) and not any(e["event"] == "error" for e in events))
            print(json.dumps({k: result[k] for k in ("question", "answer", "done")}, ensure_ascii=False), flush=True)
            return result

        try:
            identity = ask("你的模型是什么？")
            check("identity matches configuration", settings.LLM_MODEL in identity["answer"])
            check("identity comes from system", identity["done"].get("knowledge_source") == "system")
            check("identity has no document citations", identity["citations"] == [])

            science = ask("地球为什么会有昼夜交替？请用两句话解释。")
            check("science uses general knowledge", science["done"].get("knowledge_source") == "general")
            check("science explains rotation", "自转" in science["answer"])

            followup = ask("它绕着什么公转？请简短回答。", science["saved"].get("session_id"))
            check("followup retains Earth subject", "地球" in followup["done"].get("search_query", ""))
            check("followup answers Sun", "太阳" in followup["answer"])

            rag = ask("什么是检索增强生成（RAG）？请简短解释。")
            check("RAG explained using general knowledge", rag["done"].get("knowledge_source") == "general" and
                  "检索" in rag["answer"] and "生成" in rag["answer"])

            for result in (science, followup, rag):
                check(result["question"] + ": no fake citations", result["citations"] == [] and
                      not re.search(r"\[\d+\]", result["answer"]))

            kb = ask("雪纺衣服应该怎么洗？请简短回答。")
            check("document answering preserved", kb["done"].get("knowledge_source") == "knowledge_base" and
                  any("洗涤养护" in c["file_name"] for c in kb["citations"]))
        finally:
            for session_id in sessions:
                check("cleanup " + str(session_id),
                      client.delete(f"/api/sessions/{session_id}", headers=headers).status_code == 200)
            report = {"passed": sum(c["passed"] for c in checks),
                      "failed": sum(not c["passed"] for c in checks), "checks": checks, "requests": records}
            path = Path("data/logs/validation_general_knowledge_20261014.json")
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"passed": report["passed"], "failed": report["failed"],
                              "failures": [c for c in checks if not c["passed"]]}, ensure_ascii=False), flush=True)
    return int(report["failed"] > 0)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8768")
    sys.exit(main(parser.parse_args().url))
