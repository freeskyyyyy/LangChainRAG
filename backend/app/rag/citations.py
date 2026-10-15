"""按资料编号校验正文引用；支持跨模型 token 的编号和 Markdown 代码。"""

import re


class CitationFilter:
    def __init__(self, valid_indices):
        self.valid = {str(index): index for index in valid_indices}
        self.used = set()
        self.pending = ""
        self.code = None

    def feed(self, text, final=False):
        self.pending += text
        output = []
        position = 0
        while position < len(self.pending):
            rest = self.pending[position:]
            char = rest[0]
            if char in "`~":
                run = len(rest) - len(rest.lstrip(char))
                if run == len(rest) and not final:
                    break
                delimiter = char * run
                if self.code is None and (char == "`" or run >= 3):
                    self.code = delimiter
                elif self.code == delimiter:
                    self.code = None
                output.append(delimiter)
                position += run
                continue
            if self.code:
                output.append(char)
                position += 1
                continue
            if char == "\\":
                if len(rest) == 1 and not final:
                    break
                output.append(rest[:2])
                position += min(2, len(rest))
                continue
            if char != "[":
                output.append(char)
                position += 1
                continue
            end = rest.find("]")
            if end == -1:
                if not final:
                    break
                output.append(rest)
                position = len(self.pending)
                continue
            if end + 1 == len(rest) and not final:
                break
            label = rest[1:end]
            after = rest[end + 1:]
            # 数字链接标签、链接目标、引用式链接和定义不是资料引用。
            if after.startswith("("):
                depth = 0
                link_end = None
                escaped = False
                for index, value in enumerate(after):
                    if escaped:
                        escaped = False
                        continue
                    if value == "\\":
                        escaped = True
                    elif value == "(":
                        depth += 1
                    elif value == ")":
                        depth -= 1
                        if depth == 0:
                            link_end = index
                            break
                if link_end is None and not final:
                    break
                length = end + 2 + link_end if link_end is not None else len(rest)
                output.append(rest[:length])
                position += length
                continue
            if after.startswith("["):
                ref_end = after.find("]")
                if ref_end == -1 and not final:
                    break
                if ref_end != -1 and not re.fullmatch(r"[0-9]+", after[1:ref_end]):
                    length = end + 2 + ref_end
                    output.append(rest[:length])
                    position += length
                    continue
            if re.fullmatch(r"[0-9]+", label) and not after.startswith(":"):
                normalized = label.lstrip("0") or "0"
                if normalized in self.valid:
                    self.used.add(self.valid[normalized])
                    output.append(rest[:end + 1])
            else:
                output.append(rest[:end + 1])
            position += end + 1
        self.pending = self.pending[position:]
        return "".join(output)


def validate_answer(answer, citations, source):
    candidates = citations if source == "knowledge_base" else []
    filter_ = CitationFilter(c["index"] for c in candidates)
    answer = filter_.feed(answer, final=True)
    used = [c for c in candidates if c["index"] in filter_.used]
    resolved_source = "system" if source == "system" else "knowledge_base" if used else "general"
    return answer, used, resolved_source
