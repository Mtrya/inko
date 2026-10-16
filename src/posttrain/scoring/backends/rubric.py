"""Five-category VLM judging with both presentation orders."""

import time

from ...schema import require
from ..contracts import CATEGORIES, rubric_criteria

LABELS = ("内容对齐", "形态结构", "美学构图", "风格媒介", "缺陷")
SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "reasoning_effort": "medium",
}
ANSWER_PATTERN = (
    r"[ \t\r\n]*"
    + r"\n".join(
        rf"{category}\|(A|B|tie)\|[^ \t\r\n][^\r\n]*" for category in CATEGORIES
    )
    + r"[ \t\r\n]*"
)
REFERENCE = (
    "另附第三张图 R：参考图，代表目标风格的正向范例。style_medium 大类以 R 为锚点，"
    "比较两张候选图各自与 R 的技法接近度；其余大类不受 R 影响。"
)
TEMPLATE = """你是图像生成质量评审。输入：一条文生图 prompt、它的评审 rubric（五个大类，每类若干判据并标注重要性），以及候选图。

附图顺序：第一张为候选图 A，第二张为候选图 B。
{reference_clause}

任务：对每个大类输出一个总体判定——"A"、"B" 或 "tie"（确实无法区分时才用 tie）。

不要逐条输出判定。判据是检查单：你先逐条对照两张图，再按判据的重要性加权，形成该大类的总体判断。判定依据必须写明哪条判据起了决定作用。注意：artifacts 大类里"更好"指缺陷更少。

输出恰好五行，每行格式 `大类|判定|依据`，判定取 A / B / tie；依据里引用词语一律用「」。行序固定如下，此外不输出任何内容：
content_alignment|...
form_structure|...
aesthetics|...
style_medium|...
artifacts|...

prompt：
{prompt}

rubric：
{rubric_text}"""


def judge_prompt(prompt, rubric, reference):
    rubric = rubric_criteria(rubric)
    lines = []
    for category, label in zip(CATEGORIES, LABELS, strict=True):
        lines.append(f"【{label}】")
        lines.extend(
            f"- ({item['weight']}) {item['text']}" for item in rubric[category]
        )
    return TEMPLATE.format(
        prompt=prompt,
        rubric_text="\n".join(lines),
        reference_clause=REFERENCE if reference else "（无参考图。）",
    )


def parse_lines(text):
    require(isinstance(text, str), "judge content is missing")
    lines = [line for line in text.splitlines() if line.strip()]
    require(len(lines) == len(CATEGORIES), "incomplete judge categories")
    result = {}
    for expected, line in zip(CATEGORIES, lines, strict=True):
        parts = [part.strip() for part in line.split("|", 2)]
        require(len(parts) == 3, "judge line requires category, verdict and reason")
        category, verdict, basis = parts
        verdict = verdict.strip('"')
        require(
            category == expected,
            "judge categories are missing, duplicated or out of order",
        )
        require(
            verdict in {"A", "B", "tie"} and bool(basis),
            "judge verdict or reason is invalid",
        )
        result[category] = {"verdict": verdict, "basis": basis}
    return result


def merge_orders(forward, reverse, sample_ids):
    """Normalize winners to sample IDs, then turn order disagreement into ties."""
    orders, values = [], [{}, {}]
    for decisions, order in ((forward, sample_ids), (reverse, sample_ids[::-1])):
        orders.append(
            {
                category: {
                    **decisions[category],
                    "winner": (
                        None
                        if decisions[category]["verdict"] == "tie"
                        else order[0 if decisions[category]["verdict"] == "A" else 1]
                    ),
                }
                for category in CATEGORIES
            }
        )
    for category in CATEGORIES:
        a, b = (order[category]["winner"] for order in orders)
        winner = a if a == b else None
        for index, key in enumerate(sample_ids):
            values[index][f"rubric.{category}"] = (
                0.5 if winner is None else float(winner == key)
            )
    return values, orders


class RubricJudge:
    def __init__(self, endpoint, model, *, timeout, max_tokens):
        self.endpoint, self.model = endpoint, model
        self.timeout, self.max_tokens = timeout, max_tokens

    def judge(self, prompt, rubric, images):
        from ..client import http_json

        require(
            len(images) in (2, 3),
            "judge requires two candidates and optional reference",
        )
        content = [
            {"type": "image_url", "image_url": {"url": image}} for image in images
        ]
        content.append(
            {"type": "text", "text": judge_prompt(prompt, rubric, len(images) == 3)}
        )
        started = time.monotonic()
        response = http_json(
            self.endpoint,
            "/v1/chat/completions",
            self.timeout,
            {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                **SAMPLING,
                "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": True},
                "structured_outputs": {"regex": ANSWER_PATTERN},
            },
        )
        choice = response["choices"][0]
        require(
            choice["finish_reason"] == "stop",
            f"judge completion ended with {choice['finish_reason']}: "
            f"{response.get('usage', {})}",
        )
        content = choice["message"]["content"]
        try:
            decisions = parse_lines(content)
        except ValueError as exc:
            raise ValueError(f"{exc}; judge content: {content}") from exc
        return {
            "decisions": decisions,
            "usage": response.get("usage", {}),
            "seconds": time.monotonic() - started,
        }
