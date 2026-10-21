"""Build the specialized-arm pool: tagged subset of the 25k rollout pool plus
the 200 synthetic shorts with rubrics.

Classes: count / position / people (bilingual regexes on the prompt text).
Selection: near-equal class quotas filled round-robin from shuffled class
lists; overlapping rows serve every class they belong to. Deterministic under
--seed. Shorts come from generate_specialized_shorts.py + their rubric file.

Output: train.jsonl in the loader schema (id/text/aspect/rubric/reference/
scene_id), the referenced style images copied next to it, and a report.
"""

import argparse
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

from src.posttrain.prompts import load_rollout_prompts

COUNT_ZH = re.compile(
    r"数量|个数|几只|几个|几条|几张|几朵|([一两二三四五六七八九十\d]+[个只张条朵头匹幅颗株])")
COUNT_EN = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", re.I)
POS_ZH = re.compile(
    r"左侧|右侧|左边|右边|上方|下方|上面|下面|前面|后面|之间|旁边|左上|右上|左下|右下")
POS_EN = re.compile(
    r"\b(to the (left|right) of|(left|right) of|above|below|beneath|behind|"
    r"in front of|next to|beside|between|atop)\b", re.I)
PEOPLE = re.compile(
    r"人物|人像|肖像|男人|女人|男性|女性|男士|女士|男孩|女孩|孩子|儿童|婴儿|宝宝|老人|老者|青年|少年|少女|中年|夫妇|人群|观众|"
    r"医生|护士|教师|老师|厨师|农民|工人|警察|舞者|画家|歌手|律师|司机|士兵|僧人|和尚|运动员|程序员|商人|学生|科学家|工程师|摄影师|模特|演员|工匠|渔夫|牧民|"
    r"手指|手掌|双手|手臂|胳膊|双腿|双脚|四肢|肢体|拳头|左手|右手|"
    r"白人|黑人|亚洲人|非洲人|欧洲人|印度人|阿拉伯|"
    r"\b(man|men|woman|women|person|people|boy|girl|child|children|baby|kid|kids|"
    r"teenager|portrait|crowd|couple|doctor|nurse|teacher|chef|cook|farmer|worker|"
    r"police|dancer|painter|singer|lawyer|driver|soldier|monk|athlete|student|"
    r"scientist|engineer|photographer|model|actor|actress|musician|craftsman|"
    r"fisherman|waiter|waitress|hand|hands|finger|fingers|arm|arms|leg|legs|foot|"
    r"feet|fist|limb|limbs|elderly|infant|asian|african|caucasian|arab)\b|"
    r"\b(black|white)\s+(man|woman|boy|girl|person|people)\b", re.I)

CLASSES = ("count", "position", "people")
def select(tagged, total, seed):
    """Equal per-class quotas; overlapping rows count toward every class they
    carry, so realized coverage lands above the nominal thirds."""
    rng = random.Random(seed)
    by_class = {}
    for cls in CLASSES:
        ids = [i for i, (_, tags) in enumerate(tagged) if cls in tags]
        rng.shuffle(ids)
        by_class[cls] = ids
    quota, extra = divmod(total, len(CLASSES))
    quotas = {cls: quota + (1 if i < extra else 0) for i, cls in enumerate(CLASSES)}
    chosen, chosen_set = [], set()
    ptr = {cls: 0 for cls in CLASSES}
    filled = {cls: 0 for cls in CLASSES}
    while len(chosen) < total:
        progressed = False
        for cls in CLASSES:
            if len(chosen) >= total:
                break
            if filled[cls] >= quotas[cls]:
                continue
            lst = by_class[cls]
            while ptr[cls] < len(lst) and lst[ptr[cls]] in chosen_set:
                ptr[cls] += 1
            if ptr[cls] >= len(lst):
                filled[cls] = quotas[cls]
                continue
            index = lst[ptr[cls]]
            ptr[cls] += 1
            chosen.append(index)
            chosen_set.add(index)
            filled[cls] += 1
            progressed = True
        if not progressed:
            break
    return chosen


def tags_of(text):
    tags = set()
    if COUNT_ZH.search(text) or COUNT_EN.search(text):
        tags.add("count")
    if POS_ZH.search(text) or POS_EN.search(text):
        tags.add("position")
    if PEOPLE.search(text):
        tags.add("people")
    return tags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path,
                        default=Path("data/posttrain/rollout/train.jsonl"))
    parser.add_argument("--shorts", type=Path,
                        default=Path("configs/prompts/specialized_shorts.jsonl"))
    parser.add_argument("--short-rubrics", type=Path,
                        default=Path("data/posttrain/specialized/short_rubrics.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/posttrain/specialized/train.jsonl"))
    parser.add_argument("--take", type=int, default=3640)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    rows = [json.loads(l) for l in args.pool.open(encoding="utf-8")]
    tagged = [(row, tags_of(row["text"])) for row in rows]
    tally = Counter(t for _, tags in tagged for t in tags)
    print("pool class sizes:", dict(tally),
          f"(of {len(rows)} rows)")

    picked = select(tagged, args.take, args.seed)
    chosen_rows = [tagged[i][0] for i in picked]
    chosen_tags = [tagged[i][1] for i in picked]

    short_rows = [json.loads(l) for l in args.shorts.open(encoding="utf-8")]
    rubrics = {r["row_id"]: r["rubric"]
               for r in map(json.loads, args.short_rubrics.open(encoding="utf-8"))}
    missing = [r["row_id"] for r in short_rows if r["row_id"] not in rubrics]
    if missing:
        raise SystemExit(f"shorts without rubric: {missing[:5]} ({len(missing)})")

    out_rows = []
    for row, tags in zip(chosen_rows, chosen_tags):
        out_rows.append({
            "id": row["id"], "text": row["text"], "aspect": row["aspect"],
            "rubric": row["rubric"], "reference": row["reference"],
            "scene_id": row["scene_id"], "tags": sorted(tags),
        })
    for row in short_rows:
        out_rows.append({
            "id": row["row_id"], "text": row["prompt"], "aspect": row["aspect"],
            "rubric": rubrics[row["row_id"]], "reference": None,
            "scene_id": row["row_id"], "tags": row["tags"],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in sorted(out_rows, key=lambda r: r["id"]):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    references = sorted(
        {r["reference"] for r in out_rows if r["reference"] is not None})
    copied = 0
    for reference in references:
        source = args.pool.parent / reference
        destination = args.output.parent / reference
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        copied += 1

    shapes = {"1:1": (896, 896), "4:3": (768, 1024), "3:4": (1024, 768)}
    loaded = load_rollout_prompts(args.output, shapes)

    report = {
        "rows": len(loaded),
        "from_pool": len(chosen_rows),
        "shorts": len(short_rows),
        "references": copied,
        "tags": dict(Counter(t for tags in chosen_tags for t in tags)),
        "language": dict(Counter(
            "zh" if re.search(r"[一-鿿]", r["text"]) else "en"
            for r in out_rows)),
        "aspect": dict(Counter(r["aspect"] for r in out_rows)),
        "ids": [r["id"] for r in out_rows[:5]],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
