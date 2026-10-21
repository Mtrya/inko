"""Generate 200 short GenEval-style prompts for the specialized arm.

Original templates and vocab (no GenEval prompt text). Half zh, half en by
construction (even ids zh, odd ids en). Categories: counting, position,
color attribution, single object, person action. Deterministic under --seed.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

OBJECTS = [
    ("apple", "苹果", "个"), ("bird", "鸟", "只"), ("chair", "椅子", "把"),
    ("cup", "杯子", "个"), ("book", "书", "本"), ("flower", "花", "朵"),
    ("car", "汽车", "辆"), ("dog", "狗", "只"), ("cat", "猫", "只"),
    ("ball", "球", "个"), ("balloon", "气球", "个"), ("lamp", "灯", "盏"),
    ("bottle", "瓶子", "个"), ("tree", "树", "棵"), ("house", "房子", "座"),
    ("bicycle", "自行车", "辆"), ("boat", "船", "艘"), ("kite", "风筝", "个"),
    ("umbrella", "雨伞", "把"), ("guitar", "吉他", "把"),
]
COLORS = [
    ("red", "红色"), ("blue", "蓝色"), ("green", "绿色"), ("yellow", "黄色"),
    ("purple", "紫色"), ("orange", "橙色"), ("black", "黑色"), ("white", "白色"),
    ("pink", "粉色"), ("brown", "棕色"),
]
NUM_EN = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
          7: "seven", 8: "eight", 9: "nine"}
NUM_ZH = {2: "两", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}
REL_EN = ["to the left of", "to the right of", "above", "below",
          "in front of", "behind", "next to"]
REL_ZH = ["左边", "右边", "上方", "下方", "前面", "后面", "旁边"]
SETTINGS = [
    ("on a wooden table", "在木桌上"), ("on the grass", "在草地上"),
    ("in a room", "在房间里"), ("by the sea", "在海边"),
    ("against a white background", "白色背景前"),
]
ATTRS = [
    ("wearing a hat", "戴着帽子"), ("giant", "巨大的"), ("tiny", "小巧的"),
    ("wooden", "木制的"), ("fluffy", "毛茸茸的"), ("shiny", "闪亮的"),
]
PERSONS = [
    ("girl", "女孩"), ("boy", "男孩"), ("old man", "老爷爷"),
    ("old woman", "老奶奶"), ("child", "孩子"), ("dancer", "舞者"),
    ("doctor", "医生"), ("chef", "厨师"), ("farmer", "农民"),
    ("police officer", "警察"), ("painter", "画家"), ("athlete", "运动员"),
    ("young woman", "年轻女子"), ("young man", "年轻男子"),
]
ACTIONS = [
    ("flying a kite", "在放风筝"), ("reading a book", "在读书"),
    ("running", "在跑步"), ("riding a bicycle", "在骑自行车"),
    ("fishing", "在钓鱼"), ("painting a picture", "在画画"),
    ("watering flowers", "在浇花"), ("playing with a ball", "在玩球"),
    ("cooking", "在做饭"), ("playing the guitar", "在弹吉他"),
]

CATEGORIES = {"count": 40, "position": 40, "color_attr": 30,
              "single": 30, "person": 60}
ASPECTS = ["1:1"] * 4 + ["4:3"] * 2 + ["3:4"] * 2


def build(rng):
    rows, seen = [], set()

    def add(tag, en, zh):
        if (en, zh) in seen:
            return False
        seen.add((en, zh))
        rows.append({"tag": tag, "en": en, "zh": zh})
        return True

    def fill(tag, quota, make):
        made = 0
        for _ in range(quota * 50):
            if made >= quota:
                return
            made += add(tag, *make())
        raise RuntimeError(f"cannot fill {tag}: vocab exhausted")

    def make_count():
        noun_en, noun_zh, mw = rng.choice(OBJECTS)
        n = rng.randint(2, 9)
        color_en, color_zh = rng.choice(COLORS)
        setting = rng.choice(SETTINGS) if rng.random() < 0.35 else None
        en = f"{NUM_EN[n]} {color_en} {noun_en}s"
        zh = f"{NUM_ZH[n]}{mw}{color_zh}的{noun_zh}"
        if setting:
            en += f" {setting[0]}"
            zh += f"，{setting[1]}"
        return en, zh

    def make_position():
        a_en, a_zh, a_mw = rng.choice(OBJECTS)
        b_en, b_zh, b_mw = rng.choice(OBJECTS)
        rel = rng.randrange(len(REL_EN))
        return (f"a {a_en} {REL_EN[rel]} a {b_en}",
                f"一{a_mw}{a_zh}在一{b_mw}{b_zh}的{REL_ZH[rel]}")

    def make_color_attr():
        x_en, x_zh, x_mw = rng.choice(OBJECTS)
        y_en, y_zh, y_mw = rng.choice(OBJECTS)
        c1_en, c1_zh = rng.choice(COLORS)
        c2_en, c2_zh = rng.choice(COLORS)
        return (f"a {c1_en} {x_en} and a {c2_en} {y_en}",
                f"一{x_mw}{c1_zh}的{x_zh}和一{y_mw}{c2_zh}的{y_zh}")

    def make_single():
        noun_en, noun_zh, mw = rng.choice(OBJECTS)
        if rng.random() < 0.5:
            attr_en, attr_zh = rng.choice(ATTRS)
            return f"a {attr_en} {noun_en}", f"一{mw}{attr_zh}{noun_zh}"
        color_en, color_zh = rng.choice(COLORS)
        return f"a {color_en} {noun_en}", f"一{mw}{color_zh}的{noun_zh}"

    def make_person():
        person_en, person_zh = rng.choice(PERSONS)
        action_en, action_zh = rng.choice(ACTIONS)
        if rng.random() < 0.25:
            n = rng.randint(2, 4)
            return (f"{NUM_EN[n]} {person_en}s {action_en}",
                    f"{NUM_ZH[n]}个{person_zh}{action_zh}")
        return f"a {person_en} {action_en}", f"一个{person_zh}{action_zh}"

    fill("count", CATEGORIES["count"], make_count)
    fill("position", CATEGORIES["position"], make_position)
    fill("color_attr", CATEGORIES["color_attr"], make_color_attr)
    fill("single", CATEGORIES["single"], make_single)
    fill("person", CATEGORIES["person"], make_person)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("configs/prompts/specialized_shorts.jsonl"))
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = build(rng)
    assert len(rows) == sum(CATEGORIES.values()) == 200
    aspects = (ASPECTS * (len(rows) // len(ASPECTS) + 1))[: len(rows)]
    rng.shuffle(aspects)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, (row, aspect) in enumerate(zip(rows, aspects)):
            language = "zh" if index % 2 == 0 else "en"
            text = row[language]
            record = {
                "row_id": f"s{index:05d}",
                "seed_kind": "caption",
                "family": "synth",
                "language": language,
                "tier": "short",
                "aspect": aspect,
                "prompt": text,
                "tags": [row["tag"]],
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    tally = Counter(r["tag"] for r in rows)
    print(f"wrote {len(rows)} -> {args.output}")
    print("tags:", dict(tally))


if __name__ == "__main__":
    main()
