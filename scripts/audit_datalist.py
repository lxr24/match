"""Audit datalist splits and generate stratified subsets."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.path_resolver import find_datalist_file, read_datalist


def synset_of(line: str) -> str:
    parts = line.strip().split("/")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def stratified_subset(lines: list, n: int, seed: int = 123) -> list:
    rng = np.random.default_rng(seed)
    by_synset: dict = defaultdict(list)
    for line in lines:
        by_synset[synset_of(line)].append(line)

    total = len(lines)
    selected = []
    for synset, items in sorted(by_synset.items()):
        k = max(1, int(round(n * len(items) / total))) if n < total else len(items)
        k = min(k, len(items))
        idx = rng.choice(len(items), size=k, replace=False)
        selected.extend([items[i] for i in idx])

    if len(selected) > n:
        rng.shuffle(selected)
        selected = selected[:n]
    elif len(selected) < n:
        remaining = list(set(lines) - set(selected))
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalist_dir", default="./datalist")
    parser.add_argument("--out_dir", default="./datalist")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    splits = {}
    for split in ("train", "validate", "test"):
        path = find_datalist_file(args.datalist_dir, split)
        splits[split] = read_datalist(path)
        print(f"{split}: {len(splits[split])} samples from {path}")

    train_set = set(splits["train"])
    val_set = set(splits["validate"])
    test_set = set(splits["test"])

    overlap_tv = train_set & val_set
    overlap_vt = val_set & test_set
    overlap_tt = train_set & test_set
    print(f"train∩validate: {len(overlap_tv)}")
    print(f"validate∩test: {len(overlap_vt)}")
    print(f"train∩test: {len(overlap_tt)}")

    for split, lines in splits.items():
        counter = Counter(synset_of(x) for x in lines)
        print(f"\n{split} synset distribution ({len(counter)} categories):")
        for synset, cnt in counter.most_common():
            print(f"  {cnt:5d} {synset}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for n, tag in ((5000, "5k"), (10000, "10k")):
        subset = stratified_subset(splits["train"], n, seed=args.seed)
        out_path = out_dir / f"train_strat{tag}_seed{args.seed}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(subset) + "\n")
        print(f"Wrote {out_path} ({len(subset)} lines)")

    report = {
        "train": len(splits["train"]),
        "validate": len(splits["validate"]),
        "test": len(splits["test"]),
        "overlap_train_validate": len(overlap_tv),
        "overlap_validate_test": len(overlap_vt),
        "overlap_train_test": len(overlap_tt),
    }
    with open(out_dir / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
