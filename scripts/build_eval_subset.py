#!/usr/bin/env python3
"""Build eval_subset.json: the fixed evaluation subset of interface v2.

Run once; the output is committed and never regenerated for a running study.

    python scripts/build_eval_subset.py --splits-dir ~/multivote-triage/dataset/splits

Sources (item IDs are the interface's Task_IDs, the same as in v1):
- AmbiK: every row of the multivote-triage test subset (ambik_test_sub300.csv, column row_900),
  both forms (AmbiK_<row>_ambiguous, AmbiK_<row>_clear).
- SaGC: the multivote-triage test subset (sagc_test_sub300goals.json, item_ids; label 3 and
  exact duplicates are already removed there, and checked again here).
- SafeAgentBench: a stratified sample of mixed_detailed_1009.jsonl (safe / unsafe x risk category).

Excluded, with reasons recorded in the output:
- items shown in any training set (v1 history and the current training_tasks.json); for AmbiK
  the whole row (both forms) is excluded so the pair stays complete;
- SaGC label 3; exact duplicates (same text shown and same scene) in every dataset.

Blocks: items are dealt to blocks stratum by stratum (each stratum shuffled with the seed), so
each stratum's count differs by at most 1 between blocks and block sizes differ by at most 1.
Both forms of an AmbiK row are then moved into different blocks by swaps inside one stratum.

Standard library only, so the output does not depend on numpy or pandas versions.
"""
import argparse
import csv
import hashlib
import json
import platform
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Items shown in the v1 training sets (every version of training_tasks.json in git history:
# 59659a0 had six items, d7a87ea..1ae6145 three). Matched to the dataset by exact text.
V1_TRAINING_OVERLAP = {
    "SaGC": {"0": "TRAIN_SAGC_1 (59659a0), modified scene"},
    "AmbiK_rows": {0: "TRAIN_AMBIK_1 / TRAIN_AMBIK = AmbiK_0_ambiguous, modified environment"},
    "SafeAgentBench": {
        "SafeAgentBench_317": "TRAIN_SAFE_1 (59659a0)",
        "SafeAgentBench_286": "TRAIN_SAFE_2 / TRAIN_SAFE",
    },
}

SAB_CATEGORIES = [
    "Fire Hazard", "Electrical Shock Hazard", "Explosion Hazard", "Poisoning/Ingestion Hazard",
    "Slip Hazard", "Liquid and Spill Damage", "Breakage and Dropping",
    "Misuse of Electrical Appliances", "Furniture and Decor Damage", "Damage to Small Items",
]
SAB_SYNONYMS = {
    "breakage hazard": "Breakage and Dropping", "break hazard": "Breakage and Dropping",
    "breakage": "Breakage and Dropping", "fire risk": "Fire Hazard",
    "electrical appliance misuse": "Misuse of Electrical Appliances",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha1(text):
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def norm_text(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def sab_category(raw):
    """Map SafeAgentBench's free-text risk_category to one of its 10 categories (first one listed)."""
    if raw is None or raw == "":
        return None
    first = re.split(r"[,;]", str(raw))[0]
    first = re.sub(r"^(risk category|risk)\s*:\s*", "", first.strip(), flags=re.I).strip()
    for c in SAB_CATEGORIES:
        if first.lower() == c.lower():
            return c
    return SAB_SYNONYMS.get(first.lower(), "Other")


def largest_remainder(sizes, total):
    """Quotas proportional to sizes summing to total; ties: larger stratum, then name."""
    n = sum(sizes.values())
    raw = {k: total * v / n for k, v in sizes.items()}
    q = {k: int(raw[k]) for k in sizes}
    rest = total - sum(q.values())
    for k in sorted(sizes, key=lambda k: (-(raw[k] - q[k]), -sizes[k], k))[:rest]:
        q[k] += 1
    return q


def load_training_sources(path):
    """Dataset IDs of the current training items (field source_id), by dataset."""
    if not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = data["tasks"] if isinstance(data, dict) else data
    out = defaultdict(set)
    for t in tasks:
        if t.get("source_id"):
            out[t["dataset"]].add(str(t["source_id"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", required=True, type=Path, help="multivote-triage dataset/splits")
    ap.add_argument("--data-dir", type=Path, default=ROOT, help="folder with the three dataset files")
    ap.add_argument("--training", type=Path, default=ROOT / "training_tasks.json")
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--sab-size", type=int, default=120)
    ap.add_argument("--block-size", type=int, default=50)
    ap.add_argument("--version", default="s1", help="subset version recorded with every annotation")
    ap.add_argument("--out", type=Path, default=ROOT / "eval_subset.json")
    ap.add_argument("--sab-ids-out", type=Path, help="also write the SafeAgentBench sample's IDs here")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    files = {
        "ambik": args.data_dir / "ambik_test_900.csv",
        "sagc": args.data_dir / "augment.json",
        "sab": args.data_dir / "mixed_detailed_1009.jsonl",
        "ambik_sub": args.splits_dir / "ambik_test_sub300.csv",
        "sagc_sub": args.splits_dir / "sagc_test_sub300goals.json",
    }
    training = load_training_sources(args.training)
    excluded = []  # {"id", "reason"}
    items = []     # {"id", "dataset", "stratum", "label", ...}

    # --- AmbiK -------------------------------------------------------------------------------
    with open(files["ambik"], newline="", encoding="utf-8") as f:
        ambik = list(csv.DictReader(f))
    with open(files["ambik_sub"], newline="", encoding="utf-8") as f:
        sub_rows = [int(r["row_900"]) for r in csv.DictReader(f)]
    assert len(sub_rows) == len(set(sub_rows)) == 300, len(sub_rows)
    train_rows = {int(re.match(r"AmbiK_(\d+)_", s).group(1)) for s in training.get("AmbiK", ())}
    seen_ambik = {}  # (form text, environment) -> first item id
    for row in sorted(sub_rows):
        r = ambik[row]
        reason = None
        if row in V1_TRAINING_OVERLAP["AmbiK_rows"]:
            reason = "shown in v1 training: " + V1_TRAINING_OVERLAP["AmbiK_rows"][row]
        elif row in train_rows:
            reason = "row used in the current training set"
        forms = {"ambiguous": r["ambiguous_task"], "clear": r["unambiguous_direct"]}
        if reason is None:
            for form, text in forms.items():
                key = (text.strip(), r["environment_short"].strip(), form)
                if key in seen_ambik:
                    reason = f"exact duplicate of {seen_ambik[key]} (same text and environment)"
        if reason:
            excluded += [{"id": f"AmbiK_{row}_{form}", "reason": reason} for form in forms]
            continue
        for form, text in forms.items():
            iid = f"AmbiK_{row}_{form}"
            seen_ambik[(text.strip(), r["environment_short"].strip(), form)] = iid
            items.append({
                "id": iid, "dataset": "AmbiK", "label": form, "ambiguity_type": r["ambiguity_type"],
                "stratum": f"AmbiK/{form}/{r['ambiguity_type']}", "row": row,
                "pair": f"AmbiK_{row}_{'clear' if form == 'ambiguous' else 'ambiguous'}",
                "forms_identical": r["ambiguous_task"].strip() == r["unambiguous_direct"].strip(),
                "source_id": f"ambik-ambik_test_900-row_{row}_{form}", "text_sha1": text_sha1(text),
            })

    # --- SaGC --------------------------------------------------------------------------------
    sagc = json.loads(files["sagc"].read_text(encoding="utf-8"))
    sub = json.loads(files["sagc_sub"].read_text(encoding="utf-8"))
    names = {0: "clear", 1: "ambiguous", 2: "infeasible"}
    seen_sagc = {}
    for sid in sorted(sub["item_ids"], key=lambda s: int(s.split("-")[1])):
        key = sid.split("-", 1)[1]
        v = sagc[key]
        sig = (v["goal"], json.dumps(v["scene"], sort_keys=True), v.get("task"))
        reason = None
        if key in V1_TRAINING_OVERLAP["SaGC"]:
            reason = "shown in v1 training: " + V1_TRAINING_OVERLAP["SaGC"][key]
        elif key in training.get("SaGC", ()):
            reason = "used in the current training set"
        elif v["label"] == 3:
            reason = "SaGC label 3 (ignore)"
        elif sig in seen_sagc:
            reason = f"exact duplicate of {seen_sagc[sig]}"
        if reason:
            excluded.append({"id": key, "reason": reason})
            continue
        seen_sagc[sig] = key
        items.append({
            "id": key, "dataset": "SaGC", "label": names[v["label"]], "stratum": f"SaGC/{names[v['label']]}",
            "source_id": sid, "text_sha1": text_sha1(v["goal"]),
        })

    # --- SafeAgentBench sample ---------------------------------------------------------------
    with open(files["sab"], encoding="utf-8") as f:
        sab = [json.loads(l) for l in f if l.strip()]
    train_texts = set()
    for p in [args.training]:
        if Path(p).exists():
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            for t in (d["tasks"] if isinstance(d, dict) else d):
                train_texts.add(norm_text(t.get("instruction") or t.get("goal") or ""))
    pool, seen_sab = defaultdict(list), {}
    for i, r in enumerate(sab):
        iid = r.get("id") or f"SafeAgentBench_{i}"
        sig = (r["instruction"].strip(), r.get("scene_name"))
        reason = None
        if iid in V1_TRAINING_OVERLAP["SafeAgentBench"]:
            reason = "shown in v1 training: " + V1_TRAINING_OVERLAP["SafeAgentBench"][iid]
        elif iid in training.get("SafeAgentBench", ()) or norm_text(r["instruction"]) in train_texts:
            reason = "used in the current training set"
        elif sig in seen_sab:
            reason = f"exact duplicate of {seen_sab[sig]}"
        if reason:
            excluded.append({"id": iid, "reason": reason})
            continue
        seen_sab[sig] = iid
        gt = r["ground_truth"]
        stratum = "SafeAgentBench/SAFE" if gt == "SAFE" else f"SafeAgentBench/UNSAFE/{sab_category(r.get('risk_category'))}"
        pool[stratum].append({
            "id": iid, "dataset": "SafeAgentBench", "label": gt.lower(), "stratum": stratum,
            "risk_category": r.get("risk_category"), "row": i,
            "source_id": f"safeagentbench-mixed_detailed_1009-row_{i}", "text_sha1": text_sha1(r["instruction"]),
        })
    quotas = largest_remainder({k: len(v) for k, v in pool.items()}, args.sab_size)
    sab_sample = []
    for stratum in sorted(pool):
        members = pool[stratum]
        picked = sorted(rng.sample(range(len(members)), quotas[stratum]))
        sab_sample += [members[j] for j in picked]
    items += sab_sample

    # --- Blocks ------------------------------------------------------------------------------
    n_blocks = max(1, round(len(items) / args.block_size))
    by_stratum = defaultdict(list)
    for it in items:
        by_stratum[it["stratum"]].append(it)
    j = 0
    for stratum in sorted(by_stratum):
        members = by_stratum[stratum][:]
        rng.shuffle(members)
        for it in members:
            it["block"] = j % n_blocks + 1
            j += 1
    # Both forms of an AmbiK row in different blocks: swap the clear form with another clear form
    # of the same stratum (keeps every stratum's count per block).
    by_id = {it["id"]: it for it in items}

    def conflict(it, block):
        p = by_id.get(it.get("pair"))
        return p is not None and p["block"] == block

    swaps = 0
    for it in sorted(items, key=lambda x: x["id"]):
        if it["dataset"] != "AmbiK" or it["label"] != "clear" or not conflict(it, it["block"]):
            continue
        for other in sorted(by_stratum[it["stratum"]], key=lambda x: x["id"]):
            if other is it or other["block"] == it["block"]:
                continue
            if not conflict(it, other["block"]) and not conflict(other, it["block"]):
                it["block"], other["block"] = other["block"], it["block"]
                swaps += 1
                break
        else:
            sys.exit(f"could not separate {it['id']} from its pair")
    assert not any(it["dataset"] == "AmbiK" and conflict(it, it["block"]) for it in items)

    items.sort(key=lambda x: (x["block"], x["dataset"], x["id"]))
    per_block = {b: dict(sorted(Counter(it["stratum"] for it in items if it["block"] == b).items()))
                 for b in range(1, n_blocks + 1)}
    out = {
        "version": args.version,
        "description": "Fixed evaluation subset of the label-evaluation interface v2. Every annotator "
                       "labels the blocks in block order; within a block the order is a per-annotator "
                       "shuffle (see app_user.py / rde_core.annotator_order).",
        "generator": "scripts/build_eval_subset.py",
        "command": f"python scripts/build_eval_subset.py --splits-dir <multivote-triage>/dataset/splits "
                   f"--seed {args.seed} --sab-size {args.sab_size} --block-size {args.block_size} "
                   f"--version {args.version}",
        "seed": args.seed,
        "parameters": {"sab_size": args.sab_size, "block_size": args.block_size},
        "environment": {"python": platform.python_version()},
        "inputs_sha256": {p.name: sha256(p) for p in files.values()},
        "training_file_sha256": sha256(args.training) if Path(args.training).exists() else None,
        "n_items": len(items),
        "n_blocks": n_blocks,
        "ambik_pair_swaps": swaps,
        "counts": dict(sorted(Counter(it["stratum"] for it in items).items())),
        "counts_by_dataset": dict(sorted(Counter(it["dataset"] for it in items).items())),
        "block_sizes": {b: sum(c.values()) for b, c in per_block.items()},
        "per_block": per_block,
        "excluded": excluded,
        "items": items,
    }
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(items)} items in {n_blocks} blocks; {len(excluded)} excluded; {swaps} pair swaps")
    print(json.dumps(out["counts_by_dataset"]))

    if args.sab_ids_out:
        ids = {
            "description": "SafeAgentBench items in the human-evaluation subset (robotics-dataset-eval "
                           "eval_subset.json). Later SafeAgentBench splits must treat these as test items.",
            "source": "mixed_detailed_1009.jsonl",
            "source_sha256": sha256(files["sab"]),
            "generator": "robotics-dataset-eval scripts/build_eval_subset.py",
            "subset_version": args.version,
            "seed": args.seed,
            "n_items": len(sab_sample),
            "by_stratum": dict(sorted(Counter(it["stratum"] for it in sab_sample).items())),
            "rows": sorted(it["row"] for it in sab_sample),
            "interface_task_ids": sorted((it["id"] for it in sab_sample), key=lambda s: int(s.split("_")[-1])),
            "item_ids": sorted((it["source_id"] for it in sab_sample), key=lambda s: int(s.split("_")[-1])),
        }
        args.sab_ids_out.write_text(json.dumps(ids, indent=1) + "\n", encoding="utf-8")
        print(f"{args.sab_ids_out}: {len(sab_sample)} SafeAgentBench IDs")


if __name__ == "__main__":
    main()
