"""Tests of rde_core, eval_subset.json and training_tasks.json (no Streamlit runtime needed).

Run from the repository root:  python -m unittest discover -s tests -v
Optional: MVT_DIR=<multivote-triage checkout> also checks the training set against its split
files and DATA_ISSUES.md appendices.
"""
import csv
import hashlib
import json
import os
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import rde_core as core  # noqa: E402

SUBSET = json.loads((ROOT / "eval_subset.json").read_text(encoding="utf-8"))
VERSION, TRAINING = core.load_training(ROOT / "training_tasks.json")


def dataset_text():
    """{task_id: text shown} for every item of the three dataset files (interface IDs)."""
    out = {}
    with open(ROOT / "ambik_test_900.csv", newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            out[f"AmbiK_{i}_ambiguous"] = r["ambiguous_task"]
            out[f"AmbiK_{i}_clear"] = r["unambiguous_direct"]
    for k, v in json.loads((ROOT / "augment.json").read_text(encoding="utf-8")).items():
        out[k] = v["goal"]
    with open(ROOT / "mixed_detailed_1009.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(l for l in f if l.strip()):
            out[f"SafeAgentBench_{i}"] = json.loads(line)["instruction"]
    return out


TEXT = dataset_text()


class TestAccess(unittest.TestCase):
    allow = {"Alice": "a-code", "bob": "b-code"}

    def test_normalized_ids_and_codes(self):
        self.assertTrue(core.check_access(self.allow, "  ALICE ", "a-code"))
        self.assertTrue(core.check_access(self.allow, "bob", "b-code"))

    def test_wrong_code_unknown_id_no_allowlist(self):
        self.assertFalse(core.check_access(self.allow, "alice", "b-code"))
        self.assertFalse(core.check_access(self.allow, "alice", ""))
        self.assertFalse(core.check_access(self.allow, "carol", "a-code"))
        self.assertFalse(core.check_access(None, "alice", "a-code"))


class TestSheetRows(unittest.TestCase):
    header = ["Annotator_ID", "Dataset", "Task_ID", "Original_Label", "Instruction", "Workspace",
              "Clarity", "Ambiguity_Type", "Feasibility", "Safety", "Comments"]

    def test_v1_rows_are_not_v2_answers(self):
        values = [self.header, ["alice", "SaGC", "5", "0", "", "", "Clear & Actionable", "N/A", "Feasible", "Safe", ""],
                  ["alice", "SaGC", "6", "0", "", "", "x", "N/A", "y", "z", "", "v2", "t2"]]
        self.assertEqual(set(core.labeled_items(values, "alice")), {("SaGC", "6")})
        self.assertEqual(core.find_row(values, ("alice", "SaGC", "5", "v2")), None)
        self.assertEqual(core.find_row(values, ("alice", "SaGC", "6", "v2")), 3)

    def test_ids_stay_strings(self):
        # Fix B: values are compared as text, so "007" is not read back as 7.
        values = [self.header + core.HEADER_V2, ["007", "SaGC", "0012", "", "", "", "", "", "", "", "", "v2", "t2"]]
        self.assertEqual(set(core.labeled_items(values, "007")), {("SaGC", "0012")})

    def test_training_done_needs_current_version(self):
        values = [self.header + core.HEADER_V2,
                  ["alice", core.TRAINING_MARKER, "PASSED"] + ["N/A"] * 8,  # v1 marker
                  ["bob", core.TRAINING_MARKER, "PASSED"] + ["N/A"] * 8 + ["v2", "t2"]]
        self.assertFalse(core.training_done(values, "alice", "t2"))
        self.assertTrue(core.training_done(values, "bob", "t2"))
        self.assertFalse(core.training_done(values, "bob", "t3"))

    def test_header_update(self):
        self.assertEqual(core.header_update(self.header), core.HEADER_V2)
        self.assertIsNone(core.header_update(self.header + core.HEADER_V2))
        with self.assertRaises(core.SheetError):
            core.header_update(["something", "else"])
        with self.assertRaises(core.SheetError):
            core.header_update(self.header + ["Other", ""])


class TestSubset(unittest.TestCase):
    items = SUBSET["items"]

    def test_ids_unique_and_text_matches(self):
        ids = [it["id"] for it in self.items]
        self.assertEqual(len(ids), len(set(ids)))
        for it in self.items:
            self.assertEqual(hashlib.sha1(TEXT[it["id"]].encode()).hexdigest(), it["text_sha1"], it["id"])

    def test_sizes(self):
        self.assertEqual(len(self.items), SUBSET["n_items"])
        sizes = Counter(it["block"] for it in self.items)
        self.assertEqual(len(sizes), SUBSET["n_blocks"])
        self.assertLessEqual(max(sizes.values()) - min(sizes.values()), 1)
        self.assertEqual(Counter(it["dataset"] for it in self.items)["SafeAgentBench"],
                         SUBSET["parameters"]["sab_size"])

    def test_every_stratum_balanced_across_blocks(self):
        blocks = range(1, SUBSET["n_blocks"] + 1)
        for stratum in {it["stratum"] for it in self.items}:
            per = [sum(1 for it in self.items if it["stratum"] == stratum and it["block"] == b) for b in blocks]
            self.assertLessEqual(max(per) - min(per), 1, stratum)

    def test_ambik_both_forms_in_different_blocks(self):
        by_id = {it["id"]: it for it in self.items}
        ambik = [it for it in self.items if it["dataset"] == "AmbiK"]
        for it in ambik:
            self.assertIn(it["pair"], by_id, it["id"])  # both forms present
            self.assertNotEqual(it["block"], by_id[it["pair"]]["block"], it["id"])

    def test_exclusions(self):
        ids = {it["id"] for it in self.items}
        for v1 in ("0", "AmbiK_0_ambiguous", "AmbiK_0_clear", "SafeAgentBench_286", "SafeAgentBench_317"):
            self.assertNotIn(v1, ids)
        sagc = json.loads((ROOT / "augment.json").read_text(encoding="utf-8"))
        self.assertFalse(any(sagc[it["id"]]["label"] == 3 for it in self.items if it["dataset"] == "SaGC"))
        sigs = [(TEXT[it["id"]].strip(), it["dataset"], json.dumps(sagc[it["id"]]["scene"], sort_keys=True)
                 + sagc[it["id"]]["task"]) for it in self.items if it["dataset"] == "SaGC"]
        self.assertEqual(len(sigs), len(set(sigs)), "exact SaGC duplicates")

    def test_no_training_item_in_subset(self):
        ids = {it["id"] for it in self.items}
        rows = {it.get("row") for it in self.items if it["dataset"] == "AmbiK"}
        for t in TRAINING:
            self.assertNotIn(t["source_id"], ids, t["id"])
            if t["dataset"] == "AmbiK":
                self.assertNotIn(int(re.match(r"AmbiK_(\d+)_", t["source_id"]).group(1)), rows, t["id"])
        texts = {TEXT[i].strip().lower() for i in ids}
        for t in TRAINING:
            self.assertNotIn((t.get("instruction") or t.get("goal")).strip().lower(), texts, t["id"])


class TestOrder(unittest.TestCase):
    def test_reproducible_per_annotator_and_complete(self):
        a1, a2 = core.annotator_order(SUBSET, "alice"), core.annotator_order(SUBSET, "alice")
        self.assertEqual(a1, a2)
        b = core.annotator_order(SUBSET, "bob")
        self.assertNotEqual(a1, b)
        self.assertEqual([blk for blk, _ in a1], list(range(1, SUBSET["n_blocks"] + 1)))
        for (blk, ids_a), (_, ids_b) in zip(a1, b):
            self.assertEqual(sorted(ids_a), sorted(ids_b))  # same blocks, own order
            self.assertEqual(set(ids_a), {it["id"] for it in SUBSET["items"] if it["block"] == blk})

    def test_no_ambik_pair_back_to_back(self):
        pair = {it["id"]: it.get("pair") for it in SUBSET["items"]}
        for name in [f"annotator{i}" for i in range(200)]:
            flat = [tid for _, ids in core.annotator_order(SUBSET, name) for tid in ids]
            for x, y in zip(flat, flat[1:]):
                self.assertNotEqual(pair.get(x), y, (name, x, y))


class TestTraining(unittest.TestCase):
    def test_version_and_ids(self):
        self.assertEqual(VERSION, "t2")
        self.assertEqual(len({t["id"] for t in TRAINING}), len(TRAINING))

    def test_expected_answers_are_exact_option_strings(self):
        for t in TRAINING:
            self.assertIn(t["expected_clarity"], core.CLARITY_OPTIONS, t["id"])
            self.assertIn(t["expected_feasibility"], core.FEASIBILITY_OPTIONS, t["id"])
            self.assertIn(t["expected_safety"], core.SAFETY_OPTIONS, t["id"])
            types = t.get("expected_ambiguity_types") or []
            for x in types:
                self.assertIn(x, core.AMBIGUITY_OPTIONS, t["id"])
            # an item whose clarity asks for types must say which, or every answer "matches"
            self.assertEqual(core.needs_types(t["expected_clarity"]), bool(types), t["id"])

    def test_expected_answers_get_positive_feedback(self):
        for t in TRAINING:
            a = {"clarity": t["expected_clarity"], "ambiguity_types": t.get("expected_ambiguity_types") or [],
                 "feasibility": t["expected_feasibility"], "safety": t["expected_safety"]}
            self.assertTrue(core.answers_complete(a), t["id"])
            self.assertTrue(core.training_feedback(t, a)["all"], t["id"])
            wrong = dict(a, safety=next(s for s in core.SAFETY_OPTIONS if s != a["safety"]))
            self.assertFalse(core.training_feedback(t, wrong)["all"], t["id"])

    def test_text_copied_verbatim(self):
        for t in TRAINING:
            self.assertEqual(t.get("instruction") or t.get("goal"), TEXT[t["source_id"]], t["id"])

    def test_covers_every_answer_type(self):
        labels = Counter((t["dataset"], t["source_label"]) for t in TRAINING)
        for need in [("AmbiK", "clear"), ("AmbiK", "ambiguous (preferences)"),
                     ("AmbiK", "ambiguous (common_sense_knowledge)"), ("AmbiK", "ambiguous (safety)"),
                     ("SaGC", "clear"), ("SaGC", "ambiguous"), ("SaGC", "infeasible"),
                     ("SafeAgentBench", "safe"), ("SafeAgentBench", "unsafe")]:
            self.assertIn(need, labels)

    @unittest.skipUnless(os.environ.get("MVT_DIR"), "set MVT_DIR to a multivote-triage checkout")
    def test_outside_multivote_triage_lists(self):
        mvt = Path(os.environ["MVT_DIR"])
        splits = mvt / "dataset" / "splits"
        blocked = set()
        test_sagc = json.loads((splits / "sagc_split.json").read_text())
        blocked |= {s.split("-")[1] for s in test_sagc["test"]["item_ids"]}
        blocked |= {s.split("-")[1] for s in json.loads((splits / "sagc_test_sub300goals.json").read_text())["item_ids"]}
        blocked |= {s.split("-")[1] for s in json.loads((splits / "pilot_sagc.json").read_text())["item_ids"]}
        with open(splits / "ambik_test_sub300.csv", newline="") as f:
            blocked |= {f"AmbiK_{r['row_900']}_{form}" for r in csv.DictReader(f) for form in ("ambiguous", "clear")}
        issues = (mvt / "docs" / "DATA_ISSUES.md").read_text()
        app = issues[issues.index("## Appendix"):]
        for a, b in (("### A1.", "### A3."), ("### A5.", "### A6.")):
            sec = app[app.index(a):app.index(b)]
            blocked |= {s.split("-")[1] for s in re.findall(r"sagc-\d+", sec)}
        a7 = app[app.index("### A7. AmbiK `ambik_test_900.csv`"):]
        blocked |= {f"AmbiK_{r}_{f}" for r, f in re.findall(r"ambik_test_900-row_(\d+)_(ambiguous|clear)", a7)}
        dup = {s.split("-")[1] for g in test_sagc["duplicates"] for s in [g["kept"], *g["dropped"]]}
        blocked |= dup
        for t in TRAINING:
            sid = t["source_id"]
            self.assertNotIn(sid, blocked, t["id"])
            if t["dataset"] == "AmbiK":
                row = re.match(r"AmbiK_(\d+)_", sid).group(1)
                self.assertNotIn(f"AmbiK_{row}_ambiguous", blocked, t["id"])


if __name__ == "__main__":
    unittest.main()
