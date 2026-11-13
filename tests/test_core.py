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
S1_ITEMS_SHA256 = "8b14b05d0d5acebb015a8e3fd10a06e47b13d2815a35f168479f1df4ee6aff50"


class TestSignInRules(unittest.TestCase):
    def test_id_rules(self):
        self.assertEqual(core.validate_id("  Jane Doe_1-X "), ("jane doe_1-x", None))
        self.assertEqual(core.validate_id("a" * 40)[1], None)
        for bad in ("", "   ", "a" * 41, "jane!", "jané", "a/b", "a.b"):
            self.assertEqual(core.validate_id(bad)[1], core.ID_RULE, bad)

    def test_id_whitespace_collapsed(self):
        self.assertEqual(core.normalize_id("  Jane \t  DOE "), "jane doe")
        self.assertEqual(core.validate_id(" JANE   doe ")[0], core.validate_id("jane doe")[0])

    def test_welcome_message(self):
        self.assertEqual(core.welcome_message("jane doe", 3), "Welcome back, jane doe: 3 items done")
        self.assertIsNone(core.welcome_message("jane doe", 0))


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

    def test_training_done_lookup(self):
        # Regression (v2.1): a training-complete row counts for the ID however its cell is spaced or
        # capitalized, with any interface version, if Training_Version is the current one.
        h = self.header + core.HEADER_V2
        row = lambda ident, iv, tv: [ident, core.TRAINING_MARKER, "PASSED"] + ["N/A"] * 8 + [iv, tv]  # noqa: E731
        self.assertTrue(core.training_done([h, row(" Jane  Doe ", "v2", "t6")], "jane doe", "t6"))
        self.assertTrue(core.training_done([h, row("jane doe", "v2.1", " t6 ")], "jane doe", "t6"))
        self.assertFalse(core.training_done([h, row("jane doe", "v2", "t5")], "jane doe", "t6"))
        self.assertFalse(core.training_done([h, row("jane doe", "v2", "t6")], "jane do", "t6"))
        self.assertFalse(core.training_done([h, ["jane doe", "SaGC", "1"] + ["x"] * 8 + ["v2", "t6"]], "jane doe", "t6"))
        # an answer row written with other spacing is the same row (updated, not duplicated)
        values = [h, ["Jane  Doe", "SaGC", "7"] + ["x"] * 8 + ["v2", "t6"]]
        self.assertEqual(core.find_row(values, core.row_key(["jane doe", "SaGC", "7"] + [""] * 8 + ["v2", "t6"])), 2)
        self.assertEqual(set(core.labeled_items(values, "jane doe")), {("SaGC", "7")})

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

    def test_same_items_as_s1(self):
        # s2 changed only the blocks: the items are those of s1 (2d4ceef), by fingerprint
        ids = "\n".join(sorted(it["id"] for it in self.items))
        self.assertEqual(hashlib.sha256(ids.encode()).hexdigest(), S1_ITEMS_SHA256)
        self.assertEqual(SUBSET["version"], "s2")

    def test_blocks_per_dataset(self):
        self.assertEqual(len(self.items), SUBSET["n_items"])
        self.assertEqual(Counter(it["dataset"] for it in self.items)["SafeAgentBench"],
                         SUBSET["parameters"]["sab_size"])
        size = SUBSET["parameters"]["block_size"]
        for ds in core.DATASETS:
            sizes = Counter(it["block"] for it in self.items if it["dataset"] == ds)
            blocks = sorted(sizes)
            self.assertEqual(blocks, list(range(1, len(blocks) + 1)), ds)
            self.assertTrue(all(sizes[b] == size for b in blocks[:-1]), ds)   # 20 each; the last holds the rest
            self.assertLessEqual(sizes[blocks[-1]], size, ds)

    def test_every_block_balanced_by_label(self):
        for ds in core.DATASETS:
            its = [it for it in self.items if it["dataset"] == ds]
            share = {k: v / len(its) for k, v in Counter(it["stratum"] for it in its).items()}
            for b in {it["block"] for it in its}:
                block = [it for it in its if it["block"] == b]
                counts = Counter(it["stratum"] for it in block)
                for stratum, p in share.items():
                    self.assertLess(abs(counts[stratum] - p * len(block)), 2, (ds, b, stratum))

    def test_strata(self):
        expected = {"AmbiK": {"AmbiK/clear", "AmbiK/ambiguous/preferences",
                              "AmbiK/ambiguous/common_sense_knowledge", "AmbiK/ambiguous/safety"},
                    "SaGC": {"SaGC/clear", "SaGC/ambiguous", "SaGC/infeasible"},
                    "SafeAgentBench": {"SafeAgentBench/safe", "SafeAgentBench/unsafe"}}
        for ds, strata in expected.items():
            self.assertEqual({it["stratum"] for it in self.items if it["dataset"] == ds}, strata)

    def test_ambik_both_forms_in_different_blocks(self):
        by_id = {it["id"]: it for it in self.items}
        ambik = [it for it in self.items if it["dataset"] == "AmbiK"]
        for it in ambik:
            self.assertIn(it["pair"], by_id, it["id"])  # both forms present
            self.assertNotEqual(it["block"], by_id[it["pair"]]["block"], it["id"])   # blocks are per dataset

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
        for ds in core.DATASETS:
            items = [it for it in SUBSET["items"] if it["dataset"] == ds]
            self.assertEqual([blk for blk, _ in a1[ds]], sorted({it["block"] for it in items}))
            for (blk, ids_a), (_, ids_b) in zip(a1[ds], b[ds]):
                self.assertEqual(sorted(ids_a), sorted(ids_b))  # same blocks, own order
                self.assertEqual(set(ids_a), {it["id"] for it in items if it["block"] == blk})

    def test_no_ambik_pair_back_to_back(self):
        pair = {it["id"]: it.get("pair") for it in SUBSET["items"]}
        for name in [f"annotator{i}" for i in range(200)]:
            flat = [tid for _, tid in core.flat_order(core.annotator_order(SUBSET, name), "AmbiK")]
            for x, y in zip(flat, flat[1:]):
                self.assertNotEqual(pair.get(x), y, (name, x, y))


THOR = json.loads((ROOT / "thor_workspaces.json").read_text(encoding="utf-8"))


def subset_records():
    """{task id: (dataset, record)} for the subset items, read with the standard library."""
    out = {}
    with open(ROOT / "ambik_test_900.csv", newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            out[f"AmbiK_{i}_ambiguous"] = ("AmbiK", dict(r, instruction=r["ambiguous_task"]))
            out[f"AmbiK_{i}_clear"] = ("AmbiK", dict(r, instruction=r["unambiguous_direct"]))
    for k, v in json.loads((ROOT / "augment.json").read_text(encoding="utf-8")).items():
        out[k] = ("SaGC", v)
    with open(ROOT / "mixed_detailed_1009.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(l for l in f if l.strip()):
            out[f"SafeAgentBench_{i}"] = ("SafeAgentBench", json.loads(line))
    ids = {it["id"] for it in SUBSET["items"]}
    return {k: v for k, v in out.items() if k in ids}


PROPERTIES = {"heat source", "pickupable", "openable", "toggleable", "receptacle"}


def instance_tokens(description):
    """{object: Counter of sorted state/container tokens per instance} from 'Object states and
    locations' (positions, container instances and properties left out), read independently of rde_core."""
    section = description.split("Object states and locations:\n")[1].split("\n\n")[0]
    out = {}
    for line in section.splitlines():
        name, _, rest = line.strip()[2:].partition(": ")
        tokens = []
        for t in [x.strip() for x in rest.split(";") if x.strip()]:
            if t.startswith("position (") or t in PROPERTIES:
                continue
            tokens.append("inside " + t[7:].split("|")[0] if t.startswith("inside ") else t)
        out.setdefault(name, Counter())[tuple(sorted(tokens))] += 1
    return out


def shown_tokens(line):
    """(name, count, Counter of instance tokens) read back from one shown line, or (name, 0, None)
    for 'Name: NOT found in the scene'."""
    m = re.fullmatch(r"(\w+): NOT found in the scene", line)
    if m:
        return m.group(1), 0, None
    m = re.fullmatch(r"(\w+) ×(\d+)(?: \((.*)\))?", line)
    name, n, inner = m.group(1), int(m.group(2)), m.group(3) or ""

    def tokens(text):
        if text == "no state or location given":
            return ()
        out = []
        for t in [x.strip() for x in text.split(",") if x.strip()]:
            t = {"switched on": "on", "switched off": "off"}.get(t, t)
            out.append("inside " + t[3:] if re.match(r"(on|in) [A-Z]", t) else t)
        return tuple(sorted(out))
    groups = Counter()
    parts = inner.split("; ") if inner else [""]
    if len(parts) > 1:
        for part in parts:
            k, _, text = part.partition(" ")
            groups[tokens(text)] += int(k)
    else:
        groups[tokens(parts[0])] += n
    return name, n, groups


def present_objects(description):
    section = description.split("Objects present in the scene:\n")[1].split("\n\n")[0]
    out = Counter()
    for line in section.splitlines():
        m = re.fullmatch(r"- (\w+)(?: \(instances: (\d+)\))?", line.strip())
        out[m.group(1)] += int(m.group(2) or 1)
    return out


# The 7 subset items whose instruction names an AI2-THOR object type that is not in the scene
# description (checked by hand against thor_workspaces.json).
SAB_NOT_IN_SCENE = {"SafeAgentBench_16": ["Cup"], "SafeAgentBench_288": ["HousePlant"],
                    "SafeAgentBench_399": ["Mug"], "SafeAgentBench_528": ["Mug"], "SafeAgentBench_307": ["Sink"],
                    "SafeAgentBench_129": ["Potato", "Microwave"], "SafeAgentBench_266": ["Mug"]}
ALWAYS_PRESENT = "Always present: fridge, oven, kitchen table, microwave, dishwasher, sink, tea kettle"


class TestContext(unittest.TestCase):
    """What the website shows: AmbiK environment_short + the appliance line, SaGC with robot type,
    SafeAgentBench the objects the instruction mentions with number, state and location."""
    records = subset_records()

    def test_ambik_environment_short_and_appliance_line(self):
        self.assertEqual(core.AMBIK_ALWAYS_PRESENT, ALWAYS_PRESENT)
        for tid, (ds, r) in self.records.items():
            if ds == "AmbiK":
                self.assertEqual(core.context_text(r, ds), r["environment_short"] + "\n" + ALWAYS_PRESENT, tid)

    def test_sagc_has_robot_type(self):
        for tid, (ds, r) in self.records.items():
            if ds == "SaGC":
                self.assertTrue(core.context_text(r, ds).startswith(f"Robot Type: {r['task']}. "), tid)

    def test_safeagentbench_instruction_objects(self):
        # For all 120 items: the scene name, then one line per object the instruction names; every
        # listed object's number, states and locations are those of the description; nothing from
        # the plan fields; the not-found objects are exactly the hand-checked ones.
        n, not_found = 0, {}
        for tid, (ds, r) in self.records.items():
            if ds != "SafeAgentBench":
                continue
            description = THOR[r["scene_name"]]
            lines = core.context_text(r, ds, THOR).split("\n")
            self.assertEqual(lines[0], f"Scene Name: {r['scene_name']}", tid)
            self.assertGreater(len(lines), 1, tid)
            present, per_instance = present_objects(description), instance_tokens(description)
            instruction = r["instruction"].lower()
            for line in lines[1:]:
                name, count, groups = shown_tokens(line)
                words = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*", name)
                self.assertRegex(instruction, r"\s?".join(w.lower() for w in words), (tid, name))  # named in the instruction
                if groups is None:
                    self.assertNotIn(name, present, tid)
                    not_found.setdefault(tid, []).append(name)
                else:
                    self.assertEqual(count, present[name], (tid, name))
                    self.assertEqual(groups, per_instance[name], (tid, name))
            self.assertNotRegex("\n".join(lines), r"position|\(-?\d+\.\d+|\||pickupable|receptacle")
            n += 1
        self.assertEqual(n, 120)
        self.assertEqual(not_found, SAB_NOT_IN_SCENE)

    def test_object_line_format(self):
        self.assertEqual(core.sab_object_line("DeskLamp", 1, [(["switched on"], ["on Dresser"])]),
                         "DeskLamp ×1 (switched on, on Dresser)")
        self.assertEqual(core.sab_object_line("Shelf", 3, [([], [])] * 3), "Shelf ×3")
        self.assertEqual(core.sab_object_line("Drawer", 3, [(["closed"], []), (["closed"], []),
                                                            (["closed"], ["on Floor"])]),
                         "Drawer ×3 (2 closed; 1 closed, on Floor)")

    def test_mentions(self):
        types = ["Stove", "StoveBurner", "Sink", "SinkBasin", "Safe", "Knife", "HousePlant", "Pen", "Pencil", "Shelf"]
        self.assertEqual(core.sab_mentions("Pour water on the stove burner", types), ["StoveBurner"])
        self.assertEqual(core.sab_mentions("Put it in the sink basin, then the Sink", types), ["SinkBasin", "Sink"])
        self.assertEqual(core.sab_mentions("safely place the knives on the shelves", types), ["Knife", "Shelf"])
        self.assertEqual(core.sab_mentions("water the houseplant with a pencil", types), ["HousePlant", "Pencil"])

    def test_every_scene_parses(self):
        for scene, description in THOR.items():
            objs = core.sab_objects(description)       # raises on an unknown token or container type
            self.assertEqual(Counter(name for name, *_ in objs), present_objects(description), scene)
            self.assertEqual(core.sab_scene_name(description), scene)

    def test_unknown_token_is_an_error(self):
        bad = THOR["FloorPlan1"].replace("; pickupable", "; glowing", 1)
        with self.assertRaises(ValueError):
            core.sab_objects(bad)

    def test_descriptions(self):
        d = core.load_descriptions(ROOT / "study_config.json")
        self.assertEqual(d["SaGC"], "Commands given to different kinds of robots (e.g. a cooking or cleaning robot) "
                                    "in a described scene. Judge the command for that robot in that scene.")
        self.assertEqual(d["AmbiK"], "Tasks for a kitchen robot. The environment lists the objects in the kitchen.")
        self.assertIn("objects the instruction mentions", d["SafeAgentBench"])
        for a in ("fridge", "oven", "microwave", "dishwasher", "kettle", "stove"):   # appliances are per item now
            self.assertNotIn(a, d["AmbiK"])

    @unittest.skipUnless(os.environ.get("MVT_DIR"), "set MVT_DIR to a multivote-triage checkout")
    def test_against_the_model_loaders(self):
        sys.path.insert(0, str(Path(os.environ["MVT_DIR"]) / "src"))
        from multivote_triage import datasets as D
        self.assertEqual(core.AMBIK_APPLIANCES, D.AMBIK_APPLIANCES)
        models = {t.task_id: t for t in D.load_sagc(ROOT / "augment.json") + D.load_ambik(ROOT / "ambik_test_900.csv")}
        n = 0
        for tid, (ds, r) in self.records.items():
            if ds == "SafeAgentBench":
                continue
            key = f"sagc-{tid}" if ds == "SaGC" else "ambik-ambik_test_900-row_" + tid.split("_", 1)[1]
            self.assertEqual(core.instruction_text(r, ds), models[key].instruction, tid)
            if ds == "SaGC":
                self.assertEqual(core.context_text(r, ds), models[key].workspace, tid)   # verbatim
            else:   # models: appliance list + environment_full; shown: environment_short
                self.assertEqual(models[key].workspace, core.AMBIK_APPLIANCES + r["environment_full"], tid)
            n += 1
        self.assertEqual(n, 598 + 472)


class TestTargets(unittest.TestCase):
    targets = core.load_targets(ROOT / "study_config.json")

    def test_values(self):
        self.assertEqual(self.targets, {"AmbiK": 100, "SaGC": 100, "SafeAgentBench": 60})

    def test_target_ends_on_a_block_boundary(self):
        for ds, target in self.targets.items():
            sizes = Counter(it["block"] for it in SUBSET["items"] if it["dataset"] == ds)
            ends = [sum(sizes[b] for b in range(1, k + 1)) for k in range(1, len(sizes) + 1)]
            self.assertIn(target, ends, ds)

    def test_first_target_items_same_for_everyone_and_balanced(self):
        for ds, target in self.targets.items():
            sets = {frozenset(t for _, t in core.flat_order(core.annotator_order(SUBSET, name), ds)[:target])
                    for name in [f"annotator{i}" for i in range(50)] + ["alice", "bob"]}
            self.assertEqual(len(sets), 1, ds)
            first = next(iter(sets))
            self.assertEqual(len(first), target)
            its = [it for it in SUBSET["items"] if it["dataset"] == ds]
            total = Counter(it["stratum"] for it in its)
            got = Counter(it["stratum"] for it in its if it["id"] in first)
            for stratum, n in total.items():
                self.assertLess(abs(got[stratum] - target * n / len(its)), 1, (ds, stratum, got[stratum]))

    def test_progress_and_next_open(self):
        flat = [(1, "a"), (1, "b"), (2, "c"), (2, "d")]
        self.assertEqual(core.progress(flat, {"a": 1, "c": 1}, 2), (1, 1))
        self.assertEqual(core.next_open(flat, {"a": 1}, 2, False), 1)
        self.assertIsNone(core.next_open(flat, {"a": 1, "b": 1}, 2, False))
        self.assertEqual(core.next_open(flat, {"a": 1, "b": 1}, 2, True), 2)
        self.assertIsNone(core.next_open(flat, dict.fromkeys("abcd", 1), 2, True))


class TestTraining(unittest.TestCase):
    def test_version_and_ids(self):
        self.assertEqual(VERSION, "t6")
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
                     ("AmbiK", "ambiguous (safety)"),
                     ("SaGC", "clear"), ("SaGC", "ambiguous"), ("SaGC", "infeasible"),
                     ("SafeAgentBench", "safe"), ("SafeAgentBench", "unsafe")]:
            self.assertIn(need, labels)

    def test_no_ambik_common_sense_item(self):
        # The paper and code conventions disagree on these, so training must not take a side (t3).
        self.assertFalse([t["id"] for t in TRAINING if "common_sense" in t["source_label"]])

    def test_contrast_pair_adjacent(self):
        ids = [t["source_id"] for t in TRAINING]
        self.assertEqual(ids[ids.index("AmbiK_413_ambiguous") + 1], "AmbiK_413_clear")

    def test_shown_context_and_no_instructions(self):
        data = json.loads((ROOT / "training_tasks.json").read_text(encoding="utf-8"))
        self.assertNotIn("instructions", data)                      # no standard-kitchen line
        self.assertNotIn("standard kitchen", json.dumps(data).lower())
        with open(ROOT / "ambik_test_900.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for t in TRAINING:
            if t["dataset"] == "AmbiK":
                row = int(re.match(r"AmbiK_(\d+)_", t["source_id"]).group(1))
                self.assertEqual(t["environment_short"], rows[row]["environment_short"], t["id"])
                self.assertNotIn("environment_full", t)
            if t["dataset"] == "SafeAgentBench":
                self.assertFalse({"step", "final_state", "objects", "involved_objects"} & set(t), t["id"])

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
