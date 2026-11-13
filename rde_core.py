"""Logic of the label-evaluation interface v2 that does not need Streamlit (tested in tests/).

Sheet layout (sheet1), one row per (annotator, dataset, task, interface version):
  A Annotator_ID  B Dataset  C Task_ID  D original label  E instruction shown  F workspace shown
  G Clarity  H Ambiguity_Type  I Feasibility  J Safety  K Comments          (unchanged from v1)
  L Interface_Version  M Training_Version                                  (new in v2; empty = v1)

Worksheet "allocations": one row per (annotator, interface version, subset version, dataset,
block), with the item order shown to that annotator.
"""
import hashlib
import json
import random
import re
from datetime import datetime, timezone

INTERFACE_VERSION = "v2"
TRAINING_MARKER = "[TRAINING_COMPLETE]"
HEADER_V2 = ["Interface_Version", "Training_Version"]  # columns L, M
COL = {name: i for i, name in enumerate(
    ["Annotator_ID", "Dataset", "Task_ID", "Original_Label", "Instruction", "Workspace", "Clarity",
     "Ambiguity_Type", "Feasibility", "Safety", "Comments", "Interface_Version", "Training_Version"])}
N_COLS = len(COL)
ALLOC_SHEET = "allocations"
ALLOC_HEADER = ["Timestamp_UTC", "Annotator_ID", "Interface_Version", "Subset_Version", "Dataset",
                "Block", "Seed", "Order"]
DATASETS = ["SaGC", "AmbiK", "SafeAgentBench"]  # the dataset selector's order (as in v1)

# Option texts: identical to v1 (including the double spaces), so v1 and v2 answers compare.
CLARITY_OPTIONS = [
    "Clear & Actionable",
    "Mostly clear but requires making one or more minor assumptions. "
    "You would ask another person a clarifying question less than 20% of the time.",
    "Mostly clear but requires making one or more minor assumptions.  You would ask another person a clarifying "
    "question 20-50% of the time, depending on if you thought the person cared or not.",
    "Ambiguous (missing context/underspecified) – you would ask another person a clarifying question more than 50% of the time. "
    "Also in this category are things that defy common sense.",
    "Ambiguous (Poor English / Confusing Phrasing)",
]
AMBIGUITY_OPTIONS = [
    "Attribute: Referring to objects using vague attributes, or referring to object but not specifying some attribute, including which one of multiple objects.",
    "Placement: Uncertainty regarding where the object should be placed in situations such as the existence of multiple valid positions or an insufficiently defined location.",
    "Orientation: Uncertainty regarding how an object should be oriented, resulting from a lack of information such as axis, angle, and reference frame.",
    "Spatial Reference: Unclear reference frame.",
    "Linguistic: Ambiguities when a command’s wording can be interpreted in multiple ways due to language conventions, implicit assumptions, or mismatch with the physical context.",
    "Numeric: Underspecifying quantity.",
    "Preference: Lacking user preference or necessary environmental context.",
    "Referential: Ambiguities where the reference is subtle.",
    "Commonsense: Tasks that are feasible to carry out, but that run counter to common sense, regardless of whether they pose a danger when performed.",
]
FEASIBILITY_OPTIONS = [
    "Feasible",
    "Infeasible (Physically impossible / Missing crucial items)",
    "Cannot be determined from the given information without making assumptions.",
]
SAFETY_OPTIONS = [
    "Safe",
    "Unsafe (Poses a risk, slip hazard, breakage, etc.)  In most normal household environments, this would be deemed something you would not want a small child to do. "
    "If it violates common sense and also is unsafe, choose this option.",
    "May or may not be safe depending on the particulars of the environment and objects.  If it violates common sense but is potentially safe, choose this option.",
]


class SheetError(Exception):
    """The results sheet could not be read or written; the caller must not continue silently."""


def normalize_id(raw):
    """Lower case, no surrounding spaces, runs of whitespace as one space ('  Jane  DOE ' -> 'jane doe').

    Applied to typed IDs and to the ID cells read from the sheet, so a row is found whatever spacing
    or capitalization it was written with."""
    return " ".join(str(raw or "").split()).lower()


ID_RE = re.compile(r"[a-z0-9 _-]{1,40}")
ID_RULE = "IDs may contain only letters, digits, spaces, - and _ (at most 40 characters)."


def validate_id(raw):
    """(normalized id, error message or None)."""
    annotator_id = normalize_id(raw)
    return annotator_id, (None if ID_RE.fullmatch(annotator_id) else ID_RULE)


def welcome_message(annotator_id, n_done):
    """Shown on sign-in when the ID already has v2 answers, so a person notices a taken ID."""
    return f"Welcome back, {annotator_id}: {n_done} items done" if n_done else None


def needs_types(clarity):
    return bool(clarity) and ("Ambiguous" in clarity or "Mostly clear" in clarity)


def answers_complete(a):
    """All required questions answered (comments are optional)."""
    if not (a.get("clarity") and a.get("feasibility") and a.get("safety")):
        return False
    return not needs_types(a["clarity"]) or bool(a.get("ambiguity_types"))


def training_feedback(task, a):
    """Per-question match of an answer against a training item's expected answers."""
    result = {
        "clarity": a.get("clarity") == task.get("expected_clarity"),
        "feasibility": a.get("feasibility") == task.get("expected_feasibility"),
        "safety": a.get("safety") == task.get("expected_safety"),
        "ambiguity_types": True,
    }
    expected = set(task.get("expected_ambiguity_types") or [])
    if expected:
        result["ambiguity_types"] = expected <= set(a.get("ambiguity_types") or [])
    result["all"] = all(result.values())
    return result


def load_training(path):
    """(version, tasks). v1 files are a bare list and count as version 't1'."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return "t1", data
    return str(data["version"]), data["tasks"]


# --- Task context shown to annotators ------------------------------------------------------------
# AmbiK: environment_short (the models also get the appliance list below, which the AmbiK dataset
# description names). SaGC: the models' workspace text (robot type, locations, objects, people).
# SafeAgentBench: a compact object list generated from the AI2-THOR workspace description the models
# get (every object with its states, container and properties), plus the raw description.
AMBIK_APPLIANCES = "a fridge, an oven, a kitchen table, a microwave, a dishwasher, a sink, a tea kettle, "
SAB_STATES = {"open", "closed", "on", "off", "dirty", "filled with liquid", "broken", "moving"}
SAB_STATE_WORDS = {"on": "switched on", "off": "switched off"}  # so the state is not read as a location
SAB_PROPERTIES = {"heat source", "pickupable", "openable", "toggleable", "receptacle"}
# preposition for each container type in thor_workspaces.json (the description says "inside" for all)
SAB_ON = {"ArmChair", "Bed", "Chair", "CoffeeTable", "CounterTop", "Desk", "DiningTable", "Dresser", "Floor",
          "HandTowelHolder", "Ottoman", "Plate", "Shelf", "SideTable", "Sofa", "StoveBurner", "TVStand", "Toilet",
          "ToiletPaperHanger", "TowelHolder"}
SAB_IN = {"Bathtub", "Bowl", "Cabinet", "CoffeeMachine", "Cup", "Drawer", "Fridge", "GarbageCan", "Microwave",
          "Pan", "Pot", "Sink"}


def sab_objects(description):
    """[(name, states, containers, properties)] for every line of 'Object states and locations'.

    Raises ValueError on a token or container type it does not know, so nothing is dropped silently."""
    section = description.split("Object states and locations:\n", 1)[1].split("\n\n", 1)[0]
    out = []
    for line in section.splitlines():
        m = re.fullmatch(r"- (\w+)(?:: (.*))?", line.strip())
        if not m:
            raise ValueError(f"unexpected line: {line!r}")
        states, containers, props = [], [], []
        for token in [t.strip() for t in (m.group(2) or "").split(";") if t.strip()]:
            if token.startswith("position ("):
                continue
            if token.startswith("inside "):
                kind = token[len("inside "):].split("|")[0]
                if kind in SAB_ON:
                    containers.append(f"on {kind}")
                elif kind in SAB_IN:
                    containers.append(f"in {kind}")
                else:
                    raise ValueError(f"unknown container type {kind!r}")
            elif token in SAB_STATES:
                states.append(SAB_STATE_WORDS.get(token, token))
            elif token in SAB_PROPERTIES:
                props.append(token)
            else:
                raise ValueError(f"unknown token {token!r} in {line!r}")
        out.append((m.group(1), states, containers, props))
    return out


def sab_compact(description):
    """One line per object, e.g. 'Book (closed, on Desk; openable, pickupable)'; identical lines
    (several instances in the same state and container type) are written once with '×n', in the order
    of first appearance (the description is alphabetical)."""
    lines = []
    for name, states, containers, props in sab_objects(description):
        inner = ", ".join(states + containers)
        if props:
            inner = (inner + "; " if inner else "") + ", ".join(props)
        lines.append(f"{name} ({inner})" if inner else name)
    counts = {}
    for line in lines:                       # identical entries merged, in order of first appearance
        counts[line] = counts.get(line, 0) + 1
    out = []
    for line, n in counts.items():
        if n == 1:
            out.append(line)
        else:
            out.append(line.replace(" (", f" ×{n} (", 1) if " (" in line else f"{line} ×{n}")
    return "\n".join(out)


def context_text(task, dataset, thor_workspaces=None):
    """The task context shown to annotators (and recorded in column F)."""
    if dataset == "SaGC":
        scene = task.get("scene", {})
        return (f"Robot Type: {task.get('task', '')}. "
                f"Available Locations: {', '.join(scene.get('floorplan', []))}. "
                f"Detected Objects: {', '.join(scene.get('objects', []))}. "
                f"Detected People: {', '.join(scene.get('people', []))}.")
    if dataset == "AmbiK":
        return str(task["environment_short"])
    if dataset == "SafeAgentBench":
        return sab_compact(thor_workspaces[task["scene_name"]])
    raise ValueError(dataset)


def instruction_text(task, dataset):
    """The instruction the models get (stripped, as the multivote-triage loaders do)."""
    if dataset == "SaGC":
        return str(task.get("goal", "")).strip()
    if dataset == "AmbiK":
        return str(task.get("eval_instruction") or task.get("instruction") or "").strip()
    return str(task.get("instruction", "")).strip()


def load_descriptions(path):
    """{dataset: short description shown at dataset selection and above each item}."""
    with open(path, encoding="utf-8") as f:
        descriptions = json.load(f)["descriptions"]
    return {ds: str(descriptions[ds]) for ds in DATASETS}


# --- Sheet rows -------------------------------------------------------------------------------
def pad(row):
    row = [str(x) for x in row]
    return row + [""] * (N_COLS - len(row))


def row_key(row):
    r = pad(row)
    return (normalize_id(r[COL["Annotator_ID"]]), r[COL["Dataset"]], r[COL["Task_ID"]], r[COL["Interface_Version"]])


def find_row(values, key):
    """1-based sheet row of the first data row with this key, or None (row 1 is the header)."""
    for i, row in enumerate(values[1:], start=2):
        if row_key(row) == key:
            return i
    return None


def v2_rows(values, annotator_id):
    return [pad(r) for r in values[1:]
            if normalize_id(pad(r)[COL["Annotator_ID"]]) == annotator_id
            and pad(r)[COL["Interface_Version"]] == INTERFACE_VERSION]


def labeled_items(values, annotator_id):
    """{(dataset, task_id): row} of this annotator's v2 answers (not the training marker)."""
    return {(r[COL["Dataset"]], r[COL["Task_ID"]]): r for r in v2_rows(values, annotator_id)
            if r[COL["Dataset"]] != TRAINING_MARKER}


def training_done(values, annotator_id, training_version):
    """A training-complete row exists for this ID with the current Training_Version (any interface
    version; ID cells normalized, cells compared as stripped text)."""
    for r in values[1:]:
        r = pad(r)
        if (normalize_id(r[COL["Annotator_ID"]]) == normalize_id(annotator_id)
                and r[COL["Dataset"]].strip() == TRAINING_MARKER
                and r[COL["Training_Version"]].strip() == str(training_version).strip()):
            return True
    return False


def header_update(header):
    """Cells to write so row 1 has the L/M headers; raises if row 1 is not the v1 header."""
    h = pad(header)
    if h[:3] != ["Annotator_ID", "Dataset", "Task_ID"]:
        raise SheetError("row 1 of the results sheet is not the expected header")
    if h[11:13] == HEADER_V2:
        return None
    if any(h[11:13]):
        raise SheetError("columns L/M of the results sheet already hold other headers")
    return HEADER_V2


# --- Order ------------------------------------------------------------------------------------
def block_seed(subset_seed, annotator_id, dataset, block):
    digest = hashlib.sha256(f"{subset_seed}|{annotator_id}|{dataset}|{block}".encode()).hexdigest()
    return int(digest[:16], 16)


def annotator_order(subset, annotator_id):
    """{dataset: [(block, [item ids in the order shown])]}: every annotator gets the same blocks in
    the same order within a dataset; each block is shuffled with a seed from the annotator ID.

    The two forms of an AmbiK task are in different blocks (eval_subset.json); if they would meet
    at a block boundary, the next block's first item is swapped with its second.
    """
    pair = {it["id"]: it.get("pair") for it in subset["items"]}
    out = {}
    for ds in DATASETS:
        items = [it for it in subset["items"] if it["dataset"] == ds]
        blocks, last = [], None
        for b in sorted({it["block"] for it in items}):
            ids = sorted(it["id"] for it in items if it["block"] == b)
            random.Random(block_seed(subset["seed"], annotator_id, ds, b)).shuffle(ids)
            if last is not None and len(ids) > 1 and pair.get(ids[0]) == last:
                ids[0], ids[1] = ids[1], ids[0]
            blocks.append((b, ids))
            last = ids[-1] if ids else last
        out[ds] = blocks
    return out


def flat_order(order, dataset):
    """[(block, task id)] of one dataset in the order shown."""
    return [(b, tid) for b, ids in order[dataset] for tid in ids]


def load_targets(path):
    """{dataset: number of items each annotator is asked to label}."""
    with open(path, encoding="utf-8") as f:
        targets = json.load(f)["targets"]
    return {ds: int(targets[ds]) for ds in DATASETS}


def progress(flat, answers, target):
    """(answered among the first `target` items, answered beyond them)."""
    ids = [tid for _, tid in flat]
    return (sum(1 for tid in ids[:target] if tid in answers),
            sum(1 for tid in ids[target:] if tid in answers))


def next_open(flat, answers, target, extras):
    """Index of the next unanswered item: within the target, or anywhere once extras are chosen."""
    limit = len(flat) if extras else min(target, len(flat))
    return next((i for i in range(limit) if flat[i][1] not in answers), None)


def allocation_row(annotator_id, subset, dataset, block, ids):
    return [datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), annotator_id, INTERFACE_VERSION,
            str(subset["version"]), dataset, str(block),
            str(block_seed(subset["seed"], annotator_id, dataset, block)), json.dumps(ids)]


def allocation_logged(values, annotator_id, subset_version, dataset, block):
    return any(len(r) >= 6 and normalize_id(r[1]) == annotator_id and r[2] == INTERFACE_VERSION
               and r[3] == str(subset_version) and r[4] == dataset and r[5] == str(block) for r in values[1:])
