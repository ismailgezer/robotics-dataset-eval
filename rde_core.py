"""Logic of the label-evaluation interface v2 that does not need Streamlit (tested in tests/).

Sheet layout (sheet1), one row per (annotator, dataset, task, interface version):
  A Annotator_ID  B Dataset  C Task_ID  D original label  E instruction shown  F workspace shown
  G Clarity  H Ambiguity_Type  I Feasibility  J Safety  K Comments          (unchanged from v1)
  L Interface_Version  M Training_Version                                  (new in v2; empty = v1)

Worksheet "allocations": one row per (annotator, interface version, subset version, block),
with the item order shown to that annotator.
"""
import hashlib
import hmac
import json
import random
from datetime import datetime, timezone

INTERFACE_VERSION = "v2"
TRAINING_MARKER = "[TRAINING_COMPLETE]"
HEADER_V2 = ["Interface_Version", "Training_Version"]  # columns L, M
COL = {name: i for i, name in enumerate(
    ["Annotator_ID", "Dataset", "Task_ID", "Original_Label", "Instruction", "Workspace", "Clarity",
     "Ambiguity_Type", "Feasibility", "Safety", "Comments", "Interface_Version", "Training_Version"])}
N_COLS = len(COL)
ALLOC_SHEET = "allocations"
ALLOC_HEADER = ["Timestamp_UTC", "Annotator_ID", "Interface_Version", "Subset_Version", "Block",
                "Seed", "Order"]

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
    return str(raw or "").strip().lower()


def check_access(allowlist, annotator_id, code):
    """True if the normalized ID is in the allowlist and the access code matches."""
    if not allowlist:
        return False
    table = {normalize_id(k): str(v) for k, v in dict(allowlist).items()}
    expected = table.get(normalize_id(annotator_id))
    if expected is None:
        hmac.compare_digest("x", str(code or ""))  # same work for unknown IDs
        return False
    return hmac.compare_digest(expected.encode(), str(code or "").strip().encode())


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


# --- Sheet rows -------------------------------------------------------------------------------
def pad(row):
    row = [str(x) for x in row]
    return row + [""] * (N_COLS - len(row))


def row_key(row):
    r = pad(row)
    return (r[COL["Annotator_ID"]], r[COL["Dataset"]], r[COL["Task_ID"]], r[COL["Interface_Version"]])


def find_row(values, key):
    """1-based sheet row of the first data row with this key, or None (row 1 is the header)."""
    for i, row in enumerate(values[1:], start=2):
        if row_key(row) == key:
            return i
    return None


def v2_rows(values, annotator_id):
    return [pad(r) for r in values[1:]
            if pad(r)[COL["Annotator_ID"]] == annotator_id
            and pad(r)[COL["Interface_Version"]] == INTERFACE_VERSION]


def labeled_items(values, annotator_id):
    """{(dataset, task_id): row} of this annotator's v2 answers (not the training marker)."""
    return {(r[COL["Dataset"]], r[COL["Task_ID"]]): r for r in v2_rows(values, annotator_id)
            if r[COL["Dataset"]] != TRAINING_MARKER}


def training_done(values, annotator_id, training_version):
    return any(r[COL["Dataset"]] == TRAINING_MARKER and r[COL["Training_Version"]] == training_version
               for r in v2_rows(values, annotator_id))


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
def block_seed(subset_seed, annotator_id, block):
    digest = hashlib.sha256(f"{subset_seed}|{annotator_id}|{block}".encode()).hexdigest()
    return int(digest[:16], 16)


def annotator_order(subset, annotator_id):
    """[(block, [item ids in the order shown])]: blocks in order, each shuffled per annotator.

    The two forms of an AmbiK task are in different blocks (eval_subset.json); if they would meet
    at a block boundary, the next block's first item is swapped with its second.
    """
    pair = {it["id"]: it.get("pair") for it in subset["items"]}
    blocks = sorted({it["block"] for it in subset["items"]})
    out, last = [], None
    for b in blocks:
        ids = sorted(it["id"] for it in subset["items"] if it["block"] == b)
        random.Random(block_seed(subset["seed"], annotator_id, b)).shuffle(ids)
        if last is not None and len(ids) > 1 and pair.get(ids[0]) == last:
            ids[0], ids[1] = ids[1], ids[0]
        out.append((b, ids))
        last = ids[-1] if ids else last
    return out


def allocation_row(annotator_id, subset, block, ids):
    return [datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), annotator_id, INTERFACE_VERSION,
            str(subset["version"]), str(block), str(block_seed(subset["seed"], annotator_id, block)),
            json.dumps(ids)]


def allocation_logged(values, annotator_id, subset_version, block):
    return any(len(r) >= 5 and r[1] == annotator_id and r[2] == INTERFACE_VERSION
               and r[3] == str(subset_version) and r[4] == str(block) for r in values[1:])
