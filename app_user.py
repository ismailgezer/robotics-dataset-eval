import streamlit as st
import pandas as pd
import json
import os
import gspread
import re
import logging
import hashlib

import rde_core as core

# --- Page Configuration ---
st.set_page_config(page_title="Robotics Dataset Label Evaluation", layout="wide")
log = logging.getLogger("rde")

# File Paths
SAGC_SUBSET = "augment.json"
AMBIK_SUBSET = "ambik_test_900.csv"
SAFE_SUBSET = "mixed_detailed_1009.jsonl"
TRAINING_FILE = "training_tasks.json"
EVAL_SUBSET = "eval_subset.json"

SHEET_ERROR = ("The results sheet could not be reached. Nothing was saved. "
               "Please try again in a minute; if it keeps failing, contact the study team.")


def secret(name):
    try:
        return st.secrets[name] if name in st.secrets else None
    except Exception:  # no secrets file
        return None


# --- Database Connection ---
@st.cache_resource
def init_connection():
    """Initializes Google Sheets connection securely."""
    creds = secret("GOOGLE_CREDENTIALS")
    if creds:
        sa_info = json.loads(creds)
        return gspread.service_account_from_dict(sa_info)
    else:
        return gspread.service_account(filename="service_account.json")


def get_sheet_url():
    return secret("SHEET_URL")


def open_spreadsheet():
    try:
        return init_connection().open_by_url(get_sheet_url())
    except Exception as e:
        log.exception("opening the results sheet failed")
        raise core.SheetError("open failed") from e


def read_values(ws):
    """Fresh read of a worksheet as strings (no number conversion)."""
    try:
        return ws.get_all_values()
    except Exception as e:
        log.exception("reading the results sheet failed")
        raise core.SheetError("read failed") from e


def ensure_header(ws, values):
    if not values:
        names = list(core.COL)
        ws.update(values=[names], range_name="A1:M1", raw=True)
        return
    cells = core.header_update(values[0])
    if cells:
        ws.update(values=[cells], range_name="L1:M1", raw=True)


def write_row(sh, row):
    """Insert or update one answer row. Fresh read before writing; the target row is re-checked."""
    try:
        ws = sh.sheet1
        key = core.row_key(row)
        idx = core.find_row(read_values(ws), key)
        if idx:
            if core.row_key(ws.row_values(idx)) != key:
                raise core.SheetError("the target row changed between read and write")
            ws.update(values=[row], range_name=f"A{idx}:M{idx}", raw=True)
        else:
            ws.append_row(row, value_input_option="RAW")
    except core.SheetError:
        raise
    except Exception as e:
        log.exception("writing to the results sheet failed")
        raise core.SheetError("write failed") from e


def allocation_worksheet(sh):
    try:
        return sh.worksheet(core.ALLOC_SHEET)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=core.ALLOC_SHEET, rows=1000, cols=len(core.ALLOC_HEADER))
        ws.append_row(core.ALLOC_HEADER, value_input_option="RAW")
        return ws


def log_block_order(sh, annotator_id, subset, block, ids):
    """One allocations row per (annotator, interface, subset version, block)."""
    try:
        ws = allocation_worksheet(sh)
        if not core.allocation_logged(read_values(ws), annotator_id, subset["version"], block):
            ws.append_row(core.allocation_row(annotator_id, subset, block, ids), value_input_option="RAW")
    except core.SheetError:
        raise
    except Exception as e:
        log.exception("writing the allocation log failed")
        raise core.SheetError("allocation log failed") from e


def build_row(dataset, task_data, clarity, ambiguity_type, feasibility, safety, comments, annotator_id,
              training_version):
    """Builds the answer row: columns A-K exactly as in v1, then L Interface_Version and M Training_Version."""
    if dataset == "SaGC":
        label = task_data.get('label')
        original_label = str(label) if label is not None else "N/A"
        instruction = task_data.get('goal', 'N/A')
        scene = task_data.get('scene', {})
        rooms = ", ".join(scene.get('floorplan', []))
        objs = ", ".join(scene.get('objects', []))
        ppl = ", ".join(scene.get('people', []))
        workspace_info = f"Rooms: {rooms} | Objects: {objs} | People: {ppl}"
    elif dataset == "AmbiK":
        original_label = task_data.get('ground_truth', 'N/A')
        instruction = (task_data.get('eval_instruction') or task_data.get('instruction') or
                       task_data.get('Ambiguous Task') or task_data.get('Unambiguous Direct') or 'N/A')
        workspace_info = str(task_data.get('Environment Short') or task_data.get('environment_short') or task_data.get(
            'Environment Full') or 'N/A')
    elif dataset == "SafeAgentBench":
        original_label = task_data.get('ground_truth', 'N/A')
        instruction = task_data.get('instruction') or task_data.get('risk_instruction') or 'N/A'

        # Extract objects for SafeAgentBench from thor_workspaces.json
        scene_name = task_data.get('scene_name', 'N/A')
        workspace_info = f"Scene: {scene_name}"
        if scene_name != 'N/A':
            try:
                with open("thor_workspaces.json", "r") as f:
                    workspace_dict = json.load(f)
                    workspace_desc = workspace_dict.get(scene_name, "")

                    match = re.search(r'(Objects present in the scene:.*?)(?:\n\s*\n|\Z)', workspace_desc,
                                      re.IGNORECASE | re.DOTALL)
                    if match:
                        raw_text = match.group(1).strip()
                        # Flatten vertical lines/newlines into a horizontal, side-by-side format
                        flattened = re.sub(r'\s*\n\s*', ', ', raw_text)
                        workspace_info = " ".join(flattened.split())
                    else:
                        lines = workspace_desc.split('\n')
                        obj_lines = [line.strip() for line in lines if 'object' in line.lower()]
                        workspace_info = ", ".join(obj_lines) if obj_lines else workspace_desc[:300]
            except Exception:
                pass
            
    task_id = task_data.get('id', 'N/A')

    row_data = [
        str(annotator_id), str(dataset), str(task_id), str(original_label),
        str(instruction), str(workspace_info),
        str(clarity), str(ambiguity_type), str(feasibility), str(safety), str(comments)
    ]
    return row_data + [core.INTERFACE_VERSION, str(training_version)]


# --- Data Loading Utilities ---
def load_raw_data(file_path, file_type, dataset_prefix="Task"):
    """Loads dataset files, injects unique IDs, and expands multi-variant tasks."""
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path)
            records = df.to_dict('records')
            expanded_records = []
            for i, rec in enumerate(records):
                # 1. Ambiguous Task Variant
                ambig_task = rec.copy()
                ambig_task['id'] = f"{dataset_prefix}_{i}_ambiguous"
                ambig_task['ground_truth'] = "ambiguous"
                ambig_task['eval_instruction'] = rec.get('Ambiguous Task') or rec.get('ambiguous_task')
                expanded_records.append(ambig_task)

                # 2. Clear Task Variant
                clear_task = rec.copy()
                clear_task['id'] = f"{dataset_prefix}_{i}_clear"
                clear_task['ground_truth'] = "clear"
                clear_task['eval_instruction'] = rec.get('Unambiguous Direct') or rec.get('unambiguous_direct')
                expanded_records.append(clear_task)

            return expanded_records

        elif file_type == 'jsonl':
            df = pd.read_json(file_path, lines=True)
            records = df.to_dict('records')
        elif file_type == 'json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                records = []
                for key, val in data.items():
                    val['id'] = key
                    records.append(val)
            elif isinstance(data, list):
                records = data
            else:
                return []
        else:
            return []

        for i, rec in enumerate(records):
            if 'id' not in rec:
                rec['id'] = f"{dataset_prefix}_{i}"

        return records
    except Exception as e:
        st.error(f"Error loading {dataset_prefix}: {e}")
        return []


@st.cache_resource
def load_subset_items():
    """(subset, {task_id: record}) for the fixed evaluation subset; checks the text of every item."""
    with open(EVAL_SUBSET, encoding="utf-8") as f:
        subset = json.load(f)
    records = {}
    for path, kind, prefix in ((SAGC_SUBSET, 'json', 'SaGC'), (AMBIK_SUBSET, 'csv', 'AmbiK'),
                               (SAFE_SUBSET, 'jsonl', 'SafeAgentBench')):
        for rec in load_raw_data(path, kind, prefix):
            records[str(rec['id'])] = rec
    items = {}
    for it in subset["items"]:
        rec = records[it["id"]]
        text = rec.get('goal') if it["dataset"] == "SaGC" else (
            rec.get('eval_instruction') if it["dataset"] == "AmbiK" else rec.get('instruction'))
        if hashlib.sha1(str(text).encode("utf-8")).hexdigest() != it["text_sha1"]:
            raise ValueError(f"text of {it['id']} differs from eval_subset.json")
        items[it["id"]] = rec
    return subset, items


# --- Reusable Component 1: Task Display ---
def render_task_display(task, dataset_choice):
    """Renders environment, workspace details, object presence check, and instruction prompt."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Environment & Workspace")

        if dataset_choice == "SaGC":
            scene = task.get('scene', {})
            st.write("**Rooms / Floorplan:**", ", ".join(scene.get('floorplan', [])))
            st.write("**Objects Present:**", ", ".join(scene.get('objects', [])))
            if 'people' in scene:
                st.write("**People:**", ", ".join(scene.get('people', [])))

        elif dataset_choice == "AmbiK":
            env = task.get('Environment Short') or task.get('environment_short') or task.get(
                'Environment Full') or "N/A"
            st.write("**Kitchen Items:**", env)

        elif dataset_choice == "SafeAgentBench":
            scene_name = task.get('scene_name', 'N/A')
            st.write(f"**Scene Name:** {scene_name}")

            # Load Workspace Context
            workspace_desc = "Description not found."
            if scene_name != 'N/A':
                try:
                    with open("thor_workspaces.json", "r") as f:
                        workspace_dict = json.load(f)
                    workspace_desc = workspace_dict.get(scene_name, "Description not found.")
                except FileNotFoundError:
                    workspace_desc = "thor_workspaces.json file not found."

            # Parse workspace objects & counts into map
            ws_objects_map = {}
            raw_matches = re.findall(r'([A-Za-z0-9_-]+)\s*\((?:instances?|count)?:\s*(\d+)\)', workspace_desc,
                                     re.IGNORECASE)
            for obj_name, count in raw_matches:
                count_val = int(count)
                norm_plain = re.sub(r'[^a-z0-9]', '', obj_name.lower())
                ws_objects_map[norm_plain] = (obj_name, count_val)

            ws_normalized_text = re.sub(r'[^a-z0-9]', '', workspace_desc.lower())

            def normalize_obj_key(s):
                s_clean = re.sub(r'^(dirty|clean|broken|open|opened|closed|turned on|turned off|lit|unlit)\s+', '',
                                 str(s).strip(), flags=re.IGNORECASE)
                return re.sub(r'[^a-z0-9]', '', s_clean.lower())

            # Extract Target Objects
            extracted_targets = set()

            for field in ['objects', 'involved_objects']:
                items = task.get(field)
                if isinstance(items, list):
                    for item in items:
                        if item and str(item).strip():
                            extracted_targets.add(str(item).strip())

            final_state = task.get('final_state')
            if isinstance(final_state, list):
                for fs in final_state:
                    if isinstance(fs, dict):
                        if fs.get('objectType'):
                            extracted_targets.add(str(fs['objectType']).strip())
                        if isinstance(fs.get('parentReceptacles'), list):
                            for receptacle in fs['parentReceptacles']:
                                if receptacle:
                                    extracted_targets.add(str(receptacle).strip())

            steps = task.get('step')
            if isinstance(steps, list):
                for s in steps:
                    s_str = str(s).strip()
                    action_match = re.search(
                        r'^(?:find|pick|turn\s+on|turn\s+off|open|close|put|drop|dirty|clean|fillLiquid|pour|break|slice|use)\s+(.+)$',
                        s_str, re.IGNORECASE)
                    if action_match:
                        extracted_targets.add(action_match.group(1).strip())

            instruction_combined = (
                        str(task.get('instruction') or '') + " " + str(task.get('risk_instruction') or '')).lower()
            for norm_key, (raw_name, _) in ws_objects_map.items():
                raw_spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', raw_name).lower()
                if raw_spaced in instruction_combined or raw_name.lower() in instruction_combined:
                    extracted_targets.add(raw_name)

            if extracted_targets:
                st.write("**Object Presence Check:**")
                unique_targets = {}
                for tgt in extracted_targets:
                    norm_k = normalize_obj_key(tgt)
                    if norm_k not in unique_targets:
                        unique_targets[norm_k] = tgt

                for norm_k, display_tgt in sorted(unique_targets.items(), key=lambda x: x[1]):
                    match_info = ws_objects_map.get(norm_k)
                    if not match_info:
                        if norm_k.endswith('s') and norm_k[:-1] in ws_objects_map:
                            match_info = ws_objects_map.get(norm_k[:-1])
                        elif norm_k + 's' in ws_objects_map:
                            match_info = ws_objects_map.get(norm_k + 's')

                    is_in_workspace = False
                    found_count = None

                    if match_info:
                        is_in_workspace = True
                        _, found_count = match_info
                    else:
                        if norm_k in ws_normalized_text or (norm_k.endswith('s') and norm_k[:-1] in ws_normalized_text):
                            is_in_workspace = True
                            count_match = re.search(rf'{re.escape(display_tgt)}\s*\((?:instances?|count)?:\s*(\d+)\)',
                                                    workspace_desc, re.IGNORECASE)
                            if count_match:
                                found_count = int(count_match.group(1))

                    if is_in_workspace:
                        if found_count is not None:
                            st.markdown(f"- ✅ **{display_tgt}** is present (Instances: {found_count})")
                        else:
                            st.markdown(f"- ✅ **{display_tgt}** is present in the scene")
                    else:
                        st.markdown(f"- ❌ **{display_tgt}** was NOT found in the scene")
            else:
                st.write("**Object Presence Check:** No task-specific objects identified.")

            with st.expander("🔍 View Complete Workspace State (Objects & Coordinates)", expanded=False):
                st.code(workspace_desc, language="text")

    with col2:
        st.subheader("Command / Instruction")
        if dataset_choice == "SaGC":
            st.success(f"**Goal:** {task.get('goal', 'N/A')}")
            st.write(f"**Task Category:** {task.get('task', 'N/A')}")
        elif dataset_choice == "AmbiK":
            instr = task.get('eval_instruction') or task.get('instruction') or task.get('Ambiguous Task') or task.get(
                'Unambiguous Direct') or 'N/A'
            st.success(f"**Instruction:** {instr}")
        elif dataset_choice == "SafeAgentBench":
            instr = task.get('instruction') or task.get('risk_instruction') or 'N/A'
            st.success(f"**Instruction:** {instr}")


# --- Reusable Component 2: Evaluation Form Widgets ---
def render_evaluation_widgets(task_id, ambiguity_options=None):
    """
    Renders the evaluation questions with nothing preselected.
    Returns a dictionary of current selections (None for unanswered questions).
    """
    ambiguity_options = list(ambiguity_options or core.AMBIGUITY_OPTIONS)
    current_selected = st.session_state.get(f"ambig_type_{task_id}", [])
    for item in current_selected:
        if item and item not in ambiguity_options:
            ambiguity_options.append(item)

    clarity_val = st.radio(
        "1. Clarity (Does the robot have enough clear information?)",
        core.CLARITY_OPTIONS, index=None, key=f"clarity_{task_id}"
    )

    ambiguity_type_str = "N/A"
    selected_types = []
    if core.needs_types(clarity_val):
        selected_types = st.multiselect(
            "1b. Select all applicable ambiguity types:",
            ambiguity_options,
            key=f"ambig_type_{task_id}"
        )
        ambiguity_type_str = json.dumps(selected_types) if selected_types else ""

    feasibility_val = st.radio(
        "2. Feasibility (Can this be done with the given objects/environment?)",
        core.FEASIBILITY_OPTIONS, index=None, key=f"feas_{task_id}"
    )

    safety_val = st.radio(
        "3. Safety (Is this dangerous to humans, the environment, or the robot?)",
        core.SAFETY_OPTIONS, index=None, key=f"safe_{task_id}"
    )

    comments_val = st.text_area(
        "4. Additional Comments (Optional)",
        placeholder="Note any ground truth errors or edge cases...",
        key=f"comment_{task_id}"
    )

    return {
        "clarity": clarity_val,
        "ambiguity_types": selected_types,
        "ambiguity_type_str": ambiguity_type_str,
        "feasibility": feasibility_val,
        "safety": safety_val,
        "comments": comments_val
    }


def prefill_widgets(task_id, answers):
    """Show saved answers when an annotator goes back to an item (only valid option values)."""
    if f"clarity_{task_id}" in st.session_state or not answers:
        return
    for key, value, options in ((f"clarity_{task_id}", answers.get("clarity"), core.CLARITY_OPTIONS),
                                (f"feas_{task_id}", answers.get("feasibility"), core.FEASIBILITY_OPTIONS),
                                (f"safe_{task_id}", answers.get("safety"), core.SAFETY_OPTIONS)):
        if value in options:
            st.session_state[key] = value
    st.session_state[f"ambig_type_{task_id}"] = list(answers.get("ambiguity_types") or [])
    st.session_state[f"comment_{task_id}"] = answers.get("comments") or ""


def answers_from_row(row):
    raw = row[core.COL["Ambiguity_Type"]]
    try:
        types = json.loads(raw) if raw not in ("", "N/A") else []
        types = types if isinstance(types, list) else [raw]
    except json.JSONDecodeError:
        types = [x.strip() for x in raw.split(',') if x.strip()]
    return {"clarity": row[core.COL["Clarity"]], "ambiguity_types": types,
            "feasibility": row[core.COL["Feasibility"]], "safety": row[core.COL["Safety"]],
            "comments": row[core.COL["Comments"]]}


# --- Training / Calibration Session Module ---
def render_instructions(instructions):
    for line in instructions:
        st.markdown(f"📌 {line}")


def render_training_module(sh, annotator_id, training_version, train_tasks, instructions=()):
    """Calibration session with the same display and questions as the live evaluation."""
    st.info(
        "🎓 **Annotator Calibration Session**: Complete these benchmark tasks to align with evaluation standards before proceeding to live tasks.")
    render_instructions(instructions)

    if 'train_idx' not in st.session_state:
        st.session_state.train_idx = 0
    if 'show_feedback' not in st.session_state:
        st.session_state.show_feedback = False

    t_idx = st.session_state.train_idx

    if t_idx >= len(train_tasks):
        st.success("🎉 **Calibration Complete!** You have unlocked the live dataset evaluation queue.")
        if st.button("Proceed to Live Evaluations 🚀", type="primary", key="proceed_live"):
            row = [annotator_id, core.TRAINING_MARKER, "PASSED", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
                   "N/A", "Completed calibration", core.INTERFACE_VERSION, training_version]
            try:
                write_row(sh, row)
            except core.SheetError:
                st.error(SHEET_ERROR)
                return
            st.session_state.is_trained = True
            st.rerun()
        return

    task = train_tasks[t_idx]
    st.write(f"### Calibration Task {t_idx + 1} of {len(train_tasks)} ({task.get('dataset', 'Benchmark')})")

    # 1. Reuse task display
    render_task_display(task, dataset_choice=task.get('dataset', 'AmbiK'))
    st.divider()

    # 2. Reuse evaluation form
    eval_inputs = render_evaluation_widgets(
        task_id=f"train_{task.get('id', t_idx)}",
        ambiguity_options=task.get('available_ambiguity_options')
    )

    # 3. Check answer & feedback
    if not st.session_state.show_feedback:
        complete = core.answers_complete(eval_inputs)
        if not complete:
            st.caption("Answer every question to check your answer.")
        if st.button("Check Answer", type="primary", key=f"check_btn_{task.get('id', t_idx)}",
                     disabled=not complete):
            st.session_state.show_feedback = True
            st.rerun()
    else:
        fb = core.training_feedback(task, eval_inputs)
        if fb["all"]:
            st.success("✅ **Excellent! Your response matches the expert benchmark.**")
        else:
            differ = [name for name, label in (("clarity", "Clarity"), ("ambiguity_types", "Ambiguity types"),
                                               ("feasibility", "Feasibility"), ("safety", "Safety"))
                      if not fb[name]]
            st.warning("⚠️ **Your evaluation differed from the benchmark standard** on: "
                       + ", ".join({"clarity": "Clarity", "ambiguity_types": "Ambiguity types",
                                    "feasibility": "Feasibility", "safety": "Safety"}[d] for d in differ) + ".")

        with st.expander("💡 View Benchmark Ground Truth & Expert Explanation", expanded=True):
            st.markdown(f"**Expected Clarity:** {task.get('expected_clarity')}")
            if task.get('expected_ambiguity_types'):
                st.markdown(f"**Expected Ambiguity Type(s):** {', '.join(task.get('expected_ambiguity_types'))}")
            st.markdown(f"**Expected Feasibility:** {task.get('expected_feasibility')}")
            st.markdown(f"**Expected Safety:** {task.get('expected_safety')}")
            st.info(f"**Expert Explanation:** {task.get('explanation', 'N/A')}")

        if st.button("Next Training Task ➡️", type="primary", key=f"next_train_btn_{task.get('id', t_idx)}"):
            st.session_state.train_idx += 1
            st.session_state.show_feedback = False
            st.rerun()


# --- Sign-in ---
REGISTER, RETURNING = core.REGISTER, core.RETURNING


def registration_worksheet(sh):
    try:
        return sh.worksheet(core.REG_SHEET)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=core.REG_SHEET, rows=1000, cols=len(core.REG_HEADER))
        ws.append_row(core.REG_HEADER, value_input_option="RAW")
        return ws


def authenticate(sh, annotator_id, pin, pin_again, mode):
    """Register a new ID or check a returning user's PIN. Returns an error message or None."""
    try:
        ws = registration_worksheet(sh)
        reg = core.find_registration(read_values(ws), annotator_id)
        if mode == REGISTER:
            if reg is not None:
                return core.TAKEN
            error = core.validate_pin(pin)
            if error:
                return error
            if str(pin).strip() != str(pin_again or "").strip():
                return "The two PINs differ."
            row = core.registration_row(annotator_id, pin)
            ws.append_row(row, value_input_option="RAW")
            # Two registrations of one ID at the same moment: the first row wins.
            first = core.find_registration(read_values(ws), annotator_id)
            return None if first is not None and first[1] == row[1] else core.TAKEN
        if reg is None or not core.verify_pin(pin, reg[1]):
            return core.NOT_RECOGNISED
        return None
    except core.SheetError:
        raise
    except Exception as e:
        log.exception("the registrations worksheet failed")
        raise core.SheetError("registration failed") from e


def sign_in(typed_id, pin, pin_again, study_code, mode, subset, training_version):
    """Check the study code, register or check the PIN, read the sheet, restore progress.

    Returns an error message or None. Nothing is read from the sheet before the study code
    and the ID format are checked."""
    error = core.check_study_code(secret("STUDY_CODE"), study_code)
    if error:
        return error
    annotator_id, error = core.validate_id(typed_id)
    if error:
        return error
    if mode == RETURNING and not str(pin or "").strip():
        return core.NOT_RECOGNISED
    sh = open_spreadsheet()
    error = authenticate(sh, annotator_id, pin, pin_again, mode)
    if error:
        return error
    ws = sh.sheet1
    values = read_values(ws)
    ensure_header(ws, values)
    labeled = core.labeled_items(values, annotator_id)
    st.session_state.annotator_id = annotator_id
    st.session_state.is_trained = core.training_done(values, annotator_id, training_version)
    st.session_state.answers = {tid: answers_from_row(row) for (_, tid), row in labeled.items()}
    st.session_state.order = core.annotator_order(subset, annotator_id)
    st.session_state.pos = None
    st.session_state.logged_blocks = set()
    st.session_state.train_idx = 0
    st.session_state.show_feedback = False
    return None


def first_open(flat, answers):
    return next((i for i, (_, tid) in enumerate(flat) if tid not in answers), None)


# --- Main Application Session ---
def main():
    st.title("🤖 Robotics Dataset Evaluation Interface")
    try:
        subset, items = load_subset_items()
        training_version, train_tasks = core.load_training(TRAINING_FILE)
        instructions = core.load_instructions(TRAINING_FILE)
    except Exception:
        log.exception("loading the task files failed")
        st.error("The task files could not be loaded. Please contact the study team.")
        st.stop()

    # --- Sidebar Setup ---
    st.sidebar.header("Annotator Control Panel")
    if not st.session_state.get('annotator_id'):
        mode = st.sidebar.radio("Sign-in", [RETURNING, REGISTER], key="login_mode")
        typed_code = st.sidebar.text_input("Study code:", type="password", key="login_study_code")
        typed_id = st.sidebar.text_input("Your ID:", key="login_id", max_chars=40,
                                         help=core.ID_RULE)
        typed_pin = st.sidebar.text_input("PIN:", type="password", key="login_pin",
                                          help=core.PIN_RULE if mode == REGISTER else None)
        typed_pin2 = (st.sidebar.text_input("Repeat the PIN:", type="password", key="login_pin2")
                      if mode == REGISTER else None)
        if mode == REGISTER:
            st.sidebar.caption("Remember your ID and PIN: you need both to continue later.")
        label = "Register and start" if mode == REGISTER else "Start / Resume Evaluation"
        if st.sidebar.button(label, disabled=not typed_id.strip(), key="login_btn"):
            try:
                message = sign_in(typed_id, typed_pin, typed_pin2, typed_code, mode, subset, training_version)
            except core.SheetError:
                st.error(SHEET_ERROR)
                st.stop()
            if message:
                st.sidebar.error(message)
            else:
                st.rerun()
        st.info("👈 Enter the study code, then sign in with your ID and PIN (first visit: register) in the sidebar.")
        return

    annotator_id = st.session_state.annotator_id
    st.sidebar.success(f"Signed in as **{annotator_id}**")
    if st.sidebar.button("Sign out", key="logout_btn"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    try:
        sh = open_spreadsheet()
    except core.SheetError:
        st.error(SHEET_ERROR)
        st.stop()

    # Routing: Calibration vs. Live Evaluation Queue
    if not st.session_state.is_trained:
        render_training_module(sh, annotator_id, training_version, train_tasks, instructions)
        return

    order = st.session_state.order
    flat = [(b, tid) for b, ids in order for tid in ids]
    answers = st.session_state.answers
    if st.session_state.pos is None:
        st.session_state.pos = first_open(flat, answers)
    pos = st.session_state.pos
    n_done = sum(1 for _, tid in flat if tid in answers)
    st.sidebar.progress(n_done / len(flat), text=f"{n_done} of {len(flat)} items done")

    if pos is None:
        st.success("🎉 You have completed every item. Thank you!")
        return

    block, task_id = flat[pos]
    block_ids = dict(order)[block]
    if block not in st.session_state.logged_blocks:
        try:
            log_block_order(sh, annotator_id, subset, block, block_ids)
        except core.SheetError:
            st.error(SHEET_ERROR)
            st.stop()
        st.session_state.logged_blocks.add(block)

    meta = next(it for it in subset["items"] if it["id"] == task_id)
    dataset_choice = meta["dataset"]
    current_task = items[task_id]
    st.markdown(f"### Block {block} of {len(order)} · item {block_ids.index(task_id) + 1} of {len(block_ids)} "
                f"({dataset_choice})")
    st.caption("You can stop at any time; your answers are saved and you continue where you left off.")
    render_instructions(instructions)

    # Navigation Bar
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("⬅️ Previous Task", disabled=(pos == 0), key=f"prev_{task_id}"):
            st.session_state.pos -= 1
            st.rerun()
    with nav_col3:
        if st.button("Next Task ➡️", disabled=(task_id not in answers or pos >= len(flat) - 1),
                     key=f"next_{task_id}"):
            st.session_state.pos += 1
            st.rerun()

    # 1. Render Task Display
    render_task_display(current_task, dataset_choice)
    st.divider()

    # 2. Render Evaluation Widgets (saved answers shown when going back)
    prefill_widgets(task_id, answers.get(task_id))
    st.subheader("Evaluate This Task")
    eval_inputs = render_evaluation_widgets(task_id)

    # 3. Submit Action
    complete = core.answers_complete(eval_inputs)
    if not complete:
        st.caption("Answer every question to enable Submit (comments are optional).")
    if st.button("Submit Evaluation & Next", key=f"submit_{task_id}", type="primary", disabled=not complete):
        row = build_row(
            dataset_choice, current_task,
            eval_inputs["clarity"], eval_inputs["ambiguity_type_str"],
            eval_inputs["feasibility"], eval_inputs["safety"],
            eval_inputs["comments"], annotator_id, training_version
        )
        try:
            write_row(sh, row)
        except core.SheetError:
            st.error(SHEET_ERROR)
            return
        answers[task_id] = {k: eval_inputs[k] for k in
                            ("clarity", "ambiguity_types", "feasibility", "safety", "comments")}
        st.toast("Response saved to database!", icon="✅")
        st.session_state.pos = first_open(flat, answers)
        st.rerun()


if __name__ == "__main__":
    main()
