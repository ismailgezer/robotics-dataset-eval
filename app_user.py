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
STUDY_CONFIG = "study_config.json"

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


def log_block_order(sh, annotator_id, subset, dataset, block, ids):
    """One allocations row per (annotator, interface, subset version, dataset, block)."""
    try:
        ws = allocation_worksheet(sh)
        if not core.allocation_logged(read_values(ws), annotator_id, subset["version"], dataset, block):
            ws.append_row(core.allocation_row(annotator_id, subset, dataset, block, ids), value_input_option="RAW")
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
@st.cache_resource
def load_thor_workspaces():
    with open("thor_workspaces.json", "r", encoding="utf-8") as f:
        return json.load(f)


def render_task_display(task, dataset_choice, description=None):
    """Shows the dataset description (small), the task context and the instruction.

    AmbiK: environment_short; SaGC: the models' workspace text with the robot type; SafeAgentBench:
    the compact object list (states, containers, properties) and the raw description in an expander.
    Nothing the models do not see is shown (no labels, ambiguity types, plans or derived checks)."""
    if description:
        st.caption(description)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Environment & Workspace")
        st.code(core.context_text(task, dataset_choice, load_thor_workspaces()), language=None, wrap_lines=True)
        if dataset_choice == "SafeAgentBench":
            with st.expander("Full scene description", expanded=False):
                st.code(load_thor_workspaces()[task["scene_name"]], language=None)

    with col2:
        st.subheader("Command / Instruction")
        label = "Goal" if dataset_choice == "SaGC" else "Instruction"
        st.success(f"**{label}:** {core.instruction_text(task, dataset_choice)}")


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
def render_training_module(sh, annotator_id, training_version, train_tasks, descriptions=None):
    """Calibration session with the same display and questions as the live evaluation."""
    st.info(
        "🎓 **Annotator Calibration Session**: Complete these benchmark tasks to align with evaluation standards before proceeding to live tasks.")

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
    dataset = task.get('dataset', 'AmbiK')
    render_task_display(task, dataset, (descriptions or {}).get(dataset))
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
def sign_in(typed_id, subset, training_version):
    """Check the ID, read the sheet, restore progress. Returns an error message or None.

    Nothing is read from the sheet before the ID format is checked."""
    annotator_id, error = core.validate_id(typed_id)
    if error:
        return error
    sh = open_spreadsheet()
    ws = sh.sheet1
    values = read_values(ws)
    ensure_header(ws, values)
    labeled = core.labeled_items(values, annotator_id)
    st.session_state.annotator_id = annotator_id
    st.session_state.is_trained = core.training_done(values, annotator_id, training_version)
    st.session_state.answers = {tid: answers_from_row(row) for (_, tid), row in labeled.items()}
    st.session_state.welcome = core.welcome_message(annotator_id, len(st.session_state.answers))
    st.session_state.order = core.annotator_order(subset, annotator_id)
    st.session_state.pos = {}
    # an annotator who already answered optional extra items continues with them
    targets = core.load_targets(STUDY_CONFIG)
    st.session_state.extras = {
        ds: core.progress(core.flat_order(st.session_state.order, ds), st.session_state.answers, targets[ds])[1] > 0
        for ds in core.DATASETS}
    st.session_state.logged_blocks = set()
    st.session_state.train_idx = 0
    st.session_state.show_feedback = False
    return None


# --- Main Application Session ---
def main():
    st.title("🤖 Robotics Dataset Evaluation Interface")
    try:
        subset, items = load_subset_items()
        training_version, train_tasks = core.load_training(TRAINING_FILE)
        targets = core.load_targets(STUDY_CONFIG)
        descriptions = core.load_descriptions(STUDY_CONFIG)
    except Exception:
        log.exception("loading the task files failed")
        st.error("The task files could not be loaded. Please contact the study team.")
        st.stop()

    # --- Sidebar Setup ---
    st.sidebar.header("Annotator Control Panel")
    if not st.session_state.get('annotator_id'):
        typed_id = st.sidebar.text_input("Your ID (first and last name):", key="login_id", max_chars=40,
                                         help=core.ID_RULE)
        if st.sidebar.button("Start / Resume Evaluation", disabled=not typed_id.strip(), key="login_btn"):
            try:
                message = sign_in(typed_id, subset, training_version)
            except core.SheetError:
                st.error(SHEET_ERROR)
                st.stop()
            if message:
                st.sidebar.error(message)
            else:
                st.rerun()
        st.info("👈 Enter your first and last name as your ID in the sidebar, and use the same one every time.")
        return

    annotator_id = st.session_state.annotator_id
    st.sidebar.success(f"Signed in as **{annotator_id}**")
    if st.session_state.get("welcome"):
        st.info(f"👋 {st.session_state.welcome}. If this is not you, sign out and use your own ID.")
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
        render_training_module(sh, annotator_id, training_version, train_tasks, descriptions)
        return

    order = st.session_state.order
    answers = st.session_state.answers
    extras = st.session_state.extras
    st.sidebar.divider()
    dataset_choice = st.sidebar.radio("Select Dataset to Evaluate:", core.DATASETS, key="dataset_choice")
    for ds in core.DATASETS:
        done, extra = core.progress(core.flat_order(order, ds), answers, targets[ds])
        line = f"{ds}: {done} of {targets[ds]}" + (f" · {extra} extra" if extra else "")
        st.sidebar.markdown(f"**{line}**")

    flat = core.flat_order(order, dataset_choice)
    target = min(targets[dataset_choice], len(flat))
    done, extra = core.progress(flat, answers, target)
    pos_map = st.session_state.pos
    if pos_map.get(dataset_choice) is None:
        pos_map[dataset_choice] = core.next_open(flat, answers, target, extras[dataset_choice])
    pos = pos_map[dataset_choice]

    if pos is None:
        if not extras[dataset_choice] and core.next_open(flat, answers, target, True) is not None:
            st.success(f"You’ve completed the {dataset_choice} set, thank you!")
            if st.button("Continue with optional extra items", key=f"extras_{dataset_choice}"):
                extras[dataset_choice] = True
                pos_map[dataset_choice] = core.next_open(flat, answers, target, True)
                st.rerun()
        else:
            st.success(f"🎉 You have completed every {dataset_choice} item. Thank you!")
        return

    block, task_id = flat[pos]
    block_ids = dict(order[dataset_choice])[block]
    if (dataset_choice, block) not in st.session_state.logged_blocks:
        try:
            log_block_order(sh, annotator_id, subset, dataset_choice, block, block_ids)
        except core.SheetError:
            st.error(SHEET_ERROR)
            st.stop()
        st.session_state.logged_blocks.add((dataset_choice, block))

    current_task = items[task_id]
    described = st.session_state.setdefault("described", set())
    if dataset_choice not in described:  # shown when a dataset is selected, until its first answer
        st.info(f"**{dataset_choice}:** {descriptions[dataset_choice]}")
    heading = f"### Evaluating **{dataset_choice}** · {done} of {target}"
    if extras[dataset_choice]:
        heading += f" · optional extra items: {extra}"
    st.markdown(heading)
    st.caption("You can stop at any time or switch datasets; your answers are saved and you continue where you left off.")

    # Navigation Bar
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("⬅️ Previous Task", disabled=(pos == 0), key=f"prev_{task_id}"):
            pos_map[dataset_choice] = pos - 1
            st.rerun()
    with nav_col3:
        last = (len(flat) if extras[dataset_choice] else target) - 1
        if st.button("Next Task ➡️", disabled=(task_id not in answers or pos >= last),
                     key=f"next_{task_id}"):
            pos_map[dataset_choice] = pos + 1
            st.rerun()

    # 1. Render Task Display
    render_task_display(current_task, dataset_choice, descriptions[dataset_choice])
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
        # column F: the workspace text the annotator saw (the models' context)
        row[core.COL["Workspace"]] = core.context_text(current_task, dataset_choice, load_thor_workspaces())
        try:
            write_row(sh, row)
        except core.SheetError:
            st.error(SHEET_ERROR)
            return
        answers[task_id] = {k: eval_inputs[k] for k in
                            ("clarity", "ambiguity_types", "feasibility", "safety", "comments")}
        st.toast("Response saved to database!", icon="✅")
        st.session_state.welcome = None
        described.add(dataset_choice)
        pos_map[dataset_choice] = core.next_open(flat, answers, target, extras[dataset_choice])
        st.rerun()


if __name__ == "__main__":
    main()
