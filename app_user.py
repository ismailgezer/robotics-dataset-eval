import streamlit as st
import pandas as pd
import json
import os
import gspread
import re

# --- Page Configuration ---
st.set_page_config(page_title="Robotics Dataset Label Evaluation", layout="wide")

if st.session_state.get('scroll_to_top', False):
    st.markdown(
        """
        <script>
            var body = window.parent.document.querySelector(".main");
            if (body) { body.scrollTop = 0; }
        </script>
        """,
        unsafe_allow_html=True
    )
    st.session_state.scroll_to_top = False

# File Paths
SAGC_SUBSET = "augment.json"
AMBIK_SUBSET = "ambik_test_900.csv"
SAFE_SUBSET = "mixed_detailed_1009.jsonl"
TRAINING_FILE = "training_tasks.json"

# --- Database Connection ---
@st.cache_resource
def init_connection():
    """Initializes Google Sheets connection securely."""
    if "GOOGLE_CREDENTIALS" in st.secrets:
        sa_info = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
        return gspread.service_account_from_dict(sa_info)
    else:
        return gspread.service_account(filename="service_account.json")


def get_sheet_url():
    if "SHEET_URL" in st.secrets:
        return st.secrets["SHEET_URL"]


def get_existing_evaluations(gc):
    """Fetches all existing evaluations to filter assigned tasks."""
    try:
        sh = gc.open_by_url(get_sheet_url())
        worksheet = sh.sheet1
        data = worksheet.get_all_records()
        if data:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(columns=["Annotator_ID", "Dataset", "Task_ID"])
    except Exception as e:
        st.warning(f"Could not load previous evaluations. Starting fresh. ({e})")
        return pd.DataFrame(columns=["Annotator_ID", "Dataset", "Task_ID"])

def save_result(gc, dataset, task_data, clarity, ambiguity_type, feasibility, safety, comments, annotator_id):
    """Saves or updates evaluation results in Google Sheets (Retroactive Upsert)."""
    if dataset == "SaGC":
        label = task_data.get('label')
        original_label = str(label) if label is not None else "N/A"
    elif dataset in ["AmbiK", "SafeAgentBench"]:
        original_label = task_data.get('ground_truth', 'N/A')
    else:
        original_label = "N/A"

    task_id = task_data.get('id', 'N/A')

    row_data = [
        str(annotator_id), str(dataset), str(task_id), str(original_label),
        str(clarity), str(ambiguity_type), str(feasibility), str(safety), str(comments)
    ]

    try:
        sh = gc.open_by_url(get_sheet_url())
        worksheet = sh.sheet1
        
        all_records = worksheet.get_all_records()
        existing_row_idx = None
        for idx, record in enumerate(all_records, start=2):  # Row 1 is header
            if (str(record.get('Annotator_ID')) == str(annotator_id) and 
                str(record.get('Dataset')) == str(dataset) and 
                str(record.get('Task_ID')) == str(task_id)):
                existing_row_idx = idx
                break

        if existing_row_idx:
            worksheet.update(f"A{existing_row_idx}:I{existing_row_idx}", [row_data])
            st.toast("Previous evaluation updated in database!", icon="🔄")
        else:
            worksheet.append_row(row_data)
            st.toast("Response saved to database!", icon="✅")
            
        return True
    except Exception as e:
        st.error(f"Error saving to Google Sheets: {e}")
        return False

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
            else: return []
        else: return []
        
        for i, rec in enumerate(records):
            if 'id' not in rec:
                rec['id'] = f"{dataset_prefix}_{i}"
                
        return records
    except Exception as e:
        st.error(f"Error loading {dataset_prefix}: {e}")
        return []

def allocate_tasks(raw_data, existing_df, dataset_name, annotator_id, sample_size):
    """Allocates available tasks, skipping tasks completed twice or already completed by this user."""
    if existing_df.empty:
        user_completed = set()
        counts = {}
    else:
        ds_records = existing_df[existing_df['Dataset'] == dataset_name]
        user_completed = set(ds_records[ds_records['Annotator_ID'].astype(str) == str(annotator_id)]['Task_ID'].astype(str))
        counts = ds_records['Task_ID'].astype(str).value_counts().to_dict()

    eligible_tasks = []
    for task in raw_data:
        tid = str(task['id'])
        if tid in user_completed:
            continue
        if counts.get(tid, 0) < 2:
            eligible_tasks.append(task)
            
    return eligible_tasks[:sample_size]

def check_user_trained(existing_df, annotator_id):
    """Checks whether the user has completed the calibration/training session."""
    if existing_df.empty:
        return False
    trained_records = existing_df[
        (existing_df['Annotator_ID'].astype(str) == str(annotator_id)) & 
        (existing_df['Dataset'] == '[TRAINING_COMPLETE]')
    ]
    return not trained_records.empty

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
            env = task.get('Environment Short') or task.get('environment_short') or task.get('Environment Full') or "N/A"
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
            raw_matches = re.findall(r'([A-Za-z0-9_-]+)\s*\((?:instances?|count)?:\s*(\d+)\)', workspace_desc, re.IGNORECASE)
            for obj_name, count in raw_matches:
                count_val = int(count)
                norm_plain = re.sub(r'[^a-z0-9]', '', obj_name.lower())
                ws_objects_map[norm_plain] = (obj_name, count_val)

            ws_normalized_text = re.sub(r'[^a-z0-9]', '', workspace_desc.lower())

            def normalize_obj_key(s):
                s_clean = re.sub(r'^(dirty|clean|broken|open|opened|closed|turned on|turned off|lit|unlit)\s+', '', str(s).strip(), flags=re.IGNORECASE)
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
                    action_match = re.search(r'^(?:find|pick|turn\s+on|turn\s+off|open|close|put|drop|dirty|clean|fillLiquid|pour|break|slice|use)\s+(.+)$', s_str, re.IGNORECASE)
                    if action_match:
                        extracted_targets.add(action_match.group(1).strip())

            instruction_combined = (str(task.get('instruction') or '') + " " + str(task.get('risk_instruction') or '')).lower()
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
                            count_match = re.search(rf'{re.escape(display_tgt)}\s*\((?:instances?|count)?:\s*(\d+)\)', workspace_desc, re.IGNORECASE)
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
            instr = task.get('eval_instruction') or task.get('instruction') or task.get('Ambiguous Task') or task.get('Unambiguous Direct') or 'N/A'
            st.success(f"**Instruction:** {instr}")
        elif dataset_choice == "SafeAgentBench":
            instr = task.get('instruction') or task.get('risk_instruction') or 'N/A'
            st.success(f"**Instruction:** {instr}")

# --- Reusable Component 2: Evaluation Form Widgets ---
def render_evaluation_widgets(task_id, ambiguity_options=None):
    """
    Renders standard evaluation options (Clarity, Dynamic Ambiguity Types, Feasibility, Safety, Comments).
    Returns a dictionary of current selections.
    """
    if ambiguity_options is None:
        ambiguity_options = [
             "Attribute: Referring to objects using vague attributes, or referring to object but not specifying some attribute.",
             "Placement: Uncertainty regarding where the object should be placed in situations such as the existence of multiple valid positions or an insufficiently defined location.",
             "Orientation: Uncertainty regarding how an object should be oriented, resulting from a lack of information such as axis, angle, and reference frame.",
             "Spatial Reference: Unclear reference frame.",
             "Linguistic: Ambiguities when a command’s wording can be interpreted in multiple ways due to language conventions, implicit assumptions, or mismatch with the physical context.",
             "Numeric: Underspecifying quantity.",
             "Preference: Lacking user preference or necessary environmental context.",
             "Referential: Ambiguities where the reference is subtle.",
             "Commonsense: Tasks that are feasible to perform and which pose no danger when executed yet run counter to common sense."
        ]

    clarity_val = st.radio(
        "1. Clarity (Does the robot have enough clear information?)",
        ["Clear & Actionable",
         "Mostly clear but requires making one or more minor assumptions. "
         "You would ask another person a clarifying question less than 20% of the time.",
         "Mostly clear but requires making one or more minor assumptions.  You would ask another person a clarifying "
         "question 20-50% of the time, depending on if you thought the person cared or not.",
         "Ambiguous (missing context/underspecified) – you would ask another person a clarifying question more than 50% of the time. "
         "Also in this category are things that defy common sense.",
         "Ambiguous (Poor English / Confusing Phrasing)"],
        key=f"clarity_{task_id}"
    )

    ambiguity_type_str = "N/A"
    selected_types = []
    if "Ambiguous" in clarity_val or "Mostly clear" in clarity_val:
        selected_types = st.multiselect(
            "1b. Select all applicable ambiguity types:",
            ambiguity_options,
            key=f"ambig_type_{task_id}"
        )
        ambiguity_type_str = ", ".join(selected_types) if selected_types else ""

    feasibility_val = st.radio(
        "2. Feasibility (Can this be done with the given objects/environment?)",
        ["Feasible",
         "Infeasible (Physically impossible / Missing crucial items)",
         "Cannot be determined from the given information without making assumptions."],
        key=f"feas_{task_id}"
    )

    safety_val = st.radio(
        "3. Safety (Is this dangerous to humans, the environment, or the robot?)",
        ["Safe",
         "Unsafe (Poses a risk, slip hazard, breakage, etc.)  In most normal household environments, this would be deemed something you would not want a small child to do.  "
         "If it violates common sense and also is unsafe, choose this option.",
         "May or may not be safe depending on the particulars of the environment and objects.  If it violates common sense but is potentially safe, choose this option."],
        key=f"safe_{task_id}"
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

# --- Training / Calibration Session Module ---
def render_training_module(gc, annotator_id):
    """Renders calibration session using exact same UI components as live evaluation."""
    st.info("🎓 **Annotator Calibration Session**: Complete these benchmark tasks to align with evaluation standards before proceeding to live tasks.")
    
    try:
        with open(TRAINING_FILE, "r") as f:
            train_tasks = json.load(f)
    except FileNotFoundError:
        st.warning("No 'training_tasks.json' found. Proceeding directly to live evaluation.")
        return True

    if 'train_idx' not in st.session_state:
        st.session_state.train_idx = 0
    if 'show_feedback' not in st.session_state:
        st.session_state.show_feedback = False

    t_idx = st.session_state.train_idx

    if t_idx >= len(train_tasks):
        st.success("🎉 **Calibration Complete!** You have unlocked the live dataset evaluation queue.")
        if st.button("Proceed to Live Evaluations 🚀", type="primary"):
            save_result(gc, "[TRAINING_COMPLETE]", {"id": "PASSED"}, "N/A", "N/A", "N/A", "N/A", "Completed calibration", annotator_id)
            st.session_state.is_trained = True
            st.session_state.scroll_to_top = True
            st.rerun()
        return False

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
        if st.button("Check Answer", type="primary", key=f"check_btn_{task.get('id', t_idx)}"):
            if eval_inputs["clarity"] == "Ambiguous" and not eval_inputs["ambiguity_type_str"]:
                st.error("⚠️ Please select at least one ambiguity type before checking your answer.")
            else:
                st.session_state.show_feedback = True
                st.rerun()
    else:
        # Benchmark validation check
        c_match = eval_inputs["clarity"] == task.get('expected_clarity')
        f_match = eval_inputs["feasibility"] == task.get('expected_feasibility')
        s_match = eval_inputs["safety"] == task.get('expected_safety')
        
        ambig_match = True
        if task.get('expected_ambiguity_types'):
            expected_set = set(task['expected_ambiguity_types'])
            user_set = set(eval_inputs['ambiguity_types'])
            ambig_match = expected_set.issubset(user_set) or expected_set == user_set

        if c_match and f_match and s_match and ambig_match:
            st.success("✅ **Excellent! Your response matches the expert benchmark.**")
        else:
            st.warning("⚠️ **Your evaluation differed from the benchmark standard.**")

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
            st.session_state.scroll_to_top = True
            st.rerun()

    return False

# --- Main Application Session ---
def main():
    st.title("🤖 Robotics Dataset Evaluation Interface")
    
    # Initialize Session State
    if 'sampled_data' not in st.session_state:
        st.session_state.sampled_data = {}
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = {'SaGC': 0, 'AmbiK': 0, 'SafeAgentBench': 0}
    if 'annotator_id' not in st.session_state:
        st.session_state.annotator_id = None
    if 'is_trained' not in st.session_state:
        st.session_state.is_trained = False

    # --- Sidebar Setup ---
    st.sidebar.header("Annotator Control Panel")
    annotator_id = st.sidebar.text_input("Enter your Annotator ID:", value=st.session_state.annotator_id or "").strip()
    sample_size = st.sidebar.number_input("Tasks per dataset:", min_value=1, max_value=50, value=10)

    if st.sidebar.button("Start / Resume Evaluation", disabled=not annotator_id):
        gc = init_connection()
        existing_df = get_existing_evaluations(gc)
        
        st.session_state.annotator_id = annotator_id
        st.session_state.is_trained = check_user_trained(existing_df, annotator_id)

        raw_sagc = load_raw_data(SAGC_SUBSET, 'json', 'SaGC')
        raw_ambik = load_raw_data(AMBIK_SUBSET, 'csv', 'AmbiK')
        raw_safe = load_raw_data(SAFE_SUBSET, 'jsonl', 'SafeAgentBench')

        st.session_state.sampled_data['SaGC'] = allocate_tasks(raw_sagc, existing_df, "SaGC", annotator_id, sample_size)
        st.session_state.sampled_data['AmbiK'] = allocate_tasks(raw_ambik, existing_df, "AmbiK", annotator_id, sample_size)
        st.session_state.sampled_data['SafeAgentBench'] = allocate_tasks(raw_safe, existing_df, "SafeAgentBench", annotator_id, sample_size)

        st.session_state.current_idx = {'SaGC': 0, 'AmbiK': 0, 'SafeAgentBench': 0}

        is_returning = not existing_df[existing_df['Annotator_ID'].astype(str) == str(annotator_id)].empty if not existing_df.empty else False
        greeting = "Welcome back" if is_returning else "Welcome"
        total_allocated = sum(len(v) for v in st.session_state.sampled_data.values())
        
        st.sidebar.success(f"{greeting}, {annotator_id}! Allocated {total_allocated} total tasks across datasets.")
        st.session_state.scroll_to_top = True
        st.rerun()

    # Routing: Calibration vs. Live Evaluation Queue
    if st.session_state.annotator_id:
        gc = init_connection()

        if not st.session_state.is_trained:
            # Render Training Session
            completed_training = render_training_module(gc, st.session_state.annotator_id)
            if not completed_training:
                return

        # Render Live Evaluation Queue
        st.sidebar.divider()
        dataset_choice = st.sidebar.radio("Select Dataset to Evaluate:", ["SaGC", "AmbiK", "SafeAgentBench"])
        dataset_data = st.session_state.sampled_data.get(dataset_choice, [])

        if not dataset_data:
            st.info(f"No remaining tasks allocated for dataset '{dataset_choice}'. All tasks completed or max limit reached!")
            return

        current_idx = st.session_state.current_idx[dataset_choice]
        if current_idx >= len(dataset_data):
            st.success(f"🎉 You have completed all allocated tasks for '{dataset_choice}'!")
            return

        current_task = dataset_data[current_idx]
        task_id = current_task.get('id', 'N/A')

        st.markdown(f"### Evaluating **{dataset_choice}** (Task {current_idx + 1} of {len(dataset_data)})")

        # Navigation Bar
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            if st.button("⬅️ Previous Task", disabled=(current_idx == 0), key=f"prev_{dataset_choice}_{task_id}"):
                st.session_state.current_idx[dataset_choice] -= 1
                st.session_state.scroll_to_top = True
                st.rerun()
        with nav_col3:
            if st.button("Next Task ➡️", disabled=(current_idx >= len(dataset_data) - 1), key=f"next_{dataset_choice}_{task_id}"):
                st.session_state.current_idx[dataset_choice] += 1
                st.session_state.scroll_to_top = True
                st.rerun()

        # 1. Render Task Display
        render_task_display(current_task, dataset_choice)
        st.divider()

        # 2. Render Evaluation Widgets
        st.subheader("Evaluate This Task")
        eval_inputs = render_evaluation_widgets(task_id)

        # 3. Submit Action
        submit_btn = st.button("Submit Evaluation & Next", key=f"submit_{task_id}", type="primary")

        if submit_btn:
            if ("Ambiguous" in eval_inputs["clarity"] or "Mostly clear" in eval_inputs["clarity"]) and not eval_inputs["ambiguity_type_str"]:
                st.error("⚠️ Please select at least one ambiguity type before submitting.")
            else:
                success = save_result(
                    gc, dataset_choice, current_task,
                    eval_inputs["clarity"], eval_inputs["ambiguity_type_str"],
                    eval_inputs["feasibility"], eval_inputs["safety"],
                    eval_inputs["comments"], st.session_state.annotator_id
                )

                if success:
                    st.session_state.current_idx[dataset_choice] += 1
                    st.session_state.scroll_to_top = True
                    st.rerun()

    else:
        st.info("👈 Enter your Annotator ID in the sidebar to begin.")

if __name__ == "__main__":
    main()
