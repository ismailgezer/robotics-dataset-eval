import streamlit as st
import pandas as pd
import json
import os
import gspread
import re
from google.oauth2.service_account import Credentials

# --- Page Configuration ---
st.set_page_config(page_title="Robotics Dataset Label Evaluation", layout="wide")

# Invisible HTML anchor at the absolute top of the app
st.markdown("<div id='top-of-page'></div>", unsafe_allow_html=True)

# Scroll to top mechanism (Delayed JS to bypass Streamlit's scroll preservation)
if st.session_state.get('scroll_to_top', False):
    st.markdown(
        """
        <script>
            setTimeout(function() {
                var topElement = document.getElementById('top-of-page') || window.parent.document.getElementById('top-of-page');
                if (topElement) {
                    topElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 150);
        </script>
        """, 
        unsafe_allow_html=True
    )
    st.session_state.scroll_to_top = False

# --- Constants & Paths ---
SAGC_SUBSET = "augment.json"
AMBIK_SUBSET = "ambik_test_900.csv"
SAFE_SUBSET = "mixed_detailed_1009.jsonl"
TRAINING_TASKS_FILE = "training_tasks.json"
THOR_WORKSPACES_FILE = "thor_workspaces.json"

AMBIGUITY_OPTIONS = [
    "Missing context / Underspecified instruction",
    "Multiple valid target objects/locations",
    "Unclear action sequence / Order dependency",
    "Poor phrasing / Confusing language"
]

# --- Database & Auth Functions ---
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
    """Saves or updates (upserts) evaluation response in Google Sheets."""
    if dataset == "SaGC":
        label = task_data.get('label')
        original_label = str(label) if label is not None else "N/A"
    elif dataset in ["AmbiK", "SafeAgentBench"]:
        original_label = str(task_data.get('ground_truth', 'N/A'))
    else:
        original_label = "N/A"

    task_id = str(task_data.get('id', 'N/A'))

    row_data = [
        str(annotator_id), str(dataset), str(task_id), str(original_label),
        str(clarity), str(ambiguity_type), str(feasibility), str(safety), str(comments)
    ]

    try:
        sh = gc.open_by_url(get_sheet_url())
        worksheet = sh.sheet1
        
        # Check if an evaluation already exists for this annotator, dataset, and task_id
        all_records = worksheet.get_all_records()
        existing_row_idx = None
        for idx, record in enumerate(all_records, start=2):  # Row 1 is headers
            if (str(record.get('Annotator_ID')) == str(annotator_id) and 
                str(record.get('Dataset')) == str(dataset) and 
                str(record.get('Task_ID')) == str(task_id)):
                existing_row_idx = idx
                break

        if existing_row_idx:
            # Overwrite the existing row (Columns A through I)
            worksheet.update(f"A{existing_row_idx}:I{existing_row_idx}", [row_data])
            st.toast("Previous evaluation updated in database!", icon="🔄")
        else:
            worksheet.append_row(row_data)
            st.toast("Response saved to database!", icon="✅")
            
        return True
    except Exception as e:
        st.error(f"Error saving to Google Sheets: {e}")
        return False

# --- Data Loading & Preprocessing ---
def load_raw_data(file_path, file_type, dataset_prefix="Task"):
    """Loads raw dataset files and generates deterministic IDs and metadata standardizations."""
    if not os.path.exists(file_path):
        return []
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path)
            records = df.to_dict('records')
            
            # Expand AmbiK to evaluate both Clear and Ambiguous variants independently
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
        
        # Inject deterministic IDs for datasets that lack them
        for i, rec in enumerate(records):
            if 'id' not in rec:
                rec['id'] = f"{dataset_prefix}_{i}"
                
        return records
    except Exception as e:
        st.error(f"Error loading {dataset_prefix}: {e}")
        return []

def allocate_tasks(raw_tasks, existing_df, dataset_name, annotator_id, sample_size):
    """Allocates tasks ensuring max 2 unique annotators per task and excluding already completed ones."""
    if not raw_tasks:
        return []
    
    if existing_df.empty:
        return raw_tasks[:sample_size]
    
    # Filter out tasks this specific annotator has already completed
    user_evals = existing_df[
        (existing_df['Annotator_ID'].astype(str) == str(annotator_id)) & 
        (existing_df['Dataset'] == dataset_name)
    ]
    completed_ids = set(user_evals['Task_ID'].astype(str))
    
    # Count how many unique annotators have evaluated each task
    ds_evals = existing_df[existing_df['Dataset'] == dataset_name]
    counts = ds_evals.groupby('Task_ID')['Annotator_ID'].nunique().to_dict()
    
    eligible_tasks = []
    for task in raw_tasks:
        t_id = str(task['id'])
        if t_id in completed_ids:
            continue
        if counts.get(t_id, 0) < 2:
            eligible_tasks.append(task)
            
    return eligible_tasks[:sample_size]

def check_user_trained(existing_df, annotator_id):
    """Checks whether this annotator has completed the onboarding calibration training."""
    if existing_df.empty:
        return False
    trained_records = existing_df[
        (existing_df['Annotator_ID'].astype(str) == str(annotator_id)) & 
        (existing_df['Dataset'] == '[TRAINING_COMPLETE]')
    ]
    return not trained_records.empty

# --- Onboarding / Calibration Module ---
def render_training_module(gc, annotator_id):
    """Interactive training module with benchmark ground truth and expert explanations."""
    st.info("🎓 **Annotator Calibration Session**: Please complete these benchmark tasks to align with evaluation standards before proceeding to live tasks.")
    
    if not os.path.exists(TRAINING_TASKS_FILE):
        st.error(f"Missing '{TRAINING_TASKS_FILE}'. Skipping calibration phase.")
        return True

    try:
        with open(TRAINING_TASKS_FILE, "r") as f:
            train_tasks = json.load(f)
    except Exception as e:
        st.error(f"Error reading training file: {e}")
        return True

    if 'train_idx' not in st.session_state:
        st.session_state.train_idx = 0
    if 'show_feedback' not in st.session_state:
        st.session_state.show_feedback = False

    t_idx = st.session_state.train_idx

    if t_idx >= len(train_tasks):
        st.success("🎉 Calibration complete! You have unlocked the live evaluation queue.")
        if st.button("Proceed to Live Evaluations 🚀", type="primary"):
            # Record calibration completion in Google Sheets
            save_result(gc, "[TRAINING_COMPLETE]", {"id": "PASSED"}, "N/A", "N/A", "N/A", "N/A", "Completed onboarding", annotator_id)
            st.rerun()
        return False

    task = train_tasks[t_idx]
    st.write(f"### Training Task {t_idx + 1} of {len(train_tasks)} ({task['dataset']})")
    
    # Task Context Rendering
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Environment / Context")
        if task['dataset'] == "SaGC":
            if "scene" in task:
                st.write("**Floorplan Rooms:**", ", ".join(task["scene"].get("floorplan", [])))
                st.write("**Available Objects:**", ", ".join(task["scene"].get("objects", [])))
        elif task['dataset'] == "AmbiK":
            if "environment_short" in task:
                st.write("**Kitchen Items:**", task["environment_short"])
        elif task['dataset'] == "SafeAgentBench":
            if "scene_name" in task:
                st.write("**Scene Name:**", task["scene_name"])
            if "involved_objects" in task:
                st.write("**Involved Objects:**", ", ".join(task["involved_objects"]))

    with col2:
        st.subheader("Command / Instruction")
        st.success(f"**Instruction:** {task.get('instruction') or task.get('goal', 'N/A')}")

    st.divider()

    # Evaluation Form
    task_id = task['id']
    clarity_val = st.radio(
        "1. Clarity (Does the robot have enough clear information?)", 
        ["Clear & Actionable", "Ambiguous (Missing context / Underspecified)", "Ambiguous (Poor English / Confusing Phrasing)"],
        key=f"tr_clarity_{task_id}"
    )

    # Dynamic Ambiguity Type Question (Incorporated into Training as well!)
    ambiguity_type = "N/A"
    selected_types = []
    if "Ambiguous" in clarity_val:
        selected_types = st.multiselect(
            "1b. Select the specific type(s) of ambiguity:",
            AMBIGUITY_OPTIONS,
            key=f"tr_ambig_type_{task_id}"
        )
        ambiguity_type = ", ".join(selected_types) if selected_types else ""

    feas_val = st.radio(
        "2. Feasibility (Can this be done with the given objects/environment?)", 
        ["Feasible", "Infeasible (Physically impossible / Missing crucial items)"],
        key=f"tr_feas_{task_id}"
    )
    
    safety_val = st.radio(
        "3. Safety (Is this dangerous to humans, the environment, or the robot?)", 
        ["Safe", "Unsafe (Poses a risk, slip hazard, breakage, etc.)"],
        key=f"tr_safe_{task_id}"
    )

    if not st.session_state.show_feedback:
        if st.button("Check Answer", type="primary", key=f"tr_check_{task_id}"):
            # Form Validation Check for Ambiguity Selection
            if "Ambiguous" in clarity_val and not ambiguity_type:
                st.error("⚠️ Please select at least one ambiguity type before submitting.")
            else:
                st.session_state.show_feedback = True
                st.rerun()
    else:
        # Evaluate Accuracy
        c_match = clarity_val == task['expected_clarity']
        f_match = feas_val == task['expected_feasibility']
        s_match = safety_val == task['expected_safety']
        
        # Ambiguity Type Accuracy Check
        ambig_type_expected = task.get('expected_ambiguity_type')
        ambig_match = True
        if "Ambiguous" in clarity_val and ambig_type_expected:
            # Check if any expected type is in selected ambiguity types
            ambig_match = any(e.lower() in ambiguity_type.lower() for e in ambig_type_expected.split(','))

        if c_match and f_match and s_match and ambig_match:
            st.success("✅ **Excellent! All evaluations match the benchmark standard.**")
        else:
            st.warning("⚠️ **Your response differed from the benchmark standard in one or more categories.**")

        # Benchmark Explanation Section
        with st.expander("💡 View Benchmark Ground Truth & Explanation", expanded=True):
            st.markdown(f"**Expected Clarity:** {task['expected_clarity']}")
            if ambig_type_expected:
                st.markdown(f"**Expected Ambiguity Type(s):** {ambig_type_expected}")
            st.markdown(f"**Expected Feasibility:** {task['expected_feasibility']}")
            st.markdown(f"**Expected Safety:** {task['expected_safety']}")
            st.info(f"**Expert Explanation:** {task['explanation']}")

        if st.button("Next Training Task ➡️", type="primary", key=f"tr_next_{task_id}"):
            st.session_state.train_idx += 1
            st.session_state.show_feedback = False
            st.session_state.scroll_to_top = True
            st.rerun()

    return False

# --- Main Application Session Initialization ---
if 'sampled_data' not in st.session_state:
    st.session_state.sampled_data = {'SaGC': [], 'AmbiK': [], 'SafeAgentBench': []}
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = {'SaGC': 0, 'AmbiK': 0, 'SafeAgentBench': 0}
if 'annotator_id' not in st.session_state:
    st.session_state.annotator_id = ""

# --- Sidebar Controls ---
st.sidebar.title("Robotics Evaluation Panel")
annotator_id = st.sidebar.text_input("Enter Annotator ID / Username:", value=st.session_state.annotator_id).strip()
sample_size = st.sidebar.number_input("Sample size per dataset:", min_value=1, max_value=100, value=10)

if st.sidebar.button("Start / Resume Evaluation", disabled=not annotator_id):
    gc = init_connection()
    existing_df = get_existing_evaluations(gc)
    st.session_state.annotator_id = annotator_id
    
    # Check Onboarding Calibration Status
    is_trained = check_user_trained(existing_df, annotator_id)
    st.session_state.is_trained = is_trained
    
    if is_trained:
        raw_sagc = load_raw_data(SAGC_SUBSET, 'json', 'SaGC')
        raw_ambik = load_raw_data(AMBIK_SUBSET, 'csv', 'AmbiK')
        raw_safe = load_raw_data(SAFE_SUBSET, 'jsonl', 'SafeAgentBench')
        
        st.session_state.sampled_data['SaGC'] = allocate_tasks(raw_sagc, existing_df, "SaGC", annotator_id, sample_size)
        st.session_state.sampled_data['AmbiK'] = allocate_tasks(raw_ambik, existing_df, "AmbiK", annotator_id, sample_size)
        st.session_state.sampled_data['SafeAgentBench'] = allocate_tasks(raw_safe, existing_df, "SafeAgentBench", annotator_id, sample_size)
        
        st.session_state.current_idx = {'SaGC': 0, 'AmbiK': 0, 'SafeAgentBench': 0}
        
        is_returning = not existing_df[existing_df['Annotator_ID'].astype(str) == str(annotator_id)].empty
        greeting = "Welcome back" if is_returning else "Welcome"
        total_tasks = sum(len(v) for v in st.session_state.sampled_data.values())
        st.sidebar.success(f"{greeting}, {annotator_id}! Allocated {total_tasks} total tasks (Max {sample_size} per dataset).")

# --- Main App Logic Router ---
if not st.session_state.annotator_id:
    st.title("🤖 Robotics Evaluation Interface")
    st.warning("👈 Please enter your **Annotator ID** in the sidebar to begin.")
else:
    gc = init_connection()
    
    # Onboarding Routing
    if not st.session_state.get('is_trained', False):
        existing_df = get_existing_evaluations(gc)
        if not check_user_trained(existing_df, st.session_state.annotator_id):
            render_training_module(gc, st.session_state.annotator_id)
            st.stop()
        else:
            st.session_state.is_trained = True

    # --- Live Evaluation Interface ---
    st.title("🤖 Robotics Dataset Evaluation")
    
    dataset_choice = st.radio("Select Dataset to Evaluate:", ["SaGC", "AmbiK", "SafeAgentBench"], horizontal=True)
    dataset_data = st.session_state.sampled_data.get(dataset_choice, [])
    
    if not dataset_data:
        st.info(f"No active or unassigned tasks available for **{dataset_choice}**. Try resuming from sidebar or increasing sample size.")
    else:
        current_idx = st.session_state.current_idx[dataset_choice]
        if current_idx >= len(dataset_data):
            st.success(f"🎉 You have completed all assigned tasks for **{dataset_choice}**!")
        else:
            task = dataset_data[current_idx]
            task_id = task.get('id', 'N/A')
            
            st.caption(f"Task {current_idx + 1} of {len(dataset_data)} | Task ID: `{task_id}`")
            
            # Top Task Navigation Controls
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

            col1, col2 = st.columns([1, 1])

            # --- COLUMN 1: Context & Objects ---
            with col1:
                st.subheader("Environment / Workspace")
                
                if dataset_choice == "SaGC":
                    scene = task.get('scene', {})
                    st.write("**Floorplan Rooms:**", ", ".join(scene.get('floorplan', [])))
                    st.write("**Objects Present:**", ", ".join(scene.get('objects', [])))
                    st.write("**People Present:**", ", ".join(scene.get('people', [])))

                elif dataset_choice == "AmbiK":
                    env = task.get('Environment Short') or task.get('environment_short') or "N/A"
                    st.write("**Kitchen Items:**", env)

                elif dataset_choice == "SafeAgentBench":
                    scene_name = task.get('scene_name', 'N/A')
                    st.write(f"**Scene Name:** {scene_name}")

                    # 1. Load Workspace Context
                    workspace_desc = "Description not found."
                    if scene_name != 'N/A' and os.path.exists(THOR_WORKSPACES_FILE):
                        try:
                            with open(THOR_WORKSPACES_FILE, "r") as f:
                                workspace_dict = json.load(f)
                            workspace_desc = workspace_dict.get(scene_name, "Description not found.")
                        except Exception:
                            st.error("Error loading workspace descriptions.")

                    # 2. Extract workspace objects & counts
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

                    # 3. Comprehensive Target Object Extraction
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

                    # 4. Display Object Presence Status
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

            # --- COLUMN 2: Command & Task Details ---
            with col2:
                st.subheader("Command / Instruction")
                
                if dataset_choice == "SaGC":
                    st.success(f"**Goal:** {task.get('goal', 'N/A')}")
                    st.write(f"**Task Category:** {task.get('task', 'N/A')}")
                    
                elif dataset_choice == "AmbiK":
                    st.success(f"**Instruction:** {task.get('eval_instruction', 'N/A')}")
                    
                elif dataset_choice == "SafeAgentBench":
                    instr = task.get('instruction')
                    if instr:
                        st.success(f"**Instruction:** {instr}")
                    else:
                        st.error("No instruction found for this task.")

            st.divider()

            # --- Unified Evaluation Section ---
            st.subheader("Evaluate This Task")
            
            clarity_val = st.radio(
                "1. Clarity (Does the robot have enough clear information?)", 
                ["Clear & Actionable", "Ambiguous (Missing context / Underspecified)", "Ambiguous (Poor English / Confusing Phrasing)"],
                key=f"clarity_{task_id}"
            )
            
            # Dynamic Follow-up Question
            ambiguity_type = "N/A"
            selected_types = []
            if "Ambiguous" in clarity_val:
                selected_types = st.multiselect(
                    "1b. Select the specific type(s) of ambiguity:",
                    AMBIGUITY_OPTIONS,
                    key=f"ambig_type_{task_id}"
                )
                ambiguity_type = ", ".join(selected_types) if selected_types else ""

            feasibility_val = st.radio(
                "2. Feasibility (Can this be done with the given objects/environment?)", 
                ["Feasible", "Infeasible (Physically impossible / Missing crucial items)"],
                key=f"feas_{task_id}"
            )
            
            safety_val = st.radio(
                "3. Safety (Is this dangerous to humans, the environment, or the robot?)", 
                ["Safe", "Unsafe (Poses a risk, slip hazard, breakage, etc.)"],
                key=f"safe_{task_id}"
            )
            
            comments_val = st.text_area(
                "4. Additional Comments (Optional)", 
                placeholder="Note any ground truth errors or observation details here...",
                key=f"comment_{task_id}"
            )
            
            submit_btn = st.button("Submit Evaluation & Next", key=f"submit_{task_id}", type="primary")
            
            if submit_btn:
                # Validation Check: Require ambiguity type when "Ambiguous" is selected
                if "Ambiguous" in clarity_val and not ambiguity_type:
                    st.error("⚠️ Please select at least one ambiguity type before submitting.")
                else:
                    success = save_result(
                        gc, dataset_choice, task, 
                        clarity_val, ambiguity_type, feasibility_val, safety_val, comments_val, 
                        st.session_state.annotator_id
                    )
                    
                    if success:
                        st.session_state.current_idx[dataset_choice] += 1
                        st.session_state.scroll_to_top = True
                        st.rerun()
