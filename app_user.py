import streamlit as st
import pandas as pd
import json
import os
import gspread
import re

# --- Page Configuration ---
st.set_page_config(page_title="Robotics Dataset Label Evaluation", layout="wide")

# 1. Inject an invisible HTML anchor at the absolute top of the app
st.markdown("<div id='top-of-page'></div>", unsafe_allow_html=True)

# 2. Delayed scroll mechanism (Bypasses Streamlit iframe scroll preservation)
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

# --- Constants & Configuration ---
SAGC_SUBSET = "sagc_data.json"
AMBIK_SUBSET = "ambik_data.csv"
SAFE_SUBSET = "safeagentbench_data.jsonl"
TRAINING_FILE = "training_tasks.json"

# --- Session State Initialization ---
if 'sampled_data' not in st.session_state:
    st.session_state.sampled_data = {}
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = {'SaGC': 0, 'AmbiK': 0, 'SafeAgentBench': 0}
if 'annotator_id' not in st.session_state:
    st.session_state.annotator_id = ""
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False
if 'train_idx' not in st.session_state:
    st.session_state.train_idx = 0
if 'show_train_feedback' not in st.session_state:
    st.session_state.show_train_feedback = False


# --- Database / Google Sheets Setup ---
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
    """Saves or updates (upserts) evaluation responses in Google Sheets."""
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
        
        # Search for an existing record matching (Annotator_ID, Dataset, Task_ID) for retroactive edit
        all_records = worksheet.get_all_records()
        existing_row_idx = None
        for idx, record in enumerate(all_records, start=2):  # Row 1 contains column headers
            if (str(record.get('Annotator_ID')) == str(annotator_id) and 
                str(record.get('Dataset')) == str(dataset) and 
                str(record.get('Task_ID')) == str(task_id)):
                existing_row_idx = idx
                break

        if existing_row_idx:
            # Overwrite the existing evaluation row (Columns A through I)
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
    """Loads raw dataset files, generates deterministic IDs, and expands variants."""
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path)
            records = df.to_dict('records')
            
            # Expand AmbiK rows into separate ambiguous and clear tasks
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
        
        # Inject deterministic IDs if missing
        for i, rec in enumerate(records):
            if 'id' not in rec:
                rec['id'] = f"{dataset_prefix}_{i}"
                
        return records
    except Exception as e:
        st.error(f"Error loading {dataset_prefix} ({file_path}): {e}")
        return []

def allocate_tasks(raw_tasks, existing_df, dataset_name, annotator_id, sample_size=10):
    """Allocates tasks ensuring max 2 unique annotators per task and no duplicates for same user."""
    if not raw_tasks:
        return []
        
    if existing_df.empty or 'Task_ID' not in existing_df.columns:
        return raw_tasks[:sample_size]

    # Filter dataframe for current dataset
    ds_df = existing_df[existing_df['Dataset'] == dataset_name]
    
    # Identify tasks already completed by this user
    user_completed = set(ds_df[ds_df['Annotator_ID'].astype(str) == str(annotator_id)]['Task_ID'].astype(str))
    
    # Count total evaluations per task ID
    task_counts = ds_df['Task_ID'].astype(str).value_counts().to_dict()

    eligible_tasks = []
    for task in raw_tasks:
        tid = str(task.get('id'))
        # Skip if user already evaluated this task or if task reached max 2 annotators
        if tid in user_completed:
            continue
        if task_counts.get(tid, 0) >= 2:
            continue
        eligible_tasks.append(task)

    return eligible_tasks[:sample_size]


# --- Onboarding & Calibration Training Module ---
def check_user_trained(existing_df, annotator_id):
    """Checks if the user has completed onboarding calibration."""
    if existing_df.empty or 'Dataset' not in existing_df.columns:
        return False
    trained_records = existing_df[
        (existing_df['Annotator_ID'].astype(str) == str(annotator_id)) & 
        (existing_df['Dataset'] == '[TRAINING_COMPLETE]')
    ]
    return not trained_records.empty

def render_training_module(gc, annotator_id):
    """Interactive calibration module with real-time benchmark feedback."""
    st.info("🎓 **Annotator Calibration Session**: Please complete these benchmark calibration tasks to align with evaluation guidelines before proceeding to live tasks.")
    
    try:
        with open(TRAINING_FILE, "r") as f:
            train_tasks = json.load(f)
    except FileNotFoundError:
        st.warning("`training_tasks.json` not found. Skipping calibration phase.")
        return True

    t_idx = st.session_state.train_idx

    if t_idx >= len(train_tasks):
        st.success("🎉 Calibration complete! You have unlocked the live evaluation queue.")
        if st.button("Proceed to Live Evaluations 🚀", type="primary"):
            save_result(gc, "[TRAINING_COMPLETE]", {"id": "PASSED"}, "N/A", "N/A", "N/A", "N/A", "Completed calibration", annotator_id)
            st.session_state.is_trained = True
            st.rerun()
        return False

    task = train_tasks[t_idx]
    st.write(f"### Training Task {t_idx + 1} of {len(train_tasks)} ({task.get('dataset', 'Calibration')})")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Context")
        if "environment_short" in task:
            st.write("**Items:**", task["environment_short"])
        if "scene_name" in task:
            st.write("**Scene:**", task["scene_name"])
            if "involved_objects" in task:
                st.write("**Objects:**", ", ".join(task["involved_objects"]))
    with col2:
        st.subheader("Command")
        st.success(f"**Instruction:** {task.get('instruction', 'N/A')}")

    st.divider()

    clarity_val = st.radio(
        "1. Clarity", 
        ["Clear & Actionable", "Ambiguous (Missing context / Underspecified)", "Ambiguous (Poor English / Confusing Phrasing)"],
        key=f"tr_clarity_{task['id']}"
    )
    feas_val = st.radio(
        "2. Feasibility", 
        ["Feasible", "Infeasible (Physically impossible / Missing crucial items)"],
        key=f"tr_feas_{task['id']}"
    )
    safety_val = st.radio(
        "3. Safety", 
        ["Safe", "Unsafe (Poses a risk, slip hazard, breakage, etc.)"],
        key=f"tr_safe_{task['id']}"
    )

    if not st.session_state.show_train_feedback:
        if st.button("Check Answer", type="primary"):
            st.session_state.show_train_feedback = True
            st.rerun()
    else:
        c_match = clarity_val == task.get('expected_clarity')
        f_match = feas_val == task.get('expected_feasibility')
        s_match = safety_val == task.get('expected_safety')

        if c_match and f_match and s_match:
            st.success("✅ **Excellent! Your ratings match the benchmark standard.**")
        else:
            st.warning("⚠️ **Your ratings differed from the benchmark standard in one or more categories.**")

        with st.expander("💡 View Benchmark Ground Truth & Expert Explanation", expanded=True):
            st.markdown(f"**Expected Clarity:** {task.get('expected_clarity')}")
            st.markdown(f"**Expected Feasibility:** {task.get('expected_feasibility')}")
            st.markdown(f"**Expected Safety:** {task.get('expected_safety')}")
            st.info(f"**Expert Explanation:** {task.get('explanation')}")

        if st.button("Next Training Task ➡️", type="primary"):
            st.session_state.train_idx += 1
            st.session_state.show_train_feedback = False
            st.rerun()

    return False


# --- Main Application Interface ---
st.title("🤖 Robotics Dataset Label Evaluation")

# --- Sidebar Configuration ---
st.sidebar.header("Annotator Setup")
annotator_id = st.sidebar.text_input("Enter Your Annotator ID / Username:", value=st.session_state.annotator_id).strip()
sample_size = st.sidebar.number_input("Tasks per dataset batch:", min_value=1, max_value=50, value=10)

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
    st.sidebar.success(f"{greeting}, {annotator_id}! Allocated {total_allocated} total tasks (Max {sample_size} per dataset).")

# --- Routing Logic ---
if not st.session_state.annotator_id or not st.session_state.sampled_data:
    st.info("👈 Please enter your Annotator ID in the sidebar and click **Start / Resume Evaluation** to begin.")
else:
    gc = init_connection()

    # Route 1: Onboarding Training Module if not yet completed
    if not st.session_state.is_trained:
        render_training_module(gc, st.session_state.annotator_id)
    else:
        # Route 2: Main Evaluation Dashboard
        dataset_choice = st.radio("Select Dataset Queue:", ["SaGC", "AmbiK", "SafeAgentBench"], horizontal=True)
        
        dataset_data = st.session_state.sampled_data.get(dataset_choice, [])
        current_idx = st.session_state.current_idx[dataset_choice]
        
        if not dataset_data:
            st.warning(f"No remaining unassigned tasks found for dataset **{dataset_choice}**.")
        elif current_idx >= len(dataset_data):
            st.success(f"🎉 You have completed all allocated tasks in the **{dataset_choice}** queue!")
        else:
            task = dataset_data[current_idx]
            task_id = task.get('id', 'N/A')
            
            st.write(f"### Progress: Task {current_idx + 1} of {len(dataset_data)} (ID: `{task_id}`)")
            
            # --- Main Display Columns ---
            col1, col2 = st.columns(2)
            
            # --- Column 1: Context & Workspace ---
            with col1:
                st.subheader("Environment & Scene Context")
                
                if dataset_choice == "SaGC":
                    scene = task.get('scene', {})
                    st.write("**Floorplan Areas:**", ", ".join(scene.get('floorplan', [])))
                    st.write("**Objects Present:**", ", ".join(scene.get('objects', [])))
                    st.write("**People Present:**", ", ".join(scene.get('people', [])))
                    
                elif dataset_choice == "AmbiK":
                    env = task.get('Environment Short') or task.get('environment_short') or "N/A"
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
                            st.error("Missing 'thor_workspaces.json'.")

                    # Parse all workspace objects and counts into flexible lookup dictionary
                    ws_objects_map = {}
                    raw_matches = re.findall(r'([A-Za-z0-9_-]+)\s*\((?:instances?|count)?:\s*(\d+)\)', workspace_desc, re.IGNORECASE)
                    for obj_name, count in raw_matches:
                        norm_plain = re.sub(r'[^a-z0-9]', '', obj_name.lower())
                        ws_objects_map[norm_plain] = (obj_name, int(count))

                    ws_normalized_text = re.sub(r'[^a-z0-9]', '', workspace_desc.lower())

                    def normalize_obj_key(s):
                        s_clean = re.sub(r'^(dirty|clean|broken|open|opened|closed|turned on|turned off|lit|unlit)\s+', '', str(s).strip(), flags=re.IGNORECASE)
                        return re.sub(r'[^a-z0-9]', '', s_clean.lower())

                    # Target Object Extraction across all fields
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

                    # Object Presence Evaluation
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

            # --- Column 2: Instruction & Task Metadata ---
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

            # --- Navigation Controls (Previous / Next Task) ---
            st.write("")
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

            st.divider()

            # --- Unified Evaluation Input Form ---
            st.subheader("Evaluate This Task")
            
            clarity_val = st.radio(
                "1. Clarity (Does the robot have enough clear information?)", 
                ["Clear & Actionable", "Ambiguous (Missing context / Underspecified)", "Ambiguous (Poor English / Confusing Phrasing)"],
                key=f"clarity_{task_id}"
            )
            
            # Dynamic Follow-up Question for Ambiguity
            ambiguity_type = "N/A"
            if "Ambiguous" in clarity_val:
                selected_types = st.multiselect(
                    "1b. Select the specific type(s) of ambiguity:",
                    ["Scope / Entity Ambiguity", "Spatial / Location Ambiguity", "Tool / Instrument Ambiguity", "Safety / Precondition Ambiguity", "Pragmatic / Contextual Ambiguity"],
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
                placeholder="Note any ground truth errors or environment issues here...",
                key=f"comment_{task_id}"
            )
            
            submit_btn = st.button("Submit Evaluation & Next", key=f"submit_{task_id}", type="primary")
            
            if submit_btn:
                # Validation check: ensure ambiguity type is selected if task is marked ambiguous
                if "Ambiguous" in clarity_val and not ambiguity_type:
                    st.error("⚠️ Please select at least one ambiguity type before submitting.")
                else:
                    final_clarity = f"{clarity_val} [{ambiguity_type}]" if "Ambiguous" in clarity_val else clarity_val
                    
                    success = save_result(
                        gc, dataset_choice, task, 
                        final_clarity, ambiguity_type, feasibility_val, safety_val, comments_val, 
                        st.session_state.annotator_id
                    )
                    
                    if success:
                        st.session_state.current_idx[dataset_choice] += 1
                        st.session_state.scroll_to_top = True
                        st.rerun()
