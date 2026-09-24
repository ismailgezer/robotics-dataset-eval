import streamlit as st
import pandas as pd
import json
import os
import gspread

# --- Page Configuration ---
st.set_page_config(page_title="Robotics Dataset Label Evaluation", layout="wide")

# Ensure session state variables exist
if 'sampled_data' not in st.session_state:
    st.session_state.sampled_data = {'SaGC': [], 'AmbiK': [], 'SafeAgentBench': []}
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = {'SaGC': 0, 'AmbiK': 0, 'SafeAgentBench': 0}

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

def save_result(gc, dataset, task_data, clarity, feasibility, safety, comments, annotator_id):
    original_label = task_data.get('label') or task_data.get('ambiguity_type') or task_data.get('risk_category') or "N/A"
    task_id = task_data.get('id', 'N/A')

    row_data = [
        str(annotator_id), str(dataset), str(task_id), str(original_label),
        str(clarity), str(feasibility), str(safety), str(comments)
    ]

    try:
        sh = gc.open_by_url(get_sheet_url())
        worksheet = sh.sheet1
        worksheet.append_row(row_data)
        st.toast("Response saved to database!", icon="✅")
        return True
    except Exception as e:
        st.error(f"Error saving to Google Sheets: {e}")
        return False

# --- Data Loading & Task Allocation ---
def load_raw_data(file_path, file_type):
    """Loads all data from a curated subset file."""
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'jsonl':
            df = pd.read_json(file_path, lines=True)
        elif file_type == 'json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                records = []
                for key, val in data.items():
                    val['id'] = key 
                    records.append(val)
                df = pd.DataFrame(records)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else: return []
        else: return []
        return df.to_dict('records')
    except Exception:
        return []

def allocate_tasks(raw_tasks, existing_df, dataset_name, annotator_id, n_samples):
    """Filters tasks based on global completion and current annotator."""
    assigned = []
    
    for task in raw_tasks:
        task_id = str(task.get('id', 'N/A'))
        
        if not existing_df.empty:
            # Get all evaluations for this specific task
            task_evals = existing_df[(existing_df['Dataset'] == dataset_name) & (existing_df['Task_ID'].astype(str) == task_id)]
            
            # 1. Skip if THIS annotator already completed it
            if str(annotator_id) in task_evals['Annotator_ID'].astype(str).values:
                continue
                
            # 2. Skip if >= 2 OTHER unique annotators already completed it
            if task_evals['Annotator_ID'].nunique() >= 2:
                continue
                
        assigned.append(task)
        if len(assigned) >= n_samples:
            break
            
    return assigned

# --- Sidebar Setup ---
# Hardcoded Subset File Paths
SAGC_SUBSET = "augment.json"
AMBIK_SUBSET = "ambik_test_900.csv"
SAFE_SUBSET = "mixed_detailed_1009.jsonl"

st.sidebar.title("1. Setup")
annotator_id = st.sidebar.text_input("Enter your Name or Annotator ID:", placeholder="e.g., Annotator_1")
sample_size = st.sidebar.number_input("Tasks to allocate per dataset:", min_value=1, max_value=100, value=10)

if st.sidebar.button("Start / Resume Evaluation", disabled=not annotator_id):
    gc = init_connection()
    existing_df = get_existing_evaluations(gc)
    
    raw_sagc = load_raw_data(SAGC_SUBSET, 'json')
    raw_ambik = load_raw_data(AMBIK_SUBSET, 'csv')
    raw_safe = load_raw_data(SAFE_SUBSET, 'jsonl')
    
    st.session_state.sampled_data['SaGC'] = allocate_tasks(raw_sagc, existing_df, "SaGC", annotator_id, sample_size)
    st.session_state.sampled_data['AmbiK'] = allocate_tasks(raw_ambik, existing_df, "AmbiK", annotator_id, sample_size)
    st.session_state.sampled_data['SafeAgentBench'] = allocate_tasks(raw_safe, existing_df, "SafeAgentBench", annotator_id, sample_size)
    
    st.session_state.current_idx = {'SaGC': 0, 'AmbiK': 0, 'SafeAgentBench': 0}
    st.session_state.annotator_id = annotator_id
    
    total_tasks = sum(len(v) for v in st.session_state.sampled_data.values())
    st.sidebar.success(f"Welcome back, {annotator_id}! {total_tasks} new tasks assigned.")

# --- Main Interface ---
st.title("Robotics Dataset Ground Truth Reliability Testing")
st.write("Evaluate tasks across Clarity, Feasibility, and Safety. Check for poor phrasing, infeasible actions, or hidden safety risks.")

dataset_choice = st.selectbox("Select Dataset to Evaluate:", ["SaGC", "AmbiK", "SafeAgentBench"])

# Elaborated Descriptions
descriptions = {
    "SaGC": "**SaGC (Command Classification & Clarity):** Evaluates command classification for a mobile manipulator robot (used in CLARA). The robot operates in indoor environments (e.g., kitchens, living rooms, bedrooms) and is capable of navigating to rooms, detecting people, and manipulating everyday household objects.",
    "AmbiK": "**AmbiK (Uncertainty & Ambiguity):** Developed specifically for a kitchen assistant robot capable of fine manipulation (e.g., chopping, pouring, mixing). Pay attention to whether the instructions are logically feasible or pose unintended safety hazards like cross-contamination.",
    "SafeAgentBench": "**SafeAgentBench (Safety Verification):** Designed to test an embodied AI agent operating in simulated home environments (AI2-THOR). Look out for whether the instructions are clearly written, or if poor English/missing context creates unintended ambiguity alongside the safety risks."
}
st.info(descriptions[dataset_choice])

# --- Evaluation View ---
dataset_data = st.session_state.sampled_data[dataset_choice]
current_idx = st.session_state.current_idx[dataset_choice]

if not dataset_data:
    st.warning(f"No pending tasks for {dataset_choice}. Either press 'Start' or you have completed all available tasks!")
elif current_idx >= len(dataset_data):
    st.success(f"🎉 You have completed your assigned batch of {len(dataset_data)} tasks for {dataset_choice}!")
else:
    task = dataset_data[current_idx]
    task_id = task.get('id', f'idx_{current_idx}')
    
    st.write(f"### Task {current_idx + 1} of {len(dataset_data)}")
    st.progress(current_idx / len(dataset_data))
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Environment & Context")
        
        if dataset_choice == "SaGC":
            if 'scene' in task and isinstance(task['scene'], dict):
                st.write("**Floorplan:**", ", ".join(task['scene'].get('floorplan', [])))
                st.write("**Objects:**", ", ".join(task['scene'].get('objects', [])))
                st.write("**People:**", ", ".join(task['scene'].get('people', [])))
                
        elif dataset_choice == "AmbiK":
            env = task.get('environment_short') or task.get('environment_full') or "N/A"
            st.write("**Kitchen Items:**", env)
            
        elif dataset_choice == "SafeAgentBench":
            scene_name = task.get('scene_name', 'N/A')
            st.write(f"**Scene Name:** {scene_name}")
            
            # --- Robust Object Extraction ---
            inv_objs = set()
            
            # 1. Check 'objects' list
            if isinstance(task.get('objects'), list):
                inv_objs.update(task.get('objects'))
            # 2. Check 'involved_objects' list
            if isinstance(task.get('involved_objects'), list):
                inv_objs.update(task.get('involved_objects'))
            # 3. Check 'final_state' for objectTypes
            final_state = task.get('final_state')
            if isinstance(final_state, list):
                for fs in final_state:
                    if isinstance(fs, dict) and 'objectType' in fs:
                        inv_objs.add(fs['objectType'])
            # 4. Fallback: Parse the 'step' array for words after 'find'
            if not inv_objs and isinstance(task.get('step'), list):
                for s in task.get('step'):
                    if str(s).lower().startswith('find '):
                        inv_objs.add(str(s)[5:].strip())
            
            inv_objs = list(inv_objs)
            
            # --- Load Workspace Context ---
            workspace_desc = "Description not found."
            if scene_name != 'N/A':
                try:
                    with open("thor_workspaces.json", "r") as f:
                        workspace_dict = json.load(f)
                    workspace_desc = workspace_dict.get(scene_name, "Description not found.")
                except FileNotFoundError:
                    st.error("Missing 'thor_workspaces.json'.")
            
            # --- Object Presence Checker ---
            if inv_objs:
                st.write("**Object Presence Check:**")
                for obj in inv_objs:
                    # Using case-insensitive check
                    if str(obj).lower() in workspace_desc.lower():
                        st.markdown(f"- ✅ **{obj}** is present in the scene.")
                    else:
                        st.markdown(f"- ❌ **{obj}** was NOT found in the scene.")
            else:
                st.write("**Object Presence Check:** No distinct objects identified.")
            
            with st.expander("🔍 View Complete Workspace State (Objects & Coordinates)", expanded=False):
                st.code(workspace_desc, language="text")

    with col2:
        st.subheader("Command / Instruction")
        
        if dataset_choice == "SaGC":
            st.success(f"**Goal:** {task.get('goal', 'N/A')}")
            st.write(f"**Task Category:** {task.get('task', 'N/A')}")
            
        elif dataset_choice == "AmbiK":
            st.success(f"**Instruction:** {task.get('ambiguous_task') or task.get('unambiguous_direct') or 'N/A'}")
            
        elif dataset_choice == "SafeAgentBench":
            # --- Robust Instruction Extraction ---
            instr = task.get('instruction')
            risk_instr = task.get('risk_instruction')
            
            # Render both if they exist so nothing is accidentally hidden
            if instr:
                st.success(f"**Instruction:** {instr}")
            if risk_instr:
                st.warning(f"**Risk Instruction:** {risk_instr}")
                
            if not instr and not risk_instr:
                st.error("No instruction found for this task.")

    st.divider()

    # --- Unified Evaluation Form ---
    # clear_on_submit=True resets the form completely after a successful save
    with st.form(key=f"eval_form", clear_on_submit=True):
        st.subheader("Evaluate This Task")
        
        # Using task_id in the keys ensures Streamlit treats them as brand-new inputs
        clarity_val = st.radio(
            "1. Clarity (Does the robot have enough clear information?)", 
            ["Clear & Actionable", "Ambiguous (Missing context / Underspecified)", "Ambiguous (Poor English / Confusing Phrasing)"],
            key=f"clarity_{task_id}"
        )
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
            placeholder="Note any ground truth errors here...",
            key=f"comment_{task_id}"
        )
        
        submit_btn = st.form_submit_button("Submit Evaluation & Next")
        
        if submit_btn:
            gc = init_connection()
            success = save_result(
                gc, dataset_choice, task, 
                clarity_val, feasibility_val, safety_val, comments_val, 
                st.session_state.annotator_id
            )
            
            if success:
                st.session_state.current_idx[dataset_choice] += 1
                st.rerun()
