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

# Ensure session state variables exist
if 'sampled_data' not in st.session_state:
    st.session_state.sampled_data = {'SaGC': [], 'AmbiK': [], 'SafeAgentBench': []}
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = {'SaGC': 0, 'AmbiK': 0, 'SafeAgentBench': 0}
if 'is_calibrated' not in st.session_state:
    st.session_state.is_calibrated = False

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


def check_user_calibrated(existing_df, annotator_id):
    """Checks if the user has completed the onboarding calibration session."""
    if existing_df.empty:
        return False
    records = existing_df[
        (existing_df['Annotator_ID'].astype(str) == str(annotator_id)) &
        (existing_df['Dataset'].astype(str) == '[TRAINING_COMPLETE]')
    ]
    return not records.empty


def save_result(gc, dataset, task_data, clarity, ambiguity_type, feasibility, safety, comments, annotator_id):
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

        # Check if an evaluation already exists for this annotator and task
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


# --- Data Loading & Task Allocation ---
def load_raw_data(file_path, file_type, dataset_prefix="Task"):
    """Loads data, ensures unique IDs, and expands multi-state tasks."""
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path)
            records = df.to_dict('records')

            # Expand AmbiK to test both Clear and Ambiguous versions independently
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

        # Inject deterministic IDs for datasets that lack them (SafeAgentBench)
        for i, rec in enumerate(records):
            if 'id' not in rec:
                rec['id'] = f"{dataset_prefix}_{i}"

        return records
    except Exception as e:
        st.error(f"Error loading {dataset_prefix}: {e}")
        return []


def allocate_tasks(raw_tasks, existing_df, dataset_name, annotator_id, n_samples):
    """Filters tasks based on global completion and current annotator."""
    assigned = []

    for task in raw_tasks:
        task_id = str(task.get('id', 'N/A'))

        if not existing_df.empty:
            # Get all evaluations for this specific task
            task_evals = existing_df[
                (existing_df['Dataset'] == dataset_name) & (existing_df['Task_ID'].astype(str) == task_id)]

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

# --- Calibration / Training Session Module ---
def render_training_module(gc, annotator_id):
    st.info("🎓 Annotator Calibration Session: Complete these benchmark tasks to learn dataset evaluation standards before beginning live evaluation.")
    try:
        with open("training_tasks.json", "r") as f:
            train_tasks = json.load(f)
    except FileNotFoundError:
        st.error("Missing 'training_tasks.json'. Proceeding directly to evaluation queue.")
        return True

    if 'train_idx' not in st.session_state:
        st.session_state.train_idx = 0
    if 'show_feedback' not in st.session_state:
        st.session_state.show_feedback = False

    t_idx = st.session_state.train_idx

    if t_idx >= len(train_tasks):
        st.success("🎉 Calibration complete! You have unlocked the live evaluation queue.")
        if st.button("Start Live Evaluations 🚀", type="primary"):
            save_result(gc, "[TRAINING_COMPLETE]", {"id": "PASSED"}, "N/A", "N/A", "N/A", "N/A",
                        "Passed onboarding calibration", annotator_id)
            st.session_state.is_calibrated = True
            st.rerun()
        return False

    task = train_tasks[t_idx]
    task_id = task['id']
    st.write(f"### Calibration Task {t_idx + 1} of {len(train_tasks)} ({task['dataset']})")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Environment & Context")
        if "environment_short" in task:
            st.write("**Kitchen Items:**", task["environment_short"])
        if "scene" in task and isinstance(task["scene"], dict):
            st.write("**Floorplan:**", ", ".join(task["scene"].get("floorplan", [])))
            st.write("**Objects:**", ", ".join(task["scene"].get("objects", [])))
        if "scene_name" in task:
            st.write("**Scene Name:**", task["scene_name"])
            if "involved_objects" in task:
                st.write("**Involved Objects:**", ", ".join(task["involved_objects"]))

    with col2:
        st.subheader("Instruction / Goal")
        st.success(f"**Instruction:** {task.get('instruction') or task.get('goal')}")

    st.divider()

    clarity_val = st.radio(
        "1. Clarity",
        ["Clear & Actionable", "Ambiguous (Missing context / Underspecified)",
         "Ambiguous (Poor English / Confusing Phrasing)"],
        key=f"tr_clarity_{task_id}"
    )
    feas_val = st.radio(
        "2. Feasibility",
        ["Feasible", "Infeasible (Physically impossible / Missing crucial items)"],
        key=f"tr_feas_{task_id}"
    )
    safety_val = st.radio(
        "3. Safety",
        ["Safe", "Unsafe (Poses a risk, slip hazard, breakage, etc.)"],
        key=f"tr_safe_{task_id}"
    )

    if not st.session_state.show_feedback:
        if st.button("Check Calibration Answer", key=f"btn_check_{task_id}", type="primary"):
            st.session_state.show_feedback = True
            st.rerun()
    else:
        c_match = clarity_val == task['expected_clarity']
        f_match = feas_val == task['expected_feasibility']
        s_match = safety_val == task['expected_safety']

        if c_match and f_match and s_match:
            st.success("✅ **Perfect alignment with benchmark ground truth!**")
        else:
            st.warning("⚠️ **Your selection differed from the expert benchmark label.**")

        with st.expander("💡 View Benchmark Ground Truth & Expert Explanation", expanded=True):
            st.markdown(f"**Benchmark Clarity:** {task['expected_clarity']}")
            st.markdown(f"**Benchmark Feasibility:** {task['expected_feasibility']}")
            st.markdown(f"**Benchmark Safety:** {task['expected_safety']}")
            st.info(f"**Expert Explanation:** {task['explanation']}")

        if st.button("Next Calibration Task ➡️", key=f"btn_next_{task_id}", type="primary"):
            st.session_state.train_idx += 1
            st.session_state.show_feedback = False
            st.session_state.scroll_to_top = True
            st.rerun()

    return False

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

    st.session_state.is_calibrated = check_user_calibrated(existing_df, annotator_id)

    # Pass the prefix so the loader can generate IDs (e.g., "AmbiK_1")
    raw_sagc = load_raw_data(SAGC_SUBSET, 'json', 'SaGC')
    raw_ambik = load_raw_data(AMBIK_SUBSET, 'csv', 'AmbiK')
    raw_safe = load_raw_data(SAFE_SUBSET, 'jsonl', 'SafeAgentBench')

    st.session_state.sampled_data['SaGC'] = allocate_tasks(raw_sagc, existing_df, "SaGC", annotator_id, sample_size)
    st.session_state.sampled_data['AmbiK'] = allocate_tasks(raw_ambik, existing_df, "AmbiK", annotator_id, sample_size)
    st.session_state.sampled_data['SafeAgentBench'] = allocate_tasks(raw_safe, existing_df, "SafeAgentBench",
                                                                     annotator_id, sample_size)

    st.session_state.current_idx = {'SaGC': 0, 'AmbiK': 0, 'SafeAgentBench': 0}
    st.session_state.annotator_id = annotator_id

    # Check if they are a returning user for a smarter greeting
    is_returning = not existing_df[existing_df['Annotator_ID'].astype(str) == str(annotator_id)].empty
    greeting = "Welcome back" if is_returning else "Welcome"

    total_tasks = sum(len(v) for v in st.session_state.sampled_data.values())
    st.sidebar.success(
        f"{greeting}, {annotator_id}! Allocated {total_tasks} total tasks (Max {sample_size} per dataset).")

# --- Main Interface ---
st.title("🤖Robotics Dataset Ground Truth Reliability Testing")
st.write(
    "Evaluate tasks across Clarity, Feasibility, and Safety. Check for poor phrasing, infeasible actions, or hidden safety risks.")

if 'annotator_id' in st.session_state and st.session_state.annotator_id:
    gc = init_connection()

    # Run calibration session first if user is uncalibrated
    if not st.session_state.is_calibrated:
        render_training_module(gc, st.session_state.annotator_id)
    else:

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
            st.warning(
                f"No pending tasks for {dataset_choice}. Either press 'Start' or you have completed all available tasks!")
        elif current_idx >= len(dataset_data):
            st.success(f"🎉 You have completed your assigned batch of {len(dataset_data)} tasks for {dataset_choice}!")
        else:
            task = dataset_data[current_idx]
            task_id = task.get('id', f'idx_{current_idx}')

            st.write(f"### Task {current_idx + 1} of {len(dataset_data)}")
            st.progress(current_idx / len(dataset_data))

            # Navigation controls (Previous / Next task buttons)
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            with nav_col1:
                if st.button("⬅️ Previous Task", disabled=(current_idx == 0), key=f"prev_{dataset_choice}_{task_id}"):
                    st.session_state.current_idx[dataset_choice] -= 1
                    st.session_state.scroll_to_top = True
                    st.rerun()
            with nav_col3:
                if st.button("Next Task ➡️", disabled=(current_idx >= len(dataset_data) - 1),
                             key=f"next_{dataset_choice}_{task_id}"):
                    st.session_state.current_idx[dataset_choice] += 1
                    st.session_state.scroll_to_top = True
                    st.rerun()

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Environment & Context")

                if dataset_choice == "SaGC":
                    if 'scene' in task and isinstance(task['scene'], dict):
                        st.write("**Floorplan:**", ", ".join(task['scene'].get('floorplan', [])))
                        st.write("**Objects:**", ", ".join(task['scene'].get('objects', [])))
                        st.write("**People:**", ", ".join(task['scene'].get('people', [])))

                elif dataset_choice == "AmbiK":
                    env = task.get('Environment Short') or task.get('environment_short') or "N/A"
                    st.write("**Kitchen Items:**", env)


                elif dataset_choice == "SafeAgentBench":
                    scene_name = task.get('scene_name', 'N/A')
                    st.write(f"**Scene Name:** {scene_name}")

                    # 1. Load Workspace Context
                    workspace_desc = "Description not found."
                    if scene_name != 'N/A':
                        try:
                            with open("thor_workspaces.json", "r") as f:
                                workspace_dict = json.load(f)
                            workspace_desc = workspace_dict.get(scene_name, "Description not found.")
                        except FileNotFoundError:
                            st.error("Missing 'thor_workspaces.json'.")

                    # 2. Extract ALL workspace objects & counts into flexible lookup structures
                    ws_objects_map = {}
                    raw_matches = re.findall(r'([A-Za-z0-9_-]+)\s*\((?:instances?|count)?:\s*(\d+)\)', workspace_desc,
                                             re.IGNORECASE)
                    for obj_name, count in raw_matches:
                        count_val = int(count)
                        norm_plain = re.sub(r'[^a-z0-9]', '', obj_name.lower())
                        ws_objects_map[norm_plain] = (obj_name, count_val)
                    # Raw normalized workspace string for fallback presence checking
                    ws_normalized_text = re.sub(r'[^a-z0-9]', '', workspace_desc.lower())
                    # Helper to normalize object strings (strips state modifiers like "dirty", "clean", "broken")
                    def normalize_obj_key(s):
                        s_clean = re.sub(r'^(dirty|clean|broken|open|opened|closed|turned on|turned off|lit|unlit)\s+', '',
                                         str(s).strip(), flags=re.IGNORECASE)
                        return re.sub(r'[^a-z0-9]', '', s_clean.lower())

                    # 3. Comprehensive Target Object Extraction
                    extracted_targets = set()
                    # A. Explicit metadata lists
                    for field in ['objects', 'involved_objects']:
                        items = task.get(field)
                        if isinstance(items, list):
                            for item in items:
                                if item and str(item).strip():
                                    extracted_targets.add(str(item).strip())
                    # B. Check 'final_state' (both target objectType AND parentReceptacles)
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
                    # C. Parse 'step' actions (e.g., "find Sink", "open Cabinet")
                    steps = task.get('step')
                    if isinstance(steps, list):
                        for s in steps:
                            s_str = str(s).strip()
                            action_match = re.search(
                                r'^(?:find|pick|turn\s+on|turn\s+off|open|close|put|drop|dirty|clean|fillLiquid|pour|break|slice|use)\s+(.+)$',
                                s_str, re.IGNORECASE)
                            if action_match:
                                extracted_targets.add(action_match.group(1).strip())
                    # D. Instruction text cross-referencing
                    instruction_combined = (
                                str(task.get('instruction') or '') + " " + str(task.get('risk_instruction') or '')).lower()
                    for norm_key, (raw_name, _) in ws_objects_map.items():
                        raw_spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', raw_name).lower()
                        if raw_spaced in instruction_combined or raw_name.lower() in instruction_combined:
                            extracted_targets.add(raw_name)

                    # 4. Accurate Object Presence Evaluation
                    if extracted_targets:
                        st.write("**Object Presence Check:**")
                        unique_targets = {}
                        for tgt in extracted_targets:
                            norm_k = normalize_obj_key(tgt)
                            if norm_k not in unique_targets:
                                unique_targets[norm_k] = tgt
                        for norm_k, display_tgt in sorted(unique_targets.items(), key=lambda x: x[1]):
                            # Direct map lookup or singular/plural variants
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
                                # Fallback text search guarantees zero false negatives if object exists in workspace description
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
                    # Expandable complete workspace state
                    with st.expander("🔍 View Complete Workspace State (Objects & Coordinates)", expanded=False):
                        st.code(workspace_desc, language="text")

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

            # Dynamic Follow-up Question
            ambiguity_type = "N/A"
            if "Ambiguous" in clarity_val or "Mostly clear" in clarity_val:
                selected_types = st.multiselect(
                    "1b. Select the specific type of ambiguity:",
                    ["Attribute: Referring to objects using vague attributes, or referring to object but not specifying some attribute.",
                     "Placement: Uncertainty regarding where the object should be placed in situations such as the existence of multiple valid positions or an insufficiently defined location.",
                     "Orientation: Uncertainty regarding how an object should be oriented, resulting from a lack of information such as axis, angle, and reference frame.",
                     "Spatial Reference: Unclear reference frame.",
                     "Linguistic: Ambiguities when a command’s wording can be interpreted in multiple ways due to language conventions, implicit assumptions, or mismatch with the physical context.",
                     "Numeric: Underspecifying quantity.",
                     "Preference: Lacking user preference or necessary environmental context.",
                     "Referential: Ambiguities where the reference is subtle.",
                     "Commonsense: Tasks that are feasible to perform and which pose no danger when executed yet run counter to common sense."],
                    key=f"ambig_type_{task_id}"
                )

                ambiguity_type = ", ".join(selected_types) if selected_types else ""

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
                placeholder="Note any ground truth errors here...",
                key=f"comment_{task_id}"
            )

            submit_btn = st.button("Submit Evaluation & Next", key=f"submit_{task_id}", type="primary")

            if submit_btn:
                if ("Ambiguous" in clarity_val or "Mostly clear" in clarity_val) and not ambiguity_type:
                    st.error("⚠️ Please select at least one ambiguity type before submitting.")
                else:
                    gc = init_connection()

                    # Pass the separate ambiguity_type variable directly
                    success = save_result(
                        gc, dataset_choice, task,
                        clarity_val, ambiguity_type, feasibility_val, safety_val, comments_val,
                        st.session_state.annotator_id
                    )

                    if success:
                        st.session_state.current_idx[dataset_choice] += 1
                        st.session_state.scroll_to_top = True
                        st.rerun()
                    else:
                        st.info("👈 Please enter your Name or Annotator ID in the sidebar to begin.")
