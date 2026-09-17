import streamlit as st
import pandas as pd
import json
import os
import gspread

# --- Page Configuration ---
st.set_page_config(page_title="Robotics Dataset Label Evaluation", layout="wide")

# --- Initialize Session State ---
if 'sampled_data' not in st.session_state:
    st.session_state.sampled_data = {'SaGC': [], 'AmbiK': [], 'SafeAgentBench': []}
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = {'SaGC': 0, 'AmbiK': 0, 'SafeAgentBench': 0}


# --- Helper Functions ---
@st.cache_data
def load_and_sample_data(file_path, file_type, n_samples):
    """Loads data from a local file path and samples random rows."""
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
            else:
                return []
        else:
            return []

        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

        return df.to_dict('records')
    except Exception as e:
        st.error(f"Error loading file {file_path}: {e}")
        return []


import gspread
import streamlit as st


def save_result(dataset, task_data, clarity, feasibility, safety, comments, annotator_id):
    original_label = task_data.get('label') or task_data.get('ambiguity_type') or task_data.get(
        'risk_category') or "N/A"
    task_id = task_data.get('id', 'N/A')

    # Format the data as a simple list representing one row
    row_data = [
        str(annotator_id),
        str(dataset),
        str(task_id),
        str(original_label),
        str(clarity),
        str(feasibility),
        str(safety),
        str(comments)
    ]

    try:
        # 1. Authenticate securely using Streamlit Secrets
        credentials_dict = dict(st.secrets["gcp_service_account"])
        gc = gspread.service_account_from_dict(credentials_dict)

        # 2. Open the specific Google Sheet
        SHEET_URL = "https://docs.google.com/spreadsheets/d/1ZYnGCNIjOVoZqGv7TOZwkKV-k2z25J6g8UkS7dfWY-M/edit"
        sh = gc.open_by_url(SHEET_URL)

        # 3. Select the first tab (Sheet1)
        worksheet = sh.sheet1

        # 4. Append the new row to the bottom of the sheet
        worksheet.append_row(row_data)

        st.toast("Response saved to database!", icon="✅")
        return True

    except FileNotFoundError:
        st.error("Could not find 'service_account.json'. Ensure it is in the same folder as your script.")
        return False
    except gspread.exceptions.APIError as e:
        st.error(
            f"Google API Error: Check that your Service Account email is added as an Editor to the Sheet. Details: {e}")
        return False
    except Exception as e:
        st.error(f"Error saving to Google Sheets: {e}")
        return False


# --- Hardcoded File Paths ---
st.sidebar.title("1. Setup")
annotator_id = st.sidebar.text_input("Enter your Name or Annotator ID:", placeholder="e.g., Annotator_1")
sample_size = st.sidebar.number_input("Tasks to evaluate per dataset:", min_value=1, max_value=500, value=10)

# Only allow them to load datasets if they provided an ID
if st.sidebar.button("Start Evaluation", disabled=not annotator_id):
    st.session_state.sampled_data['SaGC'] = load_and_sample_data("dataset/augment.json", 'json', sample_size)
    st.session_state.sampled_data['AmbiK'] = load_and_sample_data("dataset/ambik_test_900.csv", 'csv', sample_size)
    st.session_state.sampled_data['SafeAgentBench'] = load_and_sample_data("dataset/mixed_detailed_1009.jsonl", 'jsonl', sample_size)

    st.session_state.current_idx = {'SaGC': 0, 'AmbiK': 0, 'SafeAgentBench': 0}

    # Save the annotator ID in session state so we can attach it to their saves
    st.session_state.annotator_id = annotator_id
    st.sidebar.success(f"Ready, {annotator_id}! You can begin evaluating.")

# --- Main App: Dataset Selection & Evaluation ---
st.title("Robotics Dataset Ground Truth Reliability Testing")
st.write("""
Evaluate sampled tasks across three universal dimensions: **Clarity, Feasibility, and Safety**. 
By applying this uniform lens, we can uncover flaws such as poor English framing, hidden safety risks in ambiguity datasets, and ambiguities in safety datasets.
""")

# Let the user choose which dataset they are evaluating right now
dataset_choice = st.selectbox("Select Dataset to Evaluate:", ["SaGC", "AmbiK", "SafeAgentBench"])

# Dataset Descriptions
descriptions = {
    "SaGC": "**SaGC** focuses on command classification and uncertainty resolution in general indoor environments. *Watch out for ambiguous phrasing or unintended safety risks.*",
    "AmbiK": "**AmbiK** was developed specifically for kitchen environments to detect and resolve ambiguities. *Pay attention to whether the instructions are actually feasible or pose safety hazards (e.g., cross-contamination).*",
    "SafeAgentBench": "**SafeAgentBench** is designed solely for safety verification. *Look out for whether the instructions are clearly written or if poor English/missing context creates unintended ambiguity.*"
}
st.info(descriptions[dataset_choice])

# --- Evaluation View ---
dataset_data = st.session_state.sampled_data[dataset_choice]
current_idx = st.session_state.current_idx[dataset_choice]

if not dataset_data:
    st.warning(
        f"No data loaded for {dataset_choice}. Please upload the file in the sidebar and click 'Load and Sample'.")
elif current_idx >= len(dataset_data):
    st.success(f"🎉 You have completed all {len(dataset_data)} sampled tasks for {dataset_choice}!")
else:
    task = dataset_data[current_idx]

    st.write(f"### Task {current_idx + 1} of {len(dataset_data)}")
    st.progress((current_idx) / len(dataset_data))

    # --- Context Renderer ---
    # Renders the UI dynamically based on the schema of the specific dataset
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Environment & Context")
        if dataset_choice == "SaGC":
            if 'scene' in task and isinstance(task['scene'], dict):
                st.write("**Floorplan:**", ", ".join(task['scene'].get('floorplan', [])))
                st.write("**Objects:**", ", ".join(task['scene'].get('objects', [])))
                st.write("**People:**", ", ".join(task['scene'].get('people', [])))

        elif dataset_choice == "AmbiK":
            # AmbiK uses either environment_short or environment_full
            env = task.get('environment_short') or task.get('environment_full') or "N/A"
            st.write("**Kitchen Items:**", env)


        elif dataset_choice == "SafeAgentBench":
            scene_name = task.get('scene_name', 'N/A')
            st.write(f"**Scene Name:** {scene_name}")

            # --- Text-Only Workspace Context ---
            if scene_name != 'N/A':
                try:
                    # Load the pre-generated JSON dictionary
                    with open("thor_workspaces.json", "r") as f:
                        workspace_dict = json.load(f)
                    workspace_desc = workspace_dict.get(scene_name, "Description not found.")

                    # Keep the UI clean by hiding the long text inside an expander
                    with st.expander("🔍 View Complete Workspace State (Objects & Coordinates)", expanded=False):
                        st.code(workspace_desc, language="text")
                except FileNotFoundError:
                    st.error("Missing 'thor_workspaces.json'. Please generate it and place it in this folder.")
                except Exception as e:
                    st.warning(f"Error loading workspace dictionary: {e}")

            # --- Safely check involved_objects ---
            inv_objs = task.get('involved_objects')
            if isinstance(inv_objs, list):
                st.write(f"**Involved Objects:** {', '.join(inv_objs)}")
            else:
                st.write("**Involved Objects:** N/A")

            # --- Safely check final_state ---
            final_state = task.get('final_state')
            if isinstance(final_state, list):
                st.write("**Target Final State:**", final_state)

    with col2:
        st.subheader("Command / Instruction")
        if dataset_choice == "SaGC":
            st.success(f"**Goal:** {task.get('goal', 'N/A')}")
            st.write(f"**Task Category:** {task.get('task', 'N/A')}")

        elif dataset_choice == "AmbiK":
            st.success(f"**Instruction:** {task.get('ambiguous_task') or task.get('unambiguous_direct') or 'N/A'}")
            st.write(f"**Intended Ambiguity Type:** {task.get('ambiguity_type', 'N/A')}")

        elif dataset_choice == "SafeAgentBench":
            # Sometimes safe agent bench has 'instruction' and 'risk_instruction'
            command = task.get('risk_instruction') or task.get('instruction') or "N/A"
            st.success(f"**Instruction:** {command}")
            st.write(f"**Intended Risk Category:** {task.get('risk_category', 'N/A')}")

    st.divider()

    # --- Unified Evaluation Form ---
    with st.form(key=f"unified_eval_form"):
        st.subheader("Evaluate This Task")

        clarity_val = st.radio(
            "1. Clarity (Does the robot have enough clear information?)",
            ["Clear & Actionable",
             "Ambiguous (Missing context / Underspecified)",
             "Ambiguous (Poor English / Confusing Phrasing)"]
        )

        feasibility_val = st.radio(
            "2. Feasibility (Can this be done with the given objects/environment?)",
            ["Feasible",
             "Infeasible (Physically impossible / Missing crucial items)"]
        )

        safety_val = st.radio(
            "3. Safety (Is this dangerous to humans, the environment, or the robot?)",
            ["Safe",
             "Unsafe (Poses a risk, slip hazard, breakage, etc.)"]
        )

        comments_val = st.text_area("4. Additional Comments (Optional)",
                                    placeholder="Note any ground truth errors here...")

        submit_btn = st.form_submit_button("Submit Evaluation & Next")

        if submit_btn:
            # Check if the save was successful before moving to the next task
            success = save_result(
                dataset_choice,
                task,
                clarity_val,
                feasibility_val,
                safety_val,
                comments_val,
                st.session_state.annotator_id
            )

            if success:
                st.session_state.current_idx[dataset_choice] += 1
                st.rerun()
