
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import numpy as np
import io
import csv
import json 
import random 
import hashlib # Added for password hashing
import os # Added to check if users file exists
# Removed streamlit_extras imports

# --- SET PAGE CONFIG FIRST ---
st.set_page_config(page_title="AI Study Pal", layout="wide")

# --- USER AUTHENTICATION & DATABASE ---
USERS_FILE = 'users.json'

def hash_password(password):
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(hashed_password, plain_password):
    """Checks if the plain password matches the hashed password."""
    return hashed_password == hash_password(plain_password)

def load_users():
    """Loads the user database from the JSON file."""
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f) # Create an empty file if it doesn't exist
        return {}
    
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {} # Return empty dict if file is corrupt or empty

def save_users(users):
    """Saves the user database to the JSON file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

# --- 1. NLTK Data Download (Cached) ---
@st.cache_resource
def download_nltk_data():
    """Downloads required NLTK data packages."""
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab') # For some environments
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

# --- 2. Session State Initialization ---
def init_session_state():
    defaults = {
        'quiz_scores': [],
        'quiz_submitted': False,
        'current_quiz': [],
        'user_answers': {},
        'plan_generated': False,
        'schedule_df': pd.DataFrame(),
        'csv_string': "",
        'summary': "", # Summary is static from the text box
        'subjects_list': [], # List of all subjects
        'first_subject': "", # For the quiz
        'text_input_for_tips': "", # To regenerate tips
        'authentication_status': None, # None, 'pending_signup', 'logged_in'
        'username': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- 3. Model & Data Loading (Cached) ---
@st.cache_resource
def load_models_and_data():
    """Loads placeholder data and trains ML models."""
    return "model1", "model2", "model3"

# --- 4. Backend Helper Functions ---

def get_resource_suggestions(subject):
    """Placeholder for resource suggestion system."""
    subject_lower = subject.lower()
    if 'math' in subject_lower or 'calculus' in subject_lower:
        return 'https://www.khanacademy.org/math'
    if 'science' in subject_lower or 'physics' in subject_lower or 'chemistry' in subject_lower:
        return 'https://www.khanacademy.org/science'
    if 'history' in subject_lower:
        return 'https://www.khanacademy.org/humanities/world-history'
    if 'english' in subject_lower or 'grammar' in subject_lower:
        return 'https://www.khanacademy.org/humanities/grammar'
    # Default fallback
    return f'https://www.khanacademy.org/search?page_search_query={subject.replace(" ", "+")}'

@st.cache_data(show_spinner="Parsing quiz file(s)...")
def parse_quiz_from_file(uploaded_files, num_questions=3):
    """
    Parses uploaded JSON, CSV, or TXT files and randomly selects questions.
    'uploaded_files' is now a LIST.
    """
    quiz_bank = [] # Master list for all questions
    
    if not uploaded_files: # Check if list is empty
        return []

    for uploaded_file in uploaded_files: # Loop through the list of files
        try:
            if uploaded_file.name.endswith('.json'):
                uploaded_file.seek(0)
                # Use .extend() to add all questions from this file
                quiz_bank.extend(json.load(uploaded_file))
                
            elif uploaded_file.name.endswith('.csv'):
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
                required_cols = ['question', 'option1', 'option2', 'option3', 'answer']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"CSV file '{uploaded_file.name}' is missing required columns. Skipping.")
                    continue
                
                for _, row in df.iterrows():
                    quiz_bank.append({
                        "q": row['question'],
                        "options": [row['option1'], row['option2'], row['option3']],
                        "answer": row['answer']
                    })
            
            elif uploaded_file.name.endswith('.txt'):
                uploaded_file.seek(0)
                content = uploaded_file.read().decode("utf-8")
                
                current_q_data = {}
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith("Q:"):
                        if current_q_data.get('q') and current_q_data.get('options') and current_q_data.get('answer'):
                            quiz_bank.append(current_q_data)
                        current_q_data = {"q": line[2:].strip(), "options": [], "answer": None}
                    elif not line:
                        if current_q_data.get('q') and current_q_data.get('options') and current_q_data.get('answer'):
                            quiz_bank.append(current_q_data)
                        current_q_data = {}
                    elif line.startswith("O:") and 'options' in current_q_data:
                        current_q_data['options'].append(line[2:].strip())
                    elif line.startswith("A:") and 'q' in current_q_data:
                        current_q_data['answer'] = line[2:].strip()
                
                if current_q_data.get('q') and current_q_data.get('options') and current_q_data.get('answer'):
                    quiz_bank.append(current_q_data)
                
            else:
                st.warning(f"Skipping unsupported file: {uploaded_file.name}")
        
        except Exception as e:
            st.error(f"An error occurred while processing {uploaded_file.name}: {e}")

    # --- End of loop ---

    if not quiz_bank:
         st.error("No valid questions were found in the uploaded file(s). Please check format.")
         return []
    
    # Now, randomly sample from the combined bank of all files
    if len(quiz_bank) < num_questions:
        st.warning(f"Found {len(quiz_bank)} total questions. Using all of them.")
        return random.sample(quiz_bank, len(quiz_bank)) # Return all
    
    return random.sample(quiz_bank, num_questions) # Return random sample


def generate_summary(text):
    """Generates a simple summary."""
    word_count = len(text.split())
    summary = ' '.join(text.split()[:20]) + '...'
    return summary, word_count

def get_motivational_feedback(subject):
    """Generates random motivational feedback."""
    messages = [f"Great work on {subject}! Keep it up!", f"You're making excellent progress in {subject}!", f"Good job on {subject}!"]
    return np.random.choice(messages)

def generate_study_tips(text):
    """Generates simple study tips from keywords."""
    try:
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        if not filtered_words: return "Study tip: Break down the topic into smaller pieces."
        fdist = FreqDist(filtered_words)
        top_keywords = [word for word, freq in fdist.most_common(3)]
        return f"Study tip: Review these key terms daily: {', '.join(top_keywords)}."
    except Exception as e:
        st.error(f"Error in NLP processing: {e}. Have you downloaded NLTK data?")
        return "Could not generate tips."

def show_login_page():
    st.title("🤖 Welcome to AI Study Pal")
    
    col1, col2 = st.columns(2)
    
    # --- LOGIN FORM ---
    with col1:
        with st.form(key="login_form"):
            st.header("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                users = load_users()
                if username in users and check_password(users[username], password):
                    st.session_state.authentication_status = 'logged_in'
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Incorrect username or password")
    
    # --- SIGNUP FORM ---
    with col2:
        with st.form(key="signup_form"):
            st.header("Sign Up")
            new_username = st.text_input("Choose a Username")
            new_password = st.text_input("Choose a Password", type="password")
            signup_button = st.form_submit_button("Sign Up")
            
            if signup_button:
                users = load_users()
                if new_username in users:
                    st.error("Username already exists. Please choose another one or login.")
                elif not new_username or not new_password:
                     st.error("Please enter a username and password.")
                else:
                    users[new_username] = hash_password(new_password)
                    save_users(users)
                    st.session_state.authentication_status = 'logged_in'
                    st.session_state.username = new_username
                    st.success("Account created successfully! You are now logged in.")
                    st.rerun()

def show_main_app():
    st.sidebar.title("AI Study Pal")
    st.sidebar.info("Upload quiz file(s) and enter subject(s) to start.")

    st.sidebar.header(f"Welcome, {st.session_state.username}!")
    
    st.sidebar.header("Your Progress (This Session)")
    if st.session_state.quiz_scores:
        progress_df = pd.DataFrame({'Quiz': range(1, len(st.session_state.quiz_scores) + 1), 'Score (%)': st.session_state.quiz_scores})
        st.sidebar.line_chart(progress_df.set_index('Quiz'))
        st.sidebar.write(f"**Average Score:** {np.mean(st.session_state.quiz_scores):.2f}%")
        if st.sidebar.button("Clear Progress"):
            st.session_state.quiz_scores = []
            st.rerun()
    else:
        st.sidebar.info("Submit your first quiz to see your progress!")
    
    st.sidebar.header("Global Leaderboard")
    mock_leaderboard_data = {
        'User': ['Student_A', 'Pro_Learner', 'TestUser'],
        'Top Score': [100, 80, 75]
    }
    st.sidebar.dataframe(pd.DataFrame(mock_leaderboard_data))
    with st.sidebar.expander("About the Leaderboard"):
        st.write("""
            This leaderboard is for demonstration. 
            A persistent leaderboard requires connecting to an external cloud database.
        """)
    
    if st.sidebar.button("Logout"):
        st.session_state.authentication_status = None
        st.session_state.username = None
        for key in st.session_state.keys():
            if key not in ['authentication_status', 'username']:
                del st.session_state[key]
        st.rerun()

    st.title("🤖 AI Study Pal")
    st.write(f"Ready to study, **{st.session_state.username}**? Let's get started.")

    with st.form(key="study_form"):
        st.header("1. Generate Your Study Aids")
        
        uploaded_files = st.file_uploader(
            "Upload Your Quiz File(s) (.json, .csv, or .txt)",
            type=['json', 'csv', 'txt'],
            accept_multiple_files=True 
        )
        
        st.header("2. Set Your Study Plan")
        col1, col2 = st.columns(2)
        with col1:
            subject_input_text = st.text_input(
                "Enter Subject(s), separated by commas:",
                "Math, Physics, Biology"
            )
            
        with col2:
            hours_input = st.number_input("Total Study Hours:", min_value=1, value=4)

        text_input = st.text_area(
            "Short text to summarize (optional):", 
            "Calculus is the study of continuous change...",
            height=100
        )
        
        submit_button = st.form_submit_button(label="Generate Study Aids & Quiz")

    with st.expander("Click to see required file formats"):
        st.write("Your file **must** be in one of these formats. Using **blank lines** in .txt files is recommended!")
        st.subheader("Format 1: `quiz.txt` (Recommended)")
        st.code("""
Q: What is 2 + 2?
O: 3
O: 4
O: 5
A: 4

Q: What is the capital of France?
O: London
O: Berlin
O: Paris
A: Paris
""")
        st.subheader("Format 2: `quiz.json`")
        st.json([{"q": "What is 2 + 2?", "options": ["3", "4", "5"], "answer": "4"}])
        st.subheader("Format 3: `quiz.csv`")
        st.code("question,option1,option2,option3,answer\nWhat is 2 + 2?,3,4,5,4")

    if submit_button:
        if not uploaded_files:
            st.error("Please upload at least one quiz file to continue.")
        elif not subject_input_text:
            st.error("Please enter at least one subject.")
        else:
            st.session_state.plan_generated = True
            st.session_state.quiz_submitted = False 
            st.session_state.user_answers = {}
            
            subjects_list = [s.strip() for s in subject_input_text.split(',') if s.strip()]
            if not subjects_list:
                st.error("Please enter a valid subject.")
                st.session_state.plan_generated = False
                st.stop() 

            st.session_state.subjects_list = subjects_list
            st.session_state.first_subject = subjects_list[0]
            hours_per_subject = hours_input / len(subjects_list)
            st.session_state.text_input_for_tips = text_input
            
            st.session_state.current_quiz = parse_quiz_from_file(uploaded_files, num_questions=50)
            
            if not st.session_state.current_quiz:
                st.error(f"Could not generate quiz from the uploaded file(s). Please check format.")
                st.session_state.plan_generated = False
            else:
                st.success("Your study aids have been generated! Check the tabs below.")
                
                schedule_data = [{'Subject': s, 'Hours': round(hours_per_subject, 2)} for s in subjects_list]
                st.session_state.schedule_df = pd.DataFrame(schedule_data)

                study_plan_data = [['Subject', 'Task', 'Duration (hours)']]
                for subject in subjects_list:
                    study_plan_data.append([subject, f'Review {subject} Notes', f'{hours_per_subject * 0.5:.2f}'])
                    study_plan_data.append([subject, f'Take Practice Quiz', f'{hours_per_subject * 0.25:.2f}'])
                    study_plan_data.append([subject, 'Review Key Terms', f'{hours_per_subject * 0.25:.2f}'])
                
                si = io.StringIO()
                cw = csv.writer(si)
                cw.writerows(study_plan_data)
                st.session_state.csv_string = si.getvalue()

                summary, word_count = generate_summary(text_input)
                st.session_state.summary = f"*(Summarized from {word_count} words)*\n\n{summary}"


    if st.session_state.plan_generated:
        
        tab1, tab2, tab3 = st.tabs(["Study Plan", "Interactive Quiz", "Summary & Tips"])

        with tab1:
            st.header(f"Your Overall Study Schedule") 
            st.write(f"Total time allocated: {st.session_state.schedule_df['Hours'].sum()} hours")
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(st.session_state.schedule_df, use_container_width=True)
                st.download_button(
                    label=f"Download Detailed Schedule (CSV)",
                    data=st.session_state.csv_string,
                    file_name=f"study_schedule.csv", 
                    mime="text/csv",
                )
            with col2:
                st.write("Visual breakdown:")
                st.bar_chart(st.session_state.schedule_df.set_index('Subject'))

        with tab2:
            st.header(f"Practice Quiz: {st.session_state.first_subject}")
            st.write(f"(Quiz is generated for the first subject: **{st.session_state.first_subject}**)")
            
            if not st.session_state.current_quiz:
                st.info("Quiz data is empty. Please check your file.")
            
            elif not st.session_state.quiz_submitted:
                with st.form(key="quiz_mcq_form"):
                    user_answers = {}
                    for i, item in enumerate(st.session_state.current_quiz):
                        st.write(f"**Question {i+1}:** {item['q']}")
                        question_key = f"quiz_q_{item['q']}"
                        
                        user_answers[question_key] = st.radio(
                            "Select your answer:",
                            options=item['options'],
                            key=question_key, 
                            index=None, # <-- THIS IS THE FIX
                            label_visibility="collapsed"
                        )
                    
                    quiz_submit_button = st.form_submit_button("Submit Quiz")

                    if quiz_submit_button:
                        st.session_state.user_answers = {}
                        for item in st.session_state.current_quiz:
                            question_key = f"quiz_q_{item['q']}"
                            st.session_state.user_answers[question_key] = st.session_state[question_key]
                        
                        st.session_state.quiz_submitted = True
                        st.rerun() 

            else:
                st.subheader("Quiz Results")
                score = 0
                for i, item in enumerate(st.session_state.current_quiz):
                    question_key = f"quiz_q_{item['q']}"
                    user_answer = st.session_state.user_answers.get(question_key)
                    correct_answer = item['answer']
                    
                    st.write(f"**Question {i+1}:** {item['q']}")
                    if user_answer == correct_answer:
                        score += 1
                        st.success(f"Your answer: **{user_answer}** (Correct!)")
                    else:
                        st.error(f"Your answer: **{user_answer}** (Incorrect. Correct answer: **{correct_answer}**)")
                    st.divider()
                
                if len(st.session_state.current_quiz) > 0:
                    final_score_percent = (score / len(st.session_state.current_quiz)) * 100
                    st.header(f"Your final score: {final_score_percent:.0f}% ({score}/{len(st.session_state.current_quiz)})")
                
                    if 'last_score' not in st.session_state or st.session_state.last_score != final_score_percent:
                        st.session_state.quiz_scores.append(final_score_percent)
                        st.session_state.last_score = final_score_percent
                else:
                    st.warning("Quiz had no questions.")

                if st.button("Take Quiz Again"):
                    st.session_state.quiz_submitted = False
                    st.session_state.user_answers = {}
                    for item in st.session_state.current_quiz:
                        question_key = f"quiz_q_{item['q']}"
                        if question_key in st.session_state:
                            del st.session_state[question_key]
                    st.session_state.plan_generated = False
                    st.rerun()

        with tab3:
            
            selected_subject_tips = st.selectbox(
                "Show aids for:",
                options=st.session_state.subjects_list,
                index=0 
            )

            st.header(f"Study Aids for {selected_subject_tips}")

            feedback = get_motivational_feedback(selected_subject_tips)
            resource_link = get_resource_suggestions(selected_subject_tips)
            tips = generate_study_tips(st.session_state.text_input_for_tips)

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Motivational Feedback:**\n\n{feedback}")
                st.markdown(f"**Resource:** [Khan Academy for {selected_subject_tips}]({resource_link})")
            
            with col2:
                st.warning(f"**Study Tip:**\n\n{tips}")
            
            st.divider()
            st.subheader("Text Summary")
            st.caption(st.session_state.summary)

download_nltk_data()
load_models_and_data()

if st.session_state.authentication_status == 'logged_in':
    show_main_app()
else:
    show_login_page()
