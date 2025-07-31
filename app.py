# app.py
import streamlit as st  # Main library for creating web app
import pandas as pd     # For data manipulation
import joblib           # For loading the machine learning model
import numpy as np      # For numerical operations

# =====================================================================
# SECTION 1: LOAD THE MACHINE LEARNING MODEL
# =====================================================================
# Try to load my pre-trained model file
# If there's an error, show it to the user and stop the app
try:
    # Load the kidney disease prediction model I trained 
    model = joblib.load('best_gb_model_tuned.pkl')
except Exception as e:
    # If loading fails, show the error message in red and stop the app
    st.error(f"Error loading model: {str(e)}")
    st.stop()  # Stop execution completely

# =====================================================================
# SECTION 2: SET UP THE PAGE CONFIGURATION
# =====================================================================
# Configure how web page will look
st.set_page_config(
    page_title="Kidney Health Analyser",  # Tab title in browser
    page_icon="🩺",                       # Kidney emoji for browser tab
    layout="wide",                       # Use full width of screen
    initial_sidebar_state="expanded"     # Show sidebar expanded by default
)

# =====================================================================
# SECTION 3: CUSTOM STYLING WITH CSS
# =====================================================================
# Design website appealing aesthetic theme
# Added custom colors, fonts, and layout styles
st.markdown("""
<style>
    /* Import a nice-looking font from Google */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Define the color scheme for easy reuse */
    :root {
        --primary: #0066CC;        /* Main blue color */
        --primary-light: #4D94FF;  /* Lighter blue */
        --secondary: #00A896;      /* Teal color */
        --accent: #FF6B6B;         /* Red for warnings */
        --light: #F0F7FF;          /* Very light blue */
        --dark: #003366;           /* Dark blue */
    }
    
    /* Use the chosen font everywhere */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Set a light blue background gradient for the whole page */
    body {
        background: linear-gradient(135deg, #e0f0ff 0%, #f0f7ff 100%);
        background-attachment: fixed;
    }
    
    /* Make sure Streamlit doesn't override the background */
    .stApp {
        background: transparent !important;
    }
    
    /* Style for the top header section with blue gradient */
    .header-container {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        border-radius: 0 0 20px 20px; /* Rounded bottom corners */
        padding: 2rem 0;              /* Space inside header */
        box-shadow: 0 4px 20px rgba(0,0,0,0.1); /* Soft shadow */
        margin-bottom: 2rem;           /* Space below header */
    }
    
    /* Main title style inside header */
    .header {
        color: white;                 /* White text */
        font-size: 2.5rem;            /* Large size */
        font-weight: 700;             /* Bold */
        text-align: center;            /* Center text */
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2); /* Text shadow for depth */
        margin: 0;                    /* Remove default margins */
    }
    
    /* Subtitle style under main title */
    .subheader {
        color: rgba(255,255,255,0.9); /* Semi-transparent white */
        font-size: 1.2rem;             /* Medium size */
        text-align: center;            /* Center text */
        max-width: 700px;              /* Limit width for readability */
        margin: 0.5rem auto 0;        /* Space above and center horizontally */
        font-weight: 300;              /* Light weight */
    }
    
    /* Styling for the input sections where users enter data */
    .input-section {
        background: white;             /* White background */
        border-radius: 20px;           /* Rounded corners */
        padding: 1.5rem;               /* Space inside section */
        box-shadow: 0 8px 20px rgba(0,0,0,0.05); /* Soft shadow */
        margin-bottom: 1.5rem;         /* Space below section */
        border: 1px solid rgba(0,0,0,0.05); /* Light border */
        transition: all 0.3s ease;     /* Smooth hover effect */
    }
    
    /* Hover effect for input sections */
    .input-section:hover {
        box-shadow: 0 10px 25px rgba(0,0,0,0.1); /* Stronger shadow on hover */
        transform: translateY(-3px);             /* Slight lift effect */
    }
    
    /* Style for section headings (like "Basic Health Metrics") */
    .input-section h3 {
        color: var(--primary);         /* Blue text */
        border-bottom: 2px solid var(--primary-light); /* Blue underline */
        padding-bottom: 0.5rem;        /* Space below text */
        margin-bottom: 1.5rem;         /* Space below heading */
        font-weight: 1000;             /* Extra bold */
    }
    
    /* Style for all buttons in the app */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%); /* Blue gradient */
        color: white;                 /* White text */
        border-radius: 50px;           /* Fully rounded ends */
        padding: 12px 36px;            /* Vertical and horizontal space */
        font-weight: 1000;             /* Extra bold text */
        font-size: 1.1rem;             /* Button text size */
        margin: 10px auto;             /* Space around button, center horizontally */
        display: block;                /* Make buttons full-width containers */
        transition: all 0.3s ease;     /* Smooth hover effects */
        border: none;                  /* Remove default border */
        box-shadow: 0 4px 15px rgba(0,102,204,0.3); /* Blue shadow */
        width: 100%;                   /* Full width of container */
    }
    
    /* Button hover effect */
    .stButton>button:hover {
        transform: scale(1.05);        /* Slight grow effect */
        box-shadow: 0 6px 20px rgba(0,102,204,0.4); /* Stronger shadow */
        color: white;                  /* Keep text white */
    }
    
    /* Special style for the Analyze button to make it larger */
    .analyze-button>button {
        font-size: 1.3rem !important;  /* Bigger text */
        padding: 15px 40px !important; /* More space inside */
    }
    
    /* Style for the results box */
    .result-box {
        background: white;             /* White background */
        border-radius: 20px;           /* Rounded corners */
        padding: 2.5rem;               /* Generous padding */
        margin: 2rem 0;                /* Space above and below */
        box-shadow: 0 10px 30px rgba(0,0,0,0.08); /* Subtle shadow */
        border-left: 8px solid;        /* Thick left border */
        transition: all 0.5s ease;     /* Smooth transitions */
        animation: fadeIn 0.8s ease;   /* Fade-in animation */
    }
    
    /* Fade-in animation for results */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); } /* Start hidden and lower */
        to { opacity: 1; transform: translateY(0); }      /* End visible and in place */
    }
    
    /* Style when high risk is detected */
    .risk-high {
        color: var(--accent);          /* Use red accent color */
        border-color: var(--accent);   /* Red border */
    }
    
    /* Style when low risk is detected */
    .risk-low {
        color: var(--secondary);       /* Use teal secondary color */
        border-color: var(--secondary); /* Teal border */
    }
    
    /* Result title styling */
    .result-title {
        font-size: 2rem;               /* Large text */
        font-weight: 1000;              /* Extra bold */
        text-align: center;             /* Center text */
        margin-bottom: 1.5rem;         /* Space below title */
        display: flex;                 /* For icon alignment */
        align-items: center;            /* Center vertically */
        justify-content: center;       /* Center horizontally */
        gap: 0.8rem;                   /* Space between icon and text */
    }
    
    /* Style for the result explanation text */
    .result-content {
        font-size: 1.1rem;             /* Comfortable reading size */
        line-height: 1.6;               /* Space between lines */
        color: #444;                   /* Dark gray for readability */
    }
    
    /* Style for the information/about section */
    .info-section {
        background: linear-gradient(135deg, rgba(0,168,150,0.1) 0%, rgba(0,102,204,0.1) 100%); /* Light gradient */
        border-radius: 20px;           /* Rounded corners */
        padding: 2rem;                 /* Internal spacing */
        margin: 2rem 0;                /* Space above and below */
        box-shadow: 0 5px 15px rgba(0,0,0,0.05); /* Subtle shadow */
    }
    
    /* Footer style at bottom of page */
    .footer {
        text-align: center;             /* Center text */
        color: #666;                   /* Medium gray */
        font-size: 0.9rem;             /* Small text */
        margin-top: 3rem;              /* Space above */
        padding: 1.5rem;               /* Internal padding */
        border-top: 1px solid rgba(0,0,0,0.08); /* Light top border */
    }
    
    /* Labels for input values */
    .value-label {
        font-weight: 700;              /* Bold */
        color: var(--primary);         /* Blue color */
        margin-bottom: 0.5rem;         /* Space below */
    }
    
    /* Kidney icon animation */
    .kidney-icon {
        font-size: 2rem;               /* Large icon */
        animation: pulse 2s infinite;   /* Continuous pulsing animation */
    }
    
    /* Pulsing animation for kidney icon */
    @keyframes pulse {
        0% { transform: scale(1); }    /* Normal size */
        50% { transform: scale(1.1); } /* Slightly larger */
        100% { transform: scale(1); }  /* Back to normal */
    }
    
    /* Fix styling for number input fields */
    .stNumberInput input {
        background-color: white !important; /* Force white background */
        font-weight: 600 !important;      /* Semi-bold text */
    }
    
    /* Fix styling for slider input fields */
    .stSlider input {
        background-color: white !important; /* Force white background */
    }
    
    /* Make field titles bolder and more visible */
    .stNumberInput label, .stSlider label {
        font-weight: 700 !important;    /* Bold */
        color: var(--dark) !important;  /* Dark blue color */
        margin-bottom: 8px !important;  /* Space below label */
    }
    
    /* Container for buttons to arrange them nicely */
    .button-container {
        display: flex;                  /* Use flexbox layout */
        flex-direction: column;         /* Stack buttons vertically */
        gap: 10px;                     /* Space between buttons */
        max-width: 600px;               /* Limit width */
        margin: 20px auto 0;           /* Center horizontally, space above */
    }
</style>
""", unsafe_allow_html=True)  # Tell Streamlit this is raw HTML

# =====================================================================
# SECTION 4: SET UP THE APP STATE
# =====================================================================
# Using Streamlit's session state to remember:
# - Which page user is on
# - What values the user has entered
if 'page' not in st.session_state:
    # Start on the first page (welcome screen)
    st.session_state.page = 1
    
    # Set default values for all input fields
    st.session_state.inputs = {
        'age': 40,              # Default age
        'glucose': 100,          # Default blood glucose
        'serum_creatinine': 1.00, # Default creatinine
        'potassium': 4.00,       # Default potassium
        'wbc': 7000,            # Default white blood cells
        'urine_output': 1500,    # Default urine output
        'blood_urea': 40.00,     # Default blood urea
        'upcr': 0.50,           # Default protein ratio
        'egfr': 90.00,          # Default kidney function
        'pth': 40.00,           # Default hormone level
        'il6': 5.00             # Default inflammation marker
    }

# =====================================================================
# SECTION 5: NAVIGATION FUNCTION
# =====================================================================
# This function changes pages and refreshes the app
def go_to_page(page_num):
    st.session_state.page = page_num  # Change the current page
    st.rerun()  # Refresh the app to show the new page

# =====================================================================
# SECTION 6: PAGE DEFINITIONS
# =====================================================================
# Defining each page of the app as a separate function

# ---------------------------
# PAGE 1: WELCOME SCREEN
# ---------------------------
def welcome_page():
    # Show the main header with the app title
    st.markdown("""
    <div class="header-container">
        <div class="header">Kidney Health Assessment</div>
        <div class="subheader">Welcome to your kidney health analysis portal</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show a welcome message and brief description
    st.markdown("""
    <div style="text-align:center; padding:2rem;">
        <h2>Protect Your Kidney Health</h2>
        <p style="font-size:1.1rem; max-width:800px; margin:0 auto;">
            Early detection of kidney disease can significantly improve treatment outcomes. 
            Our assessment tool analyzes key health indicators to provide insights into your kidney health.
        </p>
        <br>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns to center the Start button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        # Show the main Start button
        if st.button("Start Assessment", 
                    key="start_assessment",
                    use_container_width=True,
                    type="primary"):
            go_to_page(2)  # Go to the basic metrics page
    
    # Show the about section
    st.markdown("""
    <div class="info-section">
        <h3 style="text-align:center; color: var(--primary);">About This Assessment</h3>
        <p style="text-align:center; font-size:1.1rem;">
            This tool analyzes key kidney health indicators using a machine learning model trained on clinical data. 
            It evaluates 11 critical parameters to assess kidney function. Note that this is not a replacement 
            for professional medical diagnosis. Always consult with a healthcare provider for medical advice.
        </p>
    </div>
    
    <div class="footer">
        <p>Kidney Health Assessment Tool • Results should be verified by a healthcare professional</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# PAGE 2: BASIC HEALTH METRICS
# ---------------------------
def basic_metrics_page():
    # Show the header with page title
    st.markdown("""
    <div class="header-container">
        <div class="header">Kidney Health Assessment</div>
        <div class="subheader">Step 1: Basic Health Metrics</div>
    </div>
    """, unsafe_allow_html=True)
    
    # AGE INPUT
    st.session_state.inputs['age'] = st.slider(
        "Age of the patient (years)",  # Label
        5, 100,                        # Min and max values
        st.session_state.inputs['age'], # Current value from memory
        help="Patient's current age"    # Help text on hover
    )
    
    # GLUCOSE INPUT
    st.session_state.inputs['glucose'] = st.number_input(
        "Random blood glucose level (mg/dl)",  # Label
        50, 600,                              # Min and max
        st.session_state.inputs['glucose'],    # Current value
        step=5,                                # Increment size
        help="Random blood sugar level"        # Help text
    )
    
    # CREATININE INPUT
    st.session_state.inputs['serum_creatinine'] = st.number_input(
        "Serum creatinine (mg/dl)",            # Label
        0.50, 20.00,                          # Min and max
        st.session_state.inputs['serum_creatinine'], # Current value
        step=0.1,                              # Increment size
        format="%.1f",                         # Show 1 decimal place
        help="Waste product in blood from muscle metabolism" # Help
    )
    
    # POTASSIUM INPUT
    st.session_state.inputs['potassium'] = st.number_input(
        "Potassium level (mEq/L)",             # Label
        2.00, 8.00,                           # Min and max
        st.session_state.inputs['potassium'],  # Current value
        step=0.01,                             # Small increments
        format="%.2f",                         # Show 2 decimal places
        help="Important electrolyte for nerve/muscle function" # Help
    )
    
    # WHITE BLOOD CELL INPUT
    st.session_state.inputs['wbc'] = st.number_input(
        "White blood cell count (cells/cumm)", # Label
        1000, 20000,                          # Min and max
        st.session_state.inputs['wbc'],        # Current value
        step=50,                              # Increment size
        help="Immune system cells count"       # Help text
    )
    
    # NAVIGATION BUTTONS
    # Create two columns to place buttons side-by-side
    col1, col2 = st.columns(2)
    with col1:
        # Home button to return to start
        if st.button("🏠 Return to Home", key="home_from_basic", use_container_width=True):
            go_to_page(1)  # Go to welcome page
    
    with col2:
        # Next button to go to kidney metrics
        if st.button("Next →", key="next_to_kidney", use_container_width=True):
            go_to_page(3)  # Go to next page

# ---------------------------
# PAGE 3: KIDNEY FUNCTION METRICS
# ---------------------------
def kidney_metrics_page():
    # Show the header with page title
    st.markdown("""
    <div class="header-container">
        <div class="header">Kidney Health Assessment</div>
        <div class="subheader">Step 2: Kidney Function Metrics</div>
    </div>
    """, unsafe_allow_html=True)
    
    # URINE OUTPUT INPUT
    st.session_state.inputs['urine_output'] = st.slider(
        "Urine output (ml/day)",       # Label
        300, 4000,                     # Min and max
        st.session_state.inputs['urine_output'], # Current value
        step=1,                        # Single ml increments
        help="Total urine volume in 24 hours" # Help text
    )

    # BLOOD UREA INPUT
    st.session_state.inputs['blood_urea'] = st.number_input(
        "Blood urea (mg/dl)",          # Label
        5.00, 250.00,                  # Min and max
        st.session_state.inputs['blood_urea'], # Current value
        step=1.0,                       # Whole number increments
        format="%.2f",                  # Show 2 decimal places
        help="Waste product from protein breakdown" # Help
    )
    
    # PROTEIN RATIO INPUT
    st.session_state.inputs['upcr'] = st.number_input(
        "Urine protein-to-creatinine ratio", # Label
        0.01, 10.00,                       # Min and max
        st.session_state.inputs['upcr'],    # Current value
        step=0.1,                           # Increment size
        format="%.2f",                      # Show 2 decimal places
        help="Ratio of protein to creatinine in urine" # Help
    )
    
    # KIDNEY FUNCTION INPUT
    st.session_state.inputs['egfr'] = st.number_input(
        "Estimated Glomerular Filtration Rate (eGFR)", # Label
        0.01, 200.00,                   # Min and max
        st.session_state.inputs['egfr'], # Current value
        step=1.0,                        # Whole number increments
        help="Kidney filtration rate estimate" # Help
    )
    
    # HORMONE LEVEL INPUT
    st.session_state.inputs['pth'] = st.number_input(
        "Parathyroid hormone (PTH) level", # Label
        10.00, 100.00,                   # Min and max
        st.session_state.inputs['pth'],   # Current value
        step=1.0,                        # Whole number increments
        format="%.2f",                   # Show 2 decimal places
        help="Hormone that regulates calcium levels" # Help
    )
    
    # INFLAMMATION MARKER INPUT
    st.session_state.inputs['il6'] = st.number_input(
        "Interleukin-6 (IL-6) level (pg/ml)", # Label
        0.01, 20.00,                        # Min and max
        st.session_state.inputs['il6'],      # Current value
        step=0.05,                           # Small increments
        format="%.2f",                       # Show 2 decimal places
        help="Inflammation marker"           # Help
    )
    
    # SPECIAL LARGE ANALYZE BUTTON
    st.markdown('<div class="analyze-button">', unsafe_allow_html=True)
    if st.button("Analyze Kidney Health", key="analyze", use_container_width=True, type="primary"):
        go_to_page(4)  # Go to results page
    st.markdown('</div>', unsafe_allow_html=True)
    
    # NAVIGATION BUTTONS
    col1, col2 = st.columns(2)
    with col1:
        # Previous button to go back to basic metrics
        if st.button("← Previous", key="prev_to_basic", use_container_width=True):
            go_to_page(2)  # Go to previous page
    
    with col2:
        # Home button to return to start
        if st.button("🏠 Return to Home", key="home_from_kidney", use_container_width=True):
            go_to_page(1)  # Go to welcome page

# ---------------------------
# PAGE 4: RESULTS PAGE
# ---------------------------
def results_page():
    # Show the header with results title
    st.markdown("""
    <div class="header-container">
        <div class="header">Kidney Health Assessment</div>
        <div class="subheader">Your Kidney Health Analysis Results</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Try to generate the kidney health prediction, error handling
    try:
        # Prepare the input data for the model
        # We create a DataFrame with all the values the user entered
        input_data = pd.DataFrame({
            'Age of the patient': [st.session_state.inputs['age']],
            'Random blood glucose level (mg/dl)': [st.session_state.inputs['glucose']],
            'Blood urea (mg/dl)': [st.session_state.inputs['blood_urea']],
            'Serum creatinine (mg/dl)': [st.session_state.inputs['serum_creatinine']],
            'Potassium level (mEq/L)': [st.session_state.inputs['potassium']],
            'White blood cell count (cells/cumm)': [st.session_state.inputs['wbc']],
            'Estimated Glomerular Filtration Rate (eGFR)': [st.session_state.inputs['egfr']],
            'Urine protein-to-creatinine ratio': [st.session_state.inputs['upcr']],
            'Urine output (ml/day)': [st.session_state.inputs['urine_output']],
            'Parathyroid hormone (PTH) level': [st.session_state.inputs['pth']],
            'Interleukin-6 (IL-6) level': [st.session_state.inputs['il6']]
        })
        
        # If the model expects features in a specific order, rearrange the data
        if hasattr(model, 'feature_names_in_'):
            input_data = input_data[model.feature_names_in_]
        
        # Ask the model to predict kidney health status
        prediction = model.predict(input_data)[0]
        
        # Show the results based on prediction
        if prediction == "No_Disease":
            # Display positive result with green icon
            st.markdown('<div class="result-title"><span class="kidney-icon">💚</span> No Kidney Disease Detected</div>', unsafe_allow_html=True)
            st.markdown('<div class="result-content">Based on your input parameters, the model indicates no signs of kidney disease. Continue maintaining good kidney health with regular checkups.</div>', unsafe_allow_html=True)
        else:
            # Display warning result with alert icon
            st.markdown('<div class="result-title"><span class="kidney-icon">⚠️</span> Kidney Disease Detected</div>', unsafe_allow_html=True)
            st.markdown('<div class="result-content">Based on your input parameters, the model indicates potential kidney disease. Please consult with a healthcare professional for further evaluation.</div>', unsafe_allow_html=True)
            
    # If something goes wrong with the prediction
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")  # Show error in red
    
    # RETURN TO HOME BUTTON
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("🏠 Return to Home", key="home_from_results", use_container_width=True):
        go_to_page(1)  # Go back to welcome page
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================================
# SECTION 7: MAIN APP ROUTER
# =====================================================================
# Decide which page to show based on the current state

# If on page 1, show the welcome screen
if st.session_state.page == 1:
    welcome_page()

# If on page 2, show the basic health metrics
elif st.session_state.page == 2:
    basic_metrics_page()

# If on page 3, show the kidney function metrics
elif st.session_state.page == 3:
    kidney_metrics_page()

# If on page 4, show the results
elif st.session_state.page == 4:
    results_page()