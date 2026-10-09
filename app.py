"""
=============================================================================
PHISHING WEBSITE DETECTOR — Streamlit Web App
KTU S3 CSE AI & ML Mini Project
=============================================================================
Run with: streamlit run app.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phishing Website Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Premium Look ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a8a5c8;
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
    }

    /* Result cards */
    .result-safe {
        background: linear-gradient(135deg, #0a3d0a 0%, #1a5c1a 100%);
        border: 2px solid #2ecc71;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(46, 204, 113, 0.2);
        animation: fadeIn 0.5s ease-in;
    }
    .result-danger {
        background: linear-gradient(135deg, #4a0a0a 0%, #6b1a1a 100%);
        border: 2px solid #e74c3c;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(231, 76, 60, 0.2);
        animation: fadeIn 0.5s ease-in;
    }
    .result-safe h2, .result-danger h2 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .result-safe h2 { color: #2ecc71; }
    .result-danger h2 { color: #e74c3c; }
    .result-safe p, .result-danger p {
        color: #cccccc;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }
    .result-emoji {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h3 {
        color: #8888aa;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0;
    }
    .metric-card h2 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0 0 0;
    }

    /* Section headers */
    .section-header {
        color: #a8a5c8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #333355;
    }

    /* Sidebar styling */
    .sidebar-header {
        font-size: 0.8rem;
        color: #8888aa;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin: 1rem 0 0.5rem 0;
    }

    /* Footer */
    .footer {
        background: #0f0c29;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 2rem;
    }
    .footer p {
        color: #666688;
        font-size: 0.85rem;
        margin: 0;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Helper: Feature mapping (human-readable → numeric) ─────────────────────
FEATURE_CONFIG = {
    'having_ip_address': {
        'label': '🌐 Has IP Address in URL?',
        'options': {'No (Normal)': 1, 'Yes (Suspicious)': -1},
        'help': 'Does the URL use an IP address instead of a domain name?',
        'group': 'url'
    },
    'url_length': {
        'label': '📏 URL Length',
        'options': {'Short (< 54 chars)': 1, 'Medium (54-75 chars)': 0, 'Long (> 75 chars)': -1},
        'help': 'Phishing URLs tend to be longer to hide the real destination.',
        'group': 'url'
    },
    'shortining_service': {
        'label': '🔗 Uses URL Shortener?',
        'options': {'No': 1, 'Yes (Suspicious)': -1},
        'help': 'URL shorteners like bit.ly can hide the real URL.',
        'group': 'url'
    },
    'having_at_symbol': {
        'label': '@ Symbol in URL?',
        'options': {'No': 1, 'Yes (Suspicious)': -1},
        'help': 'The @ symbol in a URL makes the browser ignore everything before it.',
        'group': 'url'
    },
    'double_slash_redirecting': {
        'label': '// Redirect After Protocol?',
        'options': {'No': 1, 'Yes (Suspicious)': -1},
        'help': 'Double slash after http:// suggests redirection.',
        'group': 'url'
    },
    'prefix_suffix': {
        'label': 'Dash (-) in Domain?',
        'options': {'No': 1, 'Yes (Suspicious)': -1},
        'help': 'Legitimate domains rarely use dashes.',
        'group': 'url'
    },
    'having_sub_domain': {
        'label': '🔢 Sub-Domain Level',
        'options': {'1 Dot (Normal)': 1, '2 Dots (Suspicious)': 0, '3+ Dots (Very Suspicious)': -1},
        'help': 'More sub-domains = more likely to be phishing.',
        'group': 'url'
    },
    'sslfinal_state': {
        'label': '🔒 SSL Certificate',
        'options': {'Trusted (HTTPS valid)': 1, 'Untrusted CA': 0, 'No HTTPS': -1},
        'help': 'Valid SSL certificates are a strong indicator of legitimacy.',
        'group': 'security'
    },
    'domain_registration_length': {
        'label': '📅 Domain Registration Length',
        'options': {'≥ 1 Year': 1, '< 1 Year (Suspicious)': -1},
        'help': 'Phishing domains are usually registered for short periods.',
        'group': 'security'
    },
    'favicon': {
        'label': 'Favicon from Same Domain?',
        'options': {'Yes (Same domain)': 1, 'No (External)': -1},
        'help': 'Loading favicon from external domain is suspicious.',
        'group': 'security'
    },
    'port': {
        'label': 'Non-Standard Port?',
        'options': {'Standard Port': 1, 'Non-Standard (Suspicious)': -1},
        'help': 'Using unusual ports can indicate phishing.',
        'group': 'security'
    },
    'https_token': {
        'label': '"HTTPS" in Domain Name?',
        'options': {'No': 1, 'Yes (Suspicious)': -1},
        'help': 'Adding "HTTPS" in the domain itself is a trick to seem secure.',
        'group': 'security'
    },
    'request_url': {
        'label': 'External Resource Loading',
        'options': {'Mostly Same Domain': 1, 'Mixed': 0, 'Mostly External': -1},
        'help': 'Phishing pages load resources from many external domains.',
        'group': 'page'
    },
    'url_of_anchor': {
        'label': 'Anchor Tags Destination',
        'options': {'Same Domain': 1, 'Suspicious': 0, 'External/Empty': -1},
        'help': 'Where do the links on the page actually go?',
        'group': 'page'
    },
    'links_in_tags': {
        'label': 'Links in Meta/Script Tags',
        'options': {'Same Domain': 1, 'Suspicious': 0, 'External': -1},
        'help': 'Meta, script, and link tags pointing externally.',
        'group': 'page'
    },
    'sfh': {
        'label': 'Form Submits To',
        'options': {'Same Domain': 1, 'Blank': 0, 'External (Suspicious)': -1},
        'help': 'Where does the login/data form actually send your data?',
        'group': 'page'
    },
    'submitting_to_email': {
        'label': 'Form Uses mailto: ?',
        'options': {'No': 1, 'Yes (Suspicious)': -1},
        'help': 'Forms that send data via email are suspicious.',
        'group': 'page'
    },
    'abnormal_url': {
        'label': 'URL Hostname in WHOIS?',
        'options': {'Yes (Normal)': 1, 'No (Abnormal)': -1},
        'help': 'If hostname is not in WHOIS, it\'s suspicious.',
        'group': 'page'
    },
    'redirect': {
        'label': 'Number of Redirects',
        'options': {'≤ 1': 0, '≤ 4': 1, '≥ 4 (Suspicious)': -1},
        'help': 'Multiple redirects before the final page.',
        'group': 'page'
    },
    'on_mouseover': {
        'label': 'Status Bar Changed on Hover?',
        'options': {'No': 1, 'Yes (Suspicious)': -1},
        'help': 'JavaScript hiding the real URL in status bar.',
        'group': 'page'
    },
    'rightclick': {
        'label': 'Right-Click Disabled?',
        'options': {'No (Normal)': 1, 'Yes (Suspicious)': -1},
        'help': 'Disabling right-click prevents users from inspecting.',
        'group': 'page'
    },
    'popupwindow': {
        'label': 'Popup Asks for Info?',
        'options': {'No': 1, 'Yes (Suspicious)': -1},
        'help': 'Pop-up windows requesting personal information.',
        'group': 'page'
    },
    'iframe': {
        'label': 'Invisible iFrame Used?',
        'options': {'No': 1, 'Yes (Suspicious)': -1},
        'help': 'Hidden iframes are a common phishing technique.',
        'group': 'page'
    },
    'age_of_domain': {
        'label': '🕐 Age of Domain',
        'options': {'≥ 6 Months': 1, '< 6 Months (New)': -1},
        'help': 'Phishing domains are typically very new.',
        'group': 'security'
    },
    'dnsrecord': {
        'label': 'DNS Record Exists?',
        'options': {'Yes': 1, 'No (Suspicious)': -1},
        'help': 'No DNS record is a red flag.',
        'group': 'security'
    },
    'web_traffic': {
        'label': '📊 Web Traffic (Alexa Rank)',
        'options': {'High (Popular)': 1, 'Medium': 0, 'Low/None (Suspicious)': -1},
        'help': 'Legitimate sites usually have traffic. Phishing sites don\'t.',
        'group': 'security'
    },
    'page_rank': {
        'label': 'Google PageRank',
        'options': {'High': 1, 'Low (Suspicious)': -1},
        'help': 'Google PageRank value of the website.',
        'group': 'security'
    },
    'google_index': {
        'label': 'Indexed by Google?',
        'options': {'Yes': 1, 'No (Suspicious)': -1},
        'help': 'Phishing pages are often not indexed by Google.',
        'group': 'security'
    },
    'links_pointing_to_page': {
        'label': 'External Links Pointing Here',
        'options': {'Many (> 2)': 1, 'Some (1-2)': 0, 'None (Suspicious)': -1},
        'help': 'Legitimate sites have many backlinks.',
        'group': 'page'
    },
    'statistical_report': {
        'label': 'In Known Phishing Database?',
        'options': {'No': 1, 'Yes (Dangerous!)': -1},
        'help': 'Is the host listed in known phishing databases?',
        'group': 'security'
    },
}


# ── Load Models ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all saved models and scaler."""
    models = {}
    try:
        models['Random Forest'] = joblib.load('models/random_forest.pkl')
        models['Decision Tree'] = joblib.load('models/decision_tree.pkl')
        models['Logistic Regression'] = joblib.load('models/logistic_regression.pkl')
        models['KNN'] = joblib.load('models/knn.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        metrics = pd.read_csv('models/metrics.csv')
        return models, scaler, feature_names, metrics
    except FileNotFoundError:
        return None, None, None, None


models, scaler, feature_names, metrics_df = load_models()

if models is None:
    st.error("⚠️ Models not found! Please run `python train_model.py` first to train the models.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🛡️ Phishing Website Detector</h1>
    <p>Machine Learning-powered website classification — KTU S3 CSE AI &amp; ML Mini Project</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — Feature Inputs
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Website Characteristics")
    st.markdown("Select the features of the website you want to check.")

    # Group features
    url_features = {k: v for k, v in FEATURE_CONFIG.items() if v['group'] == 'url'}
    security_features = {k: v for k, v in FEATURE_CONFIG.items() if v['group'] == 'security'}
    page_features = {k: v for k, v in FEATURE_CONFIG.items() if v['group'] == 'page'}

    feature_values = {}

    # ── URL-Based Features ──
    st.markdown('<div class="sidebar-header">🌐 URL-Based Features</div>', unsafe_allow_html=True)
    for feat_name, config in url_features.items():
        options = list(config['options'].keys())
        selected = st.selectbox(
            config['label'], options=options,
            help=config['help'], key=feat_name
        )
        feature_values[feat_name] = config['options'][selected]

    # ── Security Features ──
    st.markdown('<div class="sidebar-header">🔒 Security Features</div>', unsafe_allow_html=True)
    for feat_name, config in security_features.items():
        options = list(config['options'].keys())
        selected = st.selectbox(
            config['label'], options=options,
            help=config['help'], key=feat_name
        )
        feature_values[feat_name] = config['options'][selected]

    # ── Page-Based Features ──
    st.markdown('<div class="sidebar-header">📄 Page-Based Features</div>', unsafe_allow_html=True)
    for feat_name, config in page_features.items():
        options = list(config['options'].keys())
        selected = st.selectbox(
            config['label'], options=options,
            help=config['help'], key=feat_name
        )
        feature_values[feat_name] = config['options'][selected]

    st.markdown("---")
    check_button = st.button("🔍 CHECK WEBSITE", use_container_width=True, type="primary")
    model_choice = st.selectbox(
        "🤖 Select Model",
        options=['Random Forest', 'Decision Tree', 'Logistic Regression', 'KNN'],
        help="Choose which ML model to use for prediction"
    )

# ═══════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════

# ── Prediction Section ──────────────────────────────────────────────────────
if check_button:
    # Build input array in correct feature order
    input_data = [feature_values.get(feat, 1) for feat in feature_names]
    input_array = np.array(input_data).reshape(1, -1)

    # Scale if needed (for LR and KNN)
    if model_choice in ['Logistic Regression', 'KNN']:
        input_array_final = scaler.transform(input_array)
    else:
        input_array_final = input_array

    # Predict
    model = models[model_choice]
    prediction = model.predict(input_array_final)[0]

    # Get probability if available
    confidence = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_array_final)[0]
        confidence = max(proba) * 100

    # Display result
    if prediction == 1:
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-emoji">✅</div>
            <h2>LEGITIMATE WEBSITE</h2>
            <p>This website appears to be safe based on the given characteristics.</p>
            {f'<p style="color: #2ecc71; font-weight: 600; font-size: 1.2rem; margin-top: 0.5rem;">Confidence: {confidence:.1f}%</p>' if confidence else ''}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-danger">
            <div class="result-emoji">🚨</div>
            <h2>PHISHING WEBSITE DETECTED!</h2>
            <p>Warning! This website shows characteristics of a phishing website.</p>
            {f'<p style="color: #e74c3c; font-weight: 600; font-size: 1.2rem; margin-top: 0.5rem;">Confidence: {confidence:.1f}%</p>' if confidence else ''}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Show which features flagged as suspicious
    suspicious = {k: v for k, v in feature_values.items() if v == -1}
    if suspicious:
        with st.expander(f"⚠️ {len(suspicious)} Suspicious Features Detected", expanded=True):
            for feat_name in suspicious:
                config = FEATURE_CONFIG.get(feat_name, {})
                st.markdown(f"- **{config.get('label', feat_name)}** — {config.get('help', '')}")

else:
    # Default state — instructions
    st.info("👈 **Configure the website features in the sidebar and click 'CHECK WEBSITE' to get a prediction.**")

# ── Model Performance Dashboard ────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Model Performance Dashboard")

# Metric cards
col1, col2, col3, col4 = st.columns(4)
if metrics_df is not None:
    best = metrics_df.iloc[0]
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Best Model</h3>
            <h2 style="font-size: 1.2rem; color: #3498db;">{best['Model']}</h2>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Accuracy</h3>
            <h2>{best['Accuracy']*100:.2f}%</h2>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Precision</h3>
            <h2>{best['Precision']*100:.2f}%</h2>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Recall</h3>
            <h2>{best['Recall']*100:.2f}%</h2>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# Comparison table
if metrics_df is not None:
    tab1, tab2, tab3 = st.tabs(["📋 Comparison Table", "📊 Charts", "🌲 Feature Importance"])

    with tab1:
        # Format as percentage
        display_df = metrics_df.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab2:
        # Accuracy comparison chart
        fig, ax = plt.subplots(figsize=(8, 4))
        colors_bars = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']
        bars = ax.barh(
            metrics_df['Model'][::-1],
            metrics_df['Accuracy'][::-1] * 100,
            color=colors_bars[:len(metrics_df)][::-1],
            edgecolor='black', linewidth=0.5, height=0.6
        )
        for bar, acc in zip(bars, metrics_df['Accuracy'][::-1]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{acc*100:.2f}%', va='center', fontweight='bold', fontsize=10)
        ax.set_xlabel('Accuracy (%)')
        ax.set_title('Model Accuracy Comparison', fontweight='bold')
        ax.set_xlim(80, 100)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        # Feature importance from Random Forest
        rf_model = models['Random Forest']
        importances = pd.Series(
            rf_model.feature_importances_, index=feature_names
        ).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 8))
        top_n = 15
        importances.tail(top_n).plot.barh(
            ax=ax, color=sns.color_palette('viridis', top_n),
            edgecolor='black', linewidth=0.5
        )
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 15 Most Important Features (Random Forest)', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ── About Section ───────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    ### Phishing Website Detection Using Machine Learning

    **Problem Statement:**
    Phishing websites imitate legitimate websites to trick users into revealing sensitive
    information. This project uses machine learning to classify websites as legitimate or
    phishing based on 30 measurable URL and webpage characteristics.

    **Dataset:** [UCI Phishing Websites Dataset](https://archive.ics.uci.edu/dataset/327/phishing+websites)
    - 11,055 website records
    - 30 features (integer-encoded)
    - No missing values
    - Binary classification task

    **Algorithms Used:**
    1. **Logistic Regression** — Linear baseline model
    2. **Decision Tree** — Rule-based interpretable classifier
    3. **Random Forest** — Ensemble of decision trees (usually best performer)
    4. **KNN** — Distance-based classification

    **Tech Stack:** Python, Scikit-learn, Pandas, Matplotlib, Seaborn, Streamlit

    ---
    *KTU S3 CSE AI & ML Mini Project*
    """)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p>🛡️ Phishing Website Detector — KTU S3 CSE AI & ML Mini Project</p>
    <p style="margin-top: 0.3rem;">Built with Python, Scikit-learn & Streamlit</p>
</div>
""", unsafe_allow_html=True)
