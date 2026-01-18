"""
Streamlit Dashboard for Telco Customer Churn Prediction
"""
import os

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

# Fix orjson import issue before importing plotly
try:
    import orjson
except (ImportError, AttributeError):
    # If orjson has issues, set plotly to use default JSON engine
    os.environ['PLOTLY_JSON_ENGINE'] = 'json'

import plotly.graph_objects as go
from utils import load_model, load_data, load_raw_data, predict_churn, get_feature_importance
from translations import TRANSLATIONS

# Page configuration
st.set_page_config(
    page_title="Telco Churn Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Translations are imported from translations.py

# Initialize session state
if "language" not in st.session_state:
    st.session_state.language = "tr"
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
if "prev_page" not in st.session_state:
    st.session_state.prev_page = None

def get_text(key_path):
    """Get translated text by key path"""
    keys = key_path.split(".")
    text = TRANSLATIONS[st.session_state.language]
    for key in keys:
        text = text[key]
    return text

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    .stButton>button {
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    /* Tab styling - make tabs more visible */
    button[data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        margin: 0 8px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
        border-bottom: 3px solid #667eea !important;
    }
    button[data-baseweb="tab"][aria-selected="false"] {
        background-color: rgba(102, 126, 234, 0.1) !important;
        color: #667eea !important;
    }
    button[data-baseweb="tab"]:hover {
        background-color: rgba(102, 126, 234, 0.2) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stTabs"] {
        margin-top: 20px !important;
        margin-bottom: 20px !important;
    }
    /* Language selector styling */
    div[data-testid="stRadio"] > div {
        gap: 8px;
    }
    div[data-testid="stRadio"] label {
        padding: 8px 12px;
        border-radius: 8px;
        transition: all 0.3s ease;
        font-size: 15px;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Page keys (language-independent) - define before language selector
page_keys = ["home", "prediction", "analysis", "performance"]

# Initialize current_page if not exists (before language change)
if "current_page" not in st.session_state or st.session_state.current_page not in page_keys:
    st.session_state.current_page = "home"

# Save current page before language change
saved_page = st.session_state.current_page

# Language selector - compact and clean
st.sidebar.caption("Dil / Language")
selected_lang = st.sidebar.radio(
    "Language",
    options=["TR", "EN"],
    index=0 if st.session_state.language == "tr" else 1,
    horizontal=True,
    label_visibility="collapsed",
    key="lang_toggle"
)
if selected_lang == "TR" and st.session_state.language != "tr":
    st.session_state.language = "tr"
    st.rerun()
if selected_lang == "EN" and st.session_state.language != "en":
    st.session_state.language = "en"
    st.rerun()

# Add separator between language and pages
st.sidebar.markdown("---")

# Page navigation with buttons for reliable single-click
for page_key in page_keys:
    page_label = get_text(f"pages.{page_key}")
    is_current = st.session_state.current_page == page_key
    
    if st.sidebar.button(
        page_label,
        key=f"nav_{page_key}",
        use_container_width=True,
        type="primary" if is_current else "secondary"
    ):
        st.session_state.current_page = page_key
        st.rerun()

# Get current page
page = st.session_state.current_page

# Update current page
st.session_state.current_page = page

# Check if page changed
if "prev_page" not in st.session_state:
    st.session_state.prev_page = page
    
page_changed = st.session_state.prev_page != page
if page_changed:
    st.session_state.prev_page = page

# Scroll to top - works best when page changes but try on every render
scroll_js = """
<script>
(function() {
    const parent = window.parent;
    const doc = parent.document;
    
    function scrollToTop() {
        try {
            // Scroll parent window
            parent.scrollTo(0, 0);
            doc.documentElement.scrollTop = 0;
            doc.body.scrollTop = 0;
            
            // Scroll Streamlit containers
            const containers = doc.querySelectorAll('[data-testid="stAppViewContainer"], [data-testid="stMain"], section.main');
            containers.forEach(el => {
                if (el) {
                    el.scrollTop = 0;
                    el.scrollTo && el.scrollTo(0, 0);
                }
            });
        } catch(e) {}
    }
    
    // Execute with requestAnimationFrame for smoother scrolling
    scrollToTop();
    requestAnimationFrame(scrollToTop);
    requestAnimationFrame(() => requestAnimationFrame(scrollToTop));
    
    // Also with timeouts as backup
    setTimeout(scrollToTop, 50);
    setTimeout(scrollToTop, 150);
    setTimeout(scrollToTop, 300);
})();
</script>
"""
components.html(scroll_js, height=0)

# Load data using session state caching (avoids Streamlit cache decorator warnings)
try:
    df = load_raw_data()
    if df is None:
        st.error("Veri yüklenemedi. Lütfen data klasöründe Telco-Customer-Churn.csv dosyasının olduğundan emin olun.")
        st.stop()
except Exception as e:
    st.error(f"Veri yükleme hatası: {str(e)}")
    st.stop()

# Main content
if page == "home":
    st.markdown(f'<h1 class="main-header">{get_text("home.title")}</h1>', unsafe_allow_html=True)
    
    st.markdown(f"""
    ### {get_text("home.welcome")}
    
    {get_text("home.description")}
    
    **{get_text("home.features")}**
    - {get_text("home.feature1")}
    - {get_text("home.feature2")}
    - {get_text("home.feature3")}
    """)
    
    st.markdown("---")
    
    # Key metrics with improved styling
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    avg_tenure = df["tenure"].mean()
    avg_monthly_charges = df["MonthlyCharges"].mean()
    
    with col1:
        st.metric(get_text("home.total_customers"), f"{total_customers:,}", delta=None)
    with col2:
        st.metric(get_text("home.churn_rate"), f"{churn_rate:.1f}%", delta=None)
    with col3:
        st.metric(get_text("home.avg_tenure"), f"{avg_tenure:.1f}", delta=None)
    with col4:
        st.metric(get_text("home.avg_monthly"), f"${avg_monthly_charges:.2f}", delta=None)
    
    # Quick insights
    st.markdown("---")
    st.subheader(get_text('home.quick_stats'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        churn_by_contract = df.groupby("Contract", as_index=False).agg(
            Churn=("Churn", lambda x: (x == "Yes").mean() * 100)
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=churn_by_contract["Contract"],
            y=churn_by_contract["Churn"],
            text=churn_by_contract["Churn"],
            texttemplate="%{text:.1f}%",
            textposition="outside",
            marker=dict(color=churn_by_contract["Churn"], colorscale="Reds", showscale=False)
        ))
        fig.update_layout(
            title=get_text("home.churn_by_contract"),
            xaxis_title=get_text("home.contract_type"),
            yaxis_title=get_text("home.churn_rate_label"),
            showlegend=False,
            height=400,
            yaxis=dict(range=[0, None]),
            margin=dict(b=50, t=50, l=50, r=50, pad=10)
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        churn_by_internet = df.groupby("InternetService", as_index=False).agg(
            Churn=("Churn", lambda x: (x == "Yes").mean() * 100)
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=churn_by_internet["InternetService"],
            y=churn_by_internet["Churn"],
            text=churn_by_internet["Churn"],
            texttemplate="%{text:.1f}%",
            textposition="outside",
            marker=dict(color=churn_by_internet["Churn"], colorscale="Reds", showscale=False)
        ))
        fig.update_layout(
            title=get_text("home.churn_by_internet"),
            xaxis_title=get_text("home.internet_service"),
            yaxis_title=get_text("home.churn_rate_label"),
            showlegend=False,
            height=400,
            yaxis=dict(range=[0, None]),
            margin=dict(b=50, t=50, l=50, r=50, pad=10)
        )
        st.plotly_chart(fig, width='stretch')
    
    # Additional visualizations
    col3, col4 = st.columns(2)
    
    with col3:
        # Tenure distribution
        churn_yes = df[df["Churn"] == "Yes"]["tenure"]
        churn_no = df[df["Churn"] == "No"]["tenure"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=churn_yes,
            name=get_text("analysis.churned"),
            opacity=0.7,
            marker_color="#f44336"
        ))
        fig.add_trace(go.Histogram(
            x=churn_no,
            name=get_text("analysis.not_churned"),
            opacity=0.7,
            marker_color="#4caf50"
        ))
        fig.update_layout(
            title=get_text("home.tenure_dist"),
            xaxis_title=get_text("home.tenure_months"),
            yaxis_title=get_text("home.customer_count"),
            barmode="overlay",
            height=400
        )
        st.plotly_chart(fig, width='stretch')
    
    with col4:
        # Payment method distribution
        payment_counts = df["PaymentMethod"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=payment_counts.index,
            values=payment_counts.values,
            hole=0.4
        )])
        fig.update_layout(
            title=get_text("home.payment_dist"),
            height=400
        )
        st.plotly_chart(fig, width='stretch')

elif page == "prediction":
    st.markdown(f'<h1 class="main-header">{get_text("prediction.title")}</h1>', unsafe_allow_html=True)
    st.markdown(f"### {get_text('prediction.subtitle')}")
    
    with st.form("customer_form"):
        st.subheader(get_text('prediction.customer_info'))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**{get_text('prediction.demographics')}**")
            gender = st.selectbox(
                get_text("prediction.gender"),
                ["Male", "Female"],
                format_func=lambda x: get_text("prediction.male") if x == "Male" else get_text("prediction.female"),
                key="pred_gender"
            )
            senior_citizen = st.selectbox(
                get_text("prediction.senior"),
                [0, 1],
                format_func=lambda x: get_text("yes") if x == 1 else get_text("no"),
                key="pred_senior"
            )
            partner = st.selectbox(
                get_text("prediction.partner"),
                ["Yes", "No"],
                format_func=lambda x: get_text("yes") if x == "Yes" else get_text("no"),
                key="pred_partner"
            )
            dependents = st.selectbox(
                get_text("prediction.dependents"),
                ["Yes", "No"],
                format_func=lambda x: get_text("yes") if x == "Yes" else get_text("no"),
                key="pred_dependents"
            )
            tenure = st.slider(get_text("prediction.tenure"), 0, 72, 12, key="pred_tenure")
        
        with col2:
            st.markdown(f"**{get_text('prediction.services')}**")
            phone_service = st.selectbox(
                get_text("prediction.phone_service"),
                ["Yes", "No"],
                format_func=lambda x: get_text("yes") if x == "Yes" else get_text("no"),
                key="pred_phone_service"
            )
            multiple_lines = st.selectbox(
                get_text("prediction.multiple_lines"),
                ["Yes", "No", "No phone service"],
                format_func=lambda x: get_text("prediction.no_phone") if x == "No phone service" else (get_text("yes") if x == "Yes" else get_text("no")),
                key="pred_multiple_lines"
            )
            internet_service = st.selectbox(
                get_text("prediction.internet_service"),
                ["DSL", "Fiber optic", "No"],
                key="pred_internet_service"
            )
            online_security = st.selectbox(
                get_text("prediction.online_security"),
                ["Yes", "No", "No internet service"],
                format_func=lambda x: get_text("prediction.no_internet") if x == "No internet service" else (get_text("yes") if x == "Yes" else get_text("no")),
                key="pred_online_security"
            )
            online_backup = st.selectbox(
                get_text("prediction.online_backup"),
                ["Yes", "No", "No internet service"],
                format_func=lambda x: get_text("prediction.no_internet") if x == "No internet service" else (get_text("yes") if x == "Yes" else get_text("no")),
                key="pred_online_backup"
            )
        
        with col3:
            st.markdown(f"**{get_text('prediction.additional_services')}**")
            device_protection = st.selectbox(
                get_text("prediction.device_protection"),
                ["Yes", "No", "No internet service"],
                format_func=lambda x: get_text("prediction.no_internet") if x == "No internet service" else (get_text("yes") if x == "Yes" else get_text("no")),
                key="pred_device_protection"
            )
            tech_support = st.selectbox(
                get_text("prediction.tech_support"),
                ["Yes", "No", "No internet service"],
                format_func=lambda x: get_text("prediction.no_internet") if x == "No internet service" else (get_text("yes") if x == "Yes" else get_text("no")),
                key="pred_tech_support"
            )
            streaming_tv = st.selectbox(
                get_text("prediction.streaming_tv"),
                ["Yes", "No", "No internet service"],
                format_func=lambda x: get_text("prediction.no_internet") if x == "No internet service" else (get_text("yes") if x == "Yes" else get_text("no")),
                key="pred_streaming_tv"
            )
            streaming_movies = st.selectbox(
                get_text("prediction.streaming_movies"),
                ["Yes", "No", "No internet service"],
                format_func=lambda x: get_text("prediction.no_internet") if x == "No internet service" else (get_text("yes") if x == "Yes" else get_text("no")),
                key="pred_streaming_movies"
            )
        
        st.markdown("---")
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.markdown(f"**{get_text('prediction.billing')}**")
            contract = st.selectbox(
                get_text("prediction.contract"),
                ["Month-to-month", "One year", "Two year"],
                format_func=lambda x: get_text("prediction.month_to_month") if x == "Month-to-month" else (get_text("prediction.one_year") if x == "One year" else get_text("prediction.two_year")),
                key="pred_contract"
            )
            paperless_billing = st.selectbox(
                get_text("prediction.paperless"),
                ["Yes", "No"],
                format_func=lambda x: get_text("yes") if x == "Yes" else get_text("no"),
                key="pred_paperless"
            )
            payment_method = st.selectbox(
                get_text("prediction.payment_method"),
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                format_func=lambda x: (
                    get_text("prediction.electronic_check") if x == "Electronic check" else
                    get_text("prediction.mailed_check") if x == "Mailed check" else
                    get_text("prediction.bank_transfer") if x == "Bank transfer (automatic)" else
                    get_text("prediction.credit_card")
                ),
                key="pred_payment_method"
            )
        
        with col5:
            st.markdown(f"**{get_text('prediction.charges')}**")
            monthly_charges = st.number_input(
                get_text("prediction.monthly_charges"),
                min_value=0.0,
                max_value=200.0,
                value=50.0,
                step=0.1,
                key="pred_monthly_charges"
            )
            total_charges_kwargs = dict(
                label=get_text("prediction.total_charges"),
                min_value=0.0,
                max_value=10000.0,
                step=1.0,
                help=get_text("prediction.total_charges_help"),
                key="pred_total_charges"
            )
            if "pred_total_charges" not in st.session_state:
                total_charges_kwargs["value"] = float(monthly_charges * tenure) if tenure > 0 else 0.0
            total_charges = st.number_input(**total_charges_kwargs)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(get_text("prediction.predict_button"), type="primary", width='stretch')
    
    if submitted:
        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        
        try:
            with st.spinner(get_text("prediction.predicting")):
                result = predict_churn(customer_data)
            st.session_state.prediction_result = result
            st.session_state.prediction_inputs = customer_data
            
        except Exception as e:
            st.error(f"{get_text('prediction.error')} {str(e)}")
            with st.expander(get_text("prediction.error_details")):
                st.exception(e)

    if "prediction_result" in st.session_state:
        result = st.session_state.prediction_result
        st.markdown("---")
        st.subheader(get_text('prediction.results'))
        
        churn_prob = result["churn_probability"] * 100
        
        # Determine risk level
        if churn_prob >= 70:
            risk_level = get_text("prediction.high_risk")
            risk_class = "high-risk"
        elif churn_prob >= 40:
            risk_level = get_text("prediction.medium_risk")
            risk_class = "medium-risk"
        else:
            risk_level = get_text("prediction.low_risk")
            risk_class = "low-risk"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(get_text("prediction.churn_risk"), f"{churn_prob:.1f}%", delta=None)
        with col2:
            risk_color = "#f44336" if risk_class == "high-risk" else "#fbc02d" if risk_class == "medium-risk" else "#4caf50"
            st.markdown(
                f"**{get_text('prediction.risk_level')}**<br>"
                f"<span style='color: {risk_color}; font-weight: 700;'>{risk_level}</span>",
                unsafe_allow_html=True
            )
        with col3:
            st.metric(
                get_text("prediction.prediction"),
                get_text("prediction.churn") if result["churn"] else get_text("prediction.no_churn"),
                delta=None
            )
        
        # Probability visualization - clean donut
        fig = go.Figure(data=[go.Pie(
            labels=[get_text("prediction.churn_risk_label"), get_text("prediction.no_churn_risk")],
            values=[result["churn_probability"] * 100, result["no_churn_probability"] * 100],
            hole=0.6,
            marker=dict(colors=["#f44336", "#4caf50"]),
            textinfo="percent",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>"
        )])
        fig.update_layout(
            title=get_text("prediction.churn_prob_dist"),
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            margin=dict(t=60, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, width='stretch')
            
        # Recommendations
        st.markdown("---")
        st.subheader(get_text("prediction.recommendations"))
        
        if churn_prob >= 70:
            st.warning(get_text("prediction.high_risk_warning"))
            st.markdown(f"""
            **{get_text("prediction.high_risk_actions")}**
            - {get_text("prediction.action1")}
            - {get_text("prediction.action2")}
            - {get_text("prediction.action3")}
            - {get_text("prediction.action4")}
            """)
        elif churn_prob >= 40:
            st.warning(get_text("prediction.medium_risk_warning"))
            st.markdown(f"""
            **{get_text("prediction.medium_risk_actions")}**
            - {get_text("prediction.action8")}
            - {get_text("prediction.action9")}
            - {get_text("prediction.action2")}
            """)
        else:
            st.success(get_text("prediction.low_risk_success"))
            st.markdown(f"""
            **{get_text("prediction.low_risk_actions")}**
            - {get_text("prediction.action5")}
            - {get_text("prediction.action6")}
            - {get_text("prediction.action7")}
            """)

elif page == "analysis":
    st.title(get_text('analysis.title'))
    
    # Load data using session state caching
    try:
        df = load_raw_data()
        if df is None:
            st.error("Veri yüklenemedi.")
            st.stop()
    except Exception as e:
        st.error(f"Veri yükleme hatası: {str(e)}")
        st.stop()
    
    # Overview - using full dataset
    col1, col2 = st.columns(2)
    
    with col1:
        churn_counts = df["Churn"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=churn_counts.index,
            values=churn_counts.values,
            hole=0.4,
            marker=dict(colors=["#f44336" if label == "Yes" else "#4caf50" for label in churn_counts.index]),
            textinfo="percent+label"
        )])
        fig.update_layout(
            title=get_text("analysis.churn_dist"),
            height=400
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        contract_counts = df["Contract"].value_counts()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=contract_counts.index,
            y=contract_counts.values,
            text=contract_counts.values,
            texttemplate="%{text}",
            textposition="outside",
            marker=dict(color=contract_counts.values, colorscale="Blues", showscale=False)
        ))
        fig.update_layout(
            title=get_text("analysis.contract_dist"),
            xaxis_title=get_text("analysis.contract_type"),
            yaxis_title=get_text("analysis.customer_count"),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, width='stretch')
    
    # Detailed analysis
    st.markdown("---")
    st.subheader(get_text('analysis.detailed_analysis'))
    
    # Reorder tabs: Financial Analysis first, then Category, then Feature
    tab1, tab2, tab3 = st.tabs([
        get_text("analysis.tab_financial"),
        get_text("analysis.tab_category"),
        get_text("analysis.tab_feature")
    ])
    
    with tab1:
        # Financial Analysis with elegant filter
        st.markdown(f"**{get_text('analysis.filters')}**")
        
        # Filter with radio buttons
        filter_option = st.radio(
            get_text("analysis.churn_status"),
            [get_text("analysis.all"), get_text("analysis.churned"), get_text("analysis.not_churned")],
            horizontal=True,
            key="fin_filter",
            label_visibility="collapsed"
        )
        
        # Apply filter based on selection
        if filter_option == get_text("analysis.all"):
            df_fin = df.copy()
        elif filter_option == get_text("analysis.churned"):
            df_fin = df[df["Churn"] == "Yes"].copy()
        else:
            df_fin = df[df["Churn"] == "No"].copy()
        
        st.markdown("---")
        
        # Scatter plot - optimized
        fig = go.Figure()
        
        # Use scattergl for better performance
        # Draw No Churn first (background), then Churn (foreground) so red dots are visible
        for churn_value, color, label in [
            ("No", "#4caf50", get_text("analysis.not_churned")),  # Green first
            ("Yes", "#f44336", get_text("analysis.churned"))      # Red last (more visible)
        ]:
            subset = df_fin[df_fin["Churn"] == churn_value]
            fig.add_trace(go.Scattergl(  # Using Scattergl for better performance
                x=subset["tenure"],
                y=subset["MonthlyCharges"],
                mode="markers",
                name=label,
                marker=dict(
                    size=6,
                    color=color,
                    opacity=0.7
                ),
                text=subset["Contract"],
                hovertemplate=(
                    f"{get_text('home.tenure_months')}: %{{x}}<br>"
                    f"{get_text('analysis.monthly_charges')}: %{{y}}<br>"
                    f"Contract: %{{text}}<extra></extra>"
                )
            ))
        fig.update_layout(
            title=get_text("analysis.tenure_vs_charges"),
            xaxis_title=get_text("home.tenure_months"),
            yaxis_title=get_text("analysis.monthly_charges"),
            height=500,
            hovermode='closest'
        )
        st.plotly_chart(fig, width='stretch')
        
        # Box plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            for churn_value, color, label in [
                ("Yes", "#f44336", get_text("analysis.churned")),
                ("No", "#4caf50", get_text("analysis.not_churned"))
            ]:
                subset = df_fin[df_fin["Churn"] == churn_value]
                fig.add_trace(go.Box(
                    y=subset["MonthlyCharges"],
                    name=label,
                    marker_color=color
                ))
            fig.update_layout(
                title=get_text("analysis.monthly_charges_dist"),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = go.Figure()
            for churn_value, color, label in [
                ("Yes", "#f44336", get_text("analysis.churned")),
                ("No", "#4caf50", get_text("analysis.not_churned"))
            ]:
                subset = df_fin[df_fin["Churn"] == churn_value]
                fig.add_trace(go.Box(
                    y=subset["tenure"],
                    name=label,
                    marker_color=color
                ))
            fig.update_layout(
                title=get_text("analysis.tenure_dist_title"),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, width='stretch')
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Use original df for churn rate calculation to avoid 100% when filtered
            churn_by_contract = df.groupby("Contract", as_index=False).agg(
                **{"Churn Rate (%)": ("Churn", lambda x: (x == "Yes").mean() * 100)}
            )
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=churn_by_contract["Contract"],
                y=churn_by_contract["Churn Rate (%)"],
                text=churn_by_contract["Churn Rate (%)"],
                texttemplate="%{text:.1f}%",
                textposition="outside",
                marker=dict(color=churn_by_contract["Churn Rate (%)"], colorscale="Reds", showscale=False)
            ))
            fig.update_layout(
                title=get_text("analysis.churn_by_contract"),
                xaxis_title=get_text("home.contract_type"),
                yaxis_title=get_text("home.churn_rate_label"),
                showlegend=False,
                height=400,
                yaxis=dict(range=[0, None]),
                margin=dict(b=50, t=50, l=50, r=50, pad=10)
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Use original df for churn rate calculation to avoid 100% when filtered
            churn_by_internet = df.groupby("InternetService", as_index=False).agg(
                **{"Churn Rate (%)": ("Churn", lambda x: (x == "Yes").mean() * 100)}
            )
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=churn_by_internet["InternetService"],
                y=churn_by_internet["Churn Rate (%)"],
                text=churn_by_internet["Churn Rate (%)"],
                texttemplate="%{text:.1f}%",
                textposition="outside",
                marker=dict(color=churn_by_internet["Churn Rate (%)"], colorscale="Reds", showscale=False)
            ))
            fig.update_layout(
                title=get_text("analysis.churn_by_internet"),
                xaxis_title=get_text("home.internet_service"),
                yaxis_title=get_text("home.churn_rate_label"),
                showlegend=False,
                height=400,
                yaxis=dict(range=[0, None]),
                margin=dict(b=50, t=50, l=50, r=50, pad=10)
            )
            st.plotly_chart(fig, width='stretch')
        
        # Payment method analysis - use original df for churn rate calculation
        churn_by_payment = df.groupby("PaymentMethod", as_index=False).agg(
            **{"Churn Rate (%)": ("Churn", lambda x: (x == "Yes").mean() * 100)}
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=churn_by_payment["PaymentMethod"],
            y=churn_by_payment["Churn Rate (%)"],
            text=churn_by_payment["Churn Rate (%)"],
            texttemplate="%{text:.1f}%",
            textposition="outside",
            marker=dict(color=churn_by_payment["Churn Rate (%)"], colorscale="Reds", showscale=False)
        ))
        fig.update_layout(
            title=get_text("analysis.churn_by_payment"),
            xaxis_title=get_text("prediction.payment_method"),
            yaxis_title=get_text("home.churn_rate_label"),
            showlegend=False,
            height=400,
            yaxis=dict(range=[0, None]),
            margin=dict(b=50, t=50, l=50, r=50, pad=10)
        )
        st.plotly_chart(fig, width='stretch')
    
    with tab3:
        st.subheader(get_text("analysis.feature_importance"))
        
        feature_imp_df = get_feature_importance()
        
        if feature_imp_df is not None:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=feature_imp_df["Importance"],
                y=feature_imp_df["Feature"],
                orientation="h",
                marker=dict(color=feature_imp_df["Importance"], colorscale="Viridis", showscale=False)
            ))
            fig.update_layout(
                title=get_text("analysis.top_features"),
                xaxis_title=get_text("analysis.importance"),
                yaxis_title=get_text("analysis.feature"),
                showlegend=False,
                height=600,
                yaxis=dict(categoryorder="total ascending")
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info(get_text("analysis.feature_info"))

elif page == "performance":
    st.title(get_text('performance.title'))
    
    st.markdown(f"""
    ### {get_text("performance.model_info")}
    
    {get_text("performance.model_description")}
    """)
    
    st.markdown("---")
    
    # Performance metrics
    st.subheader(get_text('performance.metrics'))
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics_data = {
        "Accuracy": 0.7466,
        "Precision": 0.5145,
        "Recall": 0.8048,
        "F1-Score": 0.6277,
        "ROC-AUC": 0.8470
    }
    
    cols = [col1, col2, col3, col4, col5]
    metric_names = ["Accuracy", get_text("performance.precision_churn"), 
                   get_text("performance.recall_churn"), get_text("performance.f1_churn"), "ROC-AUC"]
    
    for col, (key, value), name in zip(cols, metrics_data.items(), metric_names):
        with col:
            st.metric(name, f"{value:.4f}")
    
    st.markdown("---")
    
    # Model interpretation
    st.subheader(get_text('performance.interpretation'))
    
    st.info(f"""
    {get_text("performance.interpretation_text")}
    
    **{get_text("performance.important_notes")}**
    - {get_text("performance.note1")}
    - {get_text("performance.note2")}
    - {get_text("performance.note3")}
    """)
    
    # Feature importance
    st.markdown("---")
    st.subheader(get_text('performance.important_features'))
    
    important_features = [
        (get_text("performance.feature1"), get_text("performance.feature1_desc")),
        (get_text("performance.feature2"), get_text("performance.feature2_desc")),
        (get_text("performance.feature3"), get_text("performance.feature3_desc")),
        (get_text("performance.feature4"), get_text("performance.feature4_desc")),
        (get_text("performance.feature5"), get_text("performance.feature5_desc"))
    ]
    
    for feature, explanation in important_features:
        st.markdown(f"**{feature}**: {explanation}")

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: gray; padding: 20px;'>{get_text('footer')}</div>",
    unsafe_allow_html=True
)