
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from analyst import (
    profile_data,
    answer_question,
    build_forecast,
    detect_anomalies,
    make_pdf_report,
    suggest_questions
)

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Data Analyst Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS - PREMIUM DARK UI
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0B1120;
    color: #F8FAFC;
}

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263247;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F8FAFC;
}

.main-title {
    font-size: 32px;
    font-weight: 800;
    color: #60A5FA;
    letter-spacing: -1px;
}

.subtitle {
    color: #94A3B8;
    font-size: 14px;
    margin-bottom: 20px;
}

.section-title {
    color: #E2E8F0;
    font-size: 22px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 15px;
}

.metric-card {
    background: linear-gradient(135deg, #1E293B, #172033);
    border: 1px solid #334155;
    padding: 20px;
    border-radius: 15px;
    min-height: 115px;
}

.metric-label {
    color: #94A3B8;
    font-size: 13px;
    font-weight: 500;
}

.metric-value {
    color: #60A5FA;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-icon {
    font-size: 20px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1E293B, #172033);
    padding: 15px;
    border-radius: 13px;
    border: 1px solid #334155;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

div[data-testid="stMetricValue"] {
    color: #60A5FA;
}

.stButton button {
    background: #2563EB;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 600;
    min-height: 42px;
}

.stButton button:hover {
    background: #1D4ED8;
    color: white;
    border: none;
}

.stDownloadButton button {
    background: #2563EB;
    color: white;
    border-radius: 9px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 10px;
}

input, textarea {
    border-radius: 8px !important;
}

hr {
    border-color: #263247;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE
# =====================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "filename" not in st.session_state:
    st.session_state.filename = "No dataset loaded"


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🧠 AI Data Analyst Copilot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Analytics Platform | Data Insights | Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## 🧠 AI COPILOT")
    st.caption("Your Intelligent Data Assistant")

    st.divider()

    st.markdown("### 📂 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.lower().endswith(".csv"):
                new_df = pd.read_csv(uploaded_file)
            else:
                new_df = pd.read_excel(uploaded_file)

            st.session_state.df = new_df
            st.session_state.filename = uploaded_file.name

            st.success("Dataset loaded successfully!")

        except Exception as e:
            st.error(f"Upload error: {e}")

    if st.button("📊 Load Demo Dataset", use_container_width=True):

        dates = pd.date_range(
            start="2025-01-01",
            periods=12,
            freq="MS"
        )

        demo_df = pd.DataFrame({
            "Date": dates,
            "Sales": [
                12000, 14500, 13800, 16000,
                17500, 18200, 19500, 21000,
                20500, 23000, 24500, 27000
            ],
            "Profit": [
                2000, 2500, 2300, 3000,
                3500, 3700, 3900, 4200,
                4100, 4600, 4900, 5500
            ],
            "Orders": [
                100, 115, 110, 125,
                140, 145, 155, 165,
                160, 180, 190, 210
            ]
        })

        st.session_state.df = demo_df
        st.session_state.filename = "Demo Sales Dataset"

        st.success("Demo dataset loaded!")

    st.divider()

    st.markdown("### 🧭 Navigation")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Explore Data",
            "Ask AI",
            "Forecast",
            "Anomaly Detection",
            "PDF Report"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 📌 Project Information")

    st.caption("AI Data Analyst Copilot")
    st.caption("Built with Python and Streamlit")
    st.caption("Machine Learning Analytics")

    st.success("System Ready")


# =====================================================
# DATASET CHECK
# =====================================================

df = st.session_state.df

if df is None:

    st.markdown("## Welcome to AI Data Analyst Copilot")

    st.write(
        "Upload your CSV or Excel dataset to begin intelligent data analysis."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Data Analytics")
        st.caption("Analyze your dataset and discover useful patterns.")

    with col2:
        st.markdown("### 🤖 AI Assistant")
        st.caption("Ask questions and receive data-based answers.")

    with col3:
        st.markdown("### 🔮 Forecasting")
        st.caption("Explore future trends using machine learning.")

    st.info("Please upload a dataset using the left sidebar.")

    st.stop()


# =====================================================
# DATA PREPARATION
# =====================================================

df = df.copy()

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

date_columns = []

for column in df.columns:

    if pd.api.types.is_datetime64_any_dtype(df[column]):
        date_columns.append(column)

    elif any(
        word in str(column).lower()
        for word in ["date", "time"]
    ):

        converted = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        if converted.notna().sum() > 0:
            date_columns.append(column)


missing_values = int(df.isnull().sum().sum())
duplicate_rows = int(df.duplicated().sum())

total_rows = len(df)
total_columns = len(df.columns)

data_quality = (
    max(0, 100 - (missing_values / max(1, total_rows * total_columns) * 100))
)

# =====================================================
# KPI CARDS
# =====================================================

def metric_card(label, value, icon):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_kpis():

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Records", f"{total_rows:,}", "📁")

    with c2:
        metric_card("Total Columns", f"{total_columns:,}", "📊")

    with c3:
        metric_card("Missing Values", f"{missing_values:,}", "⚠️")

    with c4:
        metric_card("Data Quality", f"{data_quality:.1f}%", "✨")


# =====================================================
# OVERVIEW PAGE
# =====================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">📌 Analytics Overview</div>',
        unsafe_allow_html=True
    )

    st.caption(f"Current Dataset: {st.session_state.filename}")

    show_kpis()

    st.markdown("## 📈 Dataset Visualization")

    if numeric_columns:

        selected_metric = st.selectbox(
            "Select metric",
            numeric_columns
        )

        fig = px.histogram(
            df,
            x=selected_metric,
            title=f"Distribution of {selected_metric}",
            template="plotly_dark",
            color_discrete_sequence=["#3B82F6"]
        )

        fig.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#E2E8F0"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info("No numeric columns available for visualization.")

    st.markdown("## 📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("## 📊 Statistical Summary")

    if numeric_columns:

        st.dataframe(
            df[numeric_columns].describe(),
            use_container_width=True
        )


# =====================================================
# EXPLORE DATA PAGE
# =====================================================

elif page == "Explore Data":

    st.markdown("## 📊 Explore Your Dataset")

    st.write(
        "Discover distributions, relationships and patterns in your data."
    )

    if numeric_columns:

        selected_column = st.selectbox(
            "Choose Numeric Column",
            numeric_columns
        )

        chart_type = st.selectbox(
            "Select Chart Type",
            [
                "Histogram",
                "Box Plot",
                "Line Chart"
            ]
        )

        if chart_type == "Histogram":

            fig = px.histogram(
                df,
                x=selected_column,
                title="Data Distribution",
                template="plotly_dark",
                color_discrete_sequence=["#3B82F6"]
            )

        elif chart_type == "Box Plot":

            fig = px.box(
                df,
                y=selected_column,
                title="Outlier Distribution",
                template="plotly_dark",
                color_discrete_sequence=["#3B82F6"]
            )

        else:

            fig = px.line(
                df,
                y=selected_column,
                title="Trend Analysis",
                template="plotly_dark"
            )

        fig.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#E2E8F0"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        if len(numeric_columns) >= 2:

            st.markdown("## 🔗 Correlation Heatmap")

            corr = df[numeric_columns].corr()

            heatmap = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu",
                title="Feature Correlation",
                template="plotly_dark"
            )

            st.plotly_chart(
                heatmap,
                use_container_width=True
            )

    else:

        st.warning("No numeric columns found.")

    st.markdown("## Complete Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.download_button(
        "📥 Download Dataset",
        data=df.to_csv(index=False),
        file_name="analyzed_dataset.csv",
        mime="text/csv"
    )


# =====================================================
# ASK AI PAGE
# =====================================================

elif page == "Ask AI":

    st.markdown("## 🤖 AI Data Assistant")

    st.write(
        "Ask questions about your uploaded dataset."
    )

    st.info(
        "This version uses the question-answering logic "
        "implemented in analyst.py."
    )

    try:

        questions = suggest_questions(df)

        if questions:

            st.markdown("### 💡 Suggested Questions")

            for question in questions:

                st.markdown(f"- {question}")

    except Exception:
        pass

    user_question = st.text_input(
        "Enter your question",
        placeholder="Example: Show me the summary of the dataset"
    )

    if st.button("🚀 Generate Answer"):

        if user_question.strip():

            try:

                with st.spinner("Analyzing your dataset..."):

                    answer = answer_question(
                        df,
                        user_question
                    )

                st.markdown("### 💡 AI Answer")

                st.success(str(answer))

            except Exception as e:

                st.error(f"Analysis error: {e}")

        else:

            st.warning("Please enter a question.")


# =====================================================
# FORECAST PAGE
# =====================================================

elif page == "Forecast":

    st.markdown("## 🔮 Sales / Trend Forecasting")

    st.write(
        "Generate future trend predictions using your dataset."
    )

    if not date_columns or not numeric_columns:

        st.warning(
            "A date column and numeric column are required."
        )

        st.write("Available date columns:", date_columns)
        st.write("Available numeric columns:", numeric_columns)

    else:

        date_col = st.selectbox(
            "Select Date Column",
            date_columns
        )

        value_col = st.selectbox(
            "Select Numeric Column",
            numeric_columns
        )

        periods = st.slider(
            "Forecast Periods",
            1,
            24,
            6
        )

        if st.button("Generate Forecast"):

            try:

                with st.spinner("Building forecast model..."):

                    forecast = build_forecast(
                        df,
                        date_col,
                        value_col,
                        periods
                    )

                st.success("Forecast generated successfully!")

                if isinstance(forecast, pd.DataFrame):

                    st.dataframe(
                        forecast,
                        use_container_width=True
                    )

                    if len(forecast.columns) >= 2:

                        fig = px.line(
                            forecast,
                            x=forecast.columns[0],
                            y=forecast.columns[-1],
                            markers=True,
                            title="Future Trend Prediction",
                            template="plotly_dark",
                            color_discrete_sequence=["#60A5FA"]
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                else:

                    st.write(forecast)

            except Exception as e:

                st.error(f"Forecast error: {e}")


# =====================================================
# ANOMALY DETECTION PAGE
# =====================================================

elif page == "Anomaly Detection":

    st.markdown("## 🚨 Anomaly Detection")

    st.write(
        "Find unusual patterns and outliers in your numeric data."
    )

    if not numeric_columns:

        st.warning("Numeric columns are required.")

    else:

        selected_columns = st.multiselect(
            "Select Numeric Features",
            numeric_columns,
            default=numeric_columns
        )

        contamination = st.slider(
            "Expected Anomaly Percentage",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01
        )

        if st.button("🔍 Detect Anomalies"):

            if not selected_columns:

                st.warning("Please select at least one column.")

            else:

                try:

                    with st.spinner("Detecting unusual data points..."):

                        anomalies = detect_anomalies(
                            df,
                            selected_columns,
                            contamination
                        )

                    st.success("Anomaly detection completed!")

                    if isinstance(anomalies, pd.DataFrame):

                        st.dataframe(
                            anomalies,
                            use_container_width=True
                        )

                        st.download_button(
                            "📥 Download Results",
                            data=anomalies.to_csv(index=False),
                            file_name="anomaly_results.csv",
                            mime="text/csv"
                        )

                    else:

                        st.write(anomalies)

                except Exception as e:

                    st.error(f"Anomaly detection error: {e}")


# =====================================================
# PDF REPORT PAGE
# =====================================================

elif page == "PDF Report":

    st.markdown("## 📄 PDF Analytics Report")

    st.write(
        "Generate a professional PDF report for your dataset."
    )

    st.markdown("""
    ### Your report can contain:

    - Dataset overview
    - Statistical summary
    - Column information
    - Data analysis
    """)

    if st.button("📄 Generate PDF Report"):

        try:

            with st.spinner("Generating PDF report..."):

                pdf_file = make_pdf_report(df)

            st.success("PDF report generated successfully!")

            st.download_button(
                "📥 Download PDF",
                data=pdf_file,
                file_name="AI_Data_Analyst_Report.pdf",
                mime="application/pdf"
            )

        except Exception as e:

            st.error(f"PDF report error: {e}")


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94A3B8;
        padding:15px;
        font-size:13px;
    ">
        <b style="color:#60A5FA;">
            AI DATA ANALYST COPILOT
        </b>
        <br>
        Intelligent Analytics Platform
        <br><br>
        Built with Python | Streamlit | Pandas |
        Plotly | Machine Learning
        <br><br>
        © 2026 AI Analytics Project
    </div>
    """,
    unsafe_allow_html=True
)
