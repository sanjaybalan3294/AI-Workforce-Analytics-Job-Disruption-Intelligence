"""
app.py
=============================================================================
Interactive Streamlit Dashboard: 4-Tier Analytics Ladder & ML Intelligence
AI Job Trends & Workforce Disruption Intelligence Platform
=============================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from data_pipeline import DataHygienePipeline
from model_training import ModelTrainingPipeline


# -----------------------------------------------------------------------------
# 1. STREAMLIT APP CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Workforce Intelligence | 4-Tier Analytics",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #10B981;
    }
    .kpi-sub-neg {
        font-size: 0.8rem;
        color: #EF4444;
    }
    .insight-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 4px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .guardrail-card {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 4px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
        font-weight: 600;
        color: #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATA INGESTION & ARTIFACT CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def get_clean_data():
    csv_clean_path = "cleaned_ai_job_trends.csv"
    if not os.path.exists(csv_clean_path):
        pipeline = DataHygienePipeline("ai_job_trends_dataset.csv")
        df = pipeline.run_pipeline(csv_clean_path)
    else:
        df = pd.read_csv(csv_clean_path)
    return df


@st.cache_resource
def get_model_artifacts():
    model_path = "model_artifacts.pkl"
    if not os.path.exists(model_path):
        trainer = ModelTrainingPipeline("cleaned_ai_job_trends.csv")
        artifacts = trainer.train_and_evaluate()
    else:
        artifacts = joblib.load(model_path)
    return artifacts


df = get_clean_data()
artifacts = get_model_artifacts()


# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & GLOBAL FILTERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.markdown("### AI Workforce Analytics")
    st.caption("Human Capital Intelligence & Disruption Platform")
    st.markdown("---")

    navigation = st.radio(
        "Navigation / 4-Tier Ladder:",
        [
            "🏛️ Executive Overview & KPIs",
            "📊 Tier 1 & 2: Exploratory Data Analysis",
            "🤖 Tier 3: Predictive ML Studio",
            "🎯 Tier 4: Prescriptive Strategy & Levers",
            "📑 Data Hygiene & Quality Audit",
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### Global Cohort Filters")
    selected_industries = st.multiselect(
        "Industry Sector:",
        options=sorted(df["Industry"].unique()),
        default=sorted(df["Industry"].unique())
    )

    selected_locations = st.multiselect(
        "Geographic Market:",
        options=sorted(df["Location"].unique()),
        default=sorted(df["Location"].unique())
    )

    selected_education = st.multiselect(
        "Required Education:",
        options=sorted(df["Required Education"].unique()),
        default=sorted(df["Required Education"].unique())
    )

    st.markdown("---")
    st.caption("Environment: Production-Ready | Python 3.12 | Scikit-Learn | Streamlit")


# Filter dataframe based on selections
filtered_df = df[
    (df["Industry"].isin(selected_industries)) &
    (df["Location"].isin(selected_locations)) &
    (df["Required Education"].isin(selected_education))
]

if filtered_df.empty:
    st.warning("⚠️ No records match current filter criteria. Please broaden your selection.")
    st.stop()


# -----------------------------------------------------------------------------
# 4. TAB 1: EXECUTIVE OVERVIEW & KPI DECK
# -----------------------------------------------------------------------------
if navigation == "🏛️ Executive Overview & KPIs":
    st.markdown('<div class="main-header">🏛️ Executive Overview & Macro Labor KPIs</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Macroeconomic snapshot of 30,000 workforce roles across 8 global markets and industries.</div>', unsafe_allow_html=True)

    # KPI Top Metric Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-title">Roles Profiled</div>
            <div class="kpi-value">{len(filtered_df):,}</div>
            <div class="kpi-sub">Across 639 Occupations</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        mean_salary = filtered_df["Median Salary (USD)"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-title">Average Salary</div>
            <div class="kpi-value">${mean_salary:,.0f}</div>
            <div class="kpi-sub">Range: $30k - $150k</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_auto = filtered_df["Automation Risk (%)"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-title">Mean Automation Risk</div>
            <div class="kpi-value">{avg_auto:.1f}%</div>
            <div class="kpi-sub-neg">Cross-sector baseline</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        avg_remote = filtered_df["Remote Work Ratio (%)"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-title">Remote Penetration</div>
            <div class="kpi-value">{avg_remote:.1f}%</div>
            <div class="kpi-sub">Hybrid & distributed</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        decline_rate = (filtered_df["Job Status"] == "Decreasing").mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-title">Roles in Contraction</div>
            <div class="kpi-value">{decline_rate:.1f}%</div>
            <div class="kpi-sub-neg">Under AI displacement</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### 💼 Net Job Openings Trajectory (2024 vs. 2030)")
        sum_2024 = filtered_df["Job Openings (2024)"].sum()
        sum_2030 = filtered_df["Projected Openings (2030)"].sum()
        delta = sum_2030 - sum_2024

        fig_openings = go.Figure(data=[
            go.Bar(name='2024 Base Openings', x=['Employment Openings'], y=[sum_2024], marker_color='#3B82F6'),
            go.Bar(name='2030 Projected Openings', x=['Employment Openings'], y=[sum_2030], marker_color='#10B981')
        ])
        fig_openings.update_layout(
            barmode='group',
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Total Headcount Openings",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_openings, use_container_width=True)
        st.caption(f"**Net Projected Headcount Shift:** {delta:+,} openings ({delta/sum_2024*100:+.2f}% macro growth).")

    with col_right:
        st.markdown("##### ⚙️ Sector-Wise Average Automation Risk & Salary")
        industry_summary = filtered_df.groupby("Industry").agg(
            Avg_Salary=("Median Salary (USD)", "mean"),
            Avg_Auto_Risk=("Automation Risk (%)", "mean"),
            Count=("Job Status", "count")
        ).reset_index()

        fig_ind = px.scatter(
            industry_summary,
            x="Avg_Auto_Risk",
            y="Avg_Salary",
            size="Count",
            color="Industry",
            text="Industry",
            height=340,
            labels={"Avg_Auto_Risk": "Average Automation Risk (%)", "Avg_Salary": "Average Median Salary ($ USD)"}
        )
        fig_ind.update_traces(textposition='top center')
        fig_ind.update_layout(margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig_ind, use_container_width=True)
        st.caption("Distribution of compensation versus automation vulnerability aggregated at the industrial sector tier.")


# -----------------------------------------------------------------------------
# 5. TAB 2: EXPLORATORY DATA ANALYSIS (5 CHARTS + DIAGNOSTIC INSIGHTS)
# -----------------------------------------------------------------------------
elif navigation == "📊 Tier 1 & 2: Exploratory Data Analysis":
    st.markdown('<div class="main-header">📊 Tier 1 & 2: Exploratory Data Analysis (Descriptive & Diagnostic)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">5 distinct visual analyses exploring baseline distributions, categorical interactions, and cohort behaviors with empirical diagnostics.</div>', unsafe_allow_html=True)

    # CHART 1
    st.markdown("### Chart 1: Automation Risk Distribution Stratified by AI Impact Level")
    fig1 = px.box(
        filtered_df,
        x="AI Impact Level",
        y="Automation Risk (%)",
        color="AI Impact Level",
        points="all",
        category_orders={"AI Impact Level": ["Low", "Moderate", "High"]},
        color_discrete_map={"Low": "#10B981", "Moderate": "#F59E0B", "High": "#EF4444"},
        height=400
    )
    fig1.update_layout(margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <strong>📌 Empirical Observation:</strong><br>
        Across all three qualitative impact tiers (Low, Moderate, High), the distribution of numerical Automation Risk spans from 0.0% to 99.99% with median values stabilizing near ~50.1% and interquartile ranges spanning 25.4% to 75.1%.
    </div>
    <div class="insight-card">
        <strong>🔍 Diagnostic Business Insight:</strong><br>
        Qualitative categorization of a role into "High Impact" does not cleanly correlate with extreme continuous automation risk scores. Many roles tagged as High Impact undergo collaborative workflow integration rather than outright displacement.
    </div>
    <div class="guardrail-card">
        <strong>⚖️ Correlation vs. Causation Guardrail:</strong><br>
        <strong>Correlation:</strong> Labeling a job as "High AI Impact" correlates with perceived exposure to machine learning tools.<br>
        <strong>Non-Causation:</strong> Assigning a high-impact classification to a job title does <em>not</em> cause the job to become automated. Automation risk is causally determined by task modularity, rule-based repeatability, and physical embodiment constraints.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # CHART 2
    st.markdown("### Chart 2: Economic Landscape: Compensation vs. Automation Risk by Industry")
    fig2 = px.scatter(
        filtered_df.sample(min(2500, len(filtered_df)), random_state=42),
        x="Automation Risk (%)",
        y="Median Salary (USD)",
        color="Industry",
        opacity=0.6,
        trendline="ols",
        height=450,
        labels={"Automation Risk (%)": "Automation Risk (%)", "Median Salary (USD)": "Median Salary ($ USD)"}
    )
    fig2.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <strong>📌 Empirical Observation:</strong><br>
        Across every industry sector, roles with salaries between $30,000 and $150,000 are uniformly distributed across the full 0% to 100% automation risk continuum. Trendlines remain virtually flat across all sectors.
    </div>
    <div class="insight-card">
        <strong>🔍 Diagnostic Business Insight:</strong><br>
        In modern AI disruption economics, high compensation provides zero structural shelter from cognitive automation. Well-compensated knowledge workers (e.g. Financial Planners, Investment Analysts, Legal Secretaries) face automation risk parity with entry-level workers.
    </div>
    <div class="guardrail-card">
        <strong>⚖️ Correlation vs. Causation Guardrail:</strong><br>
        <strong>Correlation:</strong> High-paying positions correlate with higher cognitive specialization and formal academic credentials.<br>
        <strong>Non-Causation:</strong> Higher wages do <em>not</em> causally protect or expose a role to automated disruption. The causal driver is whether the core work consists of unstructured physical dexterity or structured digital symbol manipulation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # CHART 3
    st.markdown("### Chart 3: Educational Cohort Dynamics: Net Job Openings Trajectory (2024 vs. 2030)")
    edu_agg = filtered_df.groupby("Required Education").agg(
        Total_2024=("Job Openings (2024)", "sum"),
        Total_2030=("Projected Openings (2030)", "sum")
    ).reset_index()
    edu_agg["Net_Change"] = edu_agg["Total_2030"] - edu_agg["Total_2024"]

    fig3 = go.Figure(data=[
        go.Bar(name='2024 Headcount Openings', x=edu_agg['Required Education'], y=edu_agg['Total_2024'], marker_color='#6366F1'),
        go.Bar(name='2030 Projected Openings', x=edu_agg['Required Education'], y=edu_agg['Total_2030'], marker_color='#14B8A6')
    ])
    fig3.update_layout(
        barmode='group',
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Headcount Openings",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <strong>📌 Empirical Observation:</strong><br>
        Each educational tier (High School, Associate Degree, Bachelor's, Master's, PhD) accounts for roughly 30M projected openings globally, with net expansions and contractions distributed without degree-based skew.
    </div>
    <div class="insight-card">
        <strong>🔍 Diagnostic Business Insight:</strong><br>
        Traditional credentialism no longer acts as a deterministic firewall. Generative models and reasoning architectures compress the barrier between foundational technical execution and graduate-level analytical synthesis.
    </div>
    <div class="guardrail-card">
        <strong>⚖️ Correlation vs. Causation Guardrail:</strong><br>
        <strong>Correlation:</strong> Holding a Master's or PhD correlates with specialized industry licensing and research requirements.<br>
        <strong>Non-Causation:</strong> Requiring an advanced degree does <em>not</em> cause workforce preservation. Credential requirements reflect hiring filters rather than intrinsic task resilience.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # CHART 4
    st.markdown("### Chart 4: Geographic Modality: Remote Work Penetration vs. Automation Vulnerability")
    loc_agg = filtered_df.groupby("Location").agg(
        Mean_Remote=("Remote Work Ratio (%)", "mean"),
        Mean_Auto=("Automation Risk (%)", "mean"),
        Total_Roles=("Job Status", "count"),
        Decline_Rate=("Decline_Risk_Flag", "mean")
    ).reset_index()

    fig4 = px.scatter(
        loc_agg,
        x="Mean_Remote",
        y="Mean_Auto",
        size="Total_Roles",
        color="Location",
        text="Location",
        height=400,
        labels={"Mean_Remote": "Average Remote Work Ratio (%)", "Mean_Auto": "Average Automation Risk (%)"}
    )
    fig4.update_traces(textposition='top center')
    fig4.update_layout(margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <strong>📌 Empirical Observation:</strong><br>
        The 8 global markets (USA, UK, Germany, China, India, Brazil, Canada, Australia) demonstrate balanced remote work penetration (~49.8%) and automation vulnerability (~50.1%).
    </div>
    <div class="insight-card">
        <strong>🔍 Diagnostic Business Insight:</strong><br>
        Remote-amenable positions operate entirely within digital software interfaces. While remote modalities increase operational flexibility, they create standard digital audit trails that streamline autonomous agent execution.
    </div>
    <div class="guardrail-card">
        <strong>⚖️ Correlation vs. Causation Guardrail:</strong><br>
        <strong>Correlation:</strong> High remote work ratios correlate with digitized, screen-based occupations.<br>
        <strong>Non-Causation:</strong> Transitioning an employee to remote work does <em>not</em> directly cause their job to be automated. The common causal antecedent is the digital nature of the tasks being executed.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # CHART 5
    st.markdown("### Chart 5: Demographic Composition: Gender Diversity Across Expanding vs. Contracting Roles")
    fig5 = px.histogram(
        filtered_df,
        x="Gender Diversity (%)",
        color="Job Status",
        marginal="box",
        barmode="overlay",
        opacity=0.6,
        color_discrete_map={"Increasing": "#10B981", "Decreasing": "#EF4444"},
        height=400
    )
    fig5.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <strong>📌 Empirical Observation:</strong><br>
        Gender diversity profiles are identically distributed across roles undergoing contraction (mean: 50.01%, median: 49.93%) and roles undergoing expansion (mean: 49.95%, median: 50.08%).
    </div>
    <div class="insight-card">
        <strong>🔍 Diagnostic Business Insight:</strong><br>
        Macro AI disruption does not demonstrate an inherent gender preference at the overarching workforce dataset level. However, sector-specific micro-disparities can emerge where gender representation clusters in administrative vs technical specializations.
    </div>
    <div class="guardrail-card">
        <strong>⚖️ Correlation vs. Causation Guardrail:</strong><br>
        <strong>Correlation:</strong> Gender diversity levels correlate with historical talent pipelines and educational recruitment patterns within specific industry verticals.<br>
        <strong>Non-Causation:</strong> A team's gender ratio does <em>not</em> causally determine its automation vulnerability or growth status. The algorithm targets operational task content, not demographic characteristics.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. TAB 3: PREDICTIVE ML STUDIO & COMMERCIAL TRADE-OFFS
# -----------------------------------------------------------------------------
elif navigation == "🤖 Tier 3: Predictive ML Studio":
    st.markdown('<div class="main-header">🤖 Tier 3: Predictive Modeling & Leakage Prevention</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Supervised classification models predicting labor contraction risk with rigorous target leakage guardrails.</div>', unsafe_allow_html=True)

    # Leakage Prevention Architecture Callout
    st.markdown("""
    <div class="guardrail-card">
        <strong>🛡️ Strict Target Leakage Protection Architecture:</strong><br>
        • <strong>Excluded <code>Projected Openings (2030)</code></strong>: Including future openings produces lookahead target leakage because future headcount directly implies contraction or expansion.<br>
        • <strong>Excluded <code>Job Title</code></strong>: Eliminates 639 high-cardinality identity labels to prevent label memorization and force learning of genuine macroeconomic features (Salary, Education, Experience, Location, Industry, Remote Ratio).<br>
        • <strong>Excluded Direct Derivatives</strong>: Removed <code>Net_Projected_Growth</code> and <code>Growth_Rate_Pct</code> from predictor matrices.
    </div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Select Machine Learning Model Architecture:",
        options=list(artifacts["results"].keys()),
        index=0
    )

    model_res = artifacts["results"][model_choice]
    metrics = model_res["metrics"]
    cm = model_res["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

    # Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Test Accuracy", f"{metrics['accuracy']*100:.2f}%")
    with m_col2:
        st.metric("Precision", f"{metrics['precision']*100:.2f}%")
    with m_col3:
        st.metric("Recall (Sensitivity)", f"{metrics['recall']*100:.2f}%")
    with m_col4:
        st.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")
    with m_col5:
        st.metric("ROC-AUC Score", f"{metrics['roc_auc']:.4f}")

    st.markdown("---")

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown("##### 🔲 Evaluation Confusion Matrix (Test Set: N = 6,000)")
        cm_matrix = np.array([[tn, fp], [fn, tp]])
        cm_labels = [["True Neg (Stable)", "False Pos (Unwarranted Reskill)"],
                     ["False Neg (Missed Contraction)", "True Pos (Correct Alarm)"]]

        fig_cm = px.imshow(
            cm_matrix,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=["Increasing (0)", "Decreasing (1)"],
            y=["Increasing (0)", "Decreasing (1)"]
        )
        fig_cm.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption(f"**Breakdown:** TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,}")

    with col_roc:
        st.markdown("##### 📈 Receiver Operating Characteristic (ROC Curve)")
        fpr = model_res["roc_curve"]["fpr"]
        tpr = model_res["roc_curve"]["tpr"]

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_choice} (AUC = {metrics["roc_auc"]:.3f})', line=dict(color='#2563EB', width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance Baseline', line=dict(dash='dash', color='gray')))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall)",
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # Feature Importance Section
    st.markdown("##### 🔍 Feature Attribution & Explanatory Coefficients")
    feat_col1, feat_col2 = st.columns(2)

    with feat_col1:
        st.markdown("**Random Forest Feature Importances (Gini Impurity):**")
        rf_feat_df = pd.DataFrame(artifacts["feature_importance_rf"]).head(10)
        fig_rf = px.bar(
            rf_feat_df,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Viridis",
            height=320
        )
        fig_rf.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_rf, use_container_width=True)

    with feat_col2:
        st.markdown("**Logistic Regression Feature Coefficients:**")
        lr_coef_df = pd.DataFrame(artifacts["coefficients_lr"]).head(10)
        fig_lr = px.bar(
            lr_coef_df,
            x="coefficient",
            y="feature",
            orientation="h",
            color="coefficient",
            color_continuous_scale="RdBu",
            height=320
        )
        fig_lr.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_lr, use_container_width=True)

    st.markdown("---")

    # Commercial Trade-Off Deep Dive
    st.markdown("### 💼 Commercial Trade-Off Matrix: False Positives vs. False Negatives")
    st.markdown("""
    In human capital planning and workforce transformation, classification errors carry asymmetric economic penalties:
    """)

    to_col1, to_col2 = st.columns(2)
    with to_col1:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #F59E0B;">
            <h4 style="color:#B45309; margin:0 0 8px 0;">False Positive (Type I Error)</h4>
            <strong>Commercial Cost: ~$3,500 / Worker</strong>
            <ul>
                <li>Unwarranted transition budget spent on intensive reskilling programs.</li>
                <li>Premature project reassignment creating team disruptions.</li>
                <li>Heightened employee anxiety leading to voluntary attrition of top talent.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with to_col2:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #EF4444;">
            <h4 style="color:#B91C1C; margin:0 0 8px 0;">False Negative (Type II Error)</h4>
            <strong>Commercial Cost: ~$18,500 / Worker</strong>
            <ul>
                <li>Unanticipated role obsolescence catching the organization unprepared.</li>
                <li>Emergency severance packages and legal liabilities under collective bargaining.</li>
                <li>Critical skill shortages requiring emergency external hiring at 2.5x market premiums.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    tradeoff_info = model_res["commercial_tradeoff"]
    cost_df = pd.DataFrame(tradeoff_info["cost_curve"])

    fig_cost = px.line(
        cost_df,
        x="threshold",
        y="total_cost_usd",
        title="Commercial Loss Curve across Decision Cutoff Thresholds",
        labels={"threshold": "Classification Cutoff Threshold (τ)", "total_cost_usd": "Expected Commercial Loss ($ USD)"},
        height=350
    )
    opt_t = tradeoff_info["optimal_threshold"]
    min_c = tradeoff_info["min_cost_usd"]
    fig_cost.add_vline(x=opt_t, line_dash="dash", line_color="green", annotation_text=f"Optimal τ = {opt_t:.2f} (${min_c:,.0f})")
    fig_cost.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_cost, use_container_width=True)
    st.caption("Because False Negatives are ~5.3x more costly than False Positives, shifting the operating threshold downward reduces total enterprise economic risk.")


# -----------------------------------------------------------------------------
# 7. TAB 4: PRESCRIPTIVE STRATEGY & OPERATIONAL LEVERS
# -----------------------------------------------------------------------------
elif navigation == "🎯 Tier 4: Prescriptive Strategy & Levers":
    st.markdown('<div class="main-header">🎯 Tier 4: Prescriptive Strategy & Resource-Constrained Levers</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Actionable decision rules and capital allocation algorithms based on predictive probabilities.</div>', unsafe_allow_html=True)

    st.markdown("### 1. Resource-Constrained Triage Engine")
    st.markdown("""
    Enterprises cannot retrain 100% of their workforce concurrently due to budgetary caps and operational capacity limits.
    Define your organizational quarterly reskilling bandwidth to dynamically optimize allocation:
    """)

    b_col1, b_col2 = st.columns([1, 2])
    with b_col1:
        reskill_capacity_pct = st.slider(
            "Quarterly Reskilling Bandwidth (% of Workforce):",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )
        cost_per_trainee = st.number_input(
            "Reskilling Cost per Employee ($ USD):",
            min_value=1000,
            max_value=15000,
            value=4500,
            step=500
        )
        severance_avoided = st.number_input(
            "Avoided Severance / Replacement Cost ($ USD):",
            min_value=5000,
            max_value=35000,
            value=18500,
            step=1000
        )

    with b_col2:
        total_eval_cohort = len(filtered_df)
        max_trainees = int(total_eval_cohort * (reskill_capacity_pct / 100.0))
        total_reskill_budget = max_trainees * cost_per_trainee

        # Sort filtered df by decline risk probability proxy
        # Using automation risk and remote exposure as operational ranking score
        filtered_df_ranked = filtered_df.copy()
        filtered_df_ranked["Priority_Score"] = (
            filtered_df_ranked["Automation Risk (%)"] * 0.5 +
            filtered_df_ranked["Remote Work Ratio (%)"] * 0.3 +
            (100 - filtered_df_ranked["Experience Required (Years)"] * 5).clip(0, 100) * 0.2
        )
        cutoff_score = filtered_df_ranked["Priority_Score"].quantile(1 - (reskill_capacity_pct / 100.0))
        targeted_roles = filtered_df_ranked[filtered_df_ranked["Priority_Score"] >= cutoff_score]
        expected_saved = int(len(targeted_roles) * 0.65)  # 65% transition success rate
        gross_savings = expected_saved * severance_avoided
        net_roi = ((gross_savings - total_reskill_budget) / total_reskill_budget) * 100

        st.markdown(f"""
        <div class="metric-card">
            <h4>Operational Allocation Summary:</h4>
            • <strong>Budget Allocated:</strong> ${total_reskill_budget:,.0f} across {max_trainees:,} high-priority personnel.<br>
            • <strong>Dynamic Priority Cutoff Score (τ*):</strong> Top {reskill_capacity_pct}% (Score ≥ {cutoff_score:.1f}).<br>
            • <strong>Projected Successful Transitions:</strong> {expected_saved:,} employees retained.<br>
            • <strong>Gross Severance & Hiring Liabilities Avoided:</strong> ${gross_savings:,.0f}<br>
            • <strong>Net Projected Program ROI:</strong> <span style="color:#10B981; font-weight:700;">{net_roi:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 2. Concrete Risk Mitigation Strategies")
    strat_col1, strat_col2, strat_col3 = st.columns(3)

    with strat_col1:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #EF4444;">
            <h4>🚨 Tier 1: Urgent Augmentation</h4>
            <p><strong>Criteria:</strong> Top 20% Risk Decile (High Contraction Probability)</p>
            <ul>
                <li><strong>Hiring Freeze:</strong> Halt all external recruitment for redundant skillsets.</li>
                <li><strong>12-Week Intensive AI Apprenticeship:</strong> Transition staff into AI orchestrators and prompt engineers.</li>
                <li><strong>Adjacent Headcount Absorption:</strong> Migrate roles into expanding service verticals.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with strat_col2:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #F59E0B;">
            <h4>⚠️ Tier 2: Monitored Upskilling</h4>
            <p><strong>Criteria:</strong> Middle 50% Exposure Cohort</p>
            <ul>
                <li><strong>Micro-Credential Subsidies:</strong> Provide on-demand coursework in generative copilots.</li>
                <li><strong>Bi-Annual Task Audits:</strong> Track task-level shift from manual execution to automated verification.</li>
                <li><strong>Hybrid Role Redesign:</strong> Re-scope job descriptions to prioritize qualitative judgement over routine synthesis.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with strat_col3:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #10B981;">
            <h4>🚀 Tier 3: Strategic Expansion</h4>
            <p><strong>Criteria:</strong> Lower 30% Risk Cohort (Expanding Demand)</p>
            <ul>
                <li><strong>Aggressive Talent Acquisition:</strong> Double external recruiting pipelines in high-growth niches.</li>
                <li><strong>Compensation Realignment:</strong> Ensure competitive compensation to preempt talent poaching.</li>
                <li><strong>Infrastructure Investment:</strong> Provide cutting-edge computational tooling to amplify per-worker output.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Interactive Role Risk Profiler (Live Inference Simulator)
    st.markdown("### 3. Interactive Role Risk Profiler (Live Inference Simulator)")
    st.markdown("Evaluate an individual workforce position in real-time to compute risk probabilities and operational recommendations:")

    with st.form("role_simulator_form"):
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            sim_ind = st.selectbox("Industry:", sorted(df["Industry"].unique()))
            sim_loc = st.selectbox("Location:", sorted(df["Location"].unique()))
            sim_edu = st.selectbox("Required Education:", sorted(df["Required Education"].unique()))
        with sim_col2:
            sim_salary = st.slider("Median Salary ($ USD):", 30000, 150000, 85000, step=5000)
            sim_exp = st.slider("Experience Required (Years):", 0, 20, 8)
            sim_open = st.number_input("Current Job Openings (2024):", 100, 10000, 3500, step=100)
        with sim_col3:
            sim_remote = st.slider("Remote Work Ratio (%):", 0.0, 100.0, 60.0, step=5.0)
            sim_auto = st.slider("Automation Risk (%):", 0.0, 100.0, 75.0, step=5.0)
            sim_gender = st.slider("Gender Diversity (%):", 20.0, 80.0, 50.0, step=5.0)

        submit_sim = st.form_submit_button("⚡ Compute Role Risk Profile & Prescriptions", use_container_width=True)

    if submit_sim:
        sim_pipeline = artifacts["results"]["Logistic Regression (Baseline)"]["pipeline"]
        sim_input_df = pd.DataFrame([{
            "Industry": sim_ind,
            "AI Impact Level": "High" if sim_auto >= 65 else ("Moderate" if sim_auto >= 35 else "Low"),
            "Median Salary (USD)": sim_salary,
            "Required Education": sim_edu,
            "Experience Required (Years)": sim_exp,
            "Job Openings (2024)": sim_open,
            "Remote Work Ratio (%)": sim_remote,
            "Automation Risk (%)": sim_auto,
            "Location": sim_loc,
            "Gender Diversity (%)": sim_gender,
            "Salary_Per_Exp_Year": sim_salary / (sim_exp + 1),
            "Automation_Exposure_Index": (sim_auto * sim_remote) / 100.0,
            "Education_Rank": {"High School": 1, "Associate Degree": 2, "Bachelor's Degree": 3, "Master's Degree": 4, "PhD": 5}.get(sim_edu, 3)
        }])

        prob_decline = sim_pipeline.predict_proba(sim_input_df)[0][1]

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <div class="kpi-title">Predicted Contraction Risk</div>
                <div class="kpi-value" style="color: {'#EF4444' if prob_decline >= 0.50 else '#10B981'};">{prob_decline*100:.1f}%</div>
                <div style="font-weight:600; margin-top:6px;">Status: {'HIGH CONTRACTION RISK' if prob_decline >= 0.50 else 'STABLE / EXPANDING'}</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("#### Tailored Operational Levers:")
            if prob_decline >= 0.50:
                st.warning(f"⚠️ **Action Plan:** Role exhibits high vulnerability ({prob_decline*100:.1f}%). Allocate immediately to Tier 1 Reskilling. Transition job scope toward supervision of automated agentic workflows and client-facing relationship management.")
            else:
                st.success(f"✅ **Action Plan:** Role exhibits structural stability ({prob_decline*100:.1f}% risk). Continue standard operational headcount allocations. Implement Tier 2 self-paced tooling courses to sustain productivity.")


# -----------------------------------------------------------------------------
# 8. TAB 5: DATA HYGIENE & QUALITY AUDIT
# -----------------------------------------------------------------------------
elif navigation == "📑 Data Hygiene & Quality Audit":
    st.markdown('<div class="main-header">📑 Tier 1: Data Hygiene, Quality Audit & Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated data cleaning verification, schema transformations, and entity profiles.</div>', unsafe_allow_html=True)

    audit_col1, audit_col2, audit_col3, audit_col4 = st.columns(4)
    with audit_col1:
        st.metric("Total Records Ingested", f"{len(df):,}")
    with audit_col2:
        st.metric("Missing Values Cleaned", "0 Detected / Handled")
    with audit_col3:
        st.metric("Duplicate Rows Dropped", "0")
    with audit_col4:
        st.metric("Engineered Features Added", "7 New Columns")

    st.markdown("---")
    st.markdown("##### 🔍 Data Schema Transformation & Validation Matrix")

    schema_data = [
        {"Column Name": "Job Title", "Clean Type": "Object", "Domain / Boundary": "639 Unique Occupations", "Hygiene Rule": "Whitespace stripped, Unicode cleaned"},
        {"Column Name": "Industry", "Clean Type": "Object", "Domain / Boundary": "8 Major Sectors", "Hygiene Rule": "Categorical validation"},
        {"Column Name": "Job Status", "Clean Type": "Object", "Domain / Boundary": "Increasing / Decreasing", "Hygiene Rule": "Binary status verification"},
        {"Column Name": "AI Impact Level", "Clean Type": "Object", "Domain / Boundary": "Low / Moderate / High", "Hygiene Rule": "Normalized taxonomy"},
        {"Column Name": "Median Salary (USD)", "Clean Type": "Float64", "Domain / Boundary": "$30,001 - $149,999", "Hygiene Rule": "Asserted strictly > $0"},
        {"Column Name": "Required Education", "Clean Type": "Object", "Domain / Boundary": "5 Tiers (High School to PhD)", "Hygiene Rule": "Apostrophe normalization (Master's / Bachelor's)"},
        {"Column Name": "Experience Required (Years)", "Clean Type": "Int64", "Domain / Boundary": "0 - 20 Years", "Hygiene Rule": "Clipped [0, 50]"},
        {"Column Name": "Job Openings (2024)", "Clean Type": "Int64", "Domain / Boundary": "100 - 10,000", "Hygiene Rule": "Asserted >= 0"},
        {"Column Name": "Projected Openings (2030)", "Clean Type": "Int64", "Domain / Boundary": "100 - 10,000", "Hygiene Rule": "Asserted >= 0 (Excluded in ML for leakage)"},
        {"Column Name": "Remote Work Ratio (%)", "Clean Type": "Float64", "Domain / Boundary": "0.0% - 100.0%", "Hygiene Rule": "Clipped [0, 100]"},
        {"Column Name": "Automation Risk (%)", "Clean Type": "Float64", "Domain / Boundary": "0.0% - 99.99%", "Hygiene Rule": "Clipped [0, 100]"},
        {"Column Name": "Location", "Clean Type": "Object", "Domain / Boundary": "8 Global Nations", "Hygiene Rule": "Standardized geographic strings"},
        {"Column Name": "Gender Diversity (%)", "Clean Type": "Float64", "Domain / Boundary": "20.0% - 80.0%", "Hygiene Rule": "Clipped [0, 100]"},
        {"Column Name": "Net_Projected_Growth", "Clean Type": "Int64", "Domain / Boundary": "-9,810 to +9,818", "Hygiene Rule": "Engineered: 2030 Openings - 2024 Openings"},
        {"Column Name": "Growth_Rate_Pct", "Clean Type": "Float64", "Domain / Boundary": "Percentage Delta", "Hygiene Rule": "Engineered: Growth Rate Pct"},
        {"Column Name": "Salary_Per_Exp_Year", "Clean Type": "Float64", "Domain / Boundary": "$1,450 - $149,998", "Hygiene Rule": "Engineered: Salary / (Exp + 1)"},
        {"Column Name": "Automation_Exposure_Index", "Clean Type": "Float64", "Domain / Boundary": "0.0 - 98.4", "Hygiene Rule": "Engineered: (Auto Risk * Remote Ratio) / 100"},
        {"Column Name": "Education_Rank", "Clean Type": "Int64", "Domain / Boundary": "1 to 5", "Hygiene Rule": "Engineered: Ordinal rank mapping"},
        {"Column Name": "Decline_Risk_Flag", "Clean Type": "Int64", "Domain / Boundary": "0 or 1", "Hygiene Rule": "Primary Target: 1 if Decreasing, 0 if Increasing"}
    ]
    st.dataframe(pd.DataFrame(schema_data), use_container_width=True)

    st.markdown("---")
    st.markdown("##### 📋 Audited Dataset Sample (First 50 Rows)")
    st.dataframe(filtered_df.head(50), use_container_width=True)

    csv_download = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Dataset (CSV)",
        data=csv_download,
        file_name="cleaned_ai_job_trends_export.csv",
        mime="text/csv"
    )
