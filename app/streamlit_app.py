import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    f1_score, precision_score, recall_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import predict_single, DEFAULT_THRESHOLD

# Page config
st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cached loaders
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "models", "lgbm_churn_model.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "models", "feature_names.pkl"))
    return model, feature_names


@st.cache_data
def load_data():
    path = os.path.join(BASE_DIR, "data", "processed", "telco_churn_segmented.csv")
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_test_data():
    """Held-out test set saved from 03_model.ipynb — used for honest
    Model Performance metrics (never seen during training)."""
    path = os.path.join(BASE_DIR, "data", "processed", "test_set.csv")
    return pd.read_csv(path)


try:
    model, feature_names = load_model()
    df = load_data()
    DATA_LOADED = True
except FileNotFoundError as e:
    DATA_LOADED = False
    LOAD_ERROR = str(e)

try:
    test_df = load_test_data()
    TEST_DATA_LOADED = True
except FileNotFoundError:
    TEST_DATA_LOADED = False


# Shared constants
SEGMENT_COLORS = {
    "Champions": "#1D9E75",
    "At Risk": "#D85A30",
    "Fence Sitters": "#534AB7",
}

SEGMENT_OFFERS = {
    "Champions": {
        "priority": "Low",
        "action": "Loyalty rewards — keep them happy",
        "offer": "Cashback or referral bonus",
    },
    "At Risk": {
        "priority": "High",
        "action": "Contract upgrade offer + personal call",
        "offer": "20% off annual contract switch",
    },
    "Fence Sitters": {
        "priority": "Medium",
        "action": "Bundle more services at discount",
        "offer": "Free add-on service for 3 months",
    },
}

THRESHOLD = 0.35


# Sidebar navigation
st.sidebar.title("📊 Churn Intelligence")
st.sidebar.caption("Telco Customer Churn · LightGBM · K-Means")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Customer Predictor",
        "🧩 Segment Overview",
        "📈 Model Performance",
        "💡 Business Insights",
    ],
)

st.sidebar.divider()
if DATA_LOADED:
    st.sidebar.metric("Total Customers", f"{len(df):,}")
    st.sidebar.metric("Overall Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
    st.sidebar.metric("Model ROC-AUC", "0.821")
else:
    st.sidebar.error("Model/data files not found. Check models/ and data/processed/ folders.")


# PAGE 1 — CUSTOMER PREDICTOR
if page == "🏠 Customer Predictor":
    st.title("Customer Churn Predictor")
    st.caption("Enter a customer's details to predict their churn risk, segment, and recommended retention offer.")

    if not DATA_LOADED:
        st.error(f"Could not load model/data: {LOAD_ERROR}")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col2:
        st.subheader("Services")
        internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    st.divider()
    predict_btn = st.button("Predict Churn Risk", type="primary", use_container_width=True)

    if predict_btn:
        # All encoding + feature engineering now lives in src/preprocessing.py
        # and is called via src/predict.py — keeps this file free of
        # duplicated logic that must be kept in sync with the notebooks.
        raw_inputs = {
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless": paperless,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
        }

        result = predict_single(
            raw_inputs, model=model, feature_names=feature_names, threshold=THRESHOLD
        )
        churn_proba = result["churn_probability"]
        will_churn = result["will_churn"]

        # Assign segment using nearest segment centroid logic (simplified: by churn_proba rank)
        seg_means = df.groupby("segment_names")["churn_probability"].mean().sort_values()
        if churn_proba <= seg_means.iloc[0] + (seg_means.iloc[1] - seg_means.iloc[0]) / 2:
            segment_names = seg_means.index[0]
        elif churn_proba >= seg_means.iloc[-1] - (seg_means.iloc[-1] - seg_means.iloc[-2]) / 2:
            segment_names = seg_means.index[-1]
        else:
            segment_names = seg_means.index[1] if len(seg_means) > 2 else seg_means.index[0]

        offer_info = SEGMENT_OFFERS.get(segment_names, SEGMENT_OFFERS["Fence Sitters"])

        st.divider()
        st.subheader("Prediction Result")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Churn Probability", f"{churn_proba*100:.1f}%")
        with r2:
            st.metric("Prediction", "⚠️ Will Churn" if will_churn else "✅ Will Stay")
        with r3:
            st.metric("Segment", segment_names)
        with r4:
            st.metric("Priority", offer_info["priority"])

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_proba * 100,
            number={"suffix": "%"},
            title={"text": "Churn Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#D85A30" if will_churn else "#1D9E75"},
                "steps": [
                    {"range": [0, 35], "color": "#E8F5E9"},
                    {"range": [35, 65], "color": "#FFF3E0"},
                    {"range": [65, 100], "color": "#FFEBEE"},
                ],
                "threshold": {"line": {"color": "black", "width": 2}, "value": THRESHOLD * 100},
            },
        ))
        gauge.update_layout(height=300, margin=dict(t=40, b=10))
        st.plotly_chart(gauge, use_container_width=True)

        st.info(
            f"**Recommended Action:** {offer_info['action']}  \n"
            f"**Suggested Offer:** {offer_info['offer']}"
        )


# PAGE 2 — SEGMENT OVERVIEW
elif page == "🧩 Segment Overview":
    st.title("Customer Segment Overview")
    st.caption("K-Means clustering (K=3) on tenure, monthly charges, services, and churn probability.")

    if not DATA_LOADED:
        st.error(f"Could not load data: {LOAD_ERROR}")
        st.stop()

    segment_order = ["Champions", "At Risk", "Fence Sitters"]

    stats = df.groupby("segment_names").agg(
        customers=("Churn", "count"),
        churn_rate=("Churn", "mean"),
        avg_tenure=("tenure", "mean"),
        avg_charges=("MonthlyCharges", "mean"),
        avg_services=("num_services", "mean"),
        avg_CLV=("CLV", "mean"),
    ).reindex(segment_order).round(2)
    stats["churn_rate"] = (stats["churn_rate"] * 100).round(1)

    cols = st.columns(3)
    for i, seg in enumerate(segment_order):
        row = stats.loc[seg]
        with cols[i]:
            st.markdown(f"### {seg}")
            st.metric("Customers", f"{int(row['customers']):,}")
            st.metric("Churn Rate", f"{row['churn_rate']}%")
            st.metric("Avg CLV", f"${row['avg_CLV']:,.0f}")
            offer = SEGMENT_OFFERS[seg]
            st.caption(f"**Priority:** {offer['priority']}")
            st.caption(f"**Offer:** {offer['offer']}")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        churn_fig = px.bar(
            stats.reindex(segment_order).reset_index(),
            x="churn_rate", y="segment_names", orientation="h",
            color="segment_names", color_discrete_map=SEGMENT_COLORS,
            title="Churn Rate by Segment", labels={"churn_rate": "Churn %", "segment_names": ""},
        )
        avg_churn = df["Churn"].mean() * 100
        churn_fig.add_vline(x=avg_churn, line_dash="dash", line_color="gray")
        churn_fig.update_layout(showlegend=False)
        st.plotly_chart(churn_fig, use_container_width=True)

    with c2:
        count_fig = px.bar(
            stats.reindex(segment_order).reset_index(),
            x="customers", y="segment_names", orientation="h",
            color="segment_names", color_discrete_map=SEGMENT_COLORS,
            title="Customers per Segment", labels={"customers": "Count", "segment_names": ""},
        )
        count_fig.update_layout(showlegend=False)
        st.plotly_chart(count_fig, use_container_width=True)

    st.divider()
    st.subheader("PCA Cluster Visualisation")
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    cluster_features = ["tenure", "MonthlyCharges", "num_services", "churn_probability"]
    X_scaled = StandardScaler().fit_transform(df[cluster_features])
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "segment_names": df["segment_names"]})

    pca_fig = px.scatter(
        pca_df, x="PC1", y="PC2", color="segment_names",
        color_discrete_map=SEGMENT_COLORS, opacity=0.5,
        title=f"Segments in 2D (PCA, {sum(pca.explained_variance_ratio_)*100:.1f}% variance explained)",
    )
    st.plotly_chart(pca_fig, use_container_width=True)

    st.divider()
    st.subheader("Retention Strategy")
    strategy_df = pd.DataFrame({
        "Segment": segment_order,
        "Priority": [SEGMENT_OFFERS[s]["priority"] for s in segment_order],
        "Action": [SEGMENT_OFFERS[s]["action"] for s in segment_order],
        "Offer": [SEGMENT_OFFERS[s]["offer"] for s in segment_order],
    })
    st.dataframe(strategy_df, use_container_width=True, hide_index=True)


# PAGE 3 — MODEL PERFORMANCE
elif page == "📈 Model Performance":
    st.title("Model Performance")
    st.caption("LightGBM classifier trained with SMOTE balancing · threshold = 0.35")

    if not DATA_LOADED:
        st.error(f"Could not load model/data: {LOAD_ERROR}")
        st.stop()

    if not TEST_DATA_LOADED:
        st.warning(
            "`data/processed/test_set.csv` not found — showing metrics on the full "
            "dataset instead, which inflates scores since it includes training data. "
            "Re-run the save cell in `notebooks/03_model.ipynb` to generate the proper "
            "held-out test set, then refresh this page."
        )
        eval_df = df
    else:
        eval_df = test_df

    X_eval = eval_df[feature_names]
    y_eval = eval_df["Churn"]
    y_proba = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    roc_auc = roc_auc_score(y_eval, y_proba)
    f1 = f1_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred)
    recall = recall_score(y_eval, y_pred)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{roc_auc:.3f}")
    m2.metric("F1 Score", f"{f1:.3f}")
    m3.metric("Churn Precision", f"{precision:.2f}")
    m4.metric("Churn Recall", f"{recall:.2f}")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_eval, y_proba)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                      name=f"LightGBM (AUC={roc_auc:.3f})",
                                      line=dict(color="#534AB7", width=3)))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      name="Random classifier",
                                      line=dict(color="gray", dash="dash")))
        roc_fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(roc_fig, use_container_width=True)

    with c2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_eval, y_pred)
        cm_fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=["Predicted Stayed", "Predicted Churned"],
            y=["Actual Stayed", "Actual Churned"],
        )
        cm_fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(cm_fig, use_container_width=True)

    st.divider()
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=True).tail(15)
    imp_fig = px.bar(imp_df, x="importance", y="feature", orientation="h",
                      title="Top 15 Features (LightGBM gain importance)",
                      color_discrete_sequence=["#534AB7"])
    st.plotly_chart(imp_fig, use_container_width=True)

    st.caption(
        "Note: SHAP-based explainability (feature direction and magnitude per prediction) "
        "is computed in `notebooks/03_model.ipynb` — see that notebook for the full beeswarm analysis."
    )


# PAGE 4 — BUSINESS INSIGHTS
elif page == "💡 Business Insights":
    st.title("Business Insights")
    st.caption("Key findings from exploratory data analysis (Phase 1).")

    if not DATA_LOADED:
        st.error(f"Could not load data: {LOAD_ERROR}")
        st.stop()

    avg_churn = df["Churn"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Churn Rate", f"{avg_churn:.1f}%")
    c2.metric("Total Customers", f"{len(df):,}")
    c3.metric("At-Risk Segment Churn", "57.0%")
    c4.metric("Champions Avg CLV", "$4,743")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        contract_churn = df.groupby("Contract")["Churn"].mean().reset_index()
        contract_churn["Contract"] = contract_churn["Contract"].map(
            {0: "Month-to-month", 1: "One year", 2: "Two year"})
        contract_churn["Churn"] *= 100
        fig = px.bar(contract_churn, x="Contract", y="Churn", color_discrete_sequence=["#534AB7"],
                     title="Churn Rate by Contract Type", labels={"Churn": "Churn %"})
        fig.add_hline(y=avg_churn, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        internet_churn = df.groupby("InternetService")["Churn"].mean().reset_index()
        internet_churn["InternetService"] = internet_churn["InternetService"].map(
            {0: "No internet", 1: "DSL", 2: "Fiber optic"})
        internet_churn["Churn"] *= 100
        fig = px.bar(internet_churn, x="InternetService", y="Churn", color_discrete_sequence=["#1D9E75"],
                     title="Churn Rate by Internet Service", labels={"Churn": "Churn %"})
        fig.add_hline(y=avg_churn, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Key Findings")
    findings = pd.DataFrame({
        "Finding": [
            "Month-to-month contracts churn 15x more than two-year contracts",
            "Fiber optic customers churn most (41.9%) despite premium pricing",
            "Churners leave within a median of 10 months vs 38 months for retained",
            "Electronic check payment users churn at 45% — highest of all payment types",
            "Retained customers have 66% higher CLV than churned customers",
            "High-risk customers (month-to-month + new) churn at 51.4%",
        ],
        "Business Action": [
            "Incentivise annual contract upgrades",
            "Review fiber pricing / bundle value-add services",
            "Launch onboarding retention touchpoints in first 3 months",
            "Encourage switch to automatic payment methods",
            "Prioritise retention spend on high-tenure customers",
            "Target month-to-month + new customers with contract offers first",
        ],
    })
    st.dataframe(findings, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Estimated Business Impact")
    st.markdown(
        "The **At Risk** segment (2,421 customers, 57% churn rate) represents the highest-value "
        "retention opportunity. Retaining just **30%** of these customers protects approximately:"
    )

    st.success(
        "2,421 × 57% × 30% × \$883 (avg CLV) ≈ **\$366,000** in protected revenue"
    )