"""
app.py - ChurnSense: Customer Churn Prediction System (Streamlit)
Two pages only:  1) Predict Churn   2) Model Performance
Run:  streamlit run app.py
Everything shown here is read from the files created by train_model.py.
"""
import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import churn_utils as U

st.set_page_config(page_title="ChurnSense", page_icon="📉", layout="wide")

# ------------------------------------------------------------------ theme
INK, PRIMARY, TEAL = "#1E1B4B", "#4F46E5", "#14B8A6"
CHURN, STAY, AMBER = "#F43F5E", "#10B981", "#F59E0B"
MODEL_COLORS = [PRIMARY, TEAL, AMBER, CHURN]

# NOTE on layout: the page needs top padding (3.5rem) because Streamlit's fixed header bar sits on top
# of the page. Too little padding hides the first heading under it. No fixed heights are used anywhere.
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}
.stApp {{ background: #F5F6FB; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{ padding: 3.5rem 2rem 3rem 2rem; max-width: 1180px; }}
footer {{ visibility: hidden; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {INK} 0%, #312E81 100%); }}
[data-testid="stSidebar"] * {{ color: #E0E7FF !important; }}
.brand {{ font-size: 1.3rem; font-weight: 800; line-height: 1.3; margin: 0; }}
.brand-sub {{ font-size: .78rem; opacity: .75; margin: 0 0 1rem 0; line-height: 1.3; }}
.hero {{ background: linear-gradient(120deg, {INK} 0%, {PRIMARY} 62%, {TEAL} 135%);
        border-radius: 18px; padding: 1.6rem 2rem; color: white; margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(79,70,229,.22); }}
.hero .eyebrow {{ font-size: .8rem; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: #C7D2FE; margin: 0 0 .3rem 0; }}
.hero h1 {{ color: white; font-weight: 800; font-size: 2rem; line-height: 1.25; margin: 0 0 .3rem 0; padding: 0; }}
.hero .sub {{ color: #E0E7FF; font-size: 1.05rem; font-weight: 600; margin: 0 0 .5rem 0; }}
.hero p {{ color: #E0E7FF; font-size: .95rem; margin: 0; line-height: 1.55; max-width: 800px; }}
.sec {{ font-size: 1.25rem; font-weight: 700; color: {INK}; line-height: 1.35; margin: 1.8rem 0 .15rem 0; }}
.sec-sub {{ color: #64748B; font-size: .9rem; margin: 0 0 .8rem 0; line-height: 1.5; }}
.kpi {{ background: white; border-radius: 14px; padding: .85rem 1rem; box-shadow: 0 2px 10px rgba(30,27,75,.07); }}
.kpi-label {{ color: #64748B; font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .5px; line-height: 1.3; }}
.kpi-value {{ color: {INK}; font-size: 1.65rem; font-weight: 800; line-height: 1.3; }}
.kpi-sub {{ color: #94A3B8; font-size: .75rem; line-height: 1.3; }}
.card {{ background: white; border-radius: 14px; padding: 1rem 1.3rem; box-shadow: 0 2px 10px rgba(30,27,75,.07); margin-bottom: 1rem; color: #334155; line-height: 1.6; }}
.card h4 {{ margin: 0 0 .4rem 0; color: {INK}; font-size: 1.05rem; line-height: 1.4; }}
.card ul, .card ol {{ margin: .3rem 0 0 0; padding-left: 1.2rem; }}
.card code {{ font-size: .82rem; }}
.insight {{ background: white; border-left: 5px solid {PRIMARY}; border-radius: 10px; padding: .65rem 1rem;
           margin-bottom: .5rem; box-shadow: 0 2px 8px rgba(30,27,75,.06); color: #334155; line-height: 1.5; }}
.result {{ border-radius: 16px; padding: 1.2rem 1.5rem; color: white; margin-bottom: .9rem; }}
.result h2 {{ color: white; margin: 0; padding: 0; font-weight: 800; font-size: 1.6rem; line-height: 1.3; }}
.result p {{ margin: .3rem 0 0 0; color: rgba(255,255,255,.95); }}
.tip {{ background: #EEF2FF; border-radius: 10px; padding: .6rem 1rem; margin-bottom: .45rem; color: {INK}; line-height: 1.5; }}
.note {{ color: #64748B; font-size: .85rem; line-height: 1.5; }}
.form-h {{ font-weight: 700; color: {INK}; font-size: 1rem; margin: 0 0 .5rem 0; }}
div[data-testid="stForm"] {{ background: transparent; border: none; padding: 0; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: white; border-radius: 14px; }}
div[data-testid="stFormSubmitButton"] > button {{ width: 100%; background: {PRIMARY}; color: white; border: none;
    border-radius: 10px; font-weight: 700; padding: .65rem 1.6rem; }}
div[data-testid="stFormSubmitButton"] > button:hover {{ background: {INK}; color: white; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ loaders
@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load + clean the CSV with the SAME cleaning used in training (tutorial: drop blank TotalCharges)."""
    data, _ = U.clean_data(U.load_data())
    return data


@st.cache_resource
def load_model():
    """Load the final (tuned Random Forest) pipeline saved by train_model.py."""
    return joblib.load(U.RF_MODEL_PATH) if U.RF_MODEL_PATH.exists() else None


@st.cache_data
def load_metrics():
    if not U.METRICS_PATH.exists():
        return None
    with open(U.METRICS_PATH) as f:
        return json.load(f)


# ------------------------------------------------------------------ helpers
def style_fig(fig, height=400):
    """Shared chart style: generous margins + automargin so titles/labels are never clipped."""
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=70, b=30), template="plotly_white",
                      font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#334155"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      title=dict(x=0.01, xanchor="left", font=dict(size=16, color=INK)), legend_title_text="")
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def show_chart(fig, key):
    """Plotly chart without the hover toolbar (it used to sit on top of the chart titles)."""
    st.plotly_chart(fig, key=key, config={"displayModeBar": False})


def kpi(label, value, sub="", color=PRIMARY):
    st.markdown(f'<div class="kpi" style="border-top:4px solid {color}"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="sec">{title}</div>' + (f'<div class="sec-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def pretty(feature: str) -> str:
    """'cat__Contract_Month-to-month' -> 'Contract = Month-to-month'."""
    name = feature.split("__", 1)[-1]
    return name.replace("_", " = ", 1) if "_" in name else name


def churn_rate(df, mask=None):
    sub = df if mask is None else df[mask]
    return (sub["Churn"] == "Yes").mean() * 100 if len(sub) else float("nan")


def need_training():
    st.warning("Trained model files were not found. Run `python train_model.py` in the project folder, then refresh this page.")
    st.stop()


# ------------------------------------------------------------------ load data / model
try:
    DF = load_dataset()
except (FileNotFoundError, ValueError) as err:
    st.error(f"**Dataset problem:** {err}")
    st.stop()
MODEL, METRICS = load_model(), load_metrics()

PAGES = ["🔮 Predict Churn", "📈 Model Performance"]
with st.sidebar:
    st.markdown('<p class="brand">📉 ChurnSense</p><p class="brand-sub">Customer Churn<br>Prediction System</p>', unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.caption(f"{len(DF):,} customers · {len(U.FEATURES)} input features")

total, churned = len(DF), int((DF["Churn"] == "Yes").sum())
final = METRICS["models"][METRICS["meta"]["final_model"]] if METRICS else None


# ================================================================== PAGE 1: PREDICT
if page == PAGES[0]:
    st.markdown('<div class="hero"><p class="eyebrow">ChurnSense</p><h1>Customer Churn Prediction System</h1>'
                '<p class="sub">Spot customers at risk of leaving before they go.</p>'
                '<p>Enter a telecom customer\'s details and a tuned Random Forest model trained on historical customer data '
                'estimates the probability that they will churn, with practical retention ideas.</p></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Total customers", f"{total:,}", "in the dataset", PRIMARY)
    with c2: kpi("Overall churn rate", f"{churned / total:.1%}", f"{churned:,} customers left", CHURN)
    with c3: kpi("Model accuracy", f"{final['accuracy']:.1%}" if final else "n/a", "held-out test set", TEAL)
    with c4: kpi("Model ROC-AUC", f"{final['roc_auc']:.3f}" if final else "n/a", "held-out test set", AMBER)

    if MODEL is None:
        need_training()

    def opts(col):
        return sorted(DF[col].unique())

    section("Customer details", "Fill in the form and press Predict. Every field is a feature the model was trained on.")
    with st.form("predict_form"):
        with st.container(border=True):
            st.markdown('<p class="form-h">👤 Customer Information</p>', unsafe_allow_html=True)
            a1, a2, a3, a4 = st.columns(4)
            gender = a1.radio("Gender", opts("gender"), horizontal=True)
            senior = a2.radio("Senior citizen", ["No", "Yes"], horizontal=True)
            partner = a3.radio("Partner", ["No", "Yes"], horizontal=True)
            dependents = a4.radio("Dependents", ["No", "Yes"], horizontal=True)

        with st.container(border=True):
            st.markdown('<p class="form-h">📞 Services</p>', unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            phone = s1.radio("Phone service", ["Yes", "No"], horizontal=True)
            multiple = s2.radio("Multiple lines", ["No", "Yes"], horizontal=True)
            internet = s3.selectbox("Internet service", opts("InternetService"), index=opts("InternetService").index("Fiber optic"))
            security = s1.radio("Online security", ["No", "Yes"], horizontal=True)
            backup = s2.radio("Online backup", ["No", "Yes"], horizontal=True)
            protection = s3.radio("Device protection", ["No", "Yes"], horizontal=True)
            tech = s1.radio("Tech support", ["No", "Yes"], horizontal=True)
            tv = s2.radio("Streaming TV", ["No", "Yes"], horizontal=True)
            movies = s3.radio("Streaming movies", ["No", "Yes"], horizontal=True)

        left, right = st.columns(2)
        with left, st.container(border=True):
            st.markdown('<p class="form-h">📝 Contract Details</p>', unsafe_allow_html=True)
            contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
            tenure = st.slider("Tenure (months with the company)", 0, 72, 6)
        with right, st.container(border=True):
            st.markdown('<p class="form-h">💳 Billing Information</p>', unsafe_allow_html=True)
            payment = st.selectbox("Payment method", opts("PaymentMethod"),
                                   index=opts("PaymentMethod").index("Electronic check"))
            paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
            b1, b2 = st.columns(2)
            monthly = b1.number_input("Monthly charges ($)", 0.0, 200.0, 75.0, step=0.5)
            total_manual = b2.number_input("Total charges ($)", 0.0, 10000.0, 450.0, step=10.0)
            auto_total = st.checkbox("Estimate total charges as monthly charges × tenure", value=True)
        submitted = st.form_submit_button("🔮  Predict churn")

    if submitted:
        # Encode "no service" the same way the dataset does
        notes = []
        if phone == "No":
            multiple = "No phone service"
        if internet == "No":
            security = backup = protection = tech = tv = movies = "No internet service"
            notes.append("Internet service is 'No', so the internet add-ons were set to 'No internet service'.")
        total_charges = monthly * tenure if auto_total else total_manual

        row = {"gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,   # 0/1 like the dataset
               "Partner": partner, "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
               "MultipleLines": multiple, "InternetService": internet, "OnlineSecurity": security,
               "OnlineBackup": backup, "DeviceProtection": protection, "TechSupport": tech,
               "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
               "PaperlessBilling": paperless, "PaymentMethod": payment,
               "MonthlyCharges": monthly, "TotalCharges": total_charges}
        X_new = pd.DataFrame([row])[U.FEATURES]          # same columns the model was trained on

        likely_churn = bool(MODEL.predict(X_new)[0] == 1)   # the tutorial's .predict() decision rule
        prob = float(MODEL.predict_proba(X_new)[0, 1])
        base = churned / total

        section("Prediction result")
        res, gauge = st.columns([1.1, 1])
        with res:
            label = "Likely to Churn" if likely_churn else "Likely to Stay"
            st.markdown(f'<div class="result" style="background:{CHURN if likely_churn else STAY}">'
                        f'<h2>{"⚠️" if likely_churn else "✅"} {label}</h2>'
                        f'<p>Estimated churn probability: <b>{prob:.1%}</b></p></div>', unsafe_allow_html=True)
            risk = "High" if prob > .6 else "Medium" if prob > .3 else "Low"
            st.markdown(f"**What this means:** the model places this customer in the **{risk.lower()}-risk** group "
                        f"(Low below 30%, Medium 30-60%, High above 60%). In the training data {base:.0%} of customers churned, "
                        f"so this profile is {'above' if prob > base else 'below'} the average risk.")
            st.markdown('<p class="note">This is a statistical estimate, not a guaranteed outcome. '
                        'Use it to prioritise customer outreach, not to make final decisions.</p>', unsafe_allow_html=True)
            for n in notes:
                st.caption("ℹ️ " + n)
        with gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"color": INK}},
                title={"text": "Churn probability", "font": {"size": 15, "color": INK}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": CHURN if likely_churn else STAY},
                       "steps": [{"range": [0, 30], "color": "#D1FAE5"}, {"range": [30, 60], "color": "#FEF3C7"},
                                 {"range": [60, 100], "color": "#FFE4E6"}],
                       "threshold": {"line": {"color": INK, "width": 3}, "value": 50}}))
            show_chart(style_fig(fig, 280), key="gauge")

        # Explanation: notable factors, backed by real churn rates in the dataset
        signals = []

        def add(cond, text, mask):
            if cond:
                signals.append(f"{text} (churn rate in the data: {churn_rate(DF, mask):.0f}% vs {base * 100:.0f}% overall)")
        add(contract == "Month-to-month", "Month-to-month contract", DF["Contract"] == "Month-to-month")
        add(contract == "Two year", "Two-year contract (lowers risk)", DF["Contract"] == "Two year")
        add(tenure <= 12, "Short tenure (12 months or less)", DF["tenure"] <= 12)
        add(tenure > 48, "Long tenure (over 48 months, lowers risk)", DF["tenure"] > 48)
        add(payment == "Electronic check", "Pays by electronic check", DF["PaymentMethod"] == "Electronic check")
        add(internet == "Fiber optic", "Fiber optic internet", DF["InternetService"] == "Fiber optic")
        add(tech == "No" and internet != "No", "No tech support", DF["TechSupport"] == "No")
        add(tech == "Yes", "Has tech support (lowers risk)", DF["TechSupport"] == "Yes")
        section("Why this result?", "Notable factors in this profile. These are patterns in the data, not proven causes.")
        for s in signals or ["No single strongly notable factor; the estimate comes from a combination of features."]:
            st.markdown(f'<div class="insight">{s}</div>', unsafe_allow_html=True)

        tips = []
        if contract == "Month-to-month":
            tips.append("Offer a discount or perk for moving to a 1-year or 2-year contract.")
        if tenure <= 12:
            tips.append("Run onboarding check-ins during the first year, when churn is most common.")
        if payment == "Electronic check":
            tips.append("Encourage automatic payments (card or bank transfer) with a small incentive.")
        if internet == "Fiber optic" and (tech == "No" or security == "No"):
            tips.append("Bundle tech support and online security, and check service quality for this fiber customer.")
        if monthly > DF["MonthlyCharges"].quantile(.75):
            tips.append("Monthly bill is in the top 25%. Review the plan for savings or offer a loyalty discount.")
        if senior == "Yes":
            tips.append("Offer simplified plans and a dedicated support line.")
        if not tips:
            tips.append("Keep engagement high with satisfaction surveys and loyalty rewards.")
        section("💡 Retention suggestions" + ("" if likely_churn else " (to keep this customer happy)"))
        for t in tips:
            st.markdown(f'<div class="tip">{t}</div>', unsafe_allow_html=True)


# ================================================================== PAGE 2: PERFORMANCE
else:
    if METRICS is None:
        need_training()
    meta, models = METRICS["meta"], METRICS["models"]
    best = meta["final_model"]
    fm = models[best]
    names = list(models)

    st.markdown('<div class="hero"><p class="eyebrow">ChurnSense</p><h1>Model Performance</h1>'
                '<p>Model performance validated on independent test data. No hardcoded values.</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><h4>How to read these results</h4><ul>'
                f'<li><b>Test-set metrics</b> come from {meta["n_test"]:,} customers (30%) the models never saw during training.</li>'
                f'<li><b>Cross-validation (CV) metrics</b> come from repeated splits of the {meta["n_train"]:,} training customers only. '
                f'They are used to compare and tune models.</li>'
                f'<li>The final model used by the Predict page is <b>{best}</b>, as in the tutorial\'s "Final Model" steps.</li></ul></div>',
                unsafe_allow_html=True)

    section(f"Final model: {best}", "Test-set results at the default 0.5 decision threshold.")
    k = st.columns(5)
    for col, (lab, key, colr) in zip(k, [("Accuracy", "accuracy", PRIMARY), ("Precision", "precision", TEAL),
                                         ("Recall", "recall", CHURN), ("F1-score", "f1", AMBER), ("ROC-AUC", "roc_auc", INK)]):
        with col:
            kpi(lab, f"{fm[key]:.3f}", "test set", colr)

    # ---- model comparison
    section("Model comparison (test set)", "All four models evaluated on the same held-out customers.")
    table = pd.DataFrame({n: {"Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
                              "F1-score": m["f1"], "ROC-AUC": m["roc_auc"]} for n, m in models.items()}).T
    st.dataframe(table.style.format("{:.3f}").highlight_max(axis=0, color="#C7D2FE"))
    long = table.reset_index(names="Model").melt(id_vars="Model", var_name="Metric", value_name="Score")
    bar = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", title="Test-set metric comparison",
                 color_discrete_sequence=MODEL_COLORS, range_y=[0, 1])
    show_chart(style_fig(bar, 420), key="cmp_bar")
    st.caption("Highlighted cells are the best value in each column. Accuracy alone can mislead because only about 27% of customers churn.")

    # ---- CV vs test
    section("Cross-validation vs test set", "Cross-validation uses training data only; the test set is used once at the end.")
    rows = []
    for n, m in models.items():
        cv = m["cv"]
        rows.append({"Model": n,
                     "CV ROC-AUC (training data)": f"{cv['mean']:.3f} ± {cv['std']:.3f}  ({cv['folds']}-fold)" if cv else "not run in the tutorial",
                     "Test ROC-AUC (held-out)": f"{m['roc_auc']:.3f}"})
    st.dataframe(pd.DataFrame(rows).set_index("Model"))

    # ---- tuning
    section("Hyperparameter tuning", "GridSearchCV on the Random Forest, scored by ROC-AUC with 3-fold cross-validation.")
    grid_html = "".join(f"<li><code>{k.replace('model__', '')}</code>: {v}</li>" for k, v in meta["grid"].items())
    best_html = ", ".join(f"<code>{k.replace('model__', '')} = {v}</code>" for k, v in meta["best_params"].items())
    st.markdown(f'<div class="card"><h4>Search space ({meta["grid_fits"]} fits)</h4><ul>{grid_html}</ul>'
                f'<h4 style="margin-top:.8rem">Best parameters found</h4>{best_html}<br>'
                f'Best CV ROC-AUC: <b>{models[best]["cv"]["mean"]:.4f}</b></div>', unsafe_allow_html=True)

    # ---- confusion matrix + report
    section("Confusion matrix and classification report")
    chosen = st.selectbox("Model", names, index=names.index(best))
    cm_col, rep_col = st.columns(2)
    with cm_col:
        cm = models[chosen]["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, x=["Predicted: Stay", "Predicted: Churn"], y=["Actual: Stay", "Actual: Churn"],
                        color_continuous_scale=["#EEF2FF", PRIMARY], title=f"Confusion matrix: {chosen}")
        fig.update_coloraxes(showscale=False)
        show_chart(style_fig(fig, 380), key="cm")
    with rep_col:
        rep = models[chosen]["classification_report"]
        rep_df = pd.DataFrame({k: v for k, v in rep.items() if isinstance(v, dict)}).T
        rep_df = rep_df.rename(index={"0": "Stay (0)", "1": "Churn (1)"})
        rep_df["support"] = rep_df["support"].astype(int)
        st.markdown(f"<p class='form-h' style='margin-top:.6rem'>Classification report: {chosen}</p>", unsafe_allow_html=True)
        st.dataframe(rep_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:d}"}))
        tn, fp = cm[0]
        fn, tp = cm[1]
        st.caption(f"Of {tp + fn} customers who really churned, the model found {tp} (recall {tp / (tp + fn):.0%}). "
                   f"It flagged {fp} loyal customers by mistake.")

    # ---- ROC
    section("ROC curves", "The closer a curve is to the top-left corner, the better the model separates churners from stayers.")
    roc = go.Figure()
    for i, (n, m) in enumerate(models.items()):
        roc.add_scatter(x=m["roc_curve"]["fpr"], y=m["roc_curve"]["tpr"], mode="lines",
                        name=f"{n} (AUC {m['roc_auc']:.3f})", line=dict(width=4 if n == best else 2, color=MODEL_COLORS[i]))
    roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(dash="dash", color="#94A3B8"))
    roc.update_layout(title="ROC curves (test set)", xaxis_title="False positive rate", yaxis_title="True positive rate",
                      legend=dict(orientation="h", y=-0.2))
    show_chart(style_fig(roc, 460), key="roc")

    # ---- importance + SHAP
    section(f"What drives churn? ({best})")
    tab1, tab2, tab3 = st.tabs(["Feature importance", "SHAP summary", "SHAP bar"])
    with tab1:
        imp = pd.DataFrame(METRICS["feature_importance"])
        imp["feature"] = imp["feature"].map(pretty)
        fig = px.bar(imp.iloc[::-1], x="importance", y="feature", orientation="h", color_discrete_sequence=[TEAL],
                     title="Top 15 features (Random Forest importance)")
        fig.update_layout(xaxis_title="Importance", yaxis_title="")
        show_chart(style_fig(fig, 520), key="imp")
    with tab2:
        if U.SHAP_SUMMARY_PATH.exists():
            st.image(str(U.SHAP_SUMMARY_PATH))
            st.caption("Each dot is a customer. Dots to the right push the prediction toward churn; colour shows the feature value (red = high).")
        else:
            st.info("SHAP plots were not generated. Run `python train_model.py` (needs the `shap` package).")
    with tab3:
        if U.SHAP_BAR_PATH.exists():
            st.image(str(U.SHAP_BAR_PATH))
            st.caption("Average absolute SHAP value: how much each feature moves the churn prediction.")
        else:
            st.info("SHAP plots were not generated. Run `python train_model.py` (needs the `shap` package).")

    # ---- threshold + risk levels
    section("Threshold analysis and risk levels", f"Tuned Random Forest on the test set. The app predicts with the default 0.5 rule; this shows what other thresholds would do.")
    th = pd.DataFrame(METRICS["threshold_analysis"])
    t_col, r_col = st.columns([2, 1])
    with t_col:
        fig = go.Figure()
        for col, colr in [("precision", PRIMARY), ("recall", CHURN), ("f1", AMBER)]:
            fig.add_scatter(x=th["threshold"], y=th[col], mode="lines+markers", name=col.capitalize(), line=dict(color=colr))
        fig.add_vline(x=0.5, line_dash="dot", line_color="#94A3B8", annotation_text="default 0.5")
        fig.update_layout(title="Precision / recall trade-off", xaxis_title="Threshold", yaxis_title="Score", yaxis_range=[0, 1],
                          legend=dict(orientation="h", y=-0.22))
        show_chart(style_fig(fig, 400), key="thr")
    with r_col:
        risk = pd.DataFrame({"Risk level": list(METRICS["risk_levels"]), "Customers": list(METRICS["risk_levels"].values())})
        fig = px.bar(risk, x="Risk level", y="Customers", color="Risk level", text="Customers", title="Test customers by risk level",
                     color_discrete_map={"Low Risk": STAY, "Medium Risk": AMBER, "High Risk": CHURN})
        fig.update_layout(showlegend=False)
        show_chart(style_fig(fig, 400), key="risk")
    with st.expander("Show threshold table"):
        st.dataframe(th.style.format({"threshold": "{:.2f}", "precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}), hide_index=True)

    # ---- honesty notes
    section("About these results")
    st.markdown(f"""<div class="card"><ul>
<li>Training used <b>{meta['total_customers']:,}</b> customers after dropping {meta['dropped_rows']} rows with a blank <code>TotalCharges</code>
(tutorial cleaning). Split: 70% train / 30% test, stratified, seed {meta['split_seed']}.</li>
<li>The tutorial code does not show the Random Forest's base parameters, so <code>class_weight="balanced"</code> was assumed (see <code>train_model.py</code>).</li>
<li>Results come from one dataset snapshot and one split, so small differences between models may not be meaningful. Feature importance shows association, not cause.</li>
<li>Trained with scikit-learn {meta['versions']['scikit-learn']} and XGBoost {meta['versions']['xgboost']} on {meta['trained_at'][:10]}.</li></ul></div>""",
                unsafe_allow_html=True)
