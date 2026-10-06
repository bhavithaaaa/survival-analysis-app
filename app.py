from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Drug Clinical Trial: Survival Analysis", page_icon="🩺", layout="wide")
NAVY, BLUE = "#094780", "#1E88E5"
PALETTE = ["#1E88E5", "#F57C00", "#2E7D32", "#6A1B9A", "#C62828", "#00838F"]
st.markdown(f"""<style>h1,h2,h3{{color:{NAVY};}} [data-testid="stMetricValue"]{{color:{NAVY};}}</style>""", unsafe_allow_html=True)

# ---------- load data (CSV files exported from the notebook) ----------
HERE = Path(__file__).parent
def find(name):
    for folder in (HERE / "data", HERE):
        if (folder / name).exists():
            return folder / name
    return None

km_path, hr_path = find("km_curves.csv"), find("cox_hazard_ratios.csv")
if km_path is None or hr_path is None:
    st.error("Put km_curves.csv and cox_hazard_ratios.csv in the 'data' folder next to app.py.")
    st.stop()

@st.cache_data
def load(km_file, hr_file):
    return pd.read_csv(km_file), pd.read_csv(hr_file)
km, hr = load(km_path, hr_path)

NAMES = {"HER2_Pos": "HER2-positive", "ER_Pos": "ER-positive", "Age at Diagnosis": "Age (per year)",
         "Tumor Size": "Tumor size (per mm)", "Lymph nodes examined positive": "Positive lymph nodes (per node)",
         "Hormone Therapy": "Hormone therapy", "Radio Therapy": "Radiotherapy", "Tumor Stage": "Tumor stage"}
hr["Factor"] = hr["Variable"].map(NAMES).fillna(hr["Variable"])

def rgba(hex_color, a):
    h = hex_color.lstrip("#"); return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

def survival_at(g, t):
    sub = g[g["Months"] <= t]
    return np.nan if sub.empty or g["Months"].max() < t else sub["Survival"].iloc[-1]

# ---------- header ----------
st.title("🩺 Drug Clinical Trial: Survival and Subgroup Analysis")
st.caption("METABRIC breast cancer cohort | Kaplan-Meier, log-rank tests, Cox proportional hazards | Data Science Internship, Gradtwin Services")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Survival curves", "Hazard ratios", "Findings"])

with tab1:
    c = st.columns(6)
    for col, (label, val) in zip(c, [("Patients in file", "2,509"), ("With survival data", "1,981"), ("Deaths", "1,144"),
                                      ("Censored", "837"), ("Cox model (complete cases)", "1,400"), ("Test C-index", "0.687")]):
        col.metric(label, val)
    st.subheader("About the project")
    st.write("This project studies how long breast cancer patients survive and whether treatments such as chemotherapy "
             "work differently in different patient subgroups. The analysis was done in Python (Google Colab) and the "
             "results are explored here.")
    st.subheader("Workflow")
    st.write("**Data inspection** → **Data preparation** → **Exploratory analysis** → **Kaplan-Meier curves and log-rank tests** → "
             "**Chemotherapy subgroup analysis** → **Cox model** → **Validation** → **Dashboard**")

with tab2:
    left, right = st.columns([1, 3])
    with left:
        var = st.selectbox("Compare survival by", sorted(km["Variable"].unique()), index=0)
        show_ci = st.checkbox("Show 95% confidence bands", value=True)
        max_m = int(km["Months"].max())
        limit = st.slider("Follow-up shown (months)", 24, max_m, max_m)
    sub = km[km["Variable"] == var]
    fig = go.Figure()
    for i, (grp, g) in enumerate(sub.groupby("Group")):
        g = g[g["Months"] <= limit].sort_values("Months"); color = PALETTE[i % len(PALETTE)]
        if show_ci:
            fig.add_trace(go.Scatter(x=g["Months"], y=g["Upper_CI"], mode="lines", line=dict(width=0, shape="hv"), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=g["Months"], y=g["Lower_CI"], mode="lines", line=dict(width=0, shape="hv"), fill="tonexty",
                                     fillcolor=rgba(color, 0.15), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=g["Months"], y=g["Survival"], mode="lines", name=str(grp), line=dict(color=color, width=3, shape="hv")))
    fig.update_layout(height=520, xaxis_title="Months", yaxis_title="Survival probability", yaxis_range=[0, 1.02],
                      legend_title_text=var, margin=dict(t=30))
    right.plotly_chart(fig, width="stretch")
    st.subheader("Estimated survival at fixed time points")
    rows = [{"Group": grp, "5 years (60 mo)": survival_at(g, 60), "10 years (120 mo)": survival_at(g, 120), "15 years (180 mo)": survival_at(g, 180)}
            for grp, g in sub.groupby("Group")]
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.1%}" for c in ["5 years (60 mo)", "10 years (120 mo)", "15 years (180 mo)"]}, na_rep="not reached"), hide_index=True)
    st.caption("A higher curve means longer survival. Curves are unadjusted for other factors.")

with tab3:
    d = hr.sort_values("Hazard_Ratio")
    fig = go.Figure()
    for _, r in d.iterrows():
        sig = r["p_value"] < 0.05; color = BLUE if sig else "#9AA5B1"
        fig.add_trace(go.Scatter(x=[r["Lower_95_CI"], r["Upper_95_CI"]], y=[r["Factor"]] * 2, mode="lines", line=dict(color=color, width=4), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[r["Hazard_Ratio"]], y=[r["Factor"]], mode="markers", marker=dict(color=color, size=12), showlegend=False,
                                 hovertemplate=f"{r['Factor']}<br>HR %{{x:.2f}} (95% CI {r['Lower_95_CI']:.2f}-{r['Upper_95_CI']:.2f})<br>p = {r['p_value']:.4f}<extra></extra>"))
    fig.add_vline(x=1, line_dash="dash", line_color=NAVY)
    fig.update_layout(height=520, xaxis_title="Hazard ratio (log scale)", xaxis_type="log", margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")
    st.caption("Blue = statistically significant (p < 0.05). Right of the dashed line = higher risk of death, left = lower risk. Cox model on 1,400 patients.")
    table = d.sort_values("Hazard_Ratio", ascending=False)[["Factor", "Hazard_Ratio", "Lower_95_CI", "Upper_95_CI", "p_value"]]
    st.dataframe(table.style.format({"Hazard_Ratio": "{:.2f}", "Lower_95_CI": "{:.2f}", "Upper_95_CI": "{:.2f}", "p_value": "{:.4f}"}), hide_index=True)
    st.download_button("Download hazard ratio table (CSV)", table.to_csv(index=False), "hazard_ratios.csv", "text/csv")

with tab4:
    st.subheader("Key findings")
    st.markdown("""
- **Log-rank tests:** survival differed by chemotherapy (p = 0.0018), hormone therapy (p = 0.0001) and radiotherapy (p = 0.0065); age group, tumor stage and HER2 status were all p < 0.0001.
- **Higher risk:** older age (HR 1.04 per year), higher tumor stage (1.27), HER2-positive status (1.46), more positive lymph nodes (1.06 per node) and larger tumors (1.007 per mm).
- **Lower risk:** radiotherapy (HR 0.79, p = 0.0022).
- **Chemotherapy** (HR 1.23, p = 0.10) and **hormone therapy** (HR 0.88, p = 0.15) were not significant after adjustment. A chemotherapy ratio above 1 does not mean harm: sicker patients are more likely to receive chemotherapy.
- **Chemotherapy effect differs by age group (p = 0.038) and tumor stage group (p = 0.021)**, but not by ER (p = 0.379) or HER2 status (p = 0.166).
- **Model quality:** C-index 0.678 on all data and 0.687 on the held-out test set (moderate discrimination).
""")
    st.subheader("Limitations")
    st.markdown("""
- Observational data: results show association, not causation.
- The Cox model used 1,400 of 1,981 patients (complete cases).
- The proportional hazards assumption was violated for age, ER status, hormone therapy and tumor stage.
- Overall survival counts deaths from any cause.
""")
