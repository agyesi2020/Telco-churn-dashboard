"""Telco Customer Churn Dashboard (Task 6) -- run with: streamlit run app.py"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Telco Churn Dashboard", page_icon="📉", layout="wide")

CHURN_COLORS = {"No": "#4C78A8", "Yes": "#E45756"}
TENURE_LABELS = ["0-12 months", "13-36 months", "37+ months"]


@st.cache_data
def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Churned"] = (df["Churn"] == "Yes").astype(int)
    df["Senior"] = df["SeniorCitizen"].map({0: "Not senior", 1: "Senior"})
    df["Tenure Group"] = pd.cut(df["tenure"], [0, 12, 36, df["tenure"].max()],
                                labels=TENURE_LABELS, include_lowest=True)
    df["Lifecycle"] = pd.cut(df["tenure"], [0, 6, 12, 24, 36, 48, 72],
                             labels=["0-6", "7-12", "13-24", "25-36", "37-48", "49-72"], include_lowest=True)
    return df


def rate(s):  # churn rate in %
    return s.mean() * 100


def insight(text):
    st.info(f"💡 **Insight:** {text}")


# ---------------------------------------------------------------- data + sidebar
st.sidebar.header("Upload / filter")
upload = st.sidebar.file_uploader("Use a different CSV (same columns)", type="csv")
raw = load(upload) if upload else load("Telco_Dataset.csv")

contracts = st.sidebar.multiselect("Contract", raw["Contract"].unique(), default=list(raw["Contract"].unique()))
internet = st.sidebar.multiselect("Internet service", raw["Internet Service"].unique(), default=list(raw["Internet Service"].unique()))
tenure_rng = st.sidebar.slider("Tenure (months)", int(raw.tenure.min()), int(raw.tenure.max()),
                               (int(raw.tenure.min()), int(raw.tenure.max())))

df = raw[raw["Contract"].isin(contracts) & raw["Internet Service"].isin(internet)
         & raw["tenure"].between(*tenure_rng)]

st.title("📉 Telco Customer Churn Dashboard")
st.caption("EDA, customer segmentation and advanced churn analysis. Use the sidebar to filter every chart.")

if df.empty:
    st.warning("No customers match the current filters.")
    st.stop()
if len(df) < 30:
    st.warning(f"Only {len(df)} customers selected, so percentages are very noisy.")

# ---------------------------------------------------------------- KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", len(df))
k2.metric("Churn rate", f"{rate(df.Churned):.1f}%")
k3.metric("Avg monthly charge", f"${df['Monthly Charges'].mean():.2f}")
k4.metric("Avg tenure", f"{df.tenure.mean():.1f} months")
st.divider()

# ================================================================= TASK 3: EDA
st.header("1. Exploratory data analysis")

st.subheader("Summary statistics")
num = ["tenure", "Monthly Charges", "Total Charges"]
stats = pd.DataFrame({"mean": df[num].mean(), "median": df[num].median(),
                      "mode": df[num].mode().iloc[0], "std": df[num].std()}).round(2)
st.dataframe(stats, width="stretch")
skew = df["Total Charges"].mean() / df["Total Charges"].median()
insight(f"Mean Total Charges (${df['Total Charges'].mean():,.0f}) is {skew:.1f}x the median "
        f"(${df['Total Charges'].median():,.0f}): a few long-tenure, high-spend customers skew it upward, so the "
        "median describes the typical customer better.")

st.subheader("Distributions of numerical columns")
col = st.selectbox("Column", num, key="dist_col")
c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(df, x=col, nbins=15, color="Churn", color_discrete_map=CHURN_COLORS,
                       barmode="overlay", opacity=0.75, title=f"Histogram of {col}")
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = px.box(df, x="Churn", y=col, color="Churn", color_discrete_map=CHURN_COLORS,
                 category_orders={"Churn": ["No", "Yes"]}, points="all", title=f"{col} by churn status")
    st.plotly_chart(fig, width="stretch")
med = df.groupby("Churn")[col].median()
if {"Yes", "No"} <= set(med.index):
    insight(f"Median {col} is {med['Yes']:,.1f} for churned customers vs {med['No']:,.1f} for retained ones. "
            "Churners typically have much shorter tenure and pay more per month.")

st.subheader("Churn vs. non-churn")
counts = df["Churn"].value_counts().reindex(["No", "Yes"]).fillna(0)
fig = px.pie(values=counts.values, names=["Retained", "Churned"], hole=0.5,
             color_discrete_sequence=[CHURN_COLORS["No"], CHURN_COLORS["Yes"]], title="Churn proportion")
st.plotly_chart(fig, width="stretch")
insight(f"{counts['Yes']:.0f} of {counts.sum():.0f} customers ({rate(df.Churned):.1f}%) churned. The classes are "
        "imbalanced, which matters for any predictive model (use stratified splits; judge on recall/F1).")
st.divider()

# ================================================================= TASK 4: SEGMENTATION
st.header("2. Customer segmentation by tenure")
seg = df["Tenure Group"].value_counts().reindex(TENURE_LABELS).fillna(0).reset_index()
seg.columns = ["Tenure Group", "Customers"]
c1, c2 = st.columns(2)
with c1:
    fig = px.pie(seg, names="Tenure Group", values="Customers", hole=0.5,
                 color_discrete_sequence=["#F58518", "#54A24B", "#4C78A8"], title="Customers by tenure group")
    fig.update_traces(textinfo="percent+value")
    st.plotly_chart(fig, width="stretch")
    top = seg.loc[seg.Customers.idxmax()]
    insight(f"The largest segment is **{top['Tenure Group']}** ({top.Customers / seg.Customers.sum():.0%} of the selection). "
            "A loyal base of long-tenure customers sits alongside a fragile first-year group.")

with c2:
    avg = df.groupby(["Tenure Group", "Churn"], observed=True)["Monthly Charges"].mean().round(2).reset_index()
    fig = px.bar(avg, x="Tenure Group", y="Monthly Charges", color="Churn", barmode="group", text="Monthly Charges",
                 color_discrete_map=CHURN_COLORS, category_orders={"Tenure Group": TENURE_LABELS, "Churn": ["No", "Yes"]},
                 title="Average monthly charge by tenure group")
    fig.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
    fig.update_yaxes(range=[0, avg["Monthly Charges"].max() * 1.3])
    piv = avg.pivot(index="Tenure Group", columns="Churn", values="Monthly Charges")
    for g in piv.index:
        if {"Yes", "No"} <= set(piv.columns) and not piv.loc[g].isna().any():
            fig.add_annotation(x=g, y=piv.loc[g].max() * 1.15, showarrow=False, bgcolor="#FFF3CD",
                               text=f"Churners {piv.loc[g, 'Yes'] - piv.loc[g, 'No']:+.0f}$", font=dict(size=11))
    st.plotly_chart(fig, width="stretch")
    insight("Spend rises with tenure, but within each group **churners pay more than retained customers**, which "
            "points to price sensitivity as an early-churn driver. (Groups with very few churners give shaky averages.)")
st.divider()

# ================================================================= TASK 5: ADVANCED
st.header("3. Advanced analysis")

st.subheader("Tenure-group statistics")
grp = df.groupby("Tenure Group", observed=True).agg(
    customers=("customerID", "count"), churned=("Churned", "sum"),
    churn_rate_pct=("Churned", rate), avg_monthly=("Monthly Charges", "mean"),
    median_monthly=("Monthly Charges", "median"), avg_total=("Total Charges", "mean")).round(2)
st.dataframe(grp, width="stretch")
worst = grp["churn_rate_pct"].idxmax()
insight(f"Churn is highest in the **{worst}** group ({grp.loc[worst, 'churn_rate_pct']:.0f}%) and falls steadily "
        "as customers stay longer. Retention spend has the highest payoff in the first year.")

st.subheader("Churn rate by demographics, payment method and contract")
dim = st.radio("Dimension", ["gender", "Senior", "Contract", "Payment Method", "Internet Service"], horizontal=True)
t = df.groupby(dim).agg(customers=("Churned", "size"), churn_rate=("Churned", rate)).reset_index()
t["label"] = t["churn_rate"].round(0).astype(int).astype(str) + "% (n=" + t["customers"].astype(str) + ")"
fig = px.bar(t.sort_values("churn_rate", ascending=False), x=dim, y="churn_rate", text="label",
             color="churn_rate", color_continuous_scale="Reds", title=f"Churn rate by {dim}")
fig.add_hline(y=rate(df.Churned), line_dash="dash", annotation_text="overall")
fig.update_layout(coloraxis_showscale=False, yaxis_title="Churn rate (%)")
st.plotly_chart(fig, width="stretch")
hi, lo = t.loc[t.churn_rate.idxmax()], t.loc[t.churn_rate.idxmin()]
note = {
    "gender": "Gender shows little difference. It is not a useful targeting variable.",
    "Senior": "Senior-citizen churn is only slightly different and based on few customers, so it is not conclusive.",
    "Contract": "Month-to-month contracts drive nearly all churn, while one- and two-year customers almost never leave.",
    "Payment Method": "Electronic-check customers churn far more than those on automatic payment, which signals commitment.",
    "Internet Service": "Fiber-optic customers churn at a much higher rate than DSL customers, suggesting a price or quality issue.",
}[dim]
insight(f"Highest: **{hi[dim]}** ({hi.churn_rate:.0f}%); lowest: **{lo[dim]}** ({lo.churn_rate:.0f}%). {note}")

st.subheader("Lifecycle view: churn across tenure stages")
life = df.groupby("Lifecycle", observed=True).agg(customers=("Churned", "size"), churn_rate=("Churned", rate)).reset_index()
fig = go.Figure()
fig.add_bar(x=life["Lifecycle"], y=life["customers"], name="Customers", yaxis="y2", marker_color="#C9D6E8", opacity=0.7)
fig.add_scatter(x=life["Lifecycle"], y=life["churn_rate"], name="Churn rate (%)", mode="lines+markers+text",
                text=life["churn_rate"].round(0).astype(int).astype(str) + "%", textposition="top center",
                line=dict(color="#E45756", width=3))
fig.update_layout(xaxis_title="Tenure stage (months)", yaxis=dict(title="Churn rate (%)", range=[0, 110]),
                  yaxis2=dict(title="Customers", overlaying="y", side="right", showgrid=False),
                  legend=dict(orientation="h", y=-0.25), title="Churn rate by lifecycle stage")
st.plotly_chart(fig, width="stretch")
insight("Risk is front-loaded: churn is highest in the first year and decays with each later stage. "
        "The first months are the critical window for onboarding and retention offers.")

st.subheader("Contract × tenure heatmap")
heat = df.pivot_table(index="Contract", columns="Tenure Group", values="Churned", aggfunc="mean", observed=True) * 100
fig = px.imshow(heat.round(0), text_auto=True, color_continuous_scale="Reds", aspect="auto",
                labels=dict(color="Churn %"), title="Churn rate (%) by contract and tenure group")
st.plotly_chart(fig, width="stretch")
insight("The riskiest cohort is **new month-to-month customers**. Long-term contracts show almost no churn in any tenure group. "
        "Blank cells mean no customers in that combination.")
st.divider()

# ================================================================= Recommendations
st.header("4. Recommendations")
st.markdown("""
- **Onboard early:** proactive outreach in months 1-12, where churn is about 50%.
- **Convert contracts:** incentivise month-to-month customers to move to 1-2 year plans.
- **Promote auto-pay:** a small bill credit for switching from electronic check.
- **Review fiber pricing/quality:** and consider price-locks for high-spend bundles.
- **Target by behaviour, not demographics:** gender and age show no reliable effect.
""")
with st.expander("View filtered data"):
    st.dataframe(df.drop(columns=["Churned", "Senior", "Lifecycle"]), width="stretch")
st.caption("Note: the dataset has only 100 customers, so treat findings as directional.")
