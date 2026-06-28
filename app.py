import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import plotly.figure_factory as ff

st.set_page_config(
    page_title="Customer Churn Analytics Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# # ----------------------------
# PAGE CONFIGURATION
# ----------------------------


# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("cleaned_data.csv")



# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.title("🔍 Filters")

geography = st.sidebar.multiselect(
    "Select Geography",
    options=df["Geography"].unique(),
    default=df["Geography"].unique()
)

gender = st.sidebar.multiselect(
    "Select Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

age = st.sidebar.slider(
    "Select Age",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (int(df["Age"].min()), int(df["Age"].max()))
)

filtered = df[
    (df["Geography"].isin(geography)) &
    (df["Gender"].isin(gender)) &
    (df["Age"] >= age[0]) &
    (df["Age"] <= age[1])
]

# ----------------------------
# HEADER
# ----------------------------

left, right = st.columns([5,1])

with left:

    st.title("🏦 Customer Churn Analytics Dashboard")

    st.write(
        "An interactive dashboard for analyzing customer churn in a European Banking dataset."
    )

with right:

    st.info(
        f"📅 Last Updated\n\n{datetime.now().strftime('%d %b %Y %I:%M %p')}"
    )

st.divider()

# ----------------------------
# KPI SECTION
# ----------------------------

customers = len(filtered)

churned = filtered["Exited"].sum()

rate = churned/customers*100

credit = filtered["CreditScore"].mean()

balance = filtered["Balance"].mean()

c1,c2,c3,c4,c5 = st.columns(5)

c1.metric(
    "👥 Total Customers",
    f"{customers:,}"
)

c2.metric(
    "❌ Churned",
    f"{churned:,}"
)

c3.metric(
    "📉 Churn Rate",
    f"{rate:.2f}%"
)

c4.metric(
    "💳 Avg Credit",
    round(credit,2)
)

c5.metric(
    "💰 Avg Balance",
    f"€ {balance:,.0f}"
)

st.divider()


import joblib
import numpy as np

st.markdown("---")

st.header("🔮 Customer Churn Prediction")

# Load trained model
model = joblib.load("random_forest_model.pkl")

col1, col2 = st.columns([2,1])

with col1:

    credit = st.number_input(
        "Credit Score",
        300,
        900,
        650
    )

    geography = st.selectbox(
        "Geography",
        ["France","Germany","Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    age = st.slider(
        "Age",
        18,
        92,
        35
    )

    tenure = st.slider(
        "Tenure",
        0,
        10,
        5
    )

    balance = st.number_input(
        "Balance",
        0.0,
        250000.0,
        60000.0
    )

    products = st.selectbox(
        "Number of Products",
        [1,2,3,4]
    )

    has_card = st.selectbox(
        "Has Credit Card",
        ["Yes","No"]
    )

    active = st.selectbox(
        "Is Active Member",
        ["Yes","No"]
    )

    salary = st.number_input(
        "Estimated Salary",
        0.0,
        300000.0,
        70000.0
    )
    geo_germany = 1 if geography=="Germany" else 0
geo_spain = 1 if geography=="Spain" else 0

gender = 1 if gender=="Male" else 0

has_card = 1 if has_card=="Yes" else 0

active = 1 if active=="Yes" else 0

predict = st.button("Predict Churn")

if predict:

    data = np.array([[
        credit,
        gender,
        age,
        tenure,
        balance,
        products,
        has_card,
        active,
        salary,
        geo_germany,
        geo_spain
    ]])

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    with col2:

        st.subheader("Prediction Result")

        if predict:

            if prediction==1:

                st.error("⚠ Customer is likely to CHURN")

            else:

                st.success("✅ Customer is likely to STAY")

            st.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )

            st.progress(float(probability))

            st.write("### Recommendation")

            if probability>0.70:

                st.warning(
                "Offer special retention discounts and personalized support."
            )

            elif probability>0.40:

                st.info(
                "Customer should be monitored closely."
            )

            else:

                st.success(
                "Customer appears loyal."
            )
                st.markdown("""
<style>

div[data-testid="metric-container"]{

background-color:#1f2937;

padding:15px;

border-radius:12px;

border:1px solid #3b82f6;

}

</style>
""",unsafe_allow_html=True)
                st.markdown("---")
st.header("📊 Customer Analytics Dashboard")
col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        filtered,
        x="Geography",
        color="Exited",
        barmode="group",
        title="🌍 Customer Churn by Geography",
        color_discrete_sequence=["#4CAF50", "#F44336"]
    )

    fig.update_layout(
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)
with col2:

    gender_df = (
        filtered.groupby("Gender")["Exited"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        gender_df,
        names="Gender",
        values="Exited",
        hole=0.6,
        title="👨‍💼 Gender-wise Churn"
    )

    fig.update_layout(
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)
col3, col4 = st.columns(2)

with col3:

    fig = px.histogram(
        filtered,
        x="Age",
        nbins=30,
        color="Exited",
        title="📈 Age Distribution",
        color_discrete_sequence=["#00BCD4", "#FF9800"]
    )

    fig.update_layout(
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)
with col4:

    fig = px.histogram(
        filtered,
        x="CreditScore",
        nbins=30,
        title="💳 Credit Score Distribution",
        color_discrete_sequence=["#7E57C2"]
    )

    fig.update_layout(
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)
col5, col6 = st.columns(2)

with col5:

    fig = px.histogram(
        filtered,
        x="Balance",
        nbins=25,
        title="💰 Balance Distribution",
        color_discrete_sequence=["#26A69A"]
    )

    fig.update_layout(
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)
with col6:

    fig = px.scatter(
        filtered,
        x="Age",
        y="Balance",
        color="Exited",
        size="EstimatedSalary",
        hover_data=["CreditScore"],
        title="📌 Age vs Balance",
        color_discrete_sequence=["#2196F3", "#E91E63"]
    )

    fig.update_layout(
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header("📊 Advanced Customer Analytics")

# ==============================
# Row 1
# ==============================

col1, col2 = st.columns(2)

# ------------------------------
# Balance by Gender (Box Plot)
# ------------------------------
with col1:

    fig = px.box(
        filtered,
        x="Gender",
        y="Balance",
        color="Gender",
        title="💰 Balance by Gender",
        points="outliers"
    )

    fig.update_layout(
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Products Used
# ------------------------------
with col2:

    product = (
        filtered.groupby("NumOfProducts")
        .size()
        .reset_index(name="Customers")
    )

    fig = px.bar(
        product,
        x="NumOfProducts",
        y="Customers",
        color="Customers",
        title="📦 Product Usage"
    )

    fig.update_layout(
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

# ==============================
# Row 2
# ==============================

col3, col4 = st.columns(2)

# ------------------------------
# Active vs Inactive
# ------------------------------
with col3:

    active = (
        filtered.groupby("IsActiveMember")
        .size()
        .reset_index(name="Customers")
    )

    fig = px.pie(
        active,
        names="IsActiveMember",
        values="Customers",
        hole=0.60,
        title="🟢 Active vs Inactive Members"
    )

    fig.update_layout(
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Tenure Distribution
# ------------------------------
with col4:

    fig = px.histogram(
        filtered,
        x="Tenure",
        color="Exited",
        title="📅 Tenure Distribution",
        nbins=10
    )

    fig.update_layout(
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

# ==============================
# Correlation Heatmap
# ==============================

st.subheader("🔥 Correlation Heatmap")

corr_df = filtered.copy()

# Convert categorical columns
corr_df["Gender"] = corr_df["Gender"].map(
    {"Male":1,"Female":0}
)

corr_df["Geography"] = corr_df["Geography"].astype("category").cat.codes

corr = corr_df.corr(numeric_only=True)

fig = px.imshow(
    corr,
    text_auto=False,          # Hide numbers
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    aspect="auto"
)

fig.update_layout(
    title="Correlation Heatmap",
    template="plotly_dark",
    height=700
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# Salary Distribution
# ==============================

st.subheader("💼 Estimated Salary Distribution")

fig = px.histogram(
    filtered,
    x="EstimatedSalary",
    nbins=30,
    color="Exited",
    title="Salary Distribution"
)

fig.update_layout(
    template="plotly_dark",
    height=450
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# Business Insights
# ==============================

st.subheader("📌 Business Insights")

total = len(filtered)
churn = filtered["Exited"].sum()
rate = churn / total * 100

highest_geo = (
    filtered.groupby("Geography")["Exited"]
    .mean()
    .idxmax()
)

avg_balance = filtered["Balance"].mean()

avg_credit = filtered["CreditScore"].mean()

st.info(f"""
### Key Insights

• Total Customers: **{total:,}**

• Churn Rate: **{rate:.2f}%**

• Highest Churn Geography: **{highest_geo}**

• Average Balance: **€ {avg_balance:,.2f}**

• Average Credit Score: **{avg_credit:.2f}**
""")
import io

st.markdown("---")
st.header("📋 Customer Insights & Reports")

# ==================================
# SEARCH CUSTOMER
# ==================================

st.subheader("🔍 Search Customer")

customer_id = st.text_input("Enter Customer ID")

if customer_id:

    customer = filtered[
        filtered["CustomerId"].astype(str) == customer_id
    ]

    if len(customer) > 0:
        st.success("Customer Found")
        st.dataframe(customer, use_container_width=True)
    else:
        st.error("Customer Not Found")

st.markdown("---")

# ==================================
# CUSTOMER TABLE
# ==================================

st.subheader("📄 Customer Dataset")

show = st.slider(
    "Rows to Display",
    5,
    100,
    20
)

st.dataframe(
    filtered.head(show),
    use_container_width=True
)

st.markdown("---")

# ==================================
# DOWNLOAD CSV
# ==================================

st.subheader("📥 Download Filtered Dataset")

csv = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download CSV",
    data=csv,
    file_name="Filtered_Customers.csv",
    mime="text/csv"
)

st.markdown("---")

# ==================================
# BUSINESS SUMMARY
# ==================================

st.header("📊 Business Summary")

total = len(filtered)

churn = filtered["Exited"].sum()

stay = total - churn

rate = churn / total * 100

active = filtered["IsActiveMember"].sum()

inactive = total - active

avg_balance = filtered["Balance"].mean()

avg_credit = filtered["CreditScore"].mean()

col1, col2, col3 = st.columns(3)

col1.info(f"""
### 👥 Customers

Total : {total:,}

Stayed : {stay:,}

Churned : {churn:,}
""")

col2.success(f"""
### 💰 Financial

Average Balance

€ {avg_balance:,.2f}

Average Credit

{avg_credit:.2f}
""")

col3.warning(f"""
### 📉 Churn

Rate

{rate:.2f} %

Active Members

{active:,}

Inactive Members

{inactive:,}
""")

st.markdown("---")

# ==================================
# RECOMMENDATIONS
# ==================================

st.header("💡 Business Recommendations")

recommendations = []

if rate > 25:
    recommendations.append(
        "Increase customer retention campaigns because churn rate is high."
    )

if avg_credit < 600:
    recommendations.append(
        "Provide credit improvement programs for low-score customers."
    )

if inactive > active:
    recommendations.append(
        "Focus on engaging inactive customers with loyalty programs."
    )

if avg_balance < 50000:
    recommendations.append(
        "Introduce premium savings accounts and investment products."
    )

if len(recommendations) == 0:
    recommendations.append(
        "Current customer portfolio appears healthy. Continue monitoring."
    )

for rec in recommendations:
    st.write("✅", rec)

st.markdown("---")

# ==================================
# PROJECT INFORMATION
# ==================================

st.header("📌 Project Information")

st.markdown("""
### Customer Churn Analytics in European Banking

**Tech Stack**

- Python
- Streamlit
- Pandas
- Plotly
- Scikit-Learn
- Machine Learning

**Modules**

✔ Data Cleaning

✔ Exploratory Data Analysis

✔ Customer Segmentation

✔ Machine Learning

✔ Model Evaluation

✔ Business Dashboard
""")

st.markdown("---")

# ==================================
# FOOTER
# ==================================

st.markdown(
"""
---
<center>

### Developed by

**Akshata**

B.Tech Artificial Intelligence

Delhi Skill and Entrepreneurship University (DSEU)

Customer Churn Analytics in European Banking

© 2026 All Rights Reserved

</center>
""",
unsafe_allow_html=True
)