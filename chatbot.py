import streamlit as st
import pandas as pd
import plotly.express as px
import re

from transformers import pipeline

# --------------------------
# PAGE CONFIG
# --------------------------

st.set_page_config(
    page_title="Health Analytics Dashboard",
    page_icon="🩺",
    layout="wide"
)

# --------------------------
# LOAD DATA
# --------------------------

@st.cache_data
def load_data():
    return pd.read_csv("df_user.csv")

df = load_data()

# --------------------------
# LOAD MODEL
# --------------------------

@st.cache_resource
def load_model():

    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct"
    )

    return pipe

pipe = load_model()

# --------------------------
# SIDEBAR
# --------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select",
    [
        "Dashboard",
        "User Analysis",
        "AI Chat",
        "Dataset"
    ]
)

# ==========================
# DASHBOARD
# ==========================

if page=="Dashboard":

    st.title("🩺 Health Analytics Dashboard")

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Users",len(df))
    c2.metric("Average BMI",round(df["bmi_mean"].mean(),2))
    c3.metric("Average Sleep",round(df["sleep_hours_mean"].mean(),2))
    c4.metric("Average Steps",int(df["steps_mean"].mean()))

    st.divider()

    col1,col2=st.columns(2)

    with col1:

        fig=px.histogram(
            df,
            x="bmi_mean",
            nbins=30,
            title="BMI Distribution"
        )

        st.plotly_chart(fig,use_container_width=True)

    with col2:

        fig=px.histogram(
            df,
            x="sleep_hours_mean",
            nbins=30,
            title="Sleep Distribution"
        )

        st.plotly_chart(fig,use_container_width=True)

    col3,col4=st.columns(2)

    with col3:

        fig=px.pie(
            df,
            names="health_tier",
            title="Health Tier"
        )

        st.plotly_chart(fig,use_container_width=True)

    with col4:

        fig=px.histogram(
            df,
            x="risk_probability",
            title="Risk Probability"
        )

        st.plotly_chart(fig,use_container_width=True)

    st.subheader("BMI vs Health Score")

    fig=px.scatter(
        df,
        x="bmi_mean",
        y="health_score",
        color="health_tier",
        hover_data=["user_id"]
    )

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Top 10 High Risk Users")

    st.dataframe(

        df.nlargest(
            10,
            "risk_probability"
        )[[
            "user_id",
            "risk_probability",
            "health_score",
            "health_tier"
        ]]

    )

# ==========================
# USER ANALYSIS
# ==========================

elif page=="User Analysis":

    st.title("👤 Individual User Analysis")

    uid=st.number_input(
        "User ID",
        min_value=1,
        step=1
    )

    if st.button("Analyze"):

        user=df[df.user_id==uid]

        if len(user)==0:

            st.error("User not found.")

        else:

            record=user.to_dict(
                orient="records"
            )[0]

            st.subheader("User Record")

            st.json(record)

            prompt=f"""
You are a healthcare analyst.

Analyze the following patient.

{record}

Explain

1 Overall Health

2 Risk Level

3 Recommendations

4 Lifestyle Suggestions

5 Diet Suggestions

"""

            with st.spinner("Generating AI Report..."):

                out=pipe(
                    [
                        {
                            "role":"user",
                            "content":prompt
                        }
                    ],
                    max_new_tokens=300
                )

            answer=out[0]["generated_text"][-1]["content"]

            st.success(answer)

# ==========================
# AI CHAT
# ==========================

elif page=="AI Chat":

    st.title("🤖 Health Assistant")

    question=st.text_area("Ask anything")

    if st.button("Ask"):

        with st.spinner():

            out=pipe(
                [
                    {
                        "role":"user",
                        "content":question
                    }
                ],
                max_new_tokens=250
            )

        st.write(out[0]["generated_text"][-1]["content"])

# ==========================
# DATASET
# ==========================

else:

    st.title("📋 Dataset")

    st.dataframe(df)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "health_data.csv"
    )
