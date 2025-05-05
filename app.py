import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from granite_helper import GigAI
from datetime import datetime
import html

# Page configuration
st.set_page_config(
    page_title="Gig Worker Hub",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stButton>button {
    background-color: #0068c9;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: bold;
}
.recommendation-box {
    padding: 1.5rem;
    border-radius: 10px;
    background-color: transparent;
    border-left: 4px solid #0068c9;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.recommendation-content {
    line-height: 1.6;
    font-size: 15px;
}
.recommendation-header {
    color: #0068c9;
    margin-top: 0;
    margin-bottom: 1rem;
    font-size: 1.2rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

def display_recommendations(advice):
    """Display formatted recommendations"""
    if advice.startswith("Error"):
        st.error(advice)
    else:
        formatted_advice = html.escape(advice).replace("\n- ", "<br>• ").replace("\n", "<br>")
        st.markdown(
            "<div class='recommendation-box'>"
            "<h3 class='recommendation-header'>📊 Your Personalized Recommendations</h3>"
            f"<div class='recommendation-content'>{formatted_advice}</div>"
            "</div>",
            unsafe_allow_html=True
        )

@st.cache_resource
def init_gigai():
    return GigAI()

@st.cache_data
def load_sample_data():
    dates = pd.date_range("2023-01-01", periods=90)
    return pd.DataFrame({
        "date": dates,
        "earnings": np.random.randint(50, 200, size=90),
        "hours": np.random.randint(2, 8, size=90),
        "platform": np.random.choice(["Uber", "DoorDash", "Lyft"], size=90),
        "miles": np.random.uniform(5, 50, size=90).round(1)
    })

def validate_data(df):
    try:
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        required = {"date", "platform", "hours", "earnings"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isnull().any():
            raise ValueError("Invalid date format in date column")
        return df
    except Exception as e:
        st.error(f"Data validation error: {str(e)}")
        return None

def main():
    gig_ai = init_gigai()
    
    st.title("🧠 Gig Worker Analytics Hub")
    st.markdown("Optimize your gig work performance with AI-powered insights")
    
    if "data" not in st.session_state:
        st.session_state.data = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    with st.sidebar:
        st.header("📂 Data Input")
        uploaded_file = st.file_uploader("Upload your gig work data (CSV)", type=["csv"])
        col1, col2 = st.columns(2)
        with col1:
            use_sample = st.checkbox("Use sample data")
        with col2:
            if st.button("Load Data", type="primary"):
                with st.spinner("Processing data..."):
                    try:
                        if use_sample:
                            st.session_state.data = load_sample_data()
                            st.success("Sample data loaded!")
                        elif uploaded_file:
                            df = pd.read_csv(uploaded_file)
                            df = validate_data(df)
                            if df is not None:
                                st.session_state.data = df
                                st.success("Data loaded successfully!")
                        else:
                            st.warning("Please upload a file or use sample data")
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")

        with st.expander("Download sample CSV"):
            sample_df = load_sample_data()
            st.download_button(
                label="Download sample data",
                data=sample_df.to_csv(index=False),
                file_name="gig_work_sample.csv",
                mime="text/csv"
            )

    if st.session_state.data is not None:
        data = st.session_state.data
        st.subheader("📊 Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Earnings", f"${data['earnings'].sum():,}")
        col2.metric("Total Hours", f"{data['hours'].sum():,} hrs")
        hourly = data['earnings'].sum() / max(1, data['hours'].sum())
        col3.metric("Avg Hourly Rate", f"${hourly:.2f}")
        if 'miles' in data.columns:
            mile_rate = data['earnings'].sum() / max(1, data['miles'].sum())
            col4.metric("Earnings/Mile", f"${mile_rate:.2f}")
        else:
            col4.metric("Total Jobs", len(data))
        
        tab1, tab2, tab3 = st.tabs(["📈 Trends", "🔍 Breakdown", "🗺️ Map View"])
        with tab1:
            weekly = data.resample('W', on='date').sum(numeric_only=True)
            fig = px.line(weekly, x=weekly.index, y="earnings", title="Weekly Earnings Trend")
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            platform_stats = data.groupby("platform").agg({
                "earnings": ["sum", "mean"],
                "hours": ["sum", "mean"]
            })
            st.dataframe(platform_stats.style.format({
                ('earnings', 'sum'): "${:,.2f}",
                ('earnings', 'mean'): "${:.2f}",
                ('hours', 'sum'): "{:,.1f} hrs",
                ('hours', 'mean'): "{:.1f} hrs"
            }), use_container_width=True)

        st.subheader("🤖 AI Recommendations")
        if st.button("Get Personalized Advice", type="primary"):
            with st.spinner("Analyzing your gig patterns..."):
                advice = gig_ai.predict_schedule(data)
                display_recommendations(advice)
        
        st.subheader("💬 Gig Work Advisor")
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
        if prompt := st.chat_input("Ask about gig work strategies..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Generating response..."):
                response = gig_ai.ask_question(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

if __name__ == "__main__":
    main()
