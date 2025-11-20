import streamlit as st
from utils.load_data import load_energy_data
import utils.analysis_energy_production as analysis_energy_production

st.set_page_config(
    page_title="Electricity Production Analysis",
    page_icon="💡",
    layout="wide"
)
st.title("💡 Electricity Production Analysis")

production_data = load_energy_data()

#link to change price area
st.page_link(
    "pages/1_⚡_Electriciy_Production_Visualization.py",
    label="⬅️ Go to ⚡ Electricity Production Visualization to change Price Area"
)

price_area = st.session_state.get("price_area")
st.write(f"Here you can analyse the energy production data in the selected price area {price_area}.")

variable = st.selectbox(
        "What data would you like to visualize?",
        ("hydro", "solar", "thermal", "wind", "other"),
        index=0, 
        placeholder="Select an option",
        help="Choose the energy production type you want to see analyzed"
)

tab1, tab2 = st.tabs(["LOESS", "Spectrogram"])

with tab1:
    st.write("LOESS")
    #make buttons to decide between daily and weekly decomposition
    decomposition_type = st.pills(
        "Select decomposition type:",
        options = ["daily", "weekly", "monthly"],
        default="weekly",
        selection_mode="single",
        help="Choose the decomposition type for the LOESS analysis. Daily focuses on daily patterns, weekly on weekly trends, monthly on monthly cycles."
    )
    
    model_params = {
        "daily": {"period_length": 24, "seasonal_smoothing": 13, "trend_smoothing": 3 * 24 + 1},
        "weekly": {"period_length": 24*7, "seasonal_smoothing": 25, "trend_smoothing": 3 * 24*7 + 1},
        "monthly": {"period_length": 24*30, "seasonal_smoothing": 24*7+1, "trend_smoothing": 3 * 24*30 + 1},
    }
    st.info("Note: LOESS decomposition can be computationally intensive for large datasets. Please be patient while the analysis is being performed.")
    with st.spinner("Processing LOESS decomposition..."):
        fig = analysis_energy_production.LOESS_energy_production(production_data, price_area=price_area, production_group=variable, **model_params[decomposition_type])
    st.pyplot(fig)


with tab2:
    st.write("Spectrogram")
    st.info("Note: Spectrogram generation can be computationally intensive for large datasets. Please be patient while the analysis is being performed.")
    # Spectrogram parameters
    # slider to adjust NFFT
    NFFT = st.slider("NFFT (Number of data points used in each block for the FFT):", min_value=24*7, max_value=24*7*8, value=24*7*4, step=24*7, help="Higher values provide better frequency resolution but worse time resolution.")
    # slider for noverlap as percent of NFFT
    noverlap_percent = st.slider("Overlap Percentage:", min_value=0, max_value=90, value=50, step=10, help="Percentage of overlap between segments. Higher overlap provides smoother spectrograms but increases computation time.")
    noverlap = NFFT * noverlap_percent // 100

    with st.spinner("Processing Spectrogram..."):
        fig, Pxx, freqs, bins = analysis_energy_production.spectrogram_energy_production(production_data, price_area=price_area, production_group=variable, NFFT=NFFT, noverlap=noverlap)
    st.plotly_chart(fig)