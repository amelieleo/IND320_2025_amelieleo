import streamlit as st 

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Welcome to the Energy App")

st.write("This app allows you to explore and visualize weather and energy production data.")

st.image("https://t4.ftcdn.net/jpg/02/40/24/81/360_F_240248152_piluBt47ZD46vprw7C0xQ88Lk4zXLg81.jpg")
st.markdown('<span style="color:grey; font-size:12px;">Source: Adobe Stock</span>', unsafe_allow_html=True)

#initialize session state variables
if "price_area" not in st.session_state:
    st.session_state.price_area = "NO1" # default price area
    