"""
app.py - Streamlit app for Spaceship Titanic prediction
Run with: streamlit run app.py
"""
 
import streamlit as st
from source.pipeline import predict_single
 
# page config 
st.set_page_config(
    page_title="ASG 04 MD - Indira - Spaceship Titanic",
    page_icon="🚀",
    layout="centered",
)
 
st.title("ASG 04 MD - Indira - Spaceship Titanic Model Deployment")
st.markdown(
    "Enter passenger details below to predict whether they were "
    "**transported to an alternate dimension**."
)
st.divider()
 
# input form 
st.subheader("Passenger Information")
 
col1, col2 = st.columns(2)
 
with col1:
    home_planet = st.selectbox(
        "Home Planet",
        options=["Earth", "Europa", "Mars"],
        index=0,
    )
    cryo_sleep = st.selectbox(
        "CryoSleep",
        options=[False, True],
        format_func=lambda x: "Yes" if x else "No",
        index=0,
    )
    destination = st.selectbox(
        "Destination",
        options=["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"],
        index=0,
    )
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=27.0, step=1.0)
    vip = st.selectbox(
        "VIP",
        options=[False, True],
        format_func=lambda x: "Yes" if x else "No",
        index=0,
    )
 
with col2:
    cabin = st.text_input(
        "Cabin (format: Deck/Num/Side, e.g. F/100/S)",
        value="F/100/S",
        help="Deck: A–G, Side: P or S",
    )
    room_service = st.number_input("Room Service Spending ($)", min_value=0.0, value=0.0, step=10.0)
    food_court = st.number_input("Food Court Spending ($)", min_value=0.0, value=0.0, step=10.0)
    shopping_mall = st.number_input("Shopping Mall Spending ($)", min_value=0.0, value=0.0, step=10.0)
    spa = st.number_input("Spa Spending ($)", min_value=0.0, value=0.0, step=10.0)
    vr_deck = st.number_input("VR Deck Spending ($)", min_value=0.0, value=0.0, step=10.0)
 
st.divider()
 
# prediction 
if st.button(" Predict Transport Status", use_container_width=True, type="primary"):
    features = {
        "HomePlanet": home_planet,
        "CryoSleep": cryo_sleep,
        "Cabin": cabin,
        "Destination": destination,
        "Age": age,
        "VIP": vip,
        "RoomService": room_service,
        "FoodCourt": food_court,
        "ShoppingMall": shopping_mall,
        "Spa": spa,
        "VRDeck": vr_deck,
    }
 
    with st.spinner("Predicting..."):
        try:
            result = predict_single(features)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
 
    st.divider()
    if result:
        st.success("**TRANSPORTED** — This passenger was transported to an alternate dimension!")
        st.balloons()
    else:
        st.error("**NOT TRANSPORTED** — This passenger was NOT transported.")
 
# footer
st.divider()
st.caption("Model: Logistic Regression (Optuna-optimized) | Dataset: Spaceship Titanic (Kaggle)")