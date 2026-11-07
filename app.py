
import streamlit as st
import pandas as pd
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Hotel Cancellation Prediction",
    page_icon="🏨",
    layout="centered"
)

# Load trained model
model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hotel_booking_model.pkl"
)

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Application title
st.title("🏨 Hotel Booking Cancellation Prediction")

st.write(
    "Enter the booking details to predict whether "
    "the hotel booking may be cancelled."
)

st.divider()

# Booking details
st.subheader("Booking Information")

col1, col2 = st.columns(2)

with col1:
    adults = st.number_input(
        "Number of Adults", min_value=0, value=2
    )

    children = st.number_input(
        "Number of Children", min_value=0, value=0
    )

    weekend = st.number_input(
        "Weekend Nights", min_value=0, value=1
    )

    week = st.number_input(
        "Week Nights", min_value=0, value=2
    )

    meal = st.selectbox(
        "Meal Type",
        ["BB", "HB", "SC", "FB", "Undefined"]
    )

    parking = st.number_input(
        "Car Parking Spaces", min_value=0, value=0
    )

    room = st.selectbox(
        "Reserved Room Type",
        ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
    )

with col2:
    lead = st.number_input(
        "Lead Time (Days)", min_value=0, value=30
    )

    market = st.selectbox(
        "Market Segment",
        [
            "Online TA",
            "Offline TA/TO",
            "Direct",
            "Corporate",
            "Groups",
            "Complementary",
            "Aviation",
            "Undefined"
        ]
    )

    repeated = st.selectbox(
        "Repeated Guest",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    pc = st.number_input(
        "Previous Cancellations", min_value=0, value=0
    )

    pnotc = st.number_input(
        "Previous Bookings Not Cancelled", min_value=0, value=0
    )

    price = st.number_input(
        "Average Daily Rate", min_value=0.0, value=100.0
    )

    requests = st.number_input(
        "Special Requests", min_value=0, value=0
    )

st.divider()

# Prediction button
if st.button("🔍 Predict Cancellation", use_container_width=True):

    # Prepare input data
    data = pd.DataFrame([{
        "adults": adults,
        "children": children,
        "stays_in_weekend_nights": weekend,
        "stays_in_week_nights": week,
        "meal": meal,
        "required_car_parking_spaces": parking,
        "reserved_room_type": room,
        "lead_time": lead,
        "market_segment": market,
        "is_repeated_guest": repeated,
        "previous_cancellations": pc,
        "previous_bookings_not_canceled": pnotc,
        "adr": price,
        "total_of_special_requests": requests
    }])

    # Make prediction
    prediction = model.predict(data)[0]

    probabilities = model.predict_proba(data)[0]

    # Find cancellation probability (class 1)
    classes = list(model.classes_)
    cancellation_probability = probabilities[classes.index(1)]

    # Display result
    st.subheader("Prediction Result")

    st.metric(
        "Cancellation Probability",
        f"{cancellation_probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error("⚠️ Booking is predicted to be CANCELLED")
    else:
        st.success("✅ Booking is predicted NOT TO BE CANCELLED")

    st.progress(float(cancellation_probability))

    st.caption(
        "This prediction is based on the trained Random Forest model."
    )
