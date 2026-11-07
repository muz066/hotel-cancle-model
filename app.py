
import streamlit as st
import pandas as pd
import pickle
import os
import subprocess
import sys

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Hotel Cancellation Prediction",
    page_icon="🏨",
    layout="centered"
)

# Project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
model_path = os.path.join(BASE_DIR, "hotel_booking_model.pkl")
train_script = os.path.join(BASE_DIR, "train_model.py")
dataset_path = os.path.join(BASE_DIR, "hotel_bookings.csv")


# ==========================================
# MODEL TRAINING AND LOADING
# ==========================================

@st.cache_resource
def load_model():

    # Check whether model already exists
    if not os.path.exists(model_path):

        st.warning("Trained model not found. Preparing model...")

        # Check training script
        if not os.path.exists(train_script):
            raise FileNotFoundError(
                "train_model.py is missing from the project folder."
            )

        # Check dataset
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                "hotel_bookings.csv is missing from the project folder."
            )

        # Train model automatically
        with st.spinner(
            "Training the machine learning model. Please wait..."
        ):

            result = subprocess.run(
                [sys.executable, train_script],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                raise RuntimeError(
                    "Model training failed:\n" + result.stderr
                )

        # Verify model creation
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Training finished, but hotel_booking_model.pkl "
                "was not created. Check train_model.py."
            )

    # Load trained model
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


# Load model safely
try:
    model = load_model()

except Exception as e:
    st.error("Unable to load the prediction model.")
    st.exception(e)
    st.stop()


# ==========================================
# APPLICATION TITLE
# ==========================================

st.title("🏨 Hotel Booking Cancellation Prediction")

st.write(
    "Enter the booking details to predict whether "
    "the hotel booking may be cancelled."
)

st.divider()


# ==========================================
# BOOKING INFORMATION
# ==========================================

st.subheader("Booking Information")

col1, col2 = st.columns(2)

with col1:

    adults = st.number_input(
        "Number of Adults",
        min_value=0,
        value=2
    )

    children = st.number_input(
        "Number of Children",
        min_value=0,
        value=0
    )

    weekend = st.number_input(
        "Weekend Nights",
        min_value=0,
        value=1
    )

    week = st.number_input(
        "Week Nights",
        min_value=0,
        value=2
    )

    meal = st.selectbox(
        "Meal Type",
        ["BB", "HB", "SC", "FB", "Undefined"]
    )

    parking = st.number_input(
        "Car Parking Spaces",
        min_value=0,
        value=0
    )

    room = st.selectbox(
        "Reserved Room Type",
        ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
    )


with col2:

    lead = st.number_input(
        "Lead Time (Days)",
        min_value=0,
        value=30
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
        "Previous Cancellations",
        min_value=0,
        value=0
    )

    pnotc = st.number_input(
        "Previous Bookings Not Cancelled",
        min_value=0,
        value=0
    )

    price = st.number_input(
        "Average Daily Rate",
        min_value=0.0,
        value=100.0
    )

    requests = st.number_input(
        "Special Requests",
        min_value=0,
        value=0
    )


st.divider()


# ==========================================
# PREDICTION
# ==========================================

if st.button(
    "🔍 Predict Cancellation",
    use_container_width=True
):

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

    try:

        # Make prediction
        prediction = model.predict(data)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(data)[0]

        classes = list(model.classes_)

        if 1 not in classes:
            st.error(
                "Cancellation class 1 was not found in the trained model."
            )
            st.stop()

        cancellation_probability = probabilities[
            classes.index(1)
        ]

        # Display results
        st.subheader("Prediction Result")

        st.metric(
            "Cancellation Probability",
            f"{cancellation_probability * 100:.2f}%"
        )

        if prediction == 1:

            st.error(
                "⚠️ Booking is predicted to be CANCELLED"
            )

        else:

            st.success(
                "✅ Booking is predicted NOT TO BE CANCELLED"
            )

        st.progress(
            float(cancellation_probability)
        )

        st.caption(
            "This prediction is based on the trained "
            "Random Forest machine learning model."
        )

    except Exception as e:

        st.error("Prediction failed.")
        st.exception(e)
