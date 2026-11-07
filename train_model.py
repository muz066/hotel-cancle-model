
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Starting Hotel Booking Model Training...")

# 1. Load dataset
file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hotel_bookings.csv"
)

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Total records:", len(df))

# 2. Select input columns
features = [
    "adults",
    "children",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "meal",
    "required_car_parking_spaces",
    "reserved_room_type",
    "lead_time",
    "market_segment",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "adr",
    "total_of_special_requests"
]

target = "is_canceled"

# 3. Check columns
missing_columns = [
    col for col in features + [target]
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

# 4. Prepare input and output
X = df[features].copy()
y = df[target].astype(int)

# Convert children to numeric
X["children"] = pd.to_numeric(
    X["children"], errors="coerce"
)

# Remove invalid target rows
valid_rows = y.isin([0, 1])
X = X.loc[valid_rows]
y = y.loc[valid_rows]

# 5. Define categorical and numerical columns
categorical_features = [
    "meal",
    "reserved_room_type",
    "market_segment"
]

numerical_features = [
    col for col in features
    if col not in categorical_features
]

# 6. Preprocessing
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 7. Create ML model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

# 8. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 9. Train model
print("Training model... Please wait.")

model.fit(X_train, y_train)

# 10. Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL TRAINING COMPLETED!")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 11. Save model
output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hotel_booking_model.pkl"
)

with open(output_path, "wb") as file:
    pickle.dump(model, file)

print("\nSUCCESS!")
print("Model saved at:", output_path)
