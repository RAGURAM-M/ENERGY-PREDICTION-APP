import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.title("⚡ Energy Consumption Prediction Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    processed_df = df.copy()

    # Remove duplicates
    processed_df.drop_duplicates(inplace=True)

    # Handle Date column
    if 'Date' in processed_df.columns:
        processed_df['Date'] = pd.to_datetime(
            processed_df['Date'], errors='coerce'
        )
        processed_df['Days'] = (
            processed_df['Date'] - processed_df['Date'].min()
        ).dt.days

    # Fill missing values
    for col in processed_df.columns:
        if processed_df[col].dtype == 'object':
            processed_df[col].fillna(
                processed_df[col].mode()[0], inplace=True
            )
        else:
            processed_df[col].fillna(
                processed_df[col].mean(), inplace=True
            )

    # Encode categorical columns
    for col in processed_df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col])

    st.success("Data Preprocessing Completed ✅")

    # Select target
    target_column = st.selectbox(
        "Select Target Column",
        processed_df.columns
    )

    # Train model
    if st.button("Train Model"):
        X = processed_df.drop(columns=[target_column])
        y = processed_df[target_column]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        feature_columns = X.columns

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.write("## 📊 Model Metrics")
        st.write(f"R² Score: {r2:.3f}")
        st.write(f"MAE: {mae:.3f}")
        st.write(f"RMSE: {rmse:.3f}")

        st.write("## 🔍 Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        })
        st.dataframe(importance_df)

        # Prediction section
        days = st.number_input(
            "Enter number of future days to predict",
            min_value=1,
            step=1
        )

        if st.button("Predict Future"):
            last_row = X.iloc[-1].copy()
            predictions = []

            for i in range(days):
                new_row = last_row.copy()

                for col in feature_columns:
                    new_row[col] += np.random.uniform(-0.5, 0.5)

                new_row_scaled = scaler.transform([new_row])
                pred = model.predict(new_row_scaled)[0]

                predictions.append(round(pred, 2))
                last_row = new_row

            st.write("## 🔮 Predictions")
            for i, p in enumerate(predictions):
                st.write(f"Day {i+1}: {p}")

            # Graph
            st.write("## 📈 Prediction Graph")
            fig, ax = plt.subplots()
            ax.plot(predictions, marker='o')
            ax.set_title("Future Predictions")
            st.pyplot(fig)

            # Bar chart
            st.write("## 📊 Bar Chart")
            fig2, ax2 = plt.subplots()
            ax2.bar(range(len(predictions)), predictions)
            st.pyplot(fig2)