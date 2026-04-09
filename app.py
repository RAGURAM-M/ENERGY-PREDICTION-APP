import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------- TITLE ---------------- #
st.set_page_config(page_title="Energy Prediction Dashboard", layout="wide")
st.title("⚡ Energy Consumption Prediction Dashboard")

# ---------------- FILE UPLOAD ---------------- #
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# ---------------- MAIN PROCESS ---------------- #
if uploaded_file is not None:

    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.write("## 📂 Dataset Preview")
    st.dataframe(df.head())

    # Copy dataset
    processed_df = df.copy()

    # ---------------- PREPROCESSING ---------------- #
    st.write("## 🛠 Data Preprocessing Started")

    # Remove duplicates
    processed_df.drop_duplicates(inplace=True)

    # Handle Date column
    if 'Date' in processed_df.columns:
        processed_df['Date'] = pd.to_datetime(
            processed_df['Date'],
            errors='coerce'
        )

        processed_df['Days'] = (
            processed_df['Date'] - processed_df['Date'].min()
        ).dt.days

    # Fill missing values safely
    for col in processed_df.columns:

        # Try convert to numeric where possible
        processed_df[col] = pd.to_numeric(
            processed_df[col],
            errors='ignore'
        )

        if processed_df[col].dtype == 'object':
            processed_df[col].fillna(
                processed_df[col].mode()[0],
                inplace=True
            )
        else:
            processed_df[col].fillna(
                processed_df[col].mean(),
                inplace=True
            )

    # Encode categorical columns
    for col in processed_df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col])

    st.success("✅ Data Preprocessing Completed")

    # Show processed data
    st.write("## 🧹 Processed Dataset Preview")
    st.dataframe(processed_df.head())

    # ---------------- TARGET COLUMN ---------------- #
    st.write("## 🎯 Select Prediction Target")
    target_column = st.selectbox(
        "Choose Target Column",
        processed_df.columns
    )

    # ---------------- MODEL TRAINING ---------------- #
    if st.button("🚀 Train Model"):

        X = processed_df.drop(columns=[target_column])
        y = processed_df[target_column]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        feature_columns = X.columns

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.2,
            random_state=42
        )

        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Predictions for evaluation
        y_pred = model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # ---------------- DISPLAY METRICS ---------------- #
        st.write("## 📊 Model Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric("R² Score", f"{r2:.3f}")
        col2.metric("MAE", f"{mae:.3f}")
        col3.metric("RMSE", f"{rmse:.3f}")

        # ---------------- FEATURE IMPORTANCE ---------------- #
        st.write("## 🔍 Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(importance_df)

        # Plot feature importance
        fig_imp, ax_imp = plt.subplots()
        ax_imp.bar(
            importance_df["Feature"],
            importance_df["Importance"]
        )
        ax_imp.set_title("Feature Importance")
        plt.xticks(rotation=45)

        st.pyplot(fig_imp)

        # ---------------- FUTURE PREDICTION ---------------- #
        st.write("## 🔮 Future Prediction")

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

            # ---------------- SHOW PREDICTIONS ---------------- #
            st.write("## 📅 Predicted Values")

            prediction_table = pd.DataFrame({
                "Day": [f"Day {i+1}" for i in range(days)],
                "Prediction": predictions
            })

            st.dataframe(prediction_table)

            # ---------------- LINE GRAPH ---------------- #
            st.write("## 📈 Line Graph")

            fig1, ax1 = plt.subplots()
            ax1.plot(predictions, marker='o')
            ax1.set_title("Future Predictions")
            ax1.set_xlabel("Days")
            ax1.set_ylabel("Prediction")

            st.pyplot(fig1)

            # ---------------- BAR GRAPH ---------------- #
            st.write("## 📊 Bar Graph")

            fig2, ax2 = plt.subplots()
            ax2.bar(range(len(predictions)), predictions)
            ax2.set_title("Bar Chart")
            ax2.set_xlabel("Days")
            ax2.set_ylabel("Prediction")

            st.pyplot(fig2)

            # ---------------- SCATTER PLOT ---------------- #
            st.write("## 🔵 Scatter Plot")

            fig3, ax3 = plt.subplots()
            ax3.scatter(range(len(predictions)), predictions)
            ax3.set_title("Scatter Plot")
            ax3.set_xlabel("Days")
            ax3.set_ylabel("Prediction")

            st.pyplot(fig3)

            # ---------------- HISTOGRAM ---------------- #
            st.write("## 📉 Histogram")

            fig4, ax4 = plt.subplots()
            ax4.hist(predictions)
            ax4.set_title("Histogram")

            st.pyplot(fig4)

else:
    st.info("👆 Please upload a CSV file to begin.")