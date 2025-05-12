import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set page configuration
st.set_page_config(page_title="Rainfall Prediction App", layout="wide")

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Function to generate synthetic dataset
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Temperature': np.random.uniform(10, 35, n_samples),
        'Humidity': np.random.uniform(20, 100, n_samples),
        'Wind_Speed': np.random.uniform(0, 30, n_samples),
        'Pressure': np.random.uniform(980, 1020, n_samples),
        'Cloud_Cover': np.random.uniform(0, 100, n_samples)
    }
    df = pd.DataFrame(data)
    # Synthetic rule: Rain if humidity > 70 and cloud cover > 60, with some noise
    df['Rain'] = ((df['Humidity'] > 70) & (df['Cloud_Cover'] > 60)).astype(int)
    df['Rain'] = df['Rain'] + np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])  # Add noise
    df['Rain'] = df['Rain'].clip(0, 1)  # Ensure binary
    return df

# Function to train and evaluate models
@st.cache_data
def train_models(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'y_pred': y_pred,
            'scaler': scaler
        }
        # Save model and scaler
        joblib.dump(model, f'models/{name.lower().replace(" ", "_")}_model.joblib')
        if name == 'Random Forest':  # Save scaler once
            joblib.dump(scaler, 'models/scaler.joblib')
    return results

# Function to load models
def load_models(X_train, X_test, y_train, y_test):
    model_files = {
        'Random Forest': 'models/random_forest_model.joblib',
        'SVM': 'models/svm_model.joblib',
        'Logistic Regression': 'models/logistic_regression_model.joblib'
    }
    scaler_file = 'models/scaler.joblib'

    # Check if all models and scaler exist
    if all(os.path.exists(file) for file in model_files.values()) and os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = {}
        for name, file in model_files.items():
            model = joblib.load(file)
            y_pred = model.predict(X_test_scaled)
            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'y_pred': y_pred,
                'scaler': scaler
            }
        return results
    else:
        # Train and save models if not found
        return train_models(X_train, X_test, y_train, y_test)

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    return plt.gcf()

# Function to plot feature importance (for Random Forest)
def plot_feature_importance(model, features):
    importances = model.feature_importances_
    plt.figure(figsize=(6, 4))
    sns.barplot(x=importances, y=features)
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    return plt.gcf()

# Main app
def main():
    st.title("🌧️ Rainfall Prediction App")
    st.markdown("""
    This app predicts whether it will rain based on weather features using **Random Forest**, **SVM**, and **Logistic Regression**.
    Enter weather conditions in the sidebar and click 'Predict' to see results. Model performance and visualizations are displayed below.
    Models are saved using **joblib** to avoid retraining.
    """)

    # Generate or load data
    st.subheader("Dataset")
    df = generate_synthetic_data()
    st.write("Sample of the synthetic dataset used for training:")
    st.dataframe(df.head())

    # Prepare data
    X = df[['Temperature', 'Humidity', 'Wind_Speed', 'Pressure', 'Cloud_Cover']]
    y = df['Rain']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load or train models
    st.write("Loading or training models...")
    results = load_models(X_train, X_test, y_train, y_test)
    st.write("Models loaded successfully!")

    # Sidebar for user inputs
    st.sidebar.header("Input Weather Conditions")
    temperature = st.sidebar.slider("Temperature (°C)", 10.0, 35.0, 25.0)
    humidity = st.sidebar.slider("Humidity (%)", 20.0, 100.0, 50.0)
    wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 30.0, 10.0)
    pressure = st.sidebar.slider("Pressure (hPa)", 980.0, 1020.0, 1000.0)
    cloud_cover = st.sidebar.slider("Cloud Cover (%)", 0.0, 100.0, 50.0)

    # Predict button
    if st.sidebar.button("Predict"):
        user_input = np.array([[temperature, humidity, wind_speed, pressure, cloud_cover]])
        st.subheader("Prediction Results")
        for name, result in results.items():
            model = result['model']
            scaler = result['scaler']
            user_input_scaled = scaler.transform(user_input)
            prediction = model.predict(user_input_scaled)[0]
            prob = model.predict_proba(user_input_scaled)[0][1] if hasattr(model, 'predict_proba') else None
            st.write(f"**{name}**: {'Rain' if prediction == 1 else 'No Rain'}"
                     f" {'(Probability: {:.2%})'.format(prob) if prob is not None else ''}")

    # Model performance
    st.subheader("Model Performance")
    metrics_df = pd.DataFrame({
        'Model': results.keys(),
        'Accuracy': [results[name]['accuracy'] for name in results],
        'Precision': [results[name]['precision'] for name in results],
        'Recall': [results[name]['recall'] for name in results],
        'F1-Score': [results[name]['f1'] for name in results]
    })
    st.dataframe(metrics_df.style.format("{:.2%}", subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']))

    # Visualizations
    st.subheader("Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Confusion Matrices**")
        for name, result in results.items():
            fig = plot_confusion_matrix(y_test, result['y_pred'], f"Confusion Matrix ({name})")
            st.pyplot(fig)
    
    with col2:
        st.write("**Feature Importance (Random Forest)**")
        rf_model = results['Random Forest']['model']
        fig = plot_feature_importance(rf_model, X.columns)
        st.pyplot(fig)

if __name__ == "__main__":
    main()