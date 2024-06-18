import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# File path
file_path = "/Users/pedromartins/Downloads/Module 28/download/bank-additional.csv"

# Function to load the trained model
@st.cache_data
def load_model(model_file):
    model = joblib.load(model_file)
    return model

# Function to preprocess the data
def preprocess_data(data):
    # Handle empty DataFrame
    if data.empty:
        raise ValueError("Input data is empty")
    
    # Separate numeric and non-numeric columns
    numeric_columns = data.select_dtypes(include=['int', 'float']).columns
    non_numeric_columns = data.select_dtypes(exclude=['int', 'float']).columns
    
    # Check if there are numeric columns
    if numeric_columns.empty:
        raise ValueError("No numeric columns found in input data")
    
    # Handle missing values in numeric columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # Encode non-numeric columns
    label_encoders = {}
    for column in non_numeric_columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # Scale numeric columns
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    return data

# Function to make predictions
def make_predictions(model, data):
    # Select only the relevant features (assuming first 4 columns are relevant)
    relevant_features = data.iloc[:, :4]
    
    if hasattr(model, "predict_proba"):
        # If the model supports probability estimation
        predictions = model.predict_proba(relevant_features)
    else:
        # Use binary predictions if the model does not support probabilities
        predictions = model.predict(relevant_features)
        
    return predictions

# Streamlit interface
def main():
    # Get the directory of the current script
    script_directory = os.path.dirname(__file__)

    # Construct the path to the image file
    image_path = os.path.join(script_directory, "logo_ebac_copy.png")

    # Check if the image file exists
    if os.path.exists(image_path):
        st.sidebar.image(image_path, use_column_width=True)
    else:
        st.sidebar.write("Image not found. Please check the file path.")

    # Main title
    st.title("Banking - EBAC School Project")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Preprocess the data
        try:
            preprocessed_data = preprocess_data(data)
        except ValueError as e:
            st.error(str(e))
            st.info("Please upload a different CSV file.")
            return
        
        # Load the model
        model_file = 'model_final.pkl'
        model = load_model(model_file)

        # Make predictions
        try:
            predictions = make_predictions(model, preprocessed_data)
        except ValueError as e:
            st.error(str(e))
            st.info("Please check the input data and ensure it matches the expected format.")
            return
        
        # Display predictions in DataFrame format
        if predictions.ndim == 1:  # Binary predictions
            predictions_df = pd.DataFrame(predictions, columns=["Predicted Class"])
        else:  # Probability predictions
            predictions_df = pd.DataFrame(predictions, columns=[f"Probability of Class {i}" for i in range(predictions.shape[1])])
        st.write("Predictions:")
        st.dataframe(predictions_df)

if __name__ == "__main__":
    main()

                   


               