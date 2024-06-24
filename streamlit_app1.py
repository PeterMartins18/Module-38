import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Function to load the trained model
@st.cache(allow_output_mutation=True)
def load_model(model_file):
    model = joblib.load(model_file)
    return model

# Function to preprocess the data
def preprocess_data(data):
    if data.empty:
        raise ValueError("Input data is empty")
    
    numeric_columns = data.select_dtypes(include=['int', 'float']).columns
    non_numeric_columns = data.select_dtypes(exclude=['int', 'float']).columns
    
    if numeric_columns.empty:
        raise ValueError("No numeric columns found in input data")
    
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    label_encoders = {}
    for column in non_numeric_columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    return data

# Function to make predictions
def make_predictions(model, data):
    relevant_features = data.iloc[:, :4]
    
    if hasattr(model, "predict_proba"):
        predictions = model.predict_proba(relevant_features)
    else:
        predictions = model.predict(relevant_features)
        
    return predictions

# Streamlit interface
def main():
    st.sidebar.title("Banking - EBAC School Project")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        try:
            preprocessed_data = preprocess_data(data)
        except ValueError as e:
            st.error(str(e))
            st.info("Please upload a different CSV file.")
            return
        
        model_file = 'model_final.pkl'
        if os.path.exists(model_file):
            model = load_model(model_file)
        else:
            st.error(f"Model file {model_file} not found.")
            return

        try:
            predictions = make_predictions(model, preprocessed_data)
        except ValueError as e:
            st.error(str(e))
            st.info("Please check the input data and ensure it matches the expected format.")
            return
        
        if predictions.ndim == 1:
            predictions_df = pd.DataFrame(predictions, columns=["Predicted Class"])
        else:
            predictions_df = pd.DataFrame(predictions, columns=[f"Probability of Class {i}" for i in range(predictions.shape[1])])
        
        st.write("Predictions:")
        st.dataframe(predictions_df)

if __name__ == "__main__":
    main()



                   


               
