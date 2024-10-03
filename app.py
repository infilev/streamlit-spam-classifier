import streamlit as st
import joblib

# Title of the app
st.title("Spam Comment Classifier")

# Load the saved model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("spam_classifier_model.pkl")
    vectorizer = joblib.load("count_vectorizer.pkl")
    return model, vectorizer

model, cv = load_model()

# User input for prediction
st.subheader("Enter a Comment to Classify")
user_input = st.text_area("Your Comment", "")

# Add a "Check" button
if st.button("Check"):
    if user_input:
        # Transform the user input using the CountVectorizer
        transformed_input = cv.transform([user_input]).toarray()

        # Predict using the model
        prediction = model.predict(transformed_input)

        # Display prediction result
        st.write(f"Prediction: **{prediction[0]}**")
    else:
        st.write("Please enter a comment for classification.")

# Example predictions (optional)
# st.subheader("Sample Predictions")
# sample_comments = ["Check this out: https://example.com/", "This is useless information!"]
# for sample in sample_comments:
#     transformed_sample = cv.transform([sample]).toarray()
#     sample_prediction = model.predict(transformed_sample)
#     st.write(f"Comment: {sample}")
#     st.write(f"Prediction: **{sample_prediction[0]}**")
