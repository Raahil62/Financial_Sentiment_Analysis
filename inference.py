
import joblib

# Load saved model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# New financial sentence
text = input("Enter a financial sentence: ")

# Convert text into TF-IDF
text_tfidf = vectorizer.transform([text])

# Predict sentiment
prediction = model.predict(text_tfidf)

print("Predicted Sentiment:", prediction[0])
