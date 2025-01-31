import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Prediksi sudah dilakukan sebelumnya
y_pred = naivebayes.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Menampilkan hasil evaluasi di Streamlit
st.title("Evaluasi Model Naive Bayes")

# Menampilkan Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Accuracy:")
st.write(f"{accuracy:.4f}")

# Menampilkan Classification Report
report = classification_report(y_test, y_pred)
st.subheader("Classification Report:")
st.text(report)
