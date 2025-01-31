mport streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Baca data hasil prediksi
data = pd.read_csv("y_test_pred.csv")
y_test = data["y_test"]
y_pred = data["y_pred"]

# Menampilkan hasil evaluasi di Streamlit
st.title("Evaluasi Model Naive Bayes")

# Menampilkan Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Accuracy:")
st.write(f"{accuracy:.4f}")

# Mengubah classification report menjadi DataFrame
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Menampilkan classification report dalam bentuk tabel
st.subheader("Classification Report:")
st.dataframe(report_df)

# Baca data hasil prediksi
data = pd.read_csv("y_test_pred_knn.csv")
y_test_knn = data["y_test_knn"]
y_pred_knn = data["y_pred_knn"]

# Menampilkan hasil evaluasi di Streamlit
st.title("Evaluasi Model KNN")

# Menampilkan Accuracy
accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)
st.subheader("Accuracy:")
st.write(f"{accuracy_knn:.4f}")

# Mengubah classification report menjadi DataFrame
report_dict_knn = classification_report(y_test_knn, y_pred_knn, output_dict=True)
report_df_knn = pd.DataFrame(report_dict_knn).transpose()

# Menampilkan classification report dalam bentuk tabel
st.subheader("Classification Report:")
st.dataframe(report_df_knn)
