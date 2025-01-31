import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score

# **1. Membaca Data Hasil Prediksi**
data = pd.read_csv("y_test_pred.csv")
y_test = data["y_test"]
y_pred = data["y_pred"]

# **2. Membuat Diagram Distribusi Sentimen**
st.title("Analisis Sentimen")

# Menghitung distribusi sentimen
sentiment_counts = pd.Series(y_test).value_counts()

# Membuat plot distribusi sentimen
fig, ax = plt.subplots()
sns.barplot(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    hue=sentiment_counts.index,
    palette={'positif': 'green', 'negatif': 'red', 'netral': 'gray'},
    legend=False,
    ax=ax
)

ax.set_title("Distribusi Sentimen")
ax.set_xlabel("Sentimen")
ax.set_ylabel("Jumlah")

# **Menampilkan Plot di Streamlit**
st.pyplot(fig)

# **3. Evaluasi Model Naive Bayes**
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

# **4. Evaluasi Model KNN**
# Baca data hasil prediksi untuk KNN
data_knn = pd.read_csv("y_test_pred_knn.csv")
y_test_knn = data_knn["y_test_knn"]
y_pred_knn = data_knn["y_pred_knn"]

st.title("Evaluasi Model KNN")

# Menampilkan Accuracy KNN
accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)
st.subheader("Accuracy:")
st.write(f"{accuracy_knn:.4f}")

# Mengubah classification report menjadi DataFrame
report_dict_knn = classification_report(y_test_knn, y_pred_knn, output_dict=True)
report_df_knn = pd.DataFrame(report_dict_knn).transpose()

# Menampilkan classification report dalam bentuk tabel
st.subheader("Classification Report:")
st.dataframe(report_df_knn)
