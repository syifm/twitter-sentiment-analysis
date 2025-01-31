import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score

# Set page title
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Main title
st.title("Sentiment Analysis Dashboard")

# Load data
@st.cache_data  # Cache the data loading
def load_data():
    df_tweet = pd.read_csv("your_tweet_data.csv")  # Ganti dengan path file tweet Anda
    nb_results = pd.read_csv("y_test_pred.csv")
    knn_results = pd.read_csv("y_test_pred_knn.csv")
    return df_tweet, nb_results, knn_results

df_tweet, nb_results, knn_results = load_data()

# Create two columns for better layout
col1, col2 = st.columns(2)

# Sentiment Distribution Visualization
with col1:
    st.header("Distribusi Sentimen")
    
    # Calculate sentiment counts
    sentiment_counts = df_tweet['sentiment'].value_counts()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot data
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values,
                hue=sentiment_counts.index,
                palette={'positif': 'green', 'negatif': 'red', 'netral': 'gray'},
                legend=False,
                ax=ax)
    
    # Add title and labels
    ax.set_title('Distribusi Sentimen')
    ax.set_xlabel('Sentimen')
    ax.set_ylabel('Jumlah')
    
    # Display in Streamlit
    st.pyplot(fig)

# Model Evaluation
with col2:
    # Naive Bayes Evaluation
    st.header("Evaluasi Model")
    
    # Create tabs for different models
    tab1, tab2 = st.tabs(["Naive Bayes", "KNN"])
    
    with tab1:
        st.subheader("Model Naive Bayes")
        
        # Calculate and display accuracy
        accuracy = accuracy_score(nb_results["y_test"], nb_results["y_pred"])
        st.metric("Accuracy", f"{accuracy:.4f}")
        
        # Display classification report
        report_dict = classification_report(nb_results["y_test"], 
                                         nb_results["y_pred"], 
                                         output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df)
    
    with tab2:
        st.subheader("Model KNN")
        
        # Calculate and display accuracy
        accuracy_knn = accuracy_score(knn_results["y_test_knn"], 
                                    knn_results["y_pred_knn"])
        st.metric("Accuracy", f"{accuracy_knn:.4f}")
        
        # Display classification report
        report_dict_knn = classification_report(knn_results["y_test_knn"], 
                                              knn_results["y_pred_knn"], 
                                              output_dict=True)
        report_df_knn = pd.DataFrame(report_dict_knn).transpose()
        st.dataframe(report_df_knn)

# Add footer
st.markdown("---")
st.caption("Sentiment Analysis Dashboard - Created with Streamlit")
