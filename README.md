Customer Churn Analytics – European Banking

📖 Project Overview

Customer Churn Analytics – European Banking is a machine learning project that analyzes customer behavior to identify patterns associated with customer churn. The project performs data preprocessing, exploratory data analysis (EDA), customer segmentation, machine learning model training, and model evaluation. An interactive Streamlit dashboard is included to visualize insights and make churn predictions.

---

🎯 Objectives

- Analyze customer demographics and banking behavior.
- Identify factors influencing customer churn.
- Build a machine learning model to predict churn.
- Evaluate model performance using standard metrics.
- Provide an interactive dashboard for data visualization and prediction.

---

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib

---

📂 Project Structure

Customer-Churn-Analytics-European-Banking/
│── app.py
│── data_cleaning_eda.py
│── customer_segmentation.py
│── model_training.py
│── model_evaluation.py
│── random_forest_model.pkl
│── evaluation_results.txt
│── model_results.txt
│── gender_churn.png
│── geography_churn.png
│── product_usage.png
│── requirements.txt
│── README.md

---

📊 Exploratory Data Analysis (EDA)

The project analyzes customer churn based on:

- Gender
- Geography
- Product Usage
- Customer Segments

Visualizations are included to understand customer behavior and identify churn patterns.

---

🤖 Machine Learning Model

Algorithm Used:

- Random Forest Classifier

The model is trained to classify whether a customer is likely to churn based on customer attributes and banking information.

---

📈 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The evaluation results are available in:

- "evaluation_results.txt"
- "model_results.txt"

---

🌐 Streamlit Dashboard

The interactive dashboard allows users to:

- Explore customer churn analytics.
- View visualizations.
- Enter customer details.
- Predict customer churn in real time.

Run the application using:

streamlit run app.py

---

🚀 Installation

1. Clone the repository

git clone https://github.com/akshata-chaturvedi/Customer-Churn-Analytics-European-Banking.git

2. Move into the project directory

cd Customer-Churn-Analytics-European-Banking

3. Install the required packages

pip install -r requirements.txt

4. Run the Streamlit application

streamlit run app.py

---

📌 Future Improvements

- Hyperparameter tuning for improved accuracy.
- Deploy the application using Streamlit Community Cloud.
- Add more interactive visualizations.
- Compare multiple machine learning algorithms.
- Perform feature importance analysis.

---

👩‍💻 Author

Akshata 

B.Tech in Artificial Intelligence

Delhi Skill and Entrepreneurship University (DSEU)

---

📜 License

This project is created for educational and academic purposes.
