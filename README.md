🧠 AI-Powered Sales Forecasting & Analytics Dashboard

Predict future business revenue using Machine Learning with XGBoost + Salary-cycle intelligent forecasting.

🔥 Tech Stack
Component	Technology
Core Language	Python 3.10+ (venv)
Forecasting Models	XGBoost (Primary) + LightGBM
Web UI	Streamlit Dashboard
Optimization	Optuna (Hyperparameter tuning)
Visualization	Plotly, Seaborn, Matplotlib
Data Handling	Pandas, Numpy, Parquet, Dask

📂 Project Structure
Mini_Project_Shravani_Harel_Sales_Forecasting/
│── app.py
│── requirements.txt
│── final_xgboost_forecasting_model.pkl      ← place here
│── data/
│── notebooks/   (Model Training + Tuning)
│── README.md    ← paste this description here

▶ How to Run the Project
cd Mini_Project_Shravani_Harel_Sales_Forecasting
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py


Project opens at → http://localhost:8501

📈 What Makes This Project Scalable?
Scalability Feature	How it scales
Can deploy to AWS / Azure / GCP	Run globally, large traffic ready
Uses Parquet + Dask support	Handles millions of records
Model can be served via FastAPI	Convert to enterprise API
Can auto-train on new data	CRON scheduled retraining
Multi-store forecasting support	Works for retail chains & franchises
Future expansion to LSTM / TimesNet	Real AI time-series deep learning ready

🏆 Why this Project is Valuable (Minor Project Viva Highlights)
Feature Strength	Why it matters in real business
Salary-cycle based forecasting	Models human spending patterns realistically
Lag & rolling feature engineering	Captures weekly + seasonal trend shifts
Interactive dashboard	Managers forecast without coding
XGBoost tuning = best accuracy	Higher reliability than ARIMA/Prophet
Fully deployable ML application	Not just model → Real-world software

Developed by : Shravani Harel