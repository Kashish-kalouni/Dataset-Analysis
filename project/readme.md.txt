# E-commerce Sales Analysis & Prediction (Superstore Dataset)

## Description
This project analyzes the **Superstore E-commerce dataset** and builds a **predictive model** to estimate sales using machine learning. The project includes data exploration, visualization, and a Linear Regression model for training on the dataset.  

**Objectives:**
- Explore and clean the Superstore dataset
- Visualize sales, profit, and regional trends
- Prepare data for machine learning (encoding categorical variables, train-test split)
- Train a **Linear Regression model** to predict sales
- Evaluate model performance using metrics like **Mean Squared Error (MSE)** and **R² score**

## Dataset
- File: `Superstore.csv`
- Source: Public Superstore E-commerce dataset
- Key Features: Order ID, Product Name, Category, Sub-Category, Sales, Profit, Quantity, Order Date, Ship Date, Region, Customer ID

## Machine Learning Pipeline
1. **Data Preprocessing:**
   - One-hot encoding for categorical features
   - Train-test split using `train_test_split`
2. **Model:**
   - `LinearRegression` from scikit-learn
3. **Evaluation:**
   - Mean Squared Error (MSE)
   - R² Score

**Python Libraries Used:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


How to Run

Clone the repository:

git clone https://github.com/Kashish-kalouni/Superstore-Analysis.git


Navigate to the project folder:

cd Dataset-Analysis


Install required packages:

pip install -r requirements.txt


(If requirements.txt is not available, manually install: pandas, scikit-learn, matplotlib, seaborn)

Run Jupyter Notebook:

jupyter notebook Superstore_Analysis.ipynb


Follow the Notebook to explore analysis, visualizations, and train the Linear Regression model.

Author

Kashish Kalouni
GitHub
 | LinkedIn

## License
This project is licensed under the MIT License.