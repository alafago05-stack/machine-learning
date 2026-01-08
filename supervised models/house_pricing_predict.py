import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data.csv')
print(df.info())
df = df.drop(columns=["city", "date", "street", "statezip", "country"])

corr = df.corr(numeric_only=True)  # ensure only numeric columns are used
print(corr['price'].sort_values(ascending=False))

df = df.drop(columns=["sqft_lot", "condition", "yr_built", "yr_renovated"])
print(df.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X = df.drop(columns=['price'])
y = df['price']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)