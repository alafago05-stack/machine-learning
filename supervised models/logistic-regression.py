import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)
#getting the data we need
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn','callcard']]
churn_df['churn'] = churn_df['churn'].astype('int') #make all the data in churn column int integers 
#print(churn_df)
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip','callcard']])
print(X[0:5])  #print the first 5 values
y = np.asarray(churn_df['churn'])
print(y[0:5]) #print the first 5 values
X_norm = StandardScaler().fit(X).transform(X)#It is also a norm to standardize or normalize the dataset in order to have all the features at the same scale. This helps the model learn faster and improves the model performance. We may make use of StandardScalar function in the Scikit-Learn library.
print(X_norm[0:5])
X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=4)
LR = LogisticRegression().fit(X_train,y_train)
yhat = LR.predict(X_test)
print(yhat[:10])

yhat_prob = LR.predict_proba(X_test)
print(yhat_prob[:10])
coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()
print(log_loss(y_test, yhat_prob))