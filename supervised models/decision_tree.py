import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics



import warnings
warnings.filterwarnings('ignore') 

path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)
print(my_data)
print(my_data.info())

#we make the specified data into 0 and 1 to work with numbers
abel_encoder = LabelEncoder()
my_data['Sex'] = abel_encoder.fit_transform(my_data['Sex']) 
my_data['BP'] = abel_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = abel_encoder.fit_transform(my_data['Cholesterol']) 
print(my_data)

#This tells us that there are no missing values in any of the fields
print(my_data.isnull().sum())

#turning drugs types into numbers
custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)
print(my_data)

category_counts = my_data['Drug'].value_counts()

# Plot the count plot
plt.bar(category_counts.index, category_counts.values, color='blue')
plt.xlabel('Drug')
plt.ylabel('Count')
plt.title('Category Distribution')
plt.xticks(rotation=45)  # Rotate labels for better readability if needed
plt.show()

#modeling
y = my_data['Drug']
X = my_data.drop(['Drug','Drug_num'], axis=1)

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X_trainset,y_trainset)
#prediction
tree_predictions = drugTree.predict(X_testset)
print(tree_predictions)
print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))

plot_tree(drugTree)
plt.show()
