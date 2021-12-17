import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
from xgboost import XGBClassifier
from scipy.stats import norm
import xgboost

df= pd.read_csv(r"D:\python folder\PycharmProjects\pythonProject2\Gis_Final_Pune_ml_new.csv")
dd=pd.read_csv(r"D:\python folder\PycharmProjects\pythonProject2\Ref_pune.csv")

################################ project name handling #######################################

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
laben = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
dd['encoded_project_name']= laben.fit_transform(dd['Project_Name'])
dd['encoded_project_name'].unique()

def get_proj(list1):
  ss = list1
  for k,v in le_project_mapping.items():
    if ss[4] in str(f"{k}"):
      return le_project_mapping[k]
    else:
      return 0

le_project_mapping = dict(zip(laben.classes_, laben.transform(laben.classes_)))
# print(le_project_mapping)

df=df.drop(['Unnamed: 0'],axis='columns')
dd=dd.drop(['Unnamed: 0'],axis='columns')

#################### Train Test split (80:20) ###########################
dff_Train = df.iloc[:4587,:]
dff_Test = df.iloc[4587:,:]
################# From Train dataset we create x_train and y_train #####################
X_train=dff_Train.drop(['final_price'],axis=1)
y_train=dff_Train['final_price']
y_train.shape
dff_Test.drop(['final_price'],axis=1,inplace=True)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.33, random_state=7)

import xgboost
regressor = xgboost.XGBRegressor(max_depth=5, min_child_weight=2, n_estimators=500)
regressor.fit(X_train1,y_train1)

import joblib
import pickle
import xgboost as xgb
file_name = "xgb_reg_gis.pkl"
#save model
joblib.dump(regressor, file_name)

#load saved model
xgb = joblib.load(file_name)
