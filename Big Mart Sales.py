import pandas as pd
import numpy as np

#read the training csv file 
train = pd.read_csv("Train.csv")

#Handle the nan value and clean the dataset
train["Item_Fat_Content"] = train["Item_Fat_Content"].replace("LF","Low Fat")
train["Item_Fat_Content"] = train["Item_Fat_Content"].replace("low fat","Low Fat")
train["Item_Fat_Content"] = train["Item_Fat_Content"].replace("reg","Regular")

train["Item_Weight"] = train["Item_Weight"].fillna(train["Item_Weight"].mean())
train['Outlet_Size'] = train['Outlet_Size'].fillna(method="ffill")

#split the features and labels from the dataset
features = train.iloc[:,[0,1,2,3,4,5,6,8,9,10]].values
labels = train.iloc[:,-1].values


#label Encode the features data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

for col in [0,2,4,6,7,8,9]:
    le = LabelEncoder()
    features[:,col] = le.fit_transform(features[:,col])
    

#One hot encode the labels encoded data 
o1 = OneHotEncoder(categorical_features=[0,4,6,7,8,9])
features = o1.fit_transform(features).toarray()


from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test  = train_test_split(features,labels,test_size=0.25,random_state=0)

#Perform Xgboost Algorithm for training of the data

from xgboost import XGBRegressor
regressor = XGBRegressor(n_estimators=1000,learning_rate=0.05)
regressor.fit(features_train, labels_train,early_stopping_rounds=5,eval_set=[(features_test,labels_test)],verbose=False)

#Check the score of training data

print("Score_train: ",regressor.score(features_train,labels_train)
print("Score_test: ",regressor.score(features_test,labels_test))