#importing packages
import pandas as pd
import numpy as np
import pickle

df=pd.read_csv('SUV_Purchase.csv')

#Feature engineering
x=df.drop('User ID',axis=1)
y=df.drop('Gender',axis=1)

# Loading the data
x=np.array(df[['Age','EstimatedSalary']])
y=df.iloc[:,-1:].values

#splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#standard scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)


#training the model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


#predicting
y_pred=model.predict(sc.transform(x_test))
print(y_pred)

#execute once and create the file
#pickle
import pickle
pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print("Success loaded")