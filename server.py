from flask import Flask,request,render_template
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('SUV_Purchase.csv')
app=Flask(__name__)
#Deserialization
model=pickle.load(open('model.pkl','rb'))

@app.route('/')#using GET we send webpage to Client(browser)
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])#gets input data from client(browser) to Flask Server
def predict():
    features=[int(x) for x in request.form.values()]
    print(features)
    final=[np.array(features)]
    x=df.iloc[:,2:4].values
    sc = StandardScaler().fit(x)
    output=model.predict(sc.transform(final))
    print(output)
    if output[0]==0:
        return render_template('index.html',pred=f'The Person will not be able to Purchase SUV Car')
    else:
        return render_template('index.html',pred=f'The Person will  be able to Purchase SUV Car')
if __name__ =='__main__':
    app.run(debug=True)