# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 00:41:23 2020

@author: Vishal
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template , send_file
import pickle

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index_main.html')

@app.route('/check_customer_id',methods=['POST']) 
def check_customer_id():
    
    ## Customer List data frame....
    Customer_list = pickle.load(open("Customer_list.pkl" , 'rb'))
    return Customer_list.to_html()

@app.route('/predict',methods=['POST']) 
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]    
    final_features = [np.array(int_features)]
    
    Items = pickle.load(open('Item_list.pkl', 'rb'))
    prediction = []
    
    for i in Items:
        DT = pickle.load(open('DT_'+i+'.pkl', 'rb'))
        prediction.append(DT.predict(final_features).round(2))
    
    ## Creating a Dataframe of all predicted items....
    df = pd.DataFrame({"Item Name" : Items , "Predicted Quantity" : prediction})

    return df.to_html()
    
    
if __name__ == "__main__":
    app.run(debug=True)