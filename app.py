from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from time import time
import pickle
from joblib import load


app = Flask(__name__)
app.url_map.strict_slashes = False
saved_model=pickle.load(open("model.pkl","rb"))
scaler = load('scaler_filename.joblib')

@app.route('/')
def index():
	return render_template('home.html')

@app.route('/predict', methods=['POST']) 
def login_user():

	data_points = list()
	data = []
	string = 'value'
	for i in range(1,31):
		data.append(float(request.form['value'+str(i)]))

	for i in range(30):
		data_points.append(data[i])
		
	print(data_points)

	data_np = np.asarray(data, dtype = float)
	data_np = data_np.reshape(1,-1)
	data_np=pd.DataFrame(data_np)
	transformed_data=scaler.transform(data_np)

	out=saved_model.predict(transformed_data)
	
	#out, acc, t = random_forest_predict(clf, data_np)

	if(out==1):
		output = 'Malignant'
	else:
		output = 'Benign'


	return render_template('result.html', output=output)

	

if __name__=='__main__':
	app.run(debug=True,port=7451)

