import numpy as np
import pickle
from flask import Flask, request, render_template


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/50cal',methods=['POST','GET'])
def make_predict():
    if request.method=='POST':

        muzzle = request.form['muzzle']
        sabot = request.form['sabot']
        lpmass = request.form['lpmass']
        muzzle = float(muzzle)
        sabot = float(sabot)
        lpmass = float(lpmass)
        predict_request = [muzzle, sabot, lpmass]
        predict_request = np.array(predict_request)
        x = predict_request.item(1)
        y = predict_request.item(0)
        z = predict_request.item(2)
        squeeze = x - y
        cb = (x-0.4887)/(0.5835 - 0.4887)
        eb = (z-1.00404)/(2.5448 - 1.00404)
        fb = (squeeze-0.0)/(0.01100000000000001 - 0.0)
        
        knn_pkl = open("KNNRegressionModel.pkl","rb")
        model = pickle.load(knn_pkl)
        
        predict = np.array([[cb, eb, fb]])
        y_hat = model.predict(predict)
        output = y_hat[0]
        
        return render_template('result.html', output=output)

if __name__ == '__main__':
    
    app.run()