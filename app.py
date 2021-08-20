from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
from flask import Response
import numpy as np

app = Flask(__name__) 


@app.route('/predict',methods=['POST','GET'])
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            annual_income=float(request.json['annual_income'])
            spending_score = float(request.json['spending_score'])
            filename = 'finalcluster_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb')) 
            prediction=loaded_model.predict(np.asarray([[annual_income,spending_score]]))
            print('prediction is', (list(prediction)[0]))
            return Response(str((list(prediction)[0])))
        except Exception as e:
            print('The Exception message is: ',e)
            return Response('something is wrong.')



if __name__ == "__main__":
   app.run(debug=True)
