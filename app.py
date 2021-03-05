# import necessary libraries
import numpy as np
#import pickle as pkl
import joblib
import os
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

#################################################
# Database Setup
#################################################

# from flask_sqlalchemy import SQLAlchemy
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', '') or "sqlite:///db.sqlite"

# # Remove tracking modifications
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)

model = joblib.load('decisionTree_model.sav', mmap_mode=None)

# create route that renders index.html template
@app.route("/")
def home():
    return render_template("form.html")

@app.route('/predict')
def predict():
    # ini_features = [int(x) for x in request.form.values()]
    # fnl_feaures = [np.array(ini_features)]
    # predict = model.predict(fnl_feaures)
    
    # result = round(predict[0],2)
    # return render_template('form.html', predict_txt = 'Decision should be ${}'.format(result))
    return render_template('form.html')

@app.route("/send", methods=["GET", "POST"])
def send():
    #print(request.form)
    #print(request.get_json(force=True))
    data = []
    if request.method == "POST":
    
        vMatch = request.form["match"]
      
        vLike = request.form["like"]
        vAttr = request.form["attr"]
        vDec_o = request.form["dec_o"]
        vProb = request.form["prob"]
        vSinc = request.form["sinc"]
        vAttr_2_1 = request.form["attr_2_1"]
        vAttr_1 = request.form["attr_1"]
 
        
        data.append(vMatch)
        data.append(vLike)
        data.append(vAttr)
        data.append(vDec_o)
        data.append(vProb)
        data.append(vSinc)
        data.append(vAttr_2_1)
        data.append(vAttr_1)
       
    
    #print(data)
    predict = model.predict(np.array(data).reshape(1, -1))
    #print(predict)
    
    result = predict[0]
    
    
    #return jsonify(result)
    return str(result)

 

if __name__ == "__main__":
    app.run(debug=True)
