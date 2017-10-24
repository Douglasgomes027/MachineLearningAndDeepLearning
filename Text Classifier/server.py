from flask import Flask, jsonify, request, url_for
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import os, json

##
app = Flask(__name__)

@app.route("/predictions", methods=['POST'])
def test():
    ##
    filename = 'model_v1.pkl'

    ##
    with open(filename ,'rb') as f:
        loaded_model = joblib.load(f)

    ##
    vocab_filename = 'model_vocabulary.pkl'

    ##
    with open(vocab_filename, 'rb') as f2:
        loaded_vocabulary = joblib.load(f2)

    ##
    #print(loaded_model.classes_)

    ##
    vectorizer_train = CountVectorizer(vocabulary=loaded_vocabulary)

    ##
    #teste_predict=['EPSON MAGENTO','LILLO BICO','BOV AMERICANA KG','KING 100ML JASMIM']
    request_data = request.get_json()
    teste_predict = request_data['data']
    teste_predict_vect = vectorizer_train.transform(teste_predict) 
    predictions = loaded_model.predict(teste_predict_vect)
 
    return jsonify(predictions.tolist())

##
@app.route("/")
def hello_world():
    return "Hello World! <strong>I am learning Flask</strong>", 200

##
app.run()
