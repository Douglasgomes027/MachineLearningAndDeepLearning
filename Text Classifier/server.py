
from flask import Flask, jsonify, request, url_for
from flask_api import FlaskAPI,exceptions,status
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import os

##
app = Flask(__name__)

import json, requests

dados = ['EPSON MAGENTO','LILLO BICO','BOV AMERICANA KG','KING 100ML JASMIM']
data = json.dumps(dados)

with open('data.json') as json_data:
    response = requests.get("https://viacep.com.br/ws/01001000/json/")
    print(response.content)

def valores_predicao(key):
    return{
        'url':request.host_url.rstrip('/') + url_for('detalhes_predicao', key=key),
        'text': teste_predict[key]
}

@app.route("/test")
def test():
    print(data.keys)
    return [valores_predicao(id) for id in sorted(data.keys())]

@app.route('/<int:key>/', methods=['GET'])
def detalhes_predicao(key):
    if key not in teste_predict:
        raise exceptions.NotFound()
    return teste_predict(key)

@app.route('/predictions', methods=['GET'])
def predicion_list():
    teste_predict_array = [valores_predicao(id) for id in sorted(teste_predict.keys())]
    return teste_predict_array

##
@app.route("/")
def hello_world():
    return "Hello World! <strong>I am learning Flask</strong>", 200

##
@app.route('/predict', methods=['GET'])
def apicall():
    """API Call

    Pandas dataframe (sent as a payload) from API Call
    """
    try:
        test_json = request.get_json()
        test = pd.read_json(test_json, orient='records')

        #To resolve the issue of TypeError: Cannot compare types 'ndarray(dtype=int64)' and 'str'
        test['Dependents'] = [str(x) for x in list(test['Dependents'])]

        #Getting the Loan_IDs separated out
        loan_ids = test['Loan_ID']

    except Exception as e:
        raise e

    clf = 'model_v1.pk'
    vocab_filename = 'model_vocabulary.pkl'

    if test.empty:
        return(bad_request())
    else:
        #Load the saved model
        print("Loading the model...")
        loaded_model = None
        with open(clf,'rb') as f:
            loaded_model = pickle.load(f)

        print("Loading the vocabulary...")
        loaded_vocabulary = None
        with open(vocab_filename, 'rb') as f2:
            loaded_vocabulary = joblib.load(f2)

        vectorizer = CountVectorizer(vocabulary=loaded_vocabulary)

        print("The model has been loaded...doing predictions now...")
        predictions = loaded_model.predict(test)

        """Add the predictions as Series to a new pandas dataframe
                                OR
           Depending on the use-case, the entire test data appended with the new files
        """
        prediction_series = list(pd.Series(predictions))

        final_predictions = pd.DataFrame(list(zip(loan_ids, prediction_series)))

        """We can be as creative in sending the responses.
           But we need to send the response codes as well.
        """
        responses = jsonify(predictions=final_predictions.to_json(orient="records"))
        responses.status_code = 200

        return (responses)

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
teste_predict=['EPSON MAGENTO','LILLO BICO','BOV AMERICANA KG','KING 100ML JASMIM']
teste_predict_vect = vectorizer_train.transform(teste_predict) 
loaded_model.predict(teste_predict_vect)

##
app.run()
