from flask import Flask, jsonify, request, url_for
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import os, json

##
app = Flask(__name__)

##
filename = 'Classificador-v2\model_v2.pkl'

##
with open(filename ,'rb') as f:
    loaded_model = joblib.load(f)

##
vocab_filename = 'Classificador-v2\model_v2_vocabulary.pkl'

##
with open(vocab_filename, 'rb') as f2:
    loaded_vocabulary = joblib.load(f2)


@app.route("/predictions", methods=['POST'])
def test():
    ##
    #print(loaded_model.classes_)

    ##
    vectorizer_train = CountVectorizer(vocabulary=loaded_vocabulary)

    ##
    #teste_predict=['EPSON MAGENTO','LILLO BICO','BOV AMERICANA KG','KING 100ML JASMIM']
    req_data = request.get_json()
    teste_predict = []
    req_array = req_data['data']
    for i in range(len(req_array)):
        desc = req_array[i]['descricao']
        teste_predict.append(desc)
       
    teste_predict_vect = vectorizer_train.transform(teste_predict) 
    predictions = loaded_model.predict(teste_predict_vect)
 
    pred_prob_list = loaded_model.predict_proba(teste_predict_vect).tolist()
 
    prob_list = get_greatest_probabilities(pred_prob_list)
 
    output_json = json_concatenation(req_data,'data', predictions.tolist(), prob_list)
    
    return output_json

##
@app.route("/")
def hello_world():
    return "Hello World! <strong>I am learning Flask</strong>", 200

##CRIA UM JSON PARA A SAÍDA
def json_concatenation(input_json, json_key, output_list, prob_list):
    teste_predict = input_json[json_key]
    output_dict= []
    for i in range(len(output_list)):
        data = {
            "id": teste_predict[i]['id'],
            "descricao": teste_predict[i]['descricao'],
            "genero": output_list[i],
            "probabilidade": prob_list[i]
        }
        output_dict.append(data)
        
    #output_json = json.dumps(output_dict)
    output_json = {"data" : output_dict}
    #output_json = jsonify(output_dict)
    return jsonify(output_json)
    

def get_greatest_probabilities(pred_prob_list):
    max_probabilities = []
    for i in pred_prob_list:
       max_probabilities.append(max(i))
    return max_probabilities
    

##INICIANDO SERVIDOR
app.run()
