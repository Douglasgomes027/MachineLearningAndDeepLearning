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
def predictions():

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



@app.route("/predict_item", methods=['POST'])
def predict_item():
    vectorizer_train = CountVectorizer(vocabulary=loaded_vocabulary)
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
 
    output_json = format_json_probs(req_data,'data', predictions.tolist(), prob_list)
    
    return output_json



##CRIA UM JSON PARA A SAÍDA
def json_concatenation(input_json, json_key, predictions_list, prob_list):
    teste_predict = input_json[json_key]
    output_dict= []
    for i in range(len(predictions_list)):
        data = {
            "id": teste_predict[i]['id'],
            "descricao": teste_predict[i]['descricao'],
            "genero": predictions_list[i],
            "probabilidade": prob_list[i]
        }
        output_dict.append(data)
        
    #output_json = json.dumps(output_dict)
    output_json = {"data" : output_dict}
    #output_json = jsonify(output_dict)
    return jsonify(output_json)
    
def format_json_probs(input_json, json_key, predictions_list, pred_prob_list):
    teste_predict =input_json[json_key]
    output_dict = []
    probs_dict = dict()
    for i in range(len(predictions_list)):
        data = {
            "id": teste_predict[i]['id'],
            "descricao": teste_predict[i]['descricao']
        }
        for j in range(len(predictions_list[i])):
            print(pred_prob_list[i])
            probs = {
                "top" : {
                    "genero": predictions_list[i],
                    "probabilidade": pred_prob_list[i][j]
                }
            }
            print(probs)
            probs_dict.update(probs)
            
        data.update(probs_dict)
        output_dict.append(data)
        
    output_json = {"data" : output_dict}
    return jsonify(output_dict)   
        
def get_greatest_probabilities(pred_prob_list):
    max_probabilities = []
    for i in pred_prob_list:
       max_probabilities.append(max(i))
    return max_probabilities
    
def get_top_N_greatest_probabilities(list, N):
    list_copy = list
    max_probabilities = []
    for i in list_copy:
        top_probabilities = []
        for j in range(N):
            top_probabilities.append(max(i))
            top_element = i.index(max(i))
            del i[top_element]
        max_probabilities.append(top_probabilities)
        
    return max_probabilities

##INICIANDO SERVIDOR
app.run()
