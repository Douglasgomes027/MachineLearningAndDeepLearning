from flask import Flask, jsonify, request, url_for
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import os, json

##
app = Flask(__name__)

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
 
    output_json = json_concatenation(req_data,'data', predictions.tolist(), prob_list, config())
    
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
 
    prob_list = get_top_N_greatest_probabilities(pred_prob_list, 5)
 
    output_json = format_json_probs(req_data,'data', predictions.tolist(), prob_list)
    
    return output_json


@app.route("/predict-classe-genero", methods=['POST'])
def predict_classe_genero():
    req_data = request.get_json()
    class_array = predict_class(loaded_model_classe_descricao, loaded_vocabulary_classe_descricao, req_data)
    genero_array = []
    prob_array  = []
    req_array = req_data['data']
    model_path = os.path.abspath('..\modelos_para_classes')
    for i in range(len(req_array)):
        class_id = str(class_array[i])
        model_name = model_path+"\model"+class_id+".pkl"

        model_vocab = model_path+"\model"+class_id+"_vocabulary.pkl"
        
        with open(model_name, 'rb') as f:
            loaded_temp_model = joblib.load(f)
    
        with open(model_vocab, 'rb') as f1:
            loaded_temp_vocab = joblib.load(f1)        
        
        pred_genero, prob_genero = predict_genero(loaded_temp_model, loaded_temp_vocab, req_array[i]['descricao'])
        genero_array.append(pred_genero[0])
        prob_array.append(prob_genero[0][0])
        
    print(prob_array)
    output_json = json_concatenation(req_data,'data', genero_array, prob_array, config()) 
    print(output_json)
    return output_json

##CRIA UM JSON PARA A SAÍDA
def json_concatenation(input_json, json_key, predictions_list, prob_list, configuracao):
    teste_predict = input_json[json_key]
    output_dict= []
    if (bool(configuracao['ativado'])):
        for i in range(len(predictions_list)):
            if (prob_list[i]>float(configuracao['porcentagem'])):
                data = {
                    "id": teste_predict[i]['id'],
                    "descricao": teste_predict[i]['descricao'],
                    "genero": predictions_list[i],
                    "probabilidade": prob_list[i]
                }
                output_dict.append(data)
            
        output_json = {"data" : output_dict}
        return jsonify(output_json)
    else:
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
   
        predicao = list_to_dict(pred_prob_list)
        data['predicao'] = predicao[i]
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
            genero = loaded_model.classes_[i.index(max(i))]
            top_probabilities.append(genero)
            top_probabilities.append(max(i))
            top_element = i.index(max(i))
            del i[top_element]
        max_probabilities.append(top_probabilities)
        
    return max_probabilities
        

def list_to_dict(list):
    ndict = []
    for item in range(len(list)):
        predicao = dict()
        for j in range(0, len(list[item]), 2):
            key = "top" + str(int(j/2)+1)
            genero = list[item][j]
            probabilidade = list[item][j+1]
            item_genero = dict(zip(["genero"],[genero]))
            item_prob = dict(zip(["probabilidade"],[probabilidade]))
            top = dict(zip([genero],[probabilidade]))
            probs = {
                key: {
                    "genero": item_genero['genero'],
                    "probabilidade": item_prob['probabilidade']
                }
            }
            predicao.update(probs)
        ndict.append(predicao)

    return ndict


def config():
    arq = open('config.txt', 'r')
    texto = arq.readlines()
    parametros={}
    for linha in texto:
        linha=linha.replace('\n','')
        linha=linha.split(' = ')
        if (linha[1]=='False'):
            linha[1]=''
        parametros[linha[0]] = linha[1]
    return parametros

def predict_class(model, vocab, req_data):
    vectorizer_train = CountVectorizer(vocabulary=vocab)
    teste_predict = []
    req_array = req_data['data']
    for i in range(len(req_array)):
        desc = req_array[i]['descricao']
        teste_predict.append(desc)
    
    teste_predict_vect = vectorizer_train.transform(teste_predict) 
    class_predictions = model.predict(teste_predict_vect)
    
    print(class_predictions)
    return class_predictions

def predict_genero(model, vocab, data):
    vectorizer_train = CountVectorizer(vocabulary=vocab)
    
    data_v =[]
    data_v.append(data) 
    
    teste_predict_vect = vectorizer_train.transform(data_v) 
    genero_predictions = model.predict(teste_predict_vect)
    
    prob_list = model.predict_proba(teste_predict_vect).tolist()
    #print(genero_predictions)
    return genero_predictions, prob_list

def initialize_models():
    
    ##
    filename = 'Classificador-v2\model_v2.pkl'
    #filename2 = '..\modelos_para_classes\model_Classe_Descricao.pkl'
    filename2 = os.path.abspath('..\modelos_para_classes\model_Classe_Descricao.pkl')
    ##
    with open(filename ,'rb') as f:
        loaded_model = joblib.load(f)
        
    with open(filename2 ,'rb') as f2:
        loaded_model_classe_descricao = joblib.load(f2)

    ##
    vocab_filename = 'Classificador-v2\model_v2_vocabulary.pkl'
    vocab_filename2 = '..\modelos_para_classes\model_Classe_Descricao_vocabulary.pkl'
    ##
    with open(vocab_filename, 'rb') as f3:
        loaded_vocabulary = joblib.load(f3)
    
    with open(vocab_filename2, 'rb') as f4:
        loaded_vocabulary_classe_descricao = joblib.load(f4)
        
    return loaded_model, loaded_model_classe_descricao, loaded_vocabulary, loaded_vocabulary_classe_descricao 


##CARREGANDO MODELOS
loaded_model, loaded_model_classe_descricao, loaded_vocabulary, loaded_vocabulary_classe_descricao = initialize_models()

##INICIANDO SERVIDOR
app.run()
