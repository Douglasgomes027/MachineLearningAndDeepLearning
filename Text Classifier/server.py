from flask import Flask, jsonify, request, url_for
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import os, json

##
app = Flask(__name__)

##REALIZA A PREDICAO E RESPONDE COM AS CATEGORIAS
@app.route("/predictions", methods=['POST'])
def test():
    
    ##IMPORTA O MODELO E O VOCABULARIO
    filename = 'model_v1.pkl'

    ##
    with open(filename ,'rb') as f:
        loaded_model = joblib.load(f)

    ##
    vocab_filename = 'model_vocabulary.pkl'

    ##
    with open(vocab_filename, 'rb') as f2:
        loaded_vocabulary = joblib.load(f2)

    ##CARREGA O VOCABULARIO NO NOSSO VECTORIZER
    vectorizer_train = CountVectorizer(vocabulary=loaded_vocabulary)

    ##PEGA OS DADOS DA REQUISICAO E FAZ A PREDICAO
    req_data = request.get_json()
    teste_predict = req_data['data']
    teste_predict_vect = vectorizer_train.transform(teste_predict) 
    predictions = loaded_model.predict(teste_predict_vect)
 
    output_json = json_concatenation(req_data,'data', predictions.tolist())
    
    return output_json

##CRIA UM JSON COM A DESCRICAO ENTRADA E O GENERO ESPERADO
def json_concatenation(input_json, json_key, output_list):
    teste_predict = input_json[json_key]
    output_dict= dict()
    for i in range(len(output_list)):
        data = {
            teste_predict[i] : output_list[i]
        }
        output_dict.update(data)
        
    #output_json = json.dumps(output_dict)
    output_json = jsonify(output_dict)
    return output_json
    

##INICIANDO SERVIDOR
app.run()
