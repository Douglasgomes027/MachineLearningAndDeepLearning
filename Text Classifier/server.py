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
    req_data = request.get_json()
    teste_predict = []
    req_array = req_data['data']
    for i in range(len(req_array)):
        desc = req_array[i]['descricao']
        teste_predict.append(desc)
       
    teste_predict_vect = vectorizer_train.transform(teste_predict) 
    predictions = loaded_model.predict(teste_predict_vect)
 
    output_json = json_concatenation(req_data,'data', predictions.tolist())
    
    return output_json

##
@app.route("/")
def hello_world():
    return "Hello World! <strong>I am learning Flask</strong>", 200

##CRIA UM JSON PARA A SAÍDA
def json_concatenation(input_json, json_key, output_list):
    teste_predict = input_json[json_key]
    output_dict= []
    for i in range(len(output_list)):
        data = {
            "descricao" : teste_predict[i]['descricao'],
            "genero" : output_list[i]
        }
        output_dict.append(data)
        
    #output_json = json.dumps(output_dict)
    output_json = {"data" : output_dict}
    #output_json = jsonify(output_dict)
    return jsonify(output_json)
    

##INICIANDO SERVIDOR
app.run()

'''
{
	"data" : [
		{
			"descricao": "asdasds",
		},
		{
			"descricao": "asdasds",
		}
		]
}
'''

