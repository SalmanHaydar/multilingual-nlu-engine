from flask import Flask, request, jsonify,session, render_template,redirect, url_for
import numpy as np
import pandas as pd
from flair.embeddings import ELMoEmbeddings
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
from pathlib import Path
import warnings
import heapq
import random
import json
import ast
import re
import os
from flask_cors import CORS

warnings.filterwarnings('ignore')

app = Flask(__name__)
cors = CORS(app, resources={r"/getIntent/*": {"origins": "*"}})
abs_path = os.path.dirname(os.path.abspath(__file__))
allenNLP = os.path.join(abs_path,"MODELS/.allennlp")
BOT_HOME = os.path.join(abs_path,"BOT_DATA")

os.environ["ALLENNLP_CACHE_ROOT"] = allenNLP

try:
    print("Loeading Model in a RAM...")
    elmo_embedding = ELMoEmbeddings('original')
    print("Model Loaded in a RAM successfully...")
except:
    print("Cannot Load Embedding Model")
    print(os.getenv("ALLENNLP_CACHE_ROOT"))

def get_feature_vector_train(sentences,embedding=None):
  if embedding == None:
    raise Exception('Please pass flair embedding object.')
  else:
    sentence_vectors = []
    for sent in sentences:
      s = Sentence(sent)
      embedding.embed(s)
      
      temp_placeholder = []
      for token in s:
        temp_placeholder.append(token.embedding.numpy())
        
      sentence_vectors.append(np.mean(temp_placeholder,axis=0))
      
      
    return np.mean(sentence_vectors,axis=0)

def get_feature_vector_infer(sentence,embedding=None):
  if embedding == None:
    raise Exception('Please pass flair embedding object.')
  else:
    
    sent = Sentence(sentence)
    embedding.embed(sent)

    temp_placeholder = []
    for token in sent:
      temp_placeholder.append(token.embedding.numpy())

    sent_embd = np.mean(temp_placeholder,axis=0)

    return sent_embd

@app.route("/getIntent",methods=["GET","POST"])
def getIntent():
    
    
    if request.method == "GET":
        
        payloads = {"intent":"","confidence":""}
        bot_id = request.args.get("botID")
        sentence = request.args.get("sentence")
        print(request.args)
        query_vec = get_feature_vector_infer(sentence,elmo_embedding)
        print(query_vec.shape)

        if(os.path.exists(os.path.join(BOT_HOME,bot_id))):
            into_home = os.path.join(BOT_HOME,bot_id)
            if(os.path.exists(os.path.join(into_home,"trained_data"))):
                intent_files = [ f.split(".")[0] for f in os.listdir(os.path.join(into_home,"trained_data"))]
                result_dic = {}
                print(intent_files)
                trained_data_repo = os.path.join(into_home,"trained_data")
                # print(os.listdir(os.path.join(into_home,"trained_data")))
                for intent in intent_files:
                    intent_vec = np.load(os.path.join(trained_data_repo,intent+".npy"))
                    result_dic[intent] = cosine_similarity(query_vec,intent_vec)
                print(result_dic)
                res = max(result_dic, key=result_dic.get)
                payloads["intent"] = res
                payloads["confidence"] = str(result_dic)

                return json.dumps(payloads)
            else:
                return json.dumps(['Error!! Your Bot is not yet trained'])

        return json.dumps(['Error!! No Bot found with this Bot ID'])

    else:
        return json.dumps(['This method is not allowed'])




if __name__=="__main__":
    app.run(debug=True)