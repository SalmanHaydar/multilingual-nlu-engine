from flask import Flask, request, jsonify,session, render_template,redirect, url_for
import numpy as np
import pandas as pd
from flair.embeddings import ELMoEmbeddings
from flair.data import Sentence
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
from features import FeatureExtractor

warnings.filterwarnings('ignore')

app = Flask(__name__)
cors = CORS(app, resources={r"/getintent/*": {"origins": "*"}})
abs_path = os.path.dirname(os.path.abspath(__file__))
allenNLP = os.path.join(abs_path,"MODELS/.allennlp")
BOT_BASE = os.path.join(abs_path,"BOT_DATA")

os.environ["ALLENNLP_CACHE_ROOT"] = allenNLP

try:
    print("Loading Model in a RAM...")
    elmo_embedding = ELMoEmbeddings('original')
    print("Model Loaded in a RAM successfully...")
except:
    print("Cannot Load Embedding Model")
    print(os.getenv("ALLENNLP_CACHE_ROOT"))

extractor = FeatureExtractor(elmo_embedding)

@app.route("/getintent",methods=["GET","POST"])
def getIntent():

  if request.method == "GET":

    payloads = {"intent":"","confidence":""}
    bot_id = request.args.get("botID")
    sentence = request.args.get("sentence")
    # print(request.args)
    query_vec = extractor.get_feature_vector_infer(sentence)
    # print(query_vec.shape)

    if(os.path.exists(os.path.join(BOT_BASE,bot_id))):
    
      into_home = os.path.join(BOT_BASE,bot_id)
      if(os.path.exists(os.path.join(into_home,"trained_data"))):
        intent_files = [ f.split(".")[0] for f in os.listdir(os.path.join(into_home,"trained_data"))]
        result_dic = {}
        print(intent_files)
        trained_data_repo = os.path.join(into_home,"trained_data")
        # print(os.listdir(os.path.join(into_home,"trained_data")))
        for intent in intent_files:
            intent_vec = np.load(os.path.join(trained_data_repo,intent+".npy"))
            result_dic[intent] = str(cosine_similarity(query_vec,intent_vec)[0][0])
        print(result_dic)
        res = max(result_dic, key=result_dic.get)

        if float(result_dic[res]) < 0.45:
          payloads["intent"] = "unknown"
        else:
          payloads["intent"] = res

        payloads["confidence"] = result_dic

        return json.dumps(payloads)

      else:
        return json.dumps('Error!! Your Bot is not yet trained')

    else:
      return json.dumps('Error!! No Bot found with this Bot ID'), 404

  else:
      return json.dumps('This method is not allowed'), 405


@app.route("/addexample",methods=["GET","POST","PUT"])
def addexample():
  #expected that bot profile is created beforehand
  #file format .csv with 

  if request.method == "PUT":
    bot_id = request.args.get("botID")
    sentence = request.args.get("sentence")
    intent = request.args.get("intent")
    BOT_HOME = os.path.join(BOT_BASE,bot_id)
    training_data_home = os.path.join(BOT_HOME,"training_data")
    if os.path.exists(os.path.join(training_data_home,bot_id+".csv")):
      #if the file exist insert the data to the existing data repository
      df = pd.DataFrame({"sentence":[sentence],"intent":[intent]})
      df.to_csv(os.path.join(training_data_home,bot_id+".csv"),mode="a",header=False,index=False)
      return json.dumps('Added Successfully!')
    else:
      #else create a new data repository with the name <bot_id.csv>
      df = pd.DataFrame({"sentence":[sentence],"intent":[intent]})
      df.to_csv(os.path.join(training_data_home,bot_id+".csv"),columns=df.columns,index=False)
      return json.dumps('Added Successfully!')
  else:
    #for the methods other than PUT
    return json.dumps('This method is not allowed!'), 405

@app.route("/train",methods=["GET","POST"])
def train():
  if request.method=="POST":
    bot_id = request.args.get("botID")
    BOT_HOME = os.path.join(BOT_BASE,bot_id)
    trained_data_home = os.path.join(BOT_HOME,"trained_data")
    training_data_home = os.path.join(BOT_HOME,"training_data")

    if os.path.exists(os.path.join(training_data_home,bot_id+".csv")):
      df = pd.read_csv(os.path.join(training_data_home,bot_id+".csv"))
      grouped_df = df.groupby("intent")
      intents = list(grouped_df.groups.keys())
      for intent in intents:
        try:
          filtered_group = grouped_df.get_group(intent)
          sentences = list(filtered_group.sentence)
          feature_vector = extractor.get_feature_vector_train(sentences)
          np.save(os.path.join(trained_data_home,intent+".npy"),feature_vector)
        except:
          return json.dumps('Something bad happened'),500

      return json.dumps('Training has completed')

    else:
      return json.dumps('You have no data to train.'), 404
    
  else:
    return json.dumps('*This method is not allowed'), 405

@app.route("/createprofile",methods=["GET","POST"])
def createprofile():
  bot_id = request.args.get("botID")
  if request.method=="POST":
    if not os.path.exists(os.path.join(BOT_BASE,bot_id)):
      r = os.system("mkdir "+BOT_BASE+"/"+bot_id)
      if r==0:
        BOT_HOME = os.path.join(BOT_BASE,bot_id)
        rr = os.system("mkdir "+BOT_HOME+"/training_data && mkdir "+BOT_HOME+"/trained_data")
        if rr==0:
          return json.dumps('Bot profile has created successfully!')
    else:
      return json.dumps('Bot with this ID is already exists')

@app.route("/",methods=["GET","POST"])
def index():
  if request.method == "GET" or request.method=="POST":
    return render_template('index.html')
    # return json.dumps('please see the api documentation.. https://docs.google.com/document/d/1twwLx2g315XpymM60ia7XZMF7ve-lTvm0IrbgfFtErg/edit?usp=sharing')

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)