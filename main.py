from flask import Flask, request, jsonify,session, render_template,redirect, url_for
from flask import Response
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
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
from scipy.spatial import distance
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from features import FeatureExtractor

warnings.filterwarnings('ignore')

app = Flask(__name__)
cors = CORS(app, resources={r"/getintent/*": {"origins": "*"}})
abs_path = os.path.dirname(os.path.abspath(__file__))

BOT_BASE = os.path.join(abs_path,"BOT_DATA")


try:
    print("Loading Model in a RAM...")
    wv = KeyedVectors.load(os.path.join(abs_path,"MODELS/keyesvectors.kv"))
    print("Model Loaded in a RAM successfully...")
except:
    print("Cannot Load Embedding Model")

extractor = FeatureExtractor(wv)

@app.route("/getintent",methods=["GET","POST"])
def getIntent():

  if request.method == "GET":

    payloads = {"Status":"","Message":"","intent":"","confidence":""}
    bot_id = request.args.get("botID")
    sentence = request.args.get("sentence").lower()
    # print(request.args)
    
    # print(query_vec.shape)

    if(os.path.exists(os.path.join(BOT_BASE,bot_id))):
      
      into_home = os.path.join(BOT_BASE,bot_id)

      vocab_home = os.path.join(into_home,"vocab_repo")
      vocab_file_path = os.path.join(vocab_home,bot_id+".pickle")
      query_vec,matched_words_ratio = extractor.get_feature_vector_infer(sentence,vocab_file_path)

      if(os.path.exists(os.path.join(into_home,"trained_data"))):
        intent_files = [ f.split(".")[0] for f in os.listdir(os.path.join(into_home,"trained_data"))]

        result_dic = {}
        confidence = {}
        confidence_cosine = {}

        print(intent_files)
        trained_data_repo = os.path.join(into_home,"trained_data")
        # print(os.listdir(os.path.join(into_home,"trained_data")))
        for intent in intent_files:
            intent_vec = np.load(os.path.join(trained_data_repo,intent+".npy"))
            result_dic[intent] = cosine_similarity(query_vec,intent_vec)[0][0]
            # confidence[intent] = str(distance.euclidean(query_vec,intent_vec))
            confidence_cosine[intent] = str(cosine_similarity(query_vec,intent_vec)[0][0])
        print(result_dic)
        res = max(result_dic, key=result_dic.get)

        if matched_words_ratio*100>50.0 and float(result_dic[res])*100 < 50.0:
          payloads["intent"] = "unknown"
          payloads["Status"] = "success"
          payloads["Message"] = "Couldn't understand properly"
        elif matched_words_ratio*100<50.0 and float(result_dic[res])*100 < 70.0:
          payloads["intent"] = "unknown"
          payloads["Status"] = "success"
          payloads["Message"] = "Couldn't understand properly"
        else:
          payloads["intent"] = res
          payloads["Status"] = "success"
          payloads["Message"] = "null"

        # payloads["confidence"] = confidence
        payloads['confidence'] = confidence_cosine

        return Response(json.dumps(payloads),mimetype='application/json')

      else:
        
        return Response(json.dumps({"Status":"failed","Message":'Error!! Your Bot is not yet trained',"intent":"null","confidence":"null"}),mimetype='application/json')

    else:
      
      return Response(json.dumps({"Status":"failed","Message":'Error!! No Bot found with this Bot ID',"intent":"null","confidence":"null"}),status=404,mimetype='application/json')

  else:
      return Response(json.dumps({"Status":"failed","Message":'This method is not allowed',"intent":"null","confidence":"null"}),status=405,mimetype='application/json')


@app.route("/addexample",methods=["GET","POST","PUT"])
def addexample():
  #expected that bot profile is created beforehand
  #file format .csv with 

  if request.method == "PUT":
    bot_id = request.args.get("botID")
    sentence = request.args.get("sentence").lower()
    intent = request.args.get("intent")
    BOT_HOME = os.path.join(BOT_BASE,bot_id)
    training_data_home = os.path.join(BOT_HOME,"training_data")
    if os.path.exists(os.path.join(training_data_home,bot_id+".csv")):
      #if the file exist insert the data to the existing data repository
      df = pd.DataFrame({"sentence":[sentence],"intent":[intent]})
      df.to_csv(os.path.join(training_data_home,bot_id+".csv"),mode="a",header=False,index=False)
      return Response(json.dumps({"Status":"success","Message":'Added Successfully',"intent":"null","confidence":"null"}),mimetype='application/json')
    else:
      #else create a new data repository with the name <bot_id.csv>
      df = pd.DataFrame({"sentence":[sentence],"intent":[intent]})
      df.to_csv(os.path.join(training_data_home,bot_id+".csv"),columns=df.columns,index=False)
      return Response(json.dumps({"Status":"success","Message":'Added Successfully',"intent":"null","confidence":"null"}),mimetype='application/json')
  else:
    #for the methods other than PUT
    return Response(json.dumps({"Status":"failed","Message":'This method is not allowed',"intent":"null","confidence":"null"}),mimetype='application/json',status=405)

@app.route("/train",methods=["GET","POST"])
def train():
  if request.method=="POST":
    bot_id = request.args.get("botID")
    BOT_HOME = os.path.join(BOT_BASE,bot_id)
    trained_data_home = os.path.join(BOT_HOME,"trained_data")
    training_data_home = os.path.join(BOT_HOME,"training_data")
    vocab_home = os.path.join(BOT_HOME,"vocab_repo")
    vocab_file_path = os.path.join(vocab_home,bot_id+".pickle")

    if os.path.exists(os.path.join(training_data_home,bot_id+".csv")):
      df = pd.read_csv(os.path.join(training_data_home,bot_id+".csv"))
      grouped_df = df.groupby("intent")
      intents = list(grouped_df.groups.keys())
      for intent in intents:
        try:
          filtered_group = grouped_df.get_group(intent)
          sentences = list(filtered_group.sentence)
          feature_vector = extractor.get_feature_vector_train(sentences,vocab_file_path)
          np.save(os.path.join(trained_data_home,intent+".npy"),feature_vector)
        except:
          return Response(json.dumps({"Status":"failed","Message":'INTERNAL ERROR!',"intent":"null","confidence":"null"}),status=500,mimetype='application/json')

      return Response(json.dumps({"Status":"success","Message":'Training has completed',"intent":"null","confidence":"null"}),mimetype='application/json')

    else:
      return Response(json.dumps({"Status":"failed","Message":'You have no data to train',"intent":"null","confidence":"null"}),status=404,mimetype='application/json')
    
  else:
    return Response(json.dumps({"Status":"failed","Message":'This method is not allowed',"intent":"null","confidence":"null"}),status= 405,mimetype='application/json')

@app.route("/createprofile",methods=["GET","POST"])
def createprofile():
  bot_id = request.args.get("botID")
  if request.method=="POST":
    if not os.path.exists(os.path.join(BOT_BASE,bot_id)):
      try:
        os.mkdir(BOT_BASE+"/"+bot_id)
        try:
          BOT_HOME = os.path.join(BOT_BASE,bot_id)
          os.mkdir(BOT_HOME+"/training_data")
          os.mkdir(BOT_HOME+"/trained_data")
          os.mkdir(BOT_HOME+"/vocab_repo")

          return Response(json.dumps({"Status":"success","Message":'Bot Profile has created successfully',"intent":"null","confidence":"null"}),mimetype='application/json')
        except:
          return Response(json.dumps({"Status":"failed","Message":'cannot create the profile',"intent":"null","confidence":"null"}),mimetype='application/json')
      except:
        return Response(json.dumps({"Status":"failed","Message":'cannot create the profile',"intent":"null","confidence":"null"}),mimetype='application/json')
      # if r==0:
      #  
      #   BOT_HOME = os.path.join(BOT_BASE,bot_id)
      #   rr = os.mkdir(BOT_HOME+"/training_data")
      #   rrr = os.mkdir(BOT_HOME+"/trained_data")
      #   if rr==0 and rrr==0:
      #     return json.dumps('Bot profile has created successfully!')
      #   else:
      #     return json.dumps('cannot create the profile [rr]')
      # else:
      #   return json.dumps('cannot create the profile [r]')
    else:
      
      return Response(json.dumps({"Status":"failed","Message":'Bot with this ID is already exists',"intent":"null","confidence":"null"}),mimetype='application/json')

@app.route("/",methods=["GET","POST"])
def index():
  if request.method == "GET" or request.method=="POST":
    return render_template('sample-add.html')
    # return json.dumps('please see the api documentation.. https://docs.google.com/document/d/1twwLx2g315XpymM60ia7XZMF7ve-lTvm0IrbgfFtErg/edit?usp=sharing')

if __name__=="__main__":
  app.run(host="0.0.0.0",port=5000,debug=True)