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
from AddExamples import AddExamples
from trainingManager import Trainer
from inferenceManager import Inference

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

    bot_id = request.args.get("botID")
    sentence = request.args.get("sentence").lower()

    inf = Inference(bot_id, sentence, wv)

    response = inf.infer()

    return Response(json.dumps(response),status=200,mimetype='application/json')
  else:
      return Response(json.dumps({"Status":"failed","Message":'This method is not allowed',"intent":"null","confidence":"null"}),status=405,mimetype='application/json')



@app.route("/addexample",methods=["GET","POST","PUT"])
def addexample():

  if request.method == "PUT":

    data = request.get_json(force=True)

    bot_id = data.get("botID")
    sentence = data.get("sentence")
    intent = data.get("intent")
    entity = data.get("entity")

    add_obj = AddExamples(bot_id, sentence, intent, entity)
    response = add_obj.insertOne()

    return Response(json.dumps(response),mimetype='application/json')
  else:

    return Response(json.dumps({"Status":"failed","Message":'This method is not allowed',"intent":"null","confidence":"null"}),mimetype='application/json',status=405)


@app.route("/train",methods=["GET","POST"])
def train():
  if request.method=="POST":
    bot_id = request.args.get("botID")

    obj = Trainer(bot_id)
    response = obj.train(wv)

    return Response(json.dumps(response),mimetype='application/json')
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
          os.mkdir(BOT_HOME+"/entity_model")

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
