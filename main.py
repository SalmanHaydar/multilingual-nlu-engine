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
from celery import Celery

from features import FeatureExtractor
from AddExamples import AddExamples
from trainingManager import Trainer
from inferenceManager import Inference
from DButills import DButills
import config as cfg

warnings.filterwarnings('ignore')

app = Flask(__name__)
cors = CORS(app, resources={r"/getintent/*": {"origins": "*"}})

app.config['CELERY_BROKER_URL'] = cfg.BROKER_URL
app.config['CELERY_RESULT_BACKEND'] = cfg.BROKER_URL

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


try:
    print("Loading Model in a RAM...")
    wv = KeyedVectors.load(cfg.MODEL_PATH)
    print("Model Loaded in a RAM successfully...")
except:
    raise Exception("Cannot Load Embedding Model.")


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

    """{ "botID" : "train", "sentence" : "I want to book a ticket from dhaka to dinajpur", 
        "intent" : "buy_ticket", "entity" : {"dhaka" : "location", "dinajpur" : "location"}}"""

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
    # print(response)
    return Response(json.dumps(response),mimetype='application/json')
  else:
    return Response(json.dumps({"Status":"failed","Message":'This method is not allowed',"intent":"null","confidence":"null"}),status= 405,mimetype='application/json')


@app.route("/createprofile",methods=["GET","POST"])
def createprofile():
  bot_id = request.args.get("botID")
  if request.method=="POST":
    obj = DButills(bot_id)
    response = obj.createBotProfile()

    return Response(json.dumps(response),status= 200,mimetype='application/json')
  else:
    return Response(json.dumps({"Status":"failed","Message":'This method is not allowed',"intent":"null","confidence":"null"}),status= 405,mimetype='application/json')


@app.route("/",methods=["GET","POST"])
def index():
  if request.method == "GET" or request.method=="POST":

    return render_template('sample-add.html')
    # return json.dumps('please see the api documentation.. https://docs.google.com/document/d/1twwLx2g315XpymM60ia7XZMF7ve-lTvm0IrbgfFtErg/edit?usp=sharing')

if __name__=="__main__":
  app.run(host="0.0.0.0",port=5000,debug=True)
 
