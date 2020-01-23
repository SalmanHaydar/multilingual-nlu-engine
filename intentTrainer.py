import os
import numpy as np
import config as cfg
from features import FeatureExtractor
from dataTransformer import Transformer
from celery import Celery
import traceback
from gensim.models import KeyedVectors

celery = Celery('main', broker=cfg.BROKER_URL, backend=cfg.BROKER_URL)

class IntentTrainer:
    def __init__(self,botid,wv=None):
        self.wv = wv
        self.botid = botid
        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        self.BOT_BASE = os.path.join(self.abs_path,"BOT_DATA")
        self.BOT_HOME = os.path.join(self.BOT_BASE,botid)
        self.trained_data_home = os.path.join(self.BOT_HOME,"trained_data")
        # training_data_home = os.path.join(BOT_HOME,"training_data")
        self.vocab_home = os.path.join(self.BOT_HOME,"vocab_repo")
        self.vocab_file_path = os.path.join(self.vocab_home,botid+".pickle")

    def get_data(self):
        obj = Transformer(self.botid)
        dataDict, status = obj.getDataForIntentModel()

        return dataDict, status

    def get_extractor(self):
        if self.wv:
            extractor = FeatureExtractor(self.wv)
            return extractor
        else:
            return False

    def train(self):
        groupedData, status = self.get_data()

        if not status:
            return groupedData # It is the Response not Data

        extractor = self.get_extractor()

        if not extractor:
            raise Exception("Please Pass the Word2Vec Keyvector.")
            # return {"Status":"failed","Message":'Please Pass the Word2Vec Keyvector.',"intent":"null","confidence":"null"}, False
        
        try:
            global_train_and_save.delay(groupedData, self.vocab_file_path, self.trained_data_home)
            return True
        except Exception:
            traceback.print_exc()
            return False

@celery.task
def global_train_and_save(groupedData, vocab_file_path, trained_data_home):
    try:
        # print("Loading Model in a RAM...")
        wv = KeyedVectors.load(cfg.MODEL_PATH)
        print("Model Loaded in a RAM successfully...")
    except:
        raise Exception("Cannot Load Embedding Model.")

    extractor = FeatureExtractor(wv)
    print("Intent Model is Training")
    for intent in groupedData.keys():
        
        sentences = groupedData[intent]
        try:
            feature_vector = extractor.get_feature_vector_train(sentences,vocab_file_path)
            np.save(os.path.join(trained_data_home,str(intent)+".npy"),feature_vector)
        except:
            raise Exception("Something bad happened during saving the weights.")
