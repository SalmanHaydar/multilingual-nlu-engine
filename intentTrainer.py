import os
import numpy as np
import config as cfg
from features import FeatureExtractor
from dataTransformer import Transformer
from celery import Celery
import traceback
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from pickle import dump
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
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
        dataRow, status = obj.getDataForIntentModel()

        return dataRow, status

    def get_extractor(self):
        if self.wv:
            extractor = FeatureExtractor(self.wv)
            return extractor
        else:
            return {"Status":"failed","Message":'There is no data to train.',"intent":"null","confidence":"null"}

    def get_feature_for_text_classification(self, row, maxlen=10, padding_with=0):

        temp_list = []
        features = {}
        
        features["Bias"] = 1.0
    #     features["length"] = len(row['sentence'].split(" "))
        
        # offer ki ache
        for i,word in enumerate(row['sentence'].split(" ")):
            if i+1<=maxlen:
                features['word'+str(i+1)] = word
                try:
                    features['word'+str(i+1)+"-w2v"] = np.mean(self.wv[word])
                except:
                    features['word'+str(i+1)+"-w2v"] = 0.0

        if i+1<maxlen:
            for j in range(i+1, maxlen):
                features['word'+str(j+1)] = ""
                features['word'+str(j+1)+"-w2v"] = padding_with
                
        temp_list.append(features)
            
        return temp_list

    def get_label_for_text_classification(self, row):

        return [row['intent']]

    def get_classes(self,dataRow):
        intents = {}

        for row in dataRow:
            intents[row['intent']] = 1
        
        return list(intents.keys())

    def train(self):
        dataRows, status = self.get_data()
        if not status:
            return dataRows # It is the Response not Data
        dataRows = [row for row in dataRows]

        if len(dataRows)<1:

            return False

        X_train = [self.get_feature_for_text_classification(row, maxlen=10) for row in dataRows]
        y_train = [self.get_label_for_text_classification(row) for row in dataRows if 'intent' in row.keys()]

        classes = self.get_classes(dataRows)

        try:
            global_train_and_save_CRF.delay(self.botid, str(X_train), y_train, classes)
            return True
        except Exception:
            traceback.print_exc()
            return False

        # extractor = self.get_extractor()

        # if not extractor:
        #     raise Exception("Please Pass the Word2Vec Keyvector.")
        #     # return {"Status":"failed","Message":'Please Pass the Word2Vec Keyvector.',"intent":"null","confidence":"null"}, False
        
        # try:
        #     global_train_and_save.delay(groupedData, self.vocab_file_path, self.trained_data_home)
        #     return True
        # except Exception:
        #     traceback.print_exc()
        #     return False

# @celery.task
# def global_train_and_save(groupedData, vocab_file_path, trained_data_home):
#     try:
#         # print("Loading Model in a RAM...")
#         wv = KeyedVectors.load(cfg.MODEL_PATH)
#         print("Model Loaded in a RAM successfully...")
#     except:
#         raise Exception("Cannot Load Embedding Model.")

#     extractor = FeatureExtractor(wv)
#     print("Intent Model is Training")
#     for intent in groupedData.keys():
        
#         sentences = groupedData[intent]
#         try:
#             feature_vector = extractor.get_feature_vector_train(sentences,vocab_file_path)
#             np.save(os.path.join(trained_data_home,str(intent)+".npy"),feature_vector)
#         except:
#             raise Exception("Something bad happened during saving the weights.")

@celery.task
def global_train_and_save_CRF(botid, X_train, y_train, classes):
    X_train = eval(X_train)
    crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs', 
            max_iterations=100, 
            all_possible_transitions=True
            )
    params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
    }

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score, 
                            average='weighted', labels=classes)

    # search
    model = RandomizedSearchCV(crf, params_space, 
                            cv=5, 
                            verbose=1, 
                            n_jobs=-1, 
                            n_iter=50, 
                            scoring=f1_scorer)                              

    model.fit(X_train,y_train)

    BOT_HOME = os.path.join(cfg.BOT_BASE, botid)
    intent_model_path = os.path.join(BOT_HOME, 'intent_model')

    with open(os.path.join(intent_model_path, str(botid) + '.pkl'), 'wb') as f:
        try:
            dump(model.best_estimator_, f)
            print("Intent Model Saved successfully.")
        except FileNotFoundError as fnf_error:
            print(fnf_error)