import os
import numpy as np
from features import FeatureExtractor
from dataTransformer import Transformer

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
            return {"Status":"failed","Message":'Please Pass the Word2Vec Keyvector.',"intent":"null","confidence":"null"}, False
        
        for intent in groupedData.keys():
            sentences = groupedData[intent]
            try:
                feature_vector = extractor.get_feature_vector_train(sentences,self.vocab_file_path)
                print(self.trained_data_home)
                np.save(os.path.join(self.trained_data_home,str(intent)+".npy"),feature_vector)
            except:

                return {"Status":"failed","Message":'Something bad happened during saving the weights.',"intent":"null","confidence":"null"}, False
        
        return True, True

