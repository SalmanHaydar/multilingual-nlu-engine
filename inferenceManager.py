import os
import config as cfg
from features import FeatureExtractor
from intentPredictor import IntPredictor
from DButills import DButills

class Inference:

    def __init__(self,botid,sentence,wv):
        self.botid = botid
        self.sentence = sentence
        self.wv = wv 

    def get_file_path(self):

        vocabulary_file_name = str(self.botid)+".pickle"
        vocabulary_path = os.path.join(cfg.BOT_BASE,str(self.botid)+"/vocab_repo/"+vocabulary_file_name)

        return vocabulary_path


    def predict_intent(self):
        
        obj = FeatureExtractor(self.wv)

        vocabulary_path = self.get_file_path()
        
        query_vec, matched_words_ratio = obj.get_feature_vector_infer(self.sentence,vocabulary_path)

        response = IntPredictor(self.botid,query_vec,matched_words_ratio).predict()

        return response

    def predict_entity(self):
        pass

    def infer(self):

        if DButills(self.botid).doesThisBOTExist():

            response = self.predict_intent()

            return response
        else:

            return {"Status":"failed","Message":'There is no bot with this bot id',"intent":"null","confidence":"null"}

        