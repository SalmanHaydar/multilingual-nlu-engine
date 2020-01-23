import os
import config as cfg
from pickle import load
from features import FeatureExtractor
from intentPredictor import IntPredictor
from DButills import DButills
from entityPredictor import EntPredictor


class Inference:

    def __init__(self,botid,sentence,wv):
        self.botid = botid
        self.sentence = sentence
        self.wv = wv 

    def get_vocab_file_path(self):

        vocabulary_file_name = str(self.botid)+".pickle"
        vocabulary_path = os.path.join(cfg.BOT_BASE,str(self.botid)+"/vocab_repo/"+vocabulary_file_name)

        return vocabulary_path

    def get_model_file_path(self):

        model_name = str(self.botid)+".pkl"
        model_path = os.path.join(cfg.BOT_BASE,str(self.botid)+"/entity_model/"+model_name)

        return model_path

    def predict_intent(self):
        
        obj = FeatureExtractor(self.wv)

        vocabulary_path = self.get_vocab_file_path()
        
        query_vec, matched_words_ratio = obj.get_feature_vector_infer(self.sentence,vocabulary_path)

        response = IntPredictor(self.botid,query_vec,matched_words_ratio).predict()

        return response

    def load_entity_model(self):
        try:
            model = load(open(self.get_model_file_path(), 'rb'))
        except:
            raise Exception("Cannot Load the model")
    
        return model
        
    def predict_entity(self):
        model = self.load_entity_model()
        lookupTable = lookupTable = DButills(self.botid).getLookupTable()

        prediction = EntPredictor(model, self.sentence, lookupTable).predict()

        return prediction

    def infer(self):

        if DButills(self.botid).doesThisBOTExist():

            response = self.predict_intent()

            entity_response = self.predict_entity()
            response["entities"] = entity_response
            # print(entity_response)
            return response
        else:

            return {"Status":"failed","Message":'There is no bot with this bot id',"intent":"null","confidence":"null"}

        