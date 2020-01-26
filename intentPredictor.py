import os
import numpy as np
from scipy.spatial import distance
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
import config as cfg


class IntPredictor:
    def __init__(self, bot_id, features, model):
        
        self.botid = bot_id
        self.features = features
        self.model = model

    def predict(self):

        payloads = {"Status":"","Message":"","intent":"","confidence":""}
        into_home = os.path.join(cfg.BOT_BASE,str(self.botid))
        model_file = os.path.join(into_home,'intent_model/'+str(self.botid)+'.pkl')

        if(os.path.exists(model_file)):

            prediction = self.model.predict_marginals_single(self.features)
            res = max(prediction[0], key=prediction[0].get)

            confidence = {}
            for intent in prediction[0].keys():
                confidence[intent] = str(prediction[0][intent])
            
            res_confidence = prediction[0][res]
            if res_confidence*100 < 46.0:
                payloads["intent"] = "unknown"
            else:
                payloads["intent"] = res

            payloads["Status"] = "success"
            payloads["Message"] = "null"
            # intent_files = [ f.split(".")[0] for f in os.listdir(os.path.join(into_home,"trained_data"))]
            # result_dic = {}
            # confidence = {}
            # confidence_cosine = {}

            # print(intent_files)
            # trained_data_repo = os.path.join(into_home,"trained_data")
            # # print(os.listdir(os.path.join(into_home,"trained_data")))

            # for intent in intent_files:
            #     intent_vec = np.load(os.path.join(trained_data_repo,intent+".npy"))
            #     result_dic[intent] = cosine_similarity(self.query_vec,intent_vec)[0][0]
            #     # confidence[intent] = str(distance.euclidean(query_vec,intent_vec))
            #     confidence_cosine[intent] = str(cosine_similarity(self.query_vec,intent_vec)[0][0])

            # print(result_dic)

            # res = max(result_dic, key=result_dic.get)

            # if self.matched_words_ratio*100>50.0 and float(result_dic[res])*100 < 50.0:
            #     payloads["intent"] = "unknown"
            #     payloads["Status"] = "success"
            #     payloads["Message"] = "Couldn't understand properly"
            # elif self.matched_words_ratio*100<50.0 and float(result_dic[res])*100 < 70.0:
            #     payloads["intent"] = "unknown"
            #     payloads["Status"] = "success"
            #     payloads["Message"] = "Couldn't understand properly"
            # else:
            #     payloads["intent"] = res
            #     payloads["Status"] = "success"
            #     payloads["Message"] = "null"

            # payloads["confidence"] = confidence
            payloads['confidence'] = confidence

            return payloads

        else:

            return {"Status":"failed","Message":'Error!! Your Bot is not yet trained',"intent":"null","confidence":"null"}
        