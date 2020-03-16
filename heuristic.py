
import re
from fuzzywuzzy import fuzz,process
import requests
import operator

class Heuristic:
    def __init__(self, botid=None):
        self.botid = botid

    def prepare_dataset(self):
        data_repo = {}
        if self.botid:
            r = requests.get("http://182.160.104.220:5656/getintents?botID="+self.botid)
            intents = r.json()
            intents = intents["intents"]
            
            for intent in intents:
                tmp = []
                r = requests.get("http://182.160.104.220:5656/getsamples?botID="+self.botid+"&intent="+str(intent))
                sentences = r.json()
                for sent in sentences:
                    tmp.append(sent["sentence"])
                
                data_repo[intent] = tmp
                
            return data_repo

    def get_possible_intent(self, query=None):
        res_dict = {}
        if self.botid:
            data = self.prepare_dataset()
            for intent in data.keys():
                max_score = 0
                for sent in data[intent]:
                    score = fuzz.token_set_ratio(query, sent)
                    if score > max_score:
                        max_score = score
                res_dict[intent] = max_score
            sorted_ = sorted(res_dict, key=res_dict.get, reverse=True)
            
            if res_dict[sorted_[0]] >= 64:
                return sorted_[0]
            else:
                return "unknown"
            # return (sorted_[0], res_dict[sorted_[0]])
