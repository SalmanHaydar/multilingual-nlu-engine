import os
from intentTrainer import IntentTrainer
from entityTrainer import EntityTrainer
 
class Trainer:
    def __init__(self,botid):
        self.botid = botid

    def trainIntent(self,wv):
        obj = IntentTrainer(self.botid,wv)
        response, status = obj.train()
        # print(response)
        return status

    def trainEntity(self):
        obj = EntityTrainer(self.botid)
        response = obj.train()
        return response

    def train(self,wv):
        intent_response = self.trainIntent(wv)
        entity_response = self.trainEntity()

        if intent_response and entity_response:
            return {"Status":"success","Message":'Trained successfully.',"intent":"null","confidence":"null"}
        else:
            return {"Status":"failed","Message":'Something is wrong.',"intent":"null","confidence":"null"}