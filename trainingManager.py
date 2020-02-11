import os
from intentTrainer import IntentTrainer
from entityTrainer import EntityTrainer
from DButills import DButills

class Trainer:

    def __init__(self, botid):
        self.botid = botid

    def trainIntent(self, wv):
        obj = IntentTrainer(self.botid, wv)
        state = obj.train()
        # print(response)
        return state

    def trainEntity(self):
        obj = EntityTrainer(self.botid)
        state = obj.train()
        return state

    def train(self, wv):
        obj = DButills(self.botid)
        
        if obj.doesThisBOTExist():

            intent_response = self.trainIntent(wv)
            entity_response = self.trainEntity()

        else:
            
            return {"Status": "failed", "Message": 'No Bot found with this id.', "intent": "null", "confidence": "null"}

        if intent_response == True and entity_response == True:
            return {"Status":"success", "Message": 'Training has started successfully.', "intent": "null", "confidence": "null"}
        
        elif type(intent_response) == dict:
            return intent_response
        
        else:
            return {"Status": "failed", "Message": 'Something is wrong.', "intent": "null", "confidence": "null"}

