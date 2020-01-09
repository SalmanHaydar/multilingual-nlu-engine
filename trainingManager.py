import os
from intentTrainer import IntentTrainer

class Trainer:
    def __init__(self,botid):
        self.botid = botid

    def trainIntent(self,wv):
        obj = IntentTrainer(self.botid,wv)
        response = obj.train()
        return response

    def trainEntity(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError