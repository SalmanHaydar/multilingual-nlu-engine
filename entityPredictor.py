import os
from entityFeatureExt import get_feature

class EntPredictor:
    def __init__(self,model,sentence, lookupTable):
        self.model = model
        self.sentence = sentence
        self.lookupTable = lookupTable

    def predict(self):
        
        entityBucket = {}
        prediction = self.model.predict_single(get_feature({"sentence":self.sentence}))
        for index,word in enumerate(self.sentence.split()):
            if word in self.lookupTable.keys():
                entityBucket[word] = self.lookupTable[word]
            else:
                if prediction[index] != 'O':
                    entityBucket[word] = prediction[index]
                else:
                    continue
        return entityBucket