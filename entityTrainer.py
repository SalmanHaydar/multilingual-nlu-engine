import os
from itertools import chain
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import pickle
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from dataTransformer import Transformer
from entityFeatureExt import get_feature,get_label

class EntityTrainer:
    def __init__(self,botid):
        self.botid = botid

    def getDataFromDB(self):
        
        obj = Transformer(self.botid)
        data_rows = obj.getDataForEntityModel()

        return data_rows

    def getTrainingData(self):
        data_row = self.getDataFromDB()

        X_train = [get_feature(sentence_row) for sentence_row in data_row if "entity" in sentence_row.keys()]
        y_train = [get_label(sentence_row) for sentence_row in data_row if "entity" in sentence_row.keys()]

        assert len(X_train) == len(y_train)

        return X_train, y_train

    def getClasses(self,y_train):
        temp_list = []

        for row in y_train:
            for label in row:
                temp_list.append(label)
        
        unique_labels = list(set(temp_list))

        return unique_labels.remove('O')


    def getEstimator(self,cv=5,classes=None):
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
        rs = RandomizedSearchCV(crf, params_space, 
                                cv=cv, 
                                verbose=1, 
                                n_jobs=-1, 
                                n_iter=50, 
                                scoring=f1_scorer)

        return rs


    def train(self):
        X_train, y_train = self.getTrainingData()
        classes = self.getClasses(y_train)

        cls = self.getEstimator(classes=classes)

        


    