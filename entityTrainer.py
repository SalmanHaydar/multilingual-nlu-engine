import os
from itertools import chain
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from pickle import dump
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import config as cfg
from dataTransformer import Transformer
from entityFeatureExt import get_feature,get_label
from celery import Celery

celery = Celery('main', broker=cfg.BROKER_URL, backend=cfg.BROKER_URL)

class EntityTrainer:
    def __init__(self,botid):
        self.botid = botid

    def getDataFromDB(self):
        
        obj = Transformer(self.botid)
        data_rows, _ = obj.getDataForEntityModel() 

        return data_rows

    def getTrainingData(self):
        data_row = self.getDataFromDB()
        print(data_row)
        X_train = [get_feature(sentence_row) for sentence_row in data_row if "entity" in sentence_row.keys()]
        data_row = self.getDataFromDB()
        y_train = [get_label(sentence_row) for sentence_row in data_row if "entity" in sentence_row.keys()]

        # print(len(X_train))
        # print(len(y_train))

        assert len(X_train) == len(y_train)

        return X_train, y_train

    def getClasses(self,y_train):
        temp_list = []

        for row in y_train:
            for label in row:
                temp_list.append(label)
        
        unique_labels = list(set(temp_list))

        try:

            unique_labels.remove('O')

        except Exception as e:

            pass

        return unique_labels

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

    def fit_and_save(self, botid, X_train, y_train, model):
        
        model.fit(X_train,y_train)

        BOT_HOME = os.path.join(cfg.BOT_BASE, self.botid)
        entity_model_path = os.path.join(BOT_HOME, 'entity_model')

        with open(os.path.join(entity_model_path, str(self.botid) + '.pkl'), 'wb') as f:
            try:
                dump(model.best_estimator_, f)
                print("Entity Model Saved successfully.")
            except FileNotFoundError as fnf_error:
                print(fnf_error)

    def train(self):
        X_train, y_train = self.getTrainingData()
        classes = self.getClasses(y_train)

        try:
            if len(X_train)>0 and len(classes)>0:

                global_fit_and_save.delay(self.botid, X_train, y_train, classes)

                return True

            else:

                return True

        except:
            raise Exception('Entity Model is not trained.')


@celery.task
def global_fit_and_save(botid, X_train, y_train, classes):
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
    model = RandomizedSearchCV(crf, params_space, 
                            cv=5, 
                            verbose=1, 
                            n_jobs=-1, 
                            n_iter=50, 
                            scoring=f1_scorer)                              

    model.fit(X_train,y_train)

    BOT_HOME = os.path.join(cfg.BOT_BASE, botid)
    entity_model_path = os.path.join(BOT_HOME, 'entity_model')

    with open(os.path.join(entity_model_path, str(botid) + '.pkl'), 'wb') as f:
        try:
            dump(model.best_estimator_, f)
            print("Entity Model Saved successfully.")
        except FileNotFoundError as fnf_error:
            print(fnf_error)