import numpy as np


class FeatureExtractor:

    def __init__(self,wv=None):
        self.wv = wv


    def get_feature_vector_train(self,sentences):

        if self.wv == None:
            raise Exception('Please pass the KeyedVectors object.')

        else:
            sentence_vectors = []
            for sent in sentences:

                temp_placeholder = []
                # temp_placeholder.append(np.zeros(1024,dtype=np.float32))
            
                for token in sent.split():
                    try:
                        temp_placeholder.append(self.wv[token])
                    except KeyError:
                        pass

                if len(temp_placeholder)>0:
                    sentence_vectors.append(np.mean(temp_placeholder,axis=0))

            return np.mean(sentence_vectors,axis=0).reshape(1,-1)


    def get_feature_vector_infer(self,sentence):

        if self.wv == None:
            raise Exception('Please pass KeyedVector object.')

        else:
            temp_placeholder = []
            for token in sentence.split():
                try:
                    temp_placeholder.append(self.wv[token])
                except KeyError:
                    pass

            if len(temp_placeholder)>0:
                sent_embd = np.mean(temp_placeholder,axis=0)
                return sent_embd.reshape(1,-1)
            else:
                return np.zeros(1024,dtype=np.float32).reshape(1,-1)
            
