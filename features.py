from flair.data import Sentence
import numpy as np


class FeatureExtractor:

    def __init__(self,embedding=None):
        self.embedding = embedding


    def get_feature_vector_train(self,sentences):

        if self.embedding == None:
            raise Exception('Please pass flair embedding object.')

        else:
            sentence_vectors = []
            for sent in sentences:

                s = Sentence(sent)
                self.embedding.embed(s)
                
                temp_placeholder = []
                for token in s:
                    temp_placeholder.append(token.embedding.cpu().numpy().reshape(1,-1))
                
            sentence_vectors.append(np.mean(temp_placeholder,axis=0))

        return np.mean(sentence_vectors,axis=0)


    def get_feature_vector_infer(self,sentence):

        if self.embedding == None:
            raise Exception('Please pass flair embedding object.')

        else:
            sent = Sentence(sentence)
            self.embedding.embed(sent)

            temp_placeholder = []
            for token in sent:
                temp_placeholder.append(token.embedding.cpu().numpy().reshape(1,-1))

            sent_embd = np.mean(temp_placeholder,axis=0)

            return sent_embd
