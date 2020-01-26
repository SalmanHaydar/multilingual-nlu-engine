
import numpy as np
import pickle
import os
from utills import UtilityFunction



class FeatureExtractor:

    def __init__(self,embedding=None):

        self.wv = embedding
        
        self.pruning_factor_alpha = 1


    def get_feature_vector_train(self,sentences,path):

        if self.wv == None:
            raise Exception('Please pass the KeyedVectors object.')

        else:
            preprocessing = UtilityFunction(sentences)
            sentences = preprocessing.preprocess()
            self.store_vocab(sentences,path)
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
                    
#                     sentence_vectors.append(np.mean(temp_placeholder,axis=0))
                    summation = np.zeros(1024,dtype=np.float32)
                    for vector in temp_placeholder:
                        summation = np.add(summation,vector)
                    sent_embd = summation
                    sentence_vectors.append(sent_embd.reshape(1,-1))

            return np.mean(sentence_vectors,axis=0).reshape(1,-1)

    def get_feature_for_text_classification(self, row, maxlen=10, padding_with=0):

        temp_list = []
        features = {}
        
        features["Bias"] = 1.0
    #     features["length"] = len(row['sentence'].split(" "))
        
        # offer ki ache
        for i,word in enumerate(row['sentence'].split(" ")):
            if i+1<=maxlen:
                features['word'+str(i+1)] = word
                try:
                    features['word'+str(i+1)+"-w2v"] = np.mean(self.wv[word])
                except:
                    features['word'+str(i+1)+"-w2v"] = 0.0

        if i+1<maxlen:
            for j in range(i+1, maxlen):
                features['word'+str(j+1)] = ""
                features['word'+str(j+1)+"-w2v"] = padding_with
                
        temp_list.append(features)
            
        return temp_list

    def get_feature_vector_infer(self,sentence,path):

        if self.wv == None:
            raise Exception('Please pass the KeyedVectors object.')

        else:
            preprocessing = UtilityFunction(sentence)
            sentence = preprocessing.preprocess()
            
            total_matched_words = 0
            with open (path, 'rb') as fp:
                vocab_repo = pickle.load(fp)
            
            temp_placeholder = []
            for token in sentence.split():
                try:
                    if token in vocab_repo:
                        total_matched_words = total_matched_words + 1
#                         unit_vector = self.wv[token]/np.linalg.norm(self.wv[token])
                        temp_placeholder.append(self.wv[token])
                    else:
#                         pass
#                         reduced_vector = self.wv[token]*self.pruning_factor_alpha
#                         unit_vector = reduced_vector/np.linalg.norm(reduced_vector)
                        temp_placeholder.append(self.wv[token]*self.pruning_factor_alpha)
                except KeyError:
                    pass
            if len(temp_placeholder)>0:
#                 sent_embd = np.mean(temp_placeholder,axis=0)
                summation = np.zeros(1024,dtype=np.float32)
                for vector in temp_placeholder:
                    summation = np.add(summation,vector)
                sent_embd = summation
                return sent_embd.reshape(1,-1), total_matched_words/len(sentence.split(" "))
            else:
                return np.zeros(1024,dtype=np.float32).reshape(1,-1), total_matched_words/len(sentence.split(" "))

    def store_vocab(self,sentences,path):

        vocab_repo = []
        for sentence in sentences:
            for word in sentence.split(" "):
                vocab_repo.append(word)
                try:
                    for sim_word in self.wv.most_similar(word,topn=3):
                        vocab_repo.append(sim_word[0])
                except KeyError:
                    pass
        if os.path.exists(path):
            with open (path, 'rb') as fp:
                itemlist = pickle.load(fp)
            for word in itemlist:
                vocab_repo.append(word)

        with open(path,'wb') as f:
            pickle.dump(list(set(vocab_repo)),f)
