import numpy as np
import pickle
import os


class FeatureExtractor:

    def __init__(self,embedding=None):
        self.wv = embedding
        
        self.pruning_factor_alpha = 1


    def get_feature_vector_train(self,sentences,path):

        if self.wv == None:
            raise Exception('Please pass the KeyedVectors object.')

        else:
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


    def get_feature_vector_infer(self,sentence,path):

        if self.wv == None:
            raise Exception('Please pass the KeyedVectors object.')

        else:
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