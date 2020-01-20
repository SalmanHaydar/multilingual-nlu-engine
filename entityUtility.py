import pandas as pd
import numpy as np


class EntityUtility:

    def __init__(self,sentence):
        self.sentence = sentence["sentence"]

    def get_feature(self,inference=False):
        temp_list = []
        labels = []

        for i,word in enumerate(self.sentence.split(" ")):
            features = {}
            features["Bias"] = 1.0
            features["word"] = word
            features["is_title"] = word.istitle()
            features["is_upper"] = word.isupper()
            features["is_digit"] = word.isdigit()
            features["prefix"] = word[0:2]
            features["postfix"] = word[-3:]
            features["length"] = len(word)
            
            if i>0:
                prev_word = self.sentence.split(" ")[i-1]
                features["prev_word"] = prev_word
                features["prev_word_prefix"] = prev_word[0:2]
                features["prev_word_postfix"] = prev_word[-3:]
            else:
                features["BOS"] = True
                
            if i<len(self.sentence.split(" "))-1:
                next_word = self.sentence.split(" ")[i+1]
                features["next_word"] = next_word
                features["next_word_prefix"] = next_word[0:2]
                features["next_word_postfix"] = next_word[-3:]
            else:
                features["EOS"] = True
            
            temp_list.append(features)
                
        return temp_list

    def get_label(self,sent_row):

        temp_list = []
        for i,word in enumerate(self.sentence.split(" ")):
            if word in sent_row["entity"].keys():
                temp_list.append(sent_row["entity"][word])
            else:
                temp_list.append('O')
        return temp_list