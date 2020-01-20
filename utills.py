import numpy as np
import os
import re

class UtilityFunction:
    def __init__(self,sentences):
        self.sentences = sentences
        self.stop_words = ["ki","where","what","i","me","we","a","an","the","to","k","kk","you","ke","apnar","good","who",
             "whom","do","gd","did","done","have","ache","ase","kno","er","toh","bhai","bon","vai","are","of","ami",
              "apnader",'কি','ki','offer','sim','g','gb','না','অফার','আমার','airtel','আমি','টাকা','tk','na','mb',
              'সিম','আছে','ta','ami','জিবি','করতে','আপনি','to','er','you','amar','এই','taka','টাকায়','করে','number',
              'sms','বন্ধ','pack','এর','e','জি','internet','কিভাবে','সাথে','my','recharge','পারি','for','দিন','agent',
              'আপনার','কোন']

    def preprocess(self):
        if type(self.sentences) == str:
            prepared_data = []
            removeChar = '০ ১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯'.split()
            s = re.sub('[^\u0980-\u09ffa-zA-Z_]+', ' ', str(self.sentences))
            for j in removeChar:
                s = s.replace(j, ' ')
            s = ' '.join(s.split())
            for word in s.strip().lower().split():
                if word not in self.stop_words:
                    prepared_data.append(word)
            return ' '.join(prepared_data)

        elif type(self.sentences) == list:
            prepared_data = []
            
            removeChar = '০ ১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯'.split()
            for sent in self.sentences:
                s = re.sub('[^\u0980-\u09ffa-zA-Z_]+', ' ', str(sent))
                for j in removeChar:
                    s = s.replace(j, ' ')
                s = ' '.join(s.split())
                if len(s.strip()):
                    tem_repo = []
                    for word in s.lower().split():
                        if word not in self.stop_words:
                            tem_repo.append(word)
                    prepared_data.append(' '.join(tem_repo))
            return prepared_data
