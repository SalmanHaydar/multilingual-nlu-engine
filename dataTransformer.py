from pymongo import MongoClient
import config as cfg

class Transformer:
    def __init__(self,botid):
        self.botid = botid

    def initiateDB(self):
        try:
            client = MongoClient(cfg.HOST,cfg.PORT)
            db = client[cfg.DB_NAME]
            collection = db[cfg.COLLECTION_NAME]
        except:
            return False, False

        return client, collection

    def doesThisBOTExist(self,db):

        res = db.find_one({"botID":self.botid})

        if res:
            return True
        else:
            return False

    def getDataForIntentModel(self):
        client, collection = self.initiateDB()

        if collection:

            if self.doesThisBOTExist(collection):

                intents = collection.find({"botID":self.botid}, {"_id":0,"botID":0,"entity":0}).distinct("intent")
                if not intents:
                    client.close()
                    return {"Status":"failed","Message":'There is no data to train',"intent":"null","confidence":"null"}, False

                processed_dataDict = {}
                for intent in intents:
                    sentences = []
                    result = collection.find({"intent":intent},{"_id":0,"botID":0,"intent":0,"entity":0})

                    for sent in result:
                        sentences.append(sent['sentence'])
                    
                    processed_dataDict[intent] = sentences

                client.close()
                return processed_dataDict, True

            else:
                client.close()
                return {"Status":"failed","Message":'There is no bot with this name.',"intent":"null","confidence":"null"}, False
        else:
            return {"Status":"failed","Message":'Could not connect to the database.',"intent":"null","confidence":"null"}, False
        
    
    def getDataForEntityModel(self):
         client, collection = self.initiateDB()

        if collection:

            if self.doesThisBOTExist(collection):

                documents = collection.find({"botID":self.botid}, {"_id":0,"botID":0})

                if documents.collection.count_documents({}):
                    client.close()
                    return {"Status":"failed","Message":'There is no data to train',"intent":"null","confidence":"null"}, False
                
                client.close()
                
                return documents

            else:
                client.close()
                return {"Status":"failed","Message":'There is no bot with this name.',"intent":"null","confidence":"null"}, False
        else:
            return {"Status":"failed","Message":'Could not connect to the database.',"intent":"null","confidence":"null"}, False

