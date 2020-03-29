from pymongo import MongoClient
import config as cfg
from DButills import DButills
 
class Transformer:
    def __init__(self,botid):
        self.botid = botid

    def initiateDB(self):
        try:
            client = MongoClient(cfg.HOST, cfg.DB_PORT, username=cfg.USERNAME, password=cfg.PASSWORD)
            db = client[cfg.DB_NAME]
            collection = db[cfg.COLLECTION_NAME]
        except:
            return False, False

        return client, collection

    def doesThisBOTExist(self):

        obj = DButills(self.botid)

        return obj.doesThisBOTExist()

    def getDataForIntentModel(self):
        client, collection = self.initiateDB()

        if collection:

            if self.doesThisBOTExist():

                dataRows = collection.find({"botID":self.botid}, {"_id":0,"botID":0,"entity":0})

                # intents = collection.find({"botID":self.botid}, {"_id":0,"botID":0,"entity":0}).distinct("intent")
                # # print(intents)
                # if not intents:
                #     client.close()
                #     return {"Status":"failed","Message":'There is no data to train',"intent":"null","confidence":"null"}, False

                # processed_dataDict = {}
                # for intent in intents:
                #     sentences = []
                #     result = collection.find({"intent":intent},{"_id":0,"botID":0,"intent":0,"entity":0})

                #     for sent in result:
                #         sentences.append(sent['sentence'])
                    
                #     processed_dataDict[intent] = sentences

                # client.close()
                # return processed_dataDict, True
                client.close()
                return dataRows, True

            else:
                client.close()
                return {"Status":"failed","Message":'There is no data to train.',"intent":"null","confidence":"null"}, False
        else:
            return {"Status":"failed","Message":'Could not connect to the database.',"intent":"null","confidence":"null"}, False
        
    
    def getDataForEntityModel(self):

        client, collection = self.initiateDB()

        if collection:


            if self.doesThisBOTExist():

                documents = collection.find({"botID":self.botid}, {"_id":0,"botID":0})
                print(documents.collection.count_documents({}))
                if not documents.collection.count_documents({}):
                    client.close()
                    return {"Status":"failed","Message":'There is no data to train',"intent":"null","confidence":"null"}, False
                
                client.close()
                
                return documents, True

            else:
                client.close()
                return {"Status":"failed","Message":'There is no bot with this name.',"intent":"null","confidence":"null"}, False
        else:
            return {"Status":"failed","Message":'Could not connect to the database.',"intent":"null","confidence":"null"}, False

