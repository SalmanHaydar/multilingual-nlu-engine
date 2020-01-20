from pymongo import MongoClient
import config as cfg

class AddExamples:
    def __init__(self,botid,sentence,intent,entity=None):
        self.botid = botid
        self.sentence = sentence
        self.intent = intent
        self.entity = entity

    def initiateDB(self):
        try:
            client = MongoClient(cfg.HOST,cfg.PORT)
            db = client[cfg.DB_NAME]
            collection = db[cfg.COLLECTION_NAME]
        except:
            return False, False

        return client, collection
    
    def makeDocument(self):
        document = {}

        if self.botid and self.sentence and self.intent:
            document.update({"botID":self.botid, "sentence":self.sentence, "intent":self.intent})
        else:
            return False
        
        if self.entity:
            document.update({"entity":self.entity})

        return document

    def doesThisBOTExist(self,db):

        res = db.find_one({"botID":self.botid})

        if res:
            return True
        else:
            return False

    def insertOne(self):
        client, db = self.initiateDB()

        if db:
            if self.doesThisBOTExist(db):

                document = self.makeDocument()

                if document:
                    _id = db.insert_one(document).inserted_id
                    
                    if _id:

                        client.close()

                        return {"Status":"success","Message":'Added Successfully',"intent":"null","confidence":"null"}
                    else:
                        return {"Status":"failed","Message":'Could not add',"intent":"null","confidence":"null"}
                else:
                    return {"Status":"failed","Message":'One or More data is missing!',"intent":"null","confidence":"null"}
            else:
                return {"Status":"failed","Message":'There is no bot found with this name.',"intent":"null","confidence":"null"}
        else:
            return {"Status":"failed","Message":'Could not connect to the database.',"intent":"null","confidence":"null"}

              
                    


