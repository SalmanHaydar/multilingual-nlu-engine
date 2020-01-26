from pymongo import MongoClient
import config as cfg
import datetime
import secrets
import os
 
class DButills:
    def __init__(self,botid=None):
        self.botid = botid

    def initiateDB(self,table=cfg.COLLECTION_NAME):
        try:
            client = MongoClient(cfg.HOST,cfg.DB_PORT)
            db = client[cfg.DB_NAME]
            collection = db[table]
            # print(table)
        except:
            raise Exception("Cannot connect the databse.")

        return client, collection

    def doesThisBOTExist(self, table=cfg.BOT_PROFILE):

        client, collection = self.initiateDB(table=table)
        res = collection.find_one({"botID":self.botid})

        if res:
            client.close()
            return True
        else:
            return False

    def getLookupTable(self):
        client, collection = self.initiateDB(table=cfg.COLLECTION_NAME)
        dataRows = collection.find({"botID":self.botid},{"_id":0})

        dictionary = {}
        
        for row in dataRows:

            if 'entity' in row.keys():
                dictionary.update(row['entity'])
        
        client.close()
        return dictionary

    def createBotProfile(self):
        client, table = self.initiateDB(table=cfg.BOT_PROFILE)
        if not self.doesThisBOTExist():
            try:
                data = {"botID":self.botid, "botName":"Yet to Implement", "accessToken":secrets.token_hex(16),"creationTime":datetime.datetime.now()}
                table.insert_one(data)

                os.mkdir(cfg.BOT_BASE+"/"+self.botid)
                BOT_HOME = os.path.join(cfg.BOT_BASE,self.botid)
                os.mkdir(BOT_HOME+"/training_data")
                os.mkdir(BOT_HOME+"/trained_data")
                os.mkdir(BOT_HOME+"/vocab_repo")
                os.mkdir(BOT_HOME+"/entity_model")
                os.mkdir(BOT_HOME+"/intent_model")

                client.close()
                return {"Status":"success","Message":'Bot profile has created successfully.',"intent":"null","confidence":"null"}
            except:
                raise Exception("Cannot Create a BOT profile")
        else:
            return {"Status":"failed","Message":'A Bot with this ID is already exist',"intent":"null","confidence":"null"}

    