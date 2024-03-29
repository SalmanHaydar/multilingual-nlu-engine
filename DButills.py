from pymongo import MongoClient
import config as cfg
import datetime
import secrets
import os
import warnings

warnings.filterwarnings('ignore')


class DButills:

    def __init__(self, botid=None):
        self.botid = botid

    def initiateDB(self, table=cfg.COLLECTION_NAME):
        try:
            client = MongoClient(cfg.HOST, cfg.DB_PORT, username=cfg.USERNAME, password=cfg.PASSWORD)
            db = client[cfg.DB_NAME]
            collection = db[table]
            # print(table)
        except:
            raise Exception("Cannot connect to the database.")

        return client, collection

    def doesThisBOTExist(self, table=cfg.BOT_PROFILE):

        client, collection = self.initiateDB(table=table)
        res = collection.find_one({"botID": self.botid})

        if res:
            client.close()
            return True
        else:
            return False

    def getLookupTable(self):
        client, collection = self.initiateDB(table=cfg.COLLECTION_NAME)
        dataRows = collection.find({"botID": self.botid}, {"_id": 0})

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
                data = {"botID": self.botid, "botName": "Yet to Implement", "accessToken": secrets.token_hex(16), "creationTime": datetime.datetime.now()}
                table.insert_one(data)

                os.mkdir(cfg.BOT_BASE+"/"+self.botid)
                BOT_HOME = os.path.join(cfg.BOT_BASE, self.botid)
                # os.mkdir(BOT_HOME+"/training_data")
                # os.mkdir(BOT_HOME+"/trained_data")
                os.mkdir(BOT_HOME+"/vocab_repo")
                os.mkdir(BOT_HOME+"/entity_model")
                os.mkdir(BOT_HOME+"/intent_model")

                client.close()
                return {"Status": "success", "Message": 'Bot profile has created successfully.', "intent": "null", "confidence": "null"}
            except:
                raise Exception("Cannot Create a BOT profile")
        else:
            return {"Status": "failed", "Message": 'A Bot with this ID is already exist', "intent": "null", "confidence": "null"}

    def deleteOne(self, sentence):
        if sentence:
            client, collection = self.initiateDB()
            result = collection.delete_one({"botID": self.botid, "sentence": sentence})

            client.close()

            if result.deleted_count:

                return {"Status": "success", "deleted_data": result.deleted_count, "Message": 'sentence has been deleted successfully.'}
            else:
                return {"Status": "success", "deleted_data": result.deleted_count, "Message": 'No sentence has been found.'}

        else:
            return {"Status": "success", "deleted_data": '0', "Message": 'No sentence has been passed.'}

    def deleteMany(self, intent):
        if intent:
            client, collection = self.initiateDB()
            result = collection.delete_many({"botID": self.botid, "intent": intent})

            client.close()

            if result.deleted_count:

                return {"Status": "success", "deleted_data": result.deleted_count, "Message": 'intent has been deleted successfully.'}
            else:
                return {"Status": "success", "deleted_data": result.deleted_count, "Message": 'No intent has been found.'}

        else:
            return {"Status":"success", "deleted_data": 0, "Message": 'No intent name has been passed.'}

    def getsamples(self,intent=None):

        client, collection = self.initiateDB()

        if intent:
            dataRows = collection.find({"botID": self.botid, "intent": intent}, {"botID":0, "_id": 0})
        else:
            dataRows = collection.find({"botID": self.botid}, {"botID":0, "_id": 0})

        temp_list = []

        for row in dataRows:
            temp_list.append(row)

        # if len(temp_list)<1:
        #     return {"Status": "success", "Message": 'No data had been found', "intent": "null", "confidence": "null"}

        return temp_list

    def getIntents(self):

        client, collection = self.initiateDB()

        dataRows = collection.find({"botID" : self.botid}, {"_id" : 0, "botID" : 0, "entity" : 0})

        intents = {}

        for row in dataRows:
            intents[row['intent']] = 1
        
        return list(intents.keys())

