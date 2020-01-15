from pymongo import MongoClient
import config as cfg
 
class DButills:
    def __init__(self,botid):
        self.botid = botid

    def initiateDB(self,table=cfg.COLLECTION_NAME):
        try:
            client = MongoClient(cfg.HOST,cfg.PORT)
            db = client[cfg.DB_NAME]
            collection = db[table]
        except:
            raise Exception("Cannot connect the databse.")

        return client, collection

    def doesThisBOTExist(self):

        client, collection = self.initiateDB(table=cfg.BOT_PROFILE)
        res = collection.find_one({"botID":self.botid})

        if res:
            client.close()
            return True
        else:
            return False

    