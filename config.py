import os

#Database Config
DB_NAME = "Chatlogy"
COLLECTION_NAME = "DataRepo"
BOT_PROFILE = ''
HOST = "182.160.104.220"
DB_PORT = 27017

abs_path = os.path.dirname(os.path.abspath(__file__))
BOT_BASE = os.path.join(abs_path,"BOT_DATA")


