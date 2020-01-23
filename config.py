import os

#Database Config
DB_NAME = "Chatlogy"
COLLECTION_NAME = "DataRepo"
BOT_PROFILE = 'BotProfile'
HOST = "182.160.104.220"
DB_PORT = 27017

abs_path = os.path.dirname(os.path.abspath(__file__))
BOT_BASE = os.path.join(abs_path,"BOT_DATA")

MODEL_PATH = os.path.join(abs_path,"MODELS/keyesvectors.kv")

#Redis Configuration

BROKER_URL = 'redis://127.0.0.1:6379/0'


