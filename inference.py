import json
import os
from features import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
from flair.embeddings import ELMoEmbeddings
import settings
import redis

db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, db=settings.REDIS_DB)


abs_path = os.path.dirname(os.path.abspath(__file__))
allenNLP = os.path.join(abs_path,"MODELS/.allennlp")
BOT_BASE = os.path.join(abs_path,"BOT_DATA")

os.environ["ALLENNLP_CACHE_ROOT"] = allenNLP

try:
    print("Loading Model in a RAM...")
    elmo_embedding = ELMoEmbeddings('original')
    print("Model Loaded in a RAM successfully...")
except:
    print("Cannot Load Embedding Model")
    print(os.getenv("ALLENNLP_CACHE_ROOT"))

extractor = FeatureExtractor(elmo_embedding)

def getIntent():

  if request.method == "GET":

    # payloads = {"intent":"","confidence":""}
    # bot_id = request.args.get("botID")
    # sentence = request.args.get("sentence")
    
    # print(request.args)

    query_vec = extractor.get_feature_vector_infer(sentence)
    # print(query_vec.shape)

    if(os.path.exists(os.path.join(BOT_BASE,bot_id))):
    
      into_home = os.path.join(BOT_BASE,bot_id)
      if(os.path.exists(os.path.join(into_home,"trained_data"))):
        intent_files = [ f.split(".")[0] for f in os.listdir(os.path.join(into_home,"trained_data"))]
        result_dic = {}
        print(intent_files)
        trained_data_repo = os.path.join(into_home,"trained_data")
        # print(os.listdir(os.path.join(into_home,"trained_data")))
        for intent in intent_files:
            intent_vec = np.load(os.path.join(trained_data_repo,intent+".npy"))
            result_dic[intent] = cosine_similarity(query_vec,intent_vec)
        print(result_dic)
        res = max(result_dic, key=result_dic.get)
        payloads["intent"] = res
        payloads["confidence"] = str(result_dic)

        return json.dumps(payloads)

      else:
        return json.dumps('Error!! Your Bot is not yet trained')

    else:
      return json.dumps('Error!! No Bot found with this Bot ID'), 404

  else:
      return json.dumps('This method is not allowed'), 405

