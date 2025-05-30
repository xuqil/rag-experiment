import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding

load_dotenv()

embed_model = DashScopeEmbedding(
    model_name=os.getenv("EMBED_MODEL_NAME"),
    api_base=os.getenv("API_BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

embeddings = embed_model.get_text_embedding("中国的首都是北京")
print(embeddings)

embeddings2 = embed_model.get_text_embedding("中国的首都是哪里？")
embeddings3 = embed_model.get_text_embedding("苹果是一种好吃的水果")

# 比较两个向量
print(embed_model.similarity(embeddings, embeddings2))  # 0.8045008198182284
print(embed_model.similarity(embeddings, embeddings3))  # 0.053938154682054026
