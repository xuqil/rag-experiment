import os
import pprint

import chromadb
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()
llm = OpenAILike(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("API_BASE_URL"),
    is_chat_model=True,
    timeout=120  # 设置超时
)
embed_model = DashScopeEmbedding(
    model_name=os.getenv("EMBED_MODEL_NAME"),
    api_base=os.getenv("API_BASE_URL"),
    api_key=os.getenv("API_KEY"),
)
Settings.embed_model=embed_model

# 准备向量存储
chroma_client = chromadb.EphemeralClient()
chroma_client.heartbeat()
chroma_collection = chroma_client.get_or_create_collection(
    name="pipeline",
    metadata={"hnsw:space": "cosine"}  # 明确指定相似度度量
)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

docs = SimpleDirectoryReader(input_files=["../../data/yiyan.txt"]).load_data()

# 构造一个数据提取管道
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=500, chunk_overlap=0),
        TitleExtractor(llm=llm, ),
        embed_model  # 用于生成向量
    ],
    vector_store=vector_store  # 提供一个向量存储对象
)

# 运行
nodes = pipeline.run(documents=docs)
print(nodes)

results = vector_store.query(VectorStoreQuery(query_str="文心一言是什么？", similarity_top_k=3), )
pprint.pprint(results)
