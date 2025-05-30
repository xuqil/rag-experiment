import os

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core import Settings

load_dotenv()

embed_model = DashScopeEmbedding(
    model_name=os.getenv("EMBED_MODEL_NAME"),
    api_base=os.getenv("API_BASE_URL"),
    api_key=os.getenv("API_KEY"),
)
Settings.embed_model = embed_model

# 构造一个数据提取管道
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=500, chunk_overlap=0),
        embed_model  # 用于生成向量
    ],
    docstore=SimpleDocumentStore()
)

# 避免重复加载和处理
pipeline.load("./pipeline_storage")

docs = SimpleDirectoryReader(input_files=["../../data/yiyan.txt"]).load_data()
# 运行
nodes = pipeline.run(documents=docs, show_progress=True)
print(len(nodes))
pipeline.persist("./pipeline_storage")
