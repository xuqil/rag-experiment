import os

from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, RetrieverTool
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, Document

from src.tools import print_nodes

load_dotenv()

embed_model = DashScopeEmbedding(
    model_name=os.getenv("EMBED_MODEL_NAME"),
    api_base=os.getenv("API_BASE_URL"),
    api_key=os.getenv("API_KEY"),
)
llm = OpenAILike(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("API_BASE_URL"),
    is_chat_model=True,
    timeout=120  # 设置超时
)
Settings.llm = llm
Settings.embed_model = embed_model

# documents
# docs = SimpleDirectoryReader(input_files=["../../data/yiyan.txt"]).load_data()
docs = [Document(text="小麦手机是小麦公司最新出的第十代手机产品。\
                       采用了中国最先进的国产红旗CPU芯片。\
                       采用了6.95寸的OLED显示屏幕与5000毫安的电池容量。\
                       操作系统是最新的澎湃OS 3.0，兼容现有的安卓APP，是一款高性能的智能手机。\
                       与苹果手机相比较，小麦手机具有更高的性价比。", metadata={'window': 'this is window text'})]
nodes = SentenceSplitter(chunk_size=100, chunk_overlap=0).get_nodes_from_documents(docs)
vector_index = VectorStoreIndex(nodes)
retriever_xiaomai = vector_index.as_retriever()

docs2 = SimpleDirectoryReader(input_files=["../../data/yiyan.txt"]).load_data()
nodes2 = SentenceSplitter(chunk_size=100, chunk_overlap=0).get_nodes_from_documents(docs2)
vector_index2 = VectorStoreIndex(nodes2)
retriever_yiyan = vector_index2.as_retriever()

tool_xiaomai = RetrieverTool.from_defaults(
    retriever=retriever_xiaomai,
    description="用来查询小麦手机的信息",
)
tool_yiyan = RetrieverTool.from_defaults(
    retriever=retriever_yiyan,
    description="用来查询文心一言的信息",
)

retriever = RouterRetriever(
    selector=LLMSingleSelector.from_defaults(),
    retriever_tools=[
        tool_xiaomai, tool_yiyan
    ]
)
nodes = retriever.retrieve("什么是文心一言，用中文回答")
print_nodes(nodes)
