import os

from dotenv import load_dotenv
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex

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
docs_xiaomai = SimpleDirectoryReader(input_files=["../../data/xiaomai.txt"]).load_data()
docs_yiyan = SimpleDirectoryReader(input_files=["../../data/yiyan.txt"]).load_data()

vectorindex_xiaomai = VectorStoreIndex.from_documents(docs_xiaomai)
query_engine_xiaomai = vectorindex_xiaomai.as_query_engine()

vectorindex_yiyan = VectorStoreIndex.from_documents(docs_yiyan)
query_engine_yiyan = vectorindex_yiyan.as_query_engine()

tool_xiaomai = QueryEngineTool.from_defaults(
    query_engine=query_engine_xiaomai,
    description="用来查询小麦手机的信息",
)
tool_yiyan = QueryEngineTool.from_defaults(
    query_engine=query_engine_yiyan,
    description="用来查询文心一言的信息",
)

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        tool_xiaomai, tool_yiyan
    ]
)
response = query_engine.query("什么是文心一言，用中文回答")
pprint_response(response, show_source=True)
