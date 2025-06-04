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

vectorindex_xiaomai = VectorStoreIndex.from_documents(docs_xiaomai)

query_engine_quesiton = vectorindex_xiaomai.as_query_engine(response_mode="compact")
query_engine_summary = vectorindex_xiaomai.as_query_engine(response_mode="simple_summarize")

tool_question = QueryEngineTool.from_defaults(
    query_engine=query_engine_quesiton,
    description="用来回答事实性与细节性的问题",
)
tool_summarize = QueryEngineTool.from_defaults(
    query_engine=query_engine_summary,
    description="用于总结与概括",
)

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        tool_question, tool_summarize
    ], verbose=True
)
response = query_engine.query("小麦手机的优势在哪里")
pprint_response(response, show_source=True)

response = query_engine.query("简要概述小麦手机")
pprint_response(response, show_source=True)
