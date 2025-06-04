import os

from dotenv import load_dotenv
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.tools import ToolMetadata, QueryEngineTool
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, SummaryIndex, SimpleKeywordTableIndex

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
docs = SimpleDirectoryReader(input_files=["../../data/xiaomai.txt"]).load_data()

summary_index = SummaryIndex.from_documents(docs, chunk_size=100, chunk_overlap=0)
vector_index = VectorStoreIndex.from_documents(docs, chunk_size=100, chunk_overlap=0)
keyword_index = SimpleKeywordTableIndex.from_documents(docs, chunk_size=100, chunk_overlap=0)

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_index.as_query_engine(response_mode="tree_summarize", ),
    description=(
        "有助于总结小麦手机相关的问题"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_index.as_query_engine(),
    description=(
        "适合检索与小麦手机相关的特定上下文"
    ),
)

keyword_tool = QueryEngineTool.from_defaults(
    query_engine=keyword_index.as_query_engine(),
    description=(
        "适合使用关键词从文章中检索特定的上下文"
    ),
)

query_engine = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(),
    query_engine_tools=[
        summary_tool, vector_tool, keyword_tool
    ], verbose=True
)
response = query_engine.query("小麦手机的处理器是什么？")
pprint_response(response, show_source=True)
