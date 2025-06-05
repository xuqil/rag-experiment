import os

from dotenv import load_dotenv
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, PromptTemplate
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.tools import QueryEngineTool

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
docs1 = SimpleDirectoryReader(input_files=["../../../data/xiaomai.txt"]).load_data()
docs2 = SimpleDirectoryReader(input_files=["../../../data/xiaomaiUltra.txt"]).load_data()

index1 = VectorStoreIndex.from_documents(docs1)
index2 = VectorStoreIndex.from_documents(docs2)

query_xiaomai = index1.as_query_engine(response_mode="compact")
query_ultra = index2.as_query_engine(response_mode="compact")

query_tool_xiaomai = QueryEngineTool.from_defaults(
    query_engine=query_xiaomai,
    name="query_tool_xiaomai",
    description="提供小麦手机普通型号pro/max的信息")

query_tool_ultra = QueryEngineTool.from_defaults(
    query_engine=query_ultra,
    name="query_tool_ultra",
    description="提供小麦手机Ultra的信息")

from llama_index.core.tools import QueryPlanTool, QueryEngineTool
from llama_index.core import get_response_synthesizer
from llama_index.core.tools.query_plan import QueryPlan, QueryNode

response_synthesizer = get_response_synthesizer()
query_plan_tool = QueryPlanTool.from_defaults(
    query_engine_tools=[query_tool_xiaomai, query_tool_ultra],
    response_synthesizer=response_synthesizer,
)
nodes = [
    QueryNode(
        id=1,
        query_str="查询小麦Pro手机信息",
        tool_name="query_tool_xiaomai",
        dependencies=[]
    ),
    QueryNode(
        id=2,
        query_str="查询小麦Ultra信息",
        tool_name="query_tool_ultra",
        dependencies=[1]
    ),
    QueryNode(
        id=3,
        query_str="对比小麦Pro与小麦Ultrl的配置区别",
        tool_name="vs_tool",
        dependencies=[1, 2]
    )
]
output = query_plan_tool(nodes=nodes)
print(output)

""" 
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

agent = OpenAIAgent.from_tools(
    [query_plan_tool],
    max_function_calls=5,
    verbose=True,
)
response = agent.query("对比小麦手机pro和Ultra的配置区别") """
