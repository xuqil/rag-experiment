import os

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.base.embeddings.base import similarity
from llama_index.core.objects import ObjectIndex
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent import ReActAgent
from llama_index.agent.openai_legacy import ContextRetrieverOpenAIAgent


# 模拟搜索
def search_weather(query: str) -> str:
    """用于搜索天气情况"""
    # Perform search logic here
    search_results = f"明天天气晴转多云，最高温度30度，最低温度23度。天气炎热，注意防晒哦。"
    return search_results


tool_search = FunctionTool.from_defaults(fn=search_weather)


# 模拟发送邮件
def send_email(subject: str, recipient: str, message: str) -> None:
    """用于发送电子邮件"""
    # Send email logic here
    print(f"邮件已发送至 {recipient}，主题为 {subject}，内容为 {message}")


tool_send_mail = FunctionTool.from_defaults(fn=send_email)


# 模拟主题内容创作
def query_customer(phone: str) -> str:
    """用于查询客户信息"""
    # Perform creation logic here
    result = f"该客户信息为:\n姓名: 张三\n电话: {phone}\n地址: 北京市海淀区"
    return result


tool_generate = FunctionTool.from_defaults(fn=query_customer)
tools = [tool_generate, tool_search, tool_send_mail]

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

Settings.llm = llm
Settings.embed_model = embed_model

texts = [
    "Abbreviation: X = Revenue",
    "Abbreviation: YZ = Risk Factors",
    "Abbreviation: Z = Costs",
]

docs = [Document(text=t) for t in texts]
context_index = VectorStoreIndex.from_documents(docs)

context_agent = ContextRetrieverOpenAIAgent.from_tools_and_retriever(
    tools,
    context_index.as_retriever(similarity_top_k=1),
    verbose=True,
)

context_agent.chat('What is the YZ of March 2022')
