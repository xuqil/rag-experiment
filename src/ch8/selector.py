import os

from dotenv import load_dotenv
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import ToolMetadata
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings

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

""" 

# choices as a list of tool metadata
choices = [
    ToolMetadata(description="description for choice 1", name="choice_1"),
    ToolMetadata(description="description for choice 2", name="choice_2"),
]
 """
# choices as a list of strings
choices = [
    ToolMetadata(description="查询当前实时的信息", name="web_search"),
    ToolMetadata(description="知识查询或内容创作", name="query_engine")
]

selector = LLMSingleSelector.from_defaults()
selector_result = selector.select(
    choices, query="写一个悬疑小故事?"
)
print(selector_result.selections)
