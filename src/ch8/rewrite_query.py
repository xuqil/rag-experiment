import os

from dotenv import load_dotenv
from llama_index.core import PromptTemplate, Settings
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.llms.openai_like import OpenAILike

load_dotenv()

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager

llm = OpenAILike(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("API_BASE_URL"),
    is_chat_model=True,
    timeout=120  # 设置超时
)

prompt_rewrite_temp = """\
您是一个聪明的查询生成器。请生成与以下查询相关的{num_queries}个查询问题 \n
注意每个查询问题都占一行 \n
我的查询：{query}
生成查询列表：
"""

prompt_rewrite = PromptTemplate(prompt_rewrite_temp)


def rewrite_query(query: str, num: int = 3):
    # 查询转换的方法
    response = llm.predict(
        prompt=prompt_rewrite,
        num_queries=num,
        query=query,
        callback_manager=callback_manager)

    # 假设大模型将每个查询问题都放在一行上
    queries = response.split("\n")
    return queries


print(rewrite_query("中国目前大模型的发展情况如何？"))
