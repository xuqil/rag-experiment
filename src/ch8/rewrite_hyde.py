import os

from dotenv import load_dotenv
from llama_index.core.indices.query.query_transform import HyDEQueryTransform, DecomposeQueryTransform
from llama_index.core import PromptTemplate

from llama_index.llms.openai_like import OpenAILike

load_dotenv()

hyde_prompt_temp = """\
请生成一段文字来回答输入问题\n
尽可能含有更多的关键细节\n
{context_str}
生成内容：
"""
hyde_prompt = PromptTemplate(hyde_prompt_temp)

llm = OpenAILike(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("API_BASE_URL"),
    is_chat_model=True,
    timeout=120  # 设置超时
)
hyde = HyDEQueryTransform(llm=llm)
hyde.update_prompts({'hyde_prompt': hyde_prompt})

query_bundle = hyde.run("南京市的人口是多少？经济发展如何？")
print(query_bundle.__dict__)
