import os

from dotenv import load_dotenv
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import PromptTemplate
from llama_index.core import QueryBundle
from llama_index.core.tools import ToolMetadata
import pprint

load_dotenv()

llm = OpenAILike(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("API_BASE_URL"),
    is_chat_model=True,
    timeout=120  # 设置超时
)


question_gen_prompt_templ = """
你可以访问多个工具，每个工具代表一个不同的数据源或API。
每个工具都有一个名称和一个描述，格式为JSON字典。
字典的键是工具的名称，值是描述。
你的目的是通过生成一系列可以由这些工具回答的子问题来帮助回答一个复杂的用户问题。

完成任务时，请考虑以下准则：
	•	尽可能具体
	•	子问题应与用户问题相关
	•	子问题应可通过提供的工具回答
	•	你可以为每个工具生成多个子问题
	•	工具必须用它们的名称而不是描述来指定
	•	如果你认为不相关，就不需要使用工具

通过调用SubQuestionList函数输出子问题列表。

## 可用工具列表 (JSON格式)
```json
{tools_str}

## 用户问题
{query_str}

## 示例输出：
[
    {"tool_name": "tool1_name", "sub_question": "子问题1"},
    {"tool_name": "tool2_name", "sub_question": "子问题2"}
]

"""

# rewriter
# 如果支持 Function Call 可以使用 OpenAIQuestionGenerator
question_rewriter = LLMQuestionGenerator.from_defaults(llm=llm)

# 更换提示
new_prompt = PromptTemplate(question_gen_prompt_templ)
question_rewriter.update_prompts({'question_gen_prompt': new_prompt})

# tools
tool_choices = [
    ToolMetadata(
        name="query_tool_beijing",
        description=(
            "用来查询北京市各个方面的信息，如基本信息、旅游指南、城市历史等"
        ),
    ),
    ToolMetadata(
        name="query_tool_shanghai",
        description=(
            "用来查询上海市各个方面的信息，如基本信息、旅游指南、城市历史等"
        ),
    ),
]

print('-------------------------')
query_str = "北京与上海的人口差距是多少？他们的面积又相差多少？"
choices = question_rewriter.generate(tool_choices, QueryBundle(query_str=query_str))

pprint.pprint(choices)
