import json
import os

from typing import Sequence, List

from dashscope.api_entities.chat_completion_types import ChatCompletionMessageToolCall
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool, BaseTool
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage

load_dotenv()


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


# 定义一个OpenAI的Agent
class MyOpenAIAgent:

    # 初始化参数
    # tools: Sequence[BaseTool] = []，工具列表
    # llm: OpenAI = OpenAI(temperature=0, model="gpt-4o-mini")，OpenAI模型
    # chat_history: List[ChatMessage] = []，聊天历史
    def __init__(
            self,
            llm: OpenAILike,
            tools: Sequence[BaseTool] = [],
            chat_history: List[ChatMessage] = [],
    ) -> None:
        self._llm = llm
        self._tools = {tool.metadata.name: tool for tool in tools}
        self._chat_history = chat_history

    # 重置聊天历史
    def reset(self) -> None:
        self._chat_history = []

    # 聊天接口
    def chat(self, message: str) -> str:
        chat_history = self._chat_history
        chat_history.append(ChatMessage(role="user", content=message))

        # 将tools传入
        tools = [
            tool.metadata.to_openai_tool() for _, tool in self._tools.items()
        ]
        ai_message = self._llm.chat(chat_history, tools=tools).message
        additional_kwargs = ai_message.additional_kwargs
        chat_history.append(ai_message)

        # 获取响应中的工具调用
        tool_calls = additional_kwargs.get("tool_calls", None)

        # 如果有工具调用，则依次调用工具
        if tool_calls is not None:
            for tool_call in tool_calls:
                # 调用函数
                function_message = self._call_function(tool_call)
                chat_history.append(function_message)

                # 继续对话
                ai_message = self._llm.chat(chat_history).message
                chat_history.append(ai_message)

        return ai_message.content

    # 调用函数
    def _call_function(
            self, tool_call: ChatCompletionMessageToolCall
    ) -> ChatMessage:
        id_ = tool_call.id
        function_call = tool_call.function
        tool = self._tools[function_call.name]
        output = tool(**json.loads(function_call.arguments))
        return ChatMessage(
            name=function_call.name,
            content=str(output),
            role="tool",
            additional_kwargs={
                "tool_call_id": id_,
                "name": function_call.name,
            },
        )


llm = OpenAILike(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("API_BASE_URL"),
    is_chat_model=True,
    timeout=120  # 设置超时
)

agent = MyOpenAIAgent(llm=llm, tools=[tool_search, tool_send_mail, tool_generate])

while True:
    user_input = input("请输入您的消息：")
    if user_input.lower() == "quit":
        break
    response = agent.chat(user_input)
    print(response)
