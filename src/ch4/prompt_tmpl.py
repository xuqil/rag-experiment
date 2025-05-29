from llama_index.core import PromptTemplate
from llama_index.llms.openai_like import OpenAILike

template = (
    "以下提供的上下文信息: \n"
    "________________\n"
    "{contex_str}"
    "\n---------------\n"
    "根据这些信息，请回答以下问题：{query_str}\n"
)

qa_template = PromptTemplate(template)

prompt = qa_template.format(contex_str="小麦手机", query_str="你好")
print(prompt)

messages = qa_template.format_messages(context_str="秒买手机", query_str='你好')
print(messages)
