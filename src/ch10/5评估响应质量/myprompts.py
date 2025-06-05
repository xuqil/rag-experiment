from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

MY_QUESTION_GENERATION_PROMPT_TMPL = """\
以下是上下文信息:
---------------------
{context_str}
---------------------
仅根据上面的上下文信息和非预置知识，基于以下要求生成问题，不要有多余解释。
---------------------
{query_str}
---------------------
"""
MY_QUESTION_GENERATION_PROMPT = PromptTemplate(
    MY_QUESTION_GENERATION_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

MY_TEXT_QA_PROMPT_TMPL = (
    "以下是上下文信息:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "根据上面的上下文信息和非预置知识，回答以下问题。\n"
    "---------------------\n"
    "{query_str}\n"
    "---------------------\n"
    "回答: "
)
MY_TEXT_QA_PROMPT = PromptTemplate(
    MY_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

MY_EVAL_TEMPLATE = PromptTemplate(
    "您的任务是评估从文档来源中检索到的上下文是否与查询相关。\n"
    "评估应该按照以下步骤逐步进行，回答以下问题：\n"
    "1. 检索到的上下文是否与用户的查询主题匹配？\n"
    "2. 检索到的上下文是否可以独立用于提供完整的答案给用户的查询？\n"
    "每个问题的分值为2分，允许并鼓励部分得分。根据之前提到的标准问题，提供详细的反馈。\n"
    "在提供反馈后，按照以下格式严格提供最终结果："
    "'[RESULT]后跟表示分配给响应的总分的浮点数'\n\n"
    "查询：\n {query_str}\n"
    "上下文：\n {context_str}\n"
    "反馈："
)

_MY_SCORE_THRESHOLD = 4.0

MT_REFINE_TEMPLATE = PromptTemplate(
    "我们想要了解以下查询和回答是否与上下文信息一致：\n {query_str}\n"
    "我们提供了一个现有的评估分数：\n {existing_answer}\n"
    "我们有机会通过下面的更多上下文来改进现有的评估（仅在需要时）。\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    f"如果现有的评估已经是{_MY_SCORE_THRESHOLD}，仍然回答{_MY_SCORE_THRESHOLD}。"
    f"如果新的上下文中存在该信息，回答{_MY_SCORE_THRESHOLD}。"
    "否则回答{existing_answer}。\n"
)

MY_ANSWER_EVAL_TEMPLATE = PromptTemplate(
    "您的任务是评估回答是否与查询相关。\n"
    "评估应该按照以下步骤逐步进行，回答以下问题：\n"
    "1. 提供的回答是否与用户的查询主题匹配？\n"
    "2. 提供的回答是否试图回答用户查询的焦点或观点？\n"
    "上述每个问题的分值为1分。根据上述标准提供详细的反馈。\n"
    "在提供反馈后，按照以下格式严格提供最终结果："
    "'[RESULT]后跟表示分配给响应的总分的整数'\n\n"
    "查询：\n {query}\n"
    "回答：\n {response}\n"
    "反馈："
)


MY_GUILD_EVAL_TEMPLATE = PromptTemplate(
    "以下是原始问题:\n"
    "------------\n"
    "问题: {query}\n"
    "------------\n"
    "根据以下准则评价以下回答:\n"
    "------------\n"
    "回答: {response}\n"
    "------------\n"
    "准则: {guidelines}\n"
    "------------\n"
    "现在请用中文提供建设性的评价。\n"
)
