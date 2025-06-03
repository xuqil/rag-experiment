from difflib import context_diff
from typing import Optional, Sequence, Any

from langchain_core.prompts import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.service_context_elements.llm_predictor import LLMPredictor
from llama_index.core.types import RESPONSE_TEXT_TYPE

import os
import pprint

from dotenv import load_dotenv
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.data_structs import Node
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager

load_dotenv()
embed_model = OllamaEmbedding(model_name=os.getenv("LLAMA_EMBED_MODEL_NAME"),
                              embed_batch_size=50)
llm = Ollama(
    model=os.getenv("LLAMA_MODEL_NAME"),
    request_timeout=120.0,
    # Manually set the context window to limit memory usage
    context_window=8000,
)

Settings.embed_model = embed_model
Settings.llm = llm


# 自定义的响应生成器

class FunnySynthesizer(BaseSynthesizer):
    my_prompt_tmpl = (
        "根据以下上下文信息：\n"
        "--------------------"
        "{context_str}\n"
        "--------------------"
        "使用中文且幽默风趣的风格回答以下问题\n"
        "问题：{query_str}\n"
        "答案："
    )

    def __init__(self, llm: Optional[LLM]):
        super().__init__(llm=llm)
        self._input_prompt = PromptTemplate(FunnySynthesizer.my_prompt_tmpl)

    def _get_prompts(self) -> PromptDictType:
        pass

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        pass

    def get_response(
            self,
            query_str: str,
            text_chunks: Sequence[str],
            **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        context_str = "\n\n".join(n for n in text_chunks)

        return self._llm.predict(
            self._input_prompt,
            query_str=query_str,
            context_str=context_str,
            **response_kwargs,
        )

    async def aget_response(
            self,
            query_str: str,
            text_chunks: Sequence[str],
            **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        context_str = "\n\n".join(n for n in text_chunks)

        return await self._llm.predict(
            self._input_prompt,
            query_str=query_str,
            context_str=context_str,
            **response_kwargs,
        )


nodes = [
    NodeWithScore(node=Node(text="实际开发中，大多数Node对象是用Document对象通过各种数据分割器（用于解析Document对象的内容并进行分割的组件，将在5.3节介绍）生成的。\n"
                                 "在下面的例子中，我们构造一个Document对象，然后使用基于分割符的数据分割器把其转换为多个Node对象",
                            score=1.0)),
    NodeWithScore(node=Node(text="""资源提取信息。不同于静态的先检索后阅读模式，Agentic RAG涉及对LLM的迭代调用，穿插使用工具或函数调用并输出结构化结果。系统评估结果，优化查询，必要时调用更多工具，并持续循环，直到获得满意的解决方案。这种迭代的“制造者-检查者”模式提升了正确性，处理格式错误的查询，并确保结果质量。
系统主动掌控其推理过程，会重写失败的查询，选择不同的检索方式，整合多种工具——例如Azure AI Search中的向量搜索、SQL数据库或自定义API——然后才最终给出答案。agentic系统的显著特点是能够自主掌控推理过程。传统的RAG实现依赖预定义路径，而agentic系统则基于所获信息质量自主决定步骤顺序。""",
                            score=1.0)),
    NodeWithScore(node=Node(
        text="""RAG（Retrieval-Augmented Generation，检索增强生成） 是一种结合了信息检索技术与语言生成模型的人工智能技术。该技术通过从外部知识库中检索相关信息，并将其作为提示（Prompt）输入给大型语言模型（LLMs），以增强模型处理知识密集型任务的能力，如问答、文本摘要、内容生成等。RAG模型由Facebook AI Research（FAIR）团队于2020年首次提出，并迅速成为大模型应用中的热门方案。""",
        score=1.0)),
]

# 构造一个生成器
# response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)
response_synthesizer = FunnySynthesizer(llm=llm)

# 测试，调用响应生成器生成结果
reponse = response_synthesizer.synthesize(
    "RAG是什么？请使用中文回答",
    nodes=nodes,
)

print(reponse)


def print_events_llm():
    events = llama_debug.get_event_pairs('llm')

    # 发生了多少次大模型调用
    print(f"Number of LLM calls: {len(events)}")

    # 依次打印所有大模型调用的消息
    for i, event in enumerate(events):
        print(f"\n==================LLM Call {i + 1} messages=================")
        pprint.pprint(event[1].payload['messages'])
        print(f"\n==================LLM Call {i + 1} response=================")
        pprint.pprint(event[1].payload['response'].message.content)


print_events_llm()
