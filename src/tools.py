import os
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks import LlamaDebugHandler
from llama_index.core.schema import TextNode, MetadataMode
import pprint
from enum import Enum

import re


def my_chunking_tokenizer_fn(text: str):
    sentence_delimiters = re.compile(u'[。！？]')
    sentences = sentence_delimiters.split(text)
    return [s.strip() for s in sentences if s]


def print_nodes(nodes: list[TextNode], metadata_mode=MetadataMode.NONE):
    print('Count of nodes:', len(nodes))
    for index, node in enumerate(nodes):
        print("\n-----")
        print(f"Node {index}, ID: {node.node_id}")
        print(f'\ntext:{node.text}')
        print(f'\nmetadata:{node.metadata}')

        if hasattr(node, '\nrelationships'):
            print(f"Relationships: {node.relationships}")

        if hasattr(node, 'score'):
            print(f"Score: {node.score}")

        print("-----")


# 打印对象属性
def print_attrs(obj, indent=0):
    print(' ' * indent + f'Object of type: {type(obj)}')
    if hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            print(' ' * (indent + 2) + f'{key}:')
            print_attrs(value, indent + 4)
    elif isinstance(obj, (list, tuple, set)):
        for i, item in enumerate(obj):
            print(' ' * (indent + 2) + f'Item {i}:')
            print_attrs(item, indent + 4)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            print(' ' * (indent + 2) + f'{key}:')
            print_attrs(value, indent + 4)
    else:
        print(' ' * (indent + 2) + f'Value: {obj}')


# langfuse跟踪
def enable_trace():
    # Langfuse
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-e320fa32-8c2e-43d6-9380-347c76fd0122"
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-af5a7a7b-b74c-45dd-9d77-dbd63bf6e63b"
    os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

    from langfuse.llama_index import LlamaIndexInstrumentor
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.start()


class TraceType(Enum):
    LANGFUSE = "langfuse"
    LLAMA_DEBUG = "llama_debug"


class EventType(Enum):
    CHUNKING = "chunking"
    NODE_PARSING = "node_parsing"
    EMBEDDING = "embedding"
    LLM = "llm"
    QUERY = "query"
    RETRIEVE = "retrieve"
    SYNTHESIZE = "synthesize"
    TREE = "tree"
    SUB_QUESTION = "sub_question"
    TEMPLATING = "templating"
    FUNCTION_CALL = "function_call"
    RERANKING = "reranking"
    EXCEPTION = "exception"
    AGENT_STEP = "agent_step"


class DebugUtils:
    def __init__(self, trace_type: TraceType):
        self.llama_debug = None
        self.callback_manager = None

        if trace_type == TraceType.LANGFUSE:
            enable_trace()
        elif trace_type == TraceType.LLAMA_DEBUG:
            self.enable_llama_debug_trace()

    def enable_llama_debug_trace(self):
        self.llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        self.callback_manager = CallbackManager([self.llama_debug])
        Settings.callback_manager = self.callback_manager

    def print_events(self, event_type: EventType):
        """打印指定类型的事件
        
        Args:
            event_type: 事件类型,来自EventType枚举
        """
        if not self.llama_debug:
            print("Debug handler not initialized")
            return

        events = self.llama_debug.get_event_pairs(event_type.value)
        print(f'Number of {event_type.value} events: {len(events)}')

        for i, event in enumerate(events):
            print(f'\n=================={event_type.value} event {i + 1}=====================')
            if event_type == EventType.LLM:
                print("Messages:")
                pprint.pprint(event[1].payload["messages"])
                print("\nResponse:")
                pprint.pprint(event[1].payload["response"].message.content)
            else:
                pprint.pprint(event[1].payload)

    def print_events_llm(self):
        """打印LLM相关事件(为保持向后兼容)"""
        self.print_events(EventType.LLM)

# 初始化调试工具实例,如果需要调试，取消注释；也可以enable_trace()来启用langfuse跟踪    
# debug_utils = DebugUtils(TraceType.LLAMA_DEBUG)
