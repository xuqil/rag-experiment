import os

from dotenv import load_dotenv
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, PromptTemplate
from llama_index.core.text_splitter import TokenTextSplitter

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


def fn_query_engine_xiaomai():
    # documents
    docs = SimpleDirectoryReader(input_files=["../../../data/xiaomai.txt"]).load_data()
    splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=0, separator="\n", )
    nodes = splitter.get_nodes_from_documents(docs)
    index = VectorStoreIndex(nodes)

    qa_prompt_tmpl = (
        "根据以下上下文信息：\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "使用中文回答问题: {query_str}\n"
        "---------------------\n"
        "需要遵守的一些规则：\n"
        "1. 切勿在回答中直接引用给定的上下文信息。\n"
        "2. 避免使用诸如“根据上下文，...”或“上下文信息...”之类的陈述。\n"
        "3. 如果需要，请遵循下方格式要求，不要有多余格式描述\n"
        "---------------------\n"
    )
    qa_prompt = PromptTemplate(qa_prompt_tmpl)

    query_engine = index.as_query_engine(response_mode="compact", verbose=True, text_qa_template=qa_prompt)

    from llama_index.core.tools import QueryEngineTool, ToolMetadata

    # Create a tool from the query engine
    tool_xiaomai = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="tool_xiaomai",
        description="用于小麦手机信息查询",
        return_direct=False
    )

    return tool_xiaomai


def get_func_tool():
    from llama_index.core.tools import FunctionTool, ToolOutput

    def add(a: int, b: int) -> int:
        return a + b

    # Create a tool from the function
    tool_add = FunctionTool.from_defaults(
        fn=add,
        name="tool_add",
        description="用于两个整数相加",
    )

    return tool_add


def fn_retriever_xiaomai():
    # documents
    docs = SimpleDirectoryReader(input_files=["../../../data/xiaomai.txt"]).load_data()
    splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=0, separator="\n", )
    nodes = splitter.get_nodes_from_documents(docs)
    index = VectorStoreIndex(nodes)

    retriever_xiaomai = index.as_retriever(similarity_K=2, verbose=True)

    from llama_index.core.tools import RetrieverTool

    # Create a tool from the query engine
    tool_retriever_xiaomai = RetrieverTool.from_defaults(
        retriever=retriever_xiaomai,
        description="用于小麦手机信息检索"
    )

    print(tool_retriever_xiaomai.call(query_str="小麦手机采用了什么CPU？"))
    return tool_retriever_xiaomai


fn_retriever_xiaomai()

"""
    output = tool_add.call(a=1,b=3)
    print(type(output))
    print(f"Output: {output.__dict__}")
"""
