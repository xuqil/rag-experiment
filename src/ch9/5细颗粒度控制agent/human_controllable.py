import os
import pprint

import chromadb
from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore

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

chroma = chromadb.HttpClient(
    host=os.getenv("CHROMADB_HOST", "localhost"),
    port=os.getenv("CHROMADB_PORT", 8000),
    settings=chromadb.Settings(anonymized_telemetry=False)
)
chroma.heartbeat()
collection = chroma.get_or_create_collection(name="controllable_agent", metadata={"hnsw:space": "cosine"})
vector_store = ChromaVectorStore(chroma_collection=collection)

citys_dict = {
    '北京市': 'beijing',
    '南京市': 'nanjing',
    '广州市': 'guangzhou',
    '上海市': 'shanghai',
    '深圳市': 'shenzhen'
}


def create_city_tool(name: str):
    print(f'Starting to create tool agent for 【{name}】...\n')
    city_docs = SimpleDirectoryReader(input_files=[f"../../data/citys/{name}.txt"]).load_data()
    nodes = SentenceSplitter(chunk_size=500, chunk_overlap=50).get_nodes_from_documents(city_docs)

    collection = chroma.get_or_create_collection(name=f"agent_{citys_dict[name]}", metadata={"hnsw:space": "cosine"})
    vector_store = ChromaVectorStore(chroma_collection=collection)

    if not os.path.exists(f"./storage/{citys_dict[name]}"):
        print('Creating vector index...\n')
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
        vector_index.storage_context.persist(persist_dir=f"./storage/{citys_dict[name]}")
    else:
        print('Loading vector index...\n')
        storage_context = StorageContext.from_defaults(persist_dir=f"./storage/{citys_dict[name]}",
                                                       vector_store=vector_store)
        vector_index = load_index_from_storage(storage_context=storage_context)

    vector_query_engine = vector_index.as_query_engine()

    vector_query_engine_tool = QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name=f"vector_tool_{citys_dict[name]}",
            description=(
                f"Useful for questions related to specific aspects of {citys_dict[name]} (e.g. the history, arts and culture,"
                " sports, demographics, or more)."
            ),
        ),
    )

    return vector_query_engine_tool


query_engine_tools = []
for city in citys_dict.keys():
    query_engine_tools.append(create_city_tool(city))

openai_step_engine = OpenAIAgentWorker.from_tools(
    query_engine_tools, verbose=True
)
agent = AgentRunner(openai_step_engine)

task_message = None
while task_message != "exit":
    task_message = input(">> 你: ")
    if task_message == "exit":
        break

    task = agent.create_task(task_message)

    response = None
    step_output = None
    message = None

    while message != "exit" and (not step_output or not step_output.is_last):

        # 执行任务下一步
        if message is None or message == "":
            step_output = agent.run_step(task.task_id)
        else:
            step_output = agent.run_step(task.task_id, input=message)

        # 如果任务没结束，允许用户输入
        if not step_output.is_last:
            message = input(">>  请补充任务反馈信息（留空继续，exit退出）: ")

    if step_output.is_last:
        print(">> 任务运行完成。")
        response = agent.finalize_response(task.task_id)
        print(f"Final Answer: {str(response)}")
    elif not step_output.is_last:
        print(">> 任务未完成，被丢弃。")
