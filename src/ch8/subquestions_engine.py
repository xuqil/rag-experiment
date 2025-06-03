import os

import chromadb
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings

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

citys_dict = {
    '北京市': 'beijing',
    '南京市': 'nanjing',
    '广州市': 'guangzhou',
    '上海市': 'shanghai',
    '深圳市': 'shenzhen'
}


def create_city_engine(name: str):
    print(f'Starting to create tool agent for 【{name}】...\n')
    city_docs = SimpleDirectoryReader(input_files=[f"../../data/citys/{name}.txt"]).load_data()
    nodes = SentenceSplitter(chunk_size=500, chunk_overlap=50).get_nodes_from_documents(city_docs)

    collection = chroma.get_or_create_collection(name=f"agent_{citys_dict[name]}", metadata={"hnsw:space": "cosine"})
    vector_store = ChromaVectorStore(chroma_collection=collection)

    if not os.path.exists(f"./storage/{citys_dict[name]}"):
        print('Creating vector index...\n')
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
        vector_index.storage_context.persist(persist_dir=f"./storage/{citys_dict[name]}")
    else:
        print('Loading vector index...\n')
        storage_context = StorageContext.from_defaults(persist_dir=f"./storage/{citys_dict[name]}",
                                                       vector_store=vector_store)
        vector_index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)

    vector_query_engine = vector_index.as_query_engine(llm=llm, similarity_top_k=3)

    return vector_query_engine


query_engine_nanjing = create_city_engine('南京市')
query_engine_shanghai = create_city_engine('上海市')

query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine_nanjing,
        metadata=ToolMetadata(
            name="query_tool_nanjing",
            description="用来查询南京市各个方面的信息，如基本信息、旅游指南、城市历史等"
        ),
    ),
    QueryEngineTool(
        query_engine=query_engine_shanghai,
        metadata=ToolMetadata(
            name="query_tool_shanghai",
            description="用来查询上海市各个方面的信息，如基本信息、旅游指南、城市历史等"
        ),
    ),
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True,
)

response = query_engine.query(
    "南京与上海的人口差距是多少？GDP大约相差多少？使用中文回答"
)

print(response)
