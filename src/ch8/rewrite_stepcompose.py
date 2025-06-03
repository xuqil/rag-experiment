import os

import chromadb
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, \
    PromptTemplate
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.core.indices.query.query_transform import StepDecomposeQueryTransform

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
Settings.embedded_model = embed_model

prompt_templ = """
我们有机会从知识源中回答部分或全部问题。知识源的上下文信息如下，并提供了之前的推理步骤。
根据上下文和之前的推理，返回一个可以从上下文中回答的问题：
1. 这个问题可以帮助回答原问题。与原问题密切相关。
2. 可以是原问题的子问题，或者是解答原问题需要的一个步骤中需要的问题。
如果无法从上下文中提取更多信息，则提供“无”作为答案。下面给出了一个示例：

-----
问题：2020年澳大利亚网球公开赛冠军获得了多少个大满贯冠军？
知识源上下文：提供了2020年澳大利亚网球公开赛冠军的名字
之前的推理：无
新问题：谁是2020年澳大利亚网球公开赛的冠军？
-----

我的问题：{query_str}
知识源上下文：{context_str}
之前的推理：{prev_reasoning}
新问题：

"""

# vector db
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
    city_docs = SimpleDirectoryReader(
        input_files=["../../data/citys/北京市.txt", "../../data/citys/上海市.txt"]).load_data()
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

    vector_query_engine = vector_index.as_query_engine(similarity_top_k=1)

    return vector_query_engine


# query transfrom
step_transformer = StepDecomposeQueryTransform(llm=llm, verbose=True)

# 更换提示
new_prompt = PromptTemplate(prompt_templ)
step_transformer.update_prompts({'step_decompose_query_prompt': new_prompt})

query_engine = create_city_engine('南京市')

step_query_engine = MultiStepQueryEngine(query_engine=query_engine,
                                         query_transform=step_transformer,
                                         index_summary='这是一个关于城市的知识库，用来回答城市相关的信息问题')
print('\nQuerying the stepcompose city engine...')
response = step_query_engine.query("中国首都的城市人口有多少？和上海相比呢？")
pprint_response(response, show_source=False)
