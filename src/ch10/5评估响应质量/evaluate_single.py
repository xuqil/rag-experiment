import os

import chromadb
from dotenv import load_dotenv
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, ContextRelevancyEvaluator, \
    AnswerRelevancyEvaluator, CorrectnessEvaluator, SemanticSimilarityEvaluator
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

llm = OpenAILike(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("API_BASE_URL"),
    is_chat_model=True,
    timeout=120  # 设置超时
)
embed_model = DashScopeEmbedding(
    model_name=os.getenv("EMBED_MODEL_NAME"),
    api_base=os.getenv("API_BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

Settings.llm = llm
Settings.embed_model = embed_model

chroma = chromadb.HttpClient(
    host=os.getenv("CHROMADB_HOST"),
    port=os.getenv("CHROMADB_PORT", 8000),
    settings=chromadb.Settings(anonymized_telemetry=False)
)
collection = chroma.get_or_create_collection(name="docs_agent_collection")
vector_store = ChromaVectorStore(chroma_collection=collection)

DATA_DIR = '../../../data/citys'
STOR_DIR = f'./storage'

city_docs = {
    "Beijing": f'{DATA_DIR}/北京市.txt',
    "Guangzhou": f'{DATA_DIR}/广州市.txt',
    "Nanjing": f'{DATA_DIR}/南京市.txt',
    "Shanghai": f'{DATA_DIR}/上海市.txt'
}


def _create_doc_engine(name: str):
    file = city_docs[name]

    print(f'Starting to create document agent for 【{name}】...\n')
    docs = SimpleDirectoryReader(input_files=[file]).load_data()
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(docs)

    # Create a vector index
    if not os.path.exists(f"{STOR_DIR}/{name}"):
        print('Creating vector index...\n')
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
        vector_index.storage_context.persist(persist_dir=f"{STOR_DIR}/{name}")
    else:
        print('Loading vector index...\n')
        storage_context = StorageContext.from_defaults(persist_dir=f"{STOR_DIR}/{name}", vector_store=vector_store)
        vector_index = load_index_from_storage(storage_context=storage_context)

    return vector_index.as_query_engine(similarity_top_k=5)


query_engine = _create_doc_engine('Nanjing')
query = "南京的气候怎么样"
response = query_engine.query(query)
print(f'response: {response}\n')

# Evaluate the faithfulness
evaluator = FaithfulnessEvaluator()
eval_result = evaluator.evaluate_response(query=query, response=response)
print(f'faithfulness score: {eval_result.score}\n')

# Evaluate the relevancy
evaluator = RelevancyEvaluator()
eval_result = evaluator.evaluate_response(query=query,
                                          response=response)
print(f'relevancy score: {eval_result.score}\n')

# Evaluate the context relevancy
evaluator = ContextRelevancyEvaluator()
eval_result = evaluator.evaluate_response(query=query,
                                          response=response)
print(f'context relevancy score: {eval_result.score}\n')

# Evaluate the answer relevancy
evaluator = AnswerRelevancyEvaluator()
eval_result = evaluator.evaluate_response(query=query,
                                          response=response)
print(f'answer relevancy score: {eval_result.score}\n')

# Evaluate the correctness
evaluator = CorrectnessEvaluator()
eval_result = evaluator.evaluate_response(query=query,
                                          response=response,
                                          reference='南京属于较典型的北亚热带季风气候。这里四季分明，冬夏温差较大，年平均气温为16.4℃，最冷月（1月）平均气温为3.1℃，最热月（7月）平均气温为28.4℃。南京降水丰富，年平均降水量约为1144毫米，且全年有大约112.9天的降雨日。冬季受西伯利亚高压或蒙古高压控制，盛行东北风；夏季则分为初夏多雨的梅雨季节和盛夏的伏旱天气两段。')
print(f'correctness score: {eval_result.score}\n')

# Evaluate the semantic similarity
evaluator = SemanticSimilarityEvaluator()
eval_result = evaluator.evaluate_response(query=query,
                                          response=response,
                                          reference='南京四季分明，冬夏温差较大，冬季受西伯利亚高压或蒙古高压控制，盛行东北风；夏季则分为初夏多雨的梅雨季节和盛夏的伏旱天气两段。')
print(f'semantic similarity score: {eval_result.score}\n')
