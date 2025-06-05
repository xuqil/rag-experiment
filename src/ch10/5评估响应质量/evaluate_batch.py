import os
from typing import Dict, List, Any

import chromadb
import pandas as pd
from dotenv import load_dotenv
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, ContextRelevancyEvaluator, \
    AnswerRelevancyEvaluator, CorrectnessEvaluator, SemanticSimilarityEvaluator
from llama_index.core.llama_dataset import LabelledRagDataset
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

""" 
query = "南京有哪些优秀的旅游景点？请简单介绍。"
response = query_engine.query(query)
print(f'response: {response}\n') 
"""

# create evaluators
faithfulness_evaluator = FaithfulnessEvaluator()
relevancy_evaluator = RelevancyEvaluator()
correctness_evaluator = CorrectnessEvaluator()
similarity_evaluator = SemanticSimilarityEvaluator()

# Load the dataset
rag_dataset = LabelledRagDataset.from_json('./rag_eval_dataset.json')
queries = []
for example in rag_dataset.examples:
    print(f'query: {example.query}')
    queries.append(example.query)

from llama_index.core.evaluation import BatchEvalRunner
import asyncio

runner = BatchEvalRunner(
    {"faithfulness": faithfulness_evaluator,
     "relevancy": relevancy_evaluator,
     "correctness": correctness_evaluator,
     "similarity": similarity_evaluator},
    workers=4
)


async def evaluate_queries():
    """
    异步执行批量评估查询

    Returns:
        Dict: 包含各个评估指标结果的字典
    """
    eval_results = await runner.aevaluate_queries(
        query_engine=query_engine,
        queries=[example.query for example in rag_dataset.examples][:10],
        reference=[example.reference_answer for example in rag_dataset.examples][:10],
    )
    return eval_results


def calculate_metric_scores(eval_results: Dict[str, List[Any]]) -> Dict[str, List[float]]:
    """
    计算每个评估指标的得分

    Args:
        eval_results: 评估结果字典

    Returns:
        Dict: 包含每个指标得分的字典
    """
    data = {}
    for metric_name, results in eval_results.items():
        scores = [result.score for result in results]
        average = sum(scores) / len(scores)
        scores.append(average)  # 添加平均分
        data[metric_name] = scores
    return data


def format_results_dataframe(metric_scores: Dict[str, List[float]],
                             queries: List[str]) -> pd.DataFrame:
    """
    将评估结果格式化为DataFrame

    Args:
        metric_scores: 评估指标得分
        queries: 查询列表

    Returns:
        pd.DataFrame: 格式化后的评估结果
    """
    data = metric_scores.copy()
    queries = [*queries, "【Average】"]
    data["query"] = queries
    return pd.DataFrame(data)


def display_results(eval_results: Dict[str, List[Any]]) -> None:
    """
    展示评估结果

    Args:
        eval_results: 评估结果字典
    """
    metric_scores = calculate_metric_scores(eval_results)
    queries = [result.query for result in eval_results["faithfulness"]]
    results_df = format_results_dataframe(metric_scores, queries)
    print("\n=== 评估结果 ===")
    print(results_df)
    print("\n")


# 执行评估
eval_results = asyncio.run(evaluate_queries())
display_results(eval_results)
