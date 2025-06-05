import os

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike

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

documents = SimpleDirectoryReader(input_files=["../../../data/citys/南京市.txt"]).load_data()
node_parser = SentenceSplitter(chunk_size=1024)
nodes = node_parser.get_nodes_from_documents(documents)
for idx, node in enumerate(nodes):
    node.id_ = f"node_{idx}"

vector_index = VectorStoreIndex(nodes)
retriever = vector_index.as_retriever(similarity_top_k=2)

from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)

QA_GENERATE_PROMPT_TMPL = """
以下是上下文信息:
---------------------
{context_str}
---------------------
你是位专业教授。你的任务是基于以上的上下文信息，为即将到来的考试设置 {num_questions_per_chunk} 个问题。
这些问题必须基于提供的上下文信息生成，并确保上下文信息能够回答这些问题。确保每一行只有一个独立的问题。不要有多余解释。不要问题编号。"
"""

print("Generating question-context pairs...")
qa_dataset = generate_question_context_pairs(
    nodes,
    llm=llm,
    num_questions_per_chunk=1,
    qa_generate_prompt_tmpl=QA_GENERATE_PROMPT_TMPL
)

print("Saving dataset...")
qa_dataset.save_json("retriever_eval_dataset.json")

print("Loading dataset...")
qa_dataset = EmbeddingQAFinetuneDataset.from_json("retriever_eval_dataset.json")
eval_querys = list(qa_dataset.queries.items())
for eval_id, eval_query in eval_querys[:10]:
    print(f"Query: {eval_query}")

from llama_index.core.evaluation import RetrieverEvaluator

metrics = ["mrr", "hit_rate"]
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    metrics, retriever=retriever
)

for eval_id, eval_query in eval_querys[:10]:
    expect_docs = qa_dataset.relevant_docs[eval_id]

    print(f"Query: {eval_query}, Expected docs: {expect_docs}")

    eval_result = retriever_evaluator.evaluate(query=eval_query, expected_ids=expect_docs)
    print(eval_result)

# eval_results = retriever_evaluator.evaluate_dataset(qa_dataset)
