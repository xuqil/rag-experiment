import os

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, get_response_synthesizer, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank

from src.tools import print_nodes

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

docs = SimpleDirectoryReader(input_files=["../../data/yiyan.txt"]).load_data()
nodes = SentenceSplitter(chunk_size=100, chunk_overlap=0).get_nodes_from_documents(docs)
vector_index = VectorStoreIndex(nodes)

retriever = vector_index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("百度文心一言的逻辑推理能力怎么样？")
print('================before rerank================')
print_nodes(nodes)

dashscope_rerank = DashScopeRerank(top_n=5)
rerank_nodes = dashscope_rerank.postprocess_nodes(nodes, query_str='百度文心一言的逻辑推理能力怎么样?')
print('================after rerank================')
print_nodes(rerank_nodes)

synthesizer = get_response_synthesizer()
response = synthesizer.synthesize("百度文心一言的逻辑推理能力怎么样？", nodes=rerank_nodes)
pprint_response(response, show_source=True)
