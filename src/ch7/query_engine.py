import os
import pprint

import chromadb
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.llms.openai_like import OpenAILike
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
Settings.embed_model = embed_model
Settings.llm = llm

docs = [
    Document(
        text="实际开发中，大多数Node对象是用Document对象通过各种数据分割器（用于解析Document对象的内容并进行分割的组件，将在5.3节介绍）生成的。\n"
             "在下面的例子中，我们构造一个Document对象，然后使用基于分割符的数据分割器把其转换为多个Node对象",
        metadata={"title": "rag"}),
    Document(text="""资源提取信息。不同于静态的先检索后阅读模式，Agentic RAG涉及对LLM的迭代调用，穿插使用工具或函数调用并输出结构化结果。系统评估结果，优化查询，必要时调用更多工具，并持续循环，直到获得满意的解决方案。这种迭代的“制造者-检查者”模式提升了正确性，处理格式错误的查询，并确保结果质量。
系统主动掌控其推理过程，会重写失败的查询，选择不同的检索方式，整合多种工具——例如Azure AI Search中的向量搜索、SQL数据库或自定义API——然后才最终给出答案。agentic系统的显著特点是能够自主掌控推理过程。传统的RAG实现依赖预定义路径，而agentic系统则基于所获信息质量自主决定步骤顺序。""",
             metadata={"title": "agent"})]

splitter = SentenceSplitter(chunk_size=100, chunk_overlap=0)
nodes = splitter.get_nodes_from_documents(docs)

# 生成嵌入向量
nodes = embed_model(nodes)
print(nodes)

# 准备向量存储
chroma = chromadb.HttpClient(
    host=os.getenv("CHROMADB_HOST", "localhost"),
    port=os.getenv("CHROMADB_PORT", "8000"),
    settings=chromadb.Settings(anonymized_telemetry=False)
)
chroma.heartbeat()
collection = chroma.get_or_create_collection(
    name="vectorstora",
)
# 构造向量存储对象
vector_store = ChromaVectorStore(chroma_collection=collection)
ids = vector_store.add(nodes)
print(f"{len(ids)} nodes ingested into vector store")
pprint.pprint(vector_store.__dict__)

# 查看有多少
count_result = collection.count()
print("count:", count_result)

# 查询
result = vector_store.query(
    VectorStoreQuery(query_embedding=embed_model.get_text_embedding("什么是Node"), similarity_top_k=1))
print(result)

# 构造基于向量存储的向量存储对象
index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine(Settings.llm)
print(query_engine.query("什么是Agent"))
