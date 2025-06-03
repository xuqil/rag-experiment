import os
import pprint

import chromadb
from dotenv import load_dotenv
from llama_index.core.base.embeddings.base import similarity
from llama_index.core.indices.list import SummaryIndexRetriever
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, StorageContext, DocumentSummaryIndex, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()
embed_model = OllamaEmbedding(model_name=os.getenv("LLAMA_EMBED_MODEL_NAME"),
                              embed_batch_size=50)
llm = Ollama(
    model=os.getenv("LLAMA_MODEL_NAME"),
    request_timeout=120.0,
    # Manually set the context window to limit memory usage
    context_window=8000,
)

Settings.embed_model = embed_model
Settings.llm = llm

docs = [
    Document(
        text="实际开发中，大多数Node对象是用Document对象通过各种数据分割器（用于解析Document对象的内容并进行分割的组件，将在5.3节介绍）生成的。\n"
             "在下面的例子中，我们构造一个Document对象，然后使用基于分割符的数据分割器把其转换为多个Node对象",
        metadata={"title": "rag"},
        doc_id="doc2"),
    Document(
        text="""资源提取信息。不同于静态的先检索后阅读模式，Agentic RAG涉及对LLM的迭代调用，穿插使用工具或函数调用并输出结构化结果。系统评估结果，优化查询，必要时调用更多工具，并持续循环，直到获得满意的解决方案。这种迭代的“制造者-检查者”模式提升了正确性，处理格式错误的查询，并确保结果质量。
系统主动掌控其推理过程，会重写失败的查询，选择不同的检索方式，整合多种工具——例如Azure AI Search中的向量搜索、SQL数据库或自定义API——然后才最终给出答案。agentic系统的显著特点是能够自主掌控推理过程。传统的RAG实现依赖预定义路径，而agentic系统则基于所获信息质量自主决定步骤顺序。""",
        metadata={"title": "agent"},
        doc_id="doc1")]

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
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

vector_index = VectorStoreIndex.from_vector_store(vector_store)
# 获取构造检索器
retriver = vector_index.as_retriever(similarity_top_k=1)
nodes = retriver.retrieve("RAG的作用")
pprint.pprint(nodes)

# 与as_retriever作用一样
retriver = VectorIndexRetriever(index=vector_index)

# 构造摘要索引检索器
summary_index = DocumentSummaryIndex.from_documents(
    docs,
    storage_context=storage_context,
    summary_query="用中文描述所给文本的主要内容，同时描述这段文本可以回答的一些问题")
retriver = SummaryIndexRetriever(index=summary_index, choice_batch_size=5)
