import os
import pprint

from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.core.schema import Document

load_dotenv()
embed_model = OllamaEmbedding(model_name=os.getenv("LLAMA_EMBED_MODEL_NAME"),
                              embed_batch_size=50)
Settings.embed_model = embed_model

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

# 存储到向量库中
simple_vectorstore = SimpleVectorStore()
simple_vectorstore.add(nodes)

pprint.pprint(simple_vectorstore.to_dict())

# 查询
result = simple_vectorstore.query(
    VectorStoreQuery(query_embedding=embed_model.get_text_embedding("什么是Node"), similarity_top_k=1))
print(result)

# 持久化
simple_vectorstore.persist()

# 重新从文件加载
simple_vectorstore = SimpleVectorStore.from_persist_path('./storage/vector_store.json')
