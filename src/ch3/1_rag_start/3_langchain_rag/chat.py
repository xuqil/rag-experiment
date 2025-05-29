import os

import chromadb
from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatOpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url=os.environ.get("API_BASE_URL"),
    model=os.getenv("MODEL_NAME"),
)
embed_model = DashScopeEmbeddings(model=os.getenv("EMBED_MODEL_NAME"))

# 加载与读取文档
loader = DirectoryLoader('../../../../data', glob='*.txt', exclude='*tips*.txt', loader_cls=TextLoader,
                         loader_kwargs={"encoding": "utf-8"})
documents = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
splits = text_splitter.split_documents(documents)

# 准备向量存储
chroma = chromadb.HttpClient(host=os.getenv("CHROMADB_HOST"), port=int(os.getenv("CHROMADB_PORT")))
try:
    chroma.delete_collection(os.getenv("CHROMADB_COLLECTION"))
except Exception:
    pass
collection = chroma.get_or_create_collection(name=os.getenv("CHROMADB_COLLECTION"), metadata={'hnsw:space': 'cosine'})
db = Chroma(client=chroma, collection_name=os.getenv("CHROMADB_COLLECTION"), embedding_function=embed_model)

# 存储到向量库中，构造索引
db.add_documents(splits)

# 使用检索器
retrieve = db.as_retriever()

# 构造 RAG 链
prompt = hub.pull("rlm/rag-prompt")
rag_chain = (
        {"context": retrieve | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)
while True:
    user_input = input("问题：")
    if user_input.lower() == "quit":
        break

    response = rag_chain.invoke(user_input)
    print("AI助手：", response)
