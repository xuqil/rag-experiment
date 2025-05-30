import os
import pprint
import sys
import logging
import chromadb
from dotenv import load_dotenv

from llama_index.llms.openai_like import OpenAILike
from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.dashscope import DashScopeEmbedding

# 配置日志
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logger = logging.getLogger(__name__)

load_dotenv()

# 设值 embed 模型
Settings.embed_model = DashScopeEmbedding(
    model_name=os.getenv("EMBED_MODEL_NAME"),
    api_base=os.getenv("API_BASE_URL"),
    api_key=os.getenv("API_KEY"),
)


def main():
    # 加载与读取文档
    # reader = SimpleDirectoryReader(
    #     input_files=["../../../../data/yiyan.txt", "../../../../data/xiaomai.txt"])
    # documents = reader.load_data()

    # 分割文档
    # node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
    # nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

    # 准备向量存储
    chroma_client = chromadb.EphemeralClient()
    chroma_client.heartbeat()
    # try:
    #     chroma_client.delete_collection(os.getenv("CHROMADB_COLLECTION"))
    # except Exception as e:
    #     logger.info(f"删除集合失败: {str(e)}")
    chroma_collection = chroma_client.get_or_create_collection(
        name=os.getenv("CHROMADB_COLLECTION"),
        metadata={"hnsw:space": "cosine"}  # 明确指定相似度度量
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 准备向量索引
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    # index = VectorStoreIndex(
    #     nodes,
    #     storage_context=storage_context
    # )

    # 构造向量存储索引
    query_engine = index.as_query_engine(llm=OpenAILike(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE_URL"),
        is_chat_model=True,
        timeout=120  # 设置超时
    ))

    prompt_dict = query_engine.get_prompts()
    pprint.pprint(prompt_dict.keys())
    # 拿到提示词模版
    pprint.pprint(prompt_dict['response_synthesizer:text_qa_template'].get_template())

    # 修改提示词
    my_qa_prompt_tmpl_str = (
        "以下是上下文提示词。\n"
        '{context_str}\n'
        '---------------------\n'
        '根据上下文信息回答问题，不要依赖预置知识，不要编造。\n'
        '问题: {query_str}\n'
        '回答: '
    )
    my_qa_prompt_tmpl = PromptTemplate(my_qa_prompt_tmpl_str)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": my_qa_prompt_tmpl})

    prompt_dict = query_engine.get_prompts()
    pprint.pprint(prompt_dict['response_synthesizer:text_qa_template'].get_template())

    # 初始阶段修改提示词也可以
    # query_engine = index.as_query_engine(llm=OpenAILike(
    #     model=os.getenv("MODEL_NAME"),
    #     api_key=os.getenv("API_KEY"),
    #     api_base=os.getenv("API_BASE_URL"),
    #     is_chat_model=True,
    #     timeout=120  # 设置超时
    # ), text_qa_template=my_qa_prompt_tmpl)

    # while True:
    #     user_input = input("问题：")
    #     if user_input.lower() == "quit":
    #         break
    #
    #     response = query_engine.query(user_input)
    #     print("AI助手：", response.response)


if __name__ == "__main__":
    main()
