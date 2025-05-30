import os

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import SummaryExtractor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import MetadataMode
from llama_index.llms.openai_like import OpenAILike

load_dotenv()
llm = OpenAILike(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("API_BASE_URL"),
    is_chat_model=True,
    timeout=120  # 设置超时
)
# 自动生成 Document 对象的元数据
docs = SimpleDirectoryReader(input_files=["../../data/yiyan.txt"]).load_data(show_progress=True)
# 提取摘要
summary_extractor = SummaryExtractor(llm=llm,
                                     show_progress=False,
                                     prompt_template="请生成以下内容的中文摘要：{context_str}\n 摘要：",
                                     metadata_mode=MetadataMode.NONE)

parser = TokenTextSplitter(chunk_size=100, chunk_overlap=0, separator="\n")
nodes = parser.get_nodes_from_documents(docs)

# 仅支持 TextNode
print(summary_extractor.extract(nodes))
