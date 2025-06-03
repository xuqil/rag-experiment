import os

from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.base.embeddings.base import similarity
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike

from src.tools import print_nodes, my_chunking_tokenizer_fn

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

docs = [Document(text="小麦手机是小麦公司最新出的第十代手机产品。\
                      采用了中国最先进的国产红旗CPU芯片。\
                      采用了6.95寸的OLED显示屏幕与5000毫安的电池容量。")]

# splitter
node_parser = SentenceWindowNodeParser.from_defaults(
    sentence_splitter=my_chunking_tokenizer_fn,
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
nodes = node_parser.get_nodes_from_documents(docs)
print_nodes(nodes)

vector_index = VectorStoreIndex(nodes=nodes)
query_engine = vector_index.as_query_engine(
    similarity_top_k=1,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
window_response = query_engine.query(
    "小麦手机是哪个公司出品，采用什么芯片？"
)
print("===============before====================")
pprint_response(vector_index.as_query_engine(similarity_top_k=1).query("小麦手机是哪个公司出品，采用什么芯片？"),
                show_source=True)
print("===============after====================")
pprint_response(window_response, show_source=True)
