import os

from dotenv import load_dotenv
from llama_index.core import Document,  VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
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

# documents
# docs = SimpleDirectoryReader(input_files=["../../data/yiyan.txt"]).load_data()
docs = [Document(text="小麦手机是小麦公司最新出的第十代手机产品。\
                       采用了中国最先进的国产红旗CPU芯片。\
                       采用了6.95寸的OLED显示屏幕与5000毫安的电池容量。\
                       操作系统是最新的澎湃OS 3.0，兼容现有的安卓APP，是一款高性能的智能手机。\
                       与苹果手机相比较，小麦手机具有更高的性价比。", metadata={'window': 'this is window text'})]
nodes = SentenceSplitter(chunk_size=50, chunk_overlap=0).get_nodes_from_documents(docs)
vector_index = VectorStoreIndex(nodes)

""" 
# 自定义的节点后处理器
class MyNodePostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:

        pattern = r"过滤正则表达式"
        filtered_nodes = []
        for node in nodes:
            if not re.search(pattern, node.text):
                filtered_nodes.append(node)
        nodes = filtered_nodes
        return nodes

query_engine = vector_index.as_query_engine(
    node_postprocessors=[
        MyNodePostprocessor()
    ]
)
response = query_engine.query('小麦手机的显示屏是多大的？')
pprint_response(response) 
"""

retriever = VectorIndexRetriever(vector_index, similarity_top_k=1)
nodes_with_scores = retriever.retrieve('小麦手机')

print('===================Before postprocessing===================')
print_nodes(nodes_with_scores, MetadataMode.ALL)

processor = MetadataReplacementPostProcessor(target_metadata_key="window")
filtered_nodes = processor.postprocess_nodes(nodes_with_scores)

print('===================After postprocessing===================')
print_nodes(filtered_nodes, MetadataMode.ALL)
