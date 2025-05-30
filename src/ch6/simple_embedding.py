import os
import pprint

from dotenv import load_dotenv
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import TextNode, Document, MetadataMode
from llama_index.embeddings.ollama import OllamaEmbedding

load_dotenv()

# 直接生成 node
texts = ["This is a chunk1", "This is a chunk2"]
nodes = [TextNode(text=text) for text in texts]
pprint.pprint(nodes[0])

# 使用Document 生成 node
docs = [Document(
    text="实际开发中，大多数Node对象是用Document对象通过各种数据分割器（用于解析Document对象的内容并进行分割的组件，将在5.3节介绍）生成的。\n"
         "在下面的例子中，我们构造一个Document对象，然后使用基于分割符的数据分割器把其转换为多个Node对象")]

# 构造一个简单的数据分割器
parser = TokenTextSplitter(chunk_size=100, chunk_overlap=0, separator="\n")
nodes = parser.get_nodes_from_documents(docs)

ollama_embedding = OllamaEmbedding(model_name=os.getenv("LLAMA_EMBED_MODEL_NAME"))

# 调用模型接口生成向量
embeddings = ollama_embedding.get_text_embedding_batch(
    [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
    show_progress=True
)

# 把生成的向量放入 Node 对象中
for node, embedding in zip(nodes, embeddings):
    node.embedding = embedding
    print(embedding)

print(nodes[0])


# 一行代码实现
nodes = ollama_embedding(nodes)