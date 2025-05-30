import pprint

from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import TextNode, Document

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

for i, node in enumerate(nodes):
    print(f"Node {i}: {node.text}")

