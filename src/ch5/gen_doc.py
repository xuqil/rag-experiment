from llama_index.core.schema import Document, TextNode
from llama_index.core import SimpleDirectoryReader

# 直接生成
texts = ["This is a test", "This is another test", "This is a third test"]
docs = [Document(text=text) for text in texts]

# 使用数据连接器生成
# 加载一个 PDF
docs2 = SimpleDirectoryReader(input_files=["../../data/zte-report-simple.pdf"]).load_data(show_progress=True)
print("The number of documents in docs2 is:", len(docs2))  # 10

# 加载这个目录下的所有文档（共26个TXT文档）
docs3 = SimpleDirectoryReader("../../data/MiniTruthfulQADataset/source_files").load_data(show_progress=True)
print("The number of documents in docs3 is:", len(docs3))  # 26

