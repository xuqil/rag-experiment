from chromadb import Metadatas
from llama_index.core.schema import Document, MetadataMode
import pprint

doc = Document(text="RAG是一种常见的大模型应用范式，它通过检索——排序——生成的方式生成文本",
               metadata={"title": "RAG模型介绍", "auth": "llama-index"})
pprint.pprint(doc.dict())

doc4 = Document(text="百度是一家中国的搜索引擎公司",
                metadata={
                    "file_name": "test.txt",
                    "category": "technology",
                    "author": "random person"
                },
                excluded_llm_metadata_keys=["file_name"],
                excluded_embed_metadata_keys=["file_name", "author"],
                metadata_seperator=" | ",
                metadata_template="{key}=>{value}",
                text_template="Metadata: {metadata_str}\n---------\nContent:{content}"
                )
print("\n全部元数据：\n",
      doc4.get_content(metadata_mode=MetadataMode.ALL))
print("\n嵌入模型看到的：\n",
      doc4.get_content(metadata_mode=MetadataMode.EMBED))
print("\n大模型看到的：\n",
      doc4.get_content(metadata_mode=MetadataMode.LLM))
print("\n没有元数据：\n",
      doc4.get_content(metadata_mode=MetadataMode.NONE))
