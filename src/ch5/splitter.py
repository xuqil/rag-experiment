from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.node_parser.text.utils import (
    split_by_char,
    split_by_regex,
    split_by_sentence_tokenizer,
    split_by_sep
)

import tiktoken


splitter = SentenceSplitter()
nodes = splitter.get_nodes_from_documents([Document(text="This is a text.\n hello")])
print(nodes)

index = VectorStoreIndex.from_documents([Document(text="This is a text.\n hello")],
                                        transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)])
