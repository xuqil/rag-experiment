from pathlib import Path
from typing import Any, List, Dict, Optional

import psycopg2
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader


class PSQLReader(BaseReader):
    """自定义SQL解析器"""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def load_data(self, file: Path, extra_info: Optional[Dict] = None) -> List[Document]:
        with open(file) as f:
            content = f.read()

        # 执行这个文档中的SQL，获取结果
        result = execute_sql_and_return_results(content)

        metadata = {'file_suffix': 'SQL'}
        if extra_info:
            metadata = {**metadata, **extra_info}

        return [Document(text=result, metadata=metadata)]


def execute_sql_and_return_results(sql: str) -> str:
    conn = psycopg2.connect(
        host="localhost",
        user="postgres",
        password="******",
        database="postgres"
    )
    cur = conn.cursor()
    cur.execute(sql)

    results = []
    for result in cur:
        results.append(str(result))

    conn.close()
    return "\n".join(results)


def main():
    reader = SimpleDirectoryReader(input_files=["../../data/postgres.psql"],
                                   file_extractor={".psql": PSQLReader()})
    documents = reader.load_data()
    print(documents)
