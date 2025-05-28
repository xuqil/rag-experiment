import mimetypes
import os
import re
import logging
from typing import List

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def loadtext(path: str):
    """加载和读取文档"""
    try:
        path = path.strip().replace(" \n", " ")
        filename = os.path.abspath(path)

        if not os.path.exists(filename):
            logger.error(f"File {filename} not found")
            return None

        if mimetypes.guess_type(filename)[0] != 'text/plain':
            logger.warning(f"Unsupported file type: {filename}")
            return None

        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()

    except Exception as e:
        logger.error(f"Error loading file {path}: {str(e)}")
        return None


def split_text_by_sentence(source_text: str,
                           sentence_per_chunk: int = 8,
                           overlap: int = 2) -> List[str]:
    """文本分割函数"""
    if not source_text:
        return []

    if sentence_per_chunk < 1:
        raise ValueError("sentence_per_chunk must be at least 1")

    if overlap < 0 or overlap >= sentence_per_chunk:
        raise ValueError("overlap must be >=0 and < sentence_per_chunk")

    # 句子分割
    sentences = re.split(r'(?<=[。！？.?!])\s+', source_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        logger.warning("No sentences found in the text")
        return []

    chunks = []
    for i in range(0, len(sentences), sentence_per_chunk - overlap):
        end = min(i + sentence_per_chunk, len(sentences))
        chunk = ' '.join(sentences[i:end])
        chunks.append(chunk)

    return chunks


def main():
    # 配置检查
    required_env_vars = ["API_KEY", "API_BASE_URL", "CHROMADB_HOST",
                         "CHROMADB_PORT", "CHROMADB_COLLECTION", "EMBED_MODEL_NAME"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(
            f"Missing environment variables: {missing_vars}")

    # 初始化客户端
    llm_client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("API_BASE_URL"),
        timeout=30  # 添加超时设置
    )

    # 向量数据库连接
    try:
        db_client = chromadb.HttpClient(
            host=os.getenv("CHROMADB_HOST"),
            port=os.getenv("CHROMADB_PORT"),
            settings=chromadb.Settings(anonymized_telemetry=False)
        )
        db_client.heartbeat()

        # 清理并创建集合
        try:
            db_client.delete_collection(name=os.getenv("CHROMADB_COLLECTION"))
        except Exception as e:
            logger.info(f"Collection did not exist: {str(e)}")

        collection = db_client.get_or_create_collection(
            name=os.getenv("CHROMADB_COLLECTION"),
            metadata={"hnsw:space": "cosine"}  # 明确指定相似度度量
        )

    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {str(e)}")
        return

    # 处理文档
    try:
        with open("docs.txt", "r", encoding='utf-8') as f:
            for filename in f:
                filename = filename.strip()
                if not filename:
                    continue

                text = loadtext(filename)
                if not text:
                    continue

                chunks = split_text_by_sentence(
                    text, sentence_per_chunk=8, overlap=2)

                for idx, chunk in enumerate(chunks):
                    try:
                        # 添加进度显示
                        logger.info(f"Processing {filename} - chunk {idx+1}/{len(chunks)}")

                        # 生成嵌入向量
                        response = llm_client.embeddings.create(
                            model=os.getenv("EMBED_MODEL_NAME"),
                            input=chunk,
                            encoding_format="float"
                        )

                        # 存储到向量库
                        collection.add(
                            ids=f"{filename}_{idx}",
                            embeddings=[response.data[0].embedding],
                            documents=[chunk],
                            metadatas={"source": filename}
                        )

                    except Exception as e:
                        logger.error(
                            f"Error processing chunk {idx} of {filename}: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")


if __name__ == "__main__":
    main()
