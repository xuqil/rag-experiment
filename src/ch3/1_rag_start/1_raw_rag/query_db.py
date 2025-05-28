import os
import logging
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def initialize_clients():
    """初始化数据库和LLM客户端"""
    try:
        # 检查必要的环境变量
        required_vars = ["CHROMADB_HOST", "CHROMADB_PORT",
                         "API_KEY", "API_BASE_URL", "EMBED_MODEL_NAME",
                         "CHROMADB_COLLECTION"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"缺少必要的环境变量: {missing_vars}")

        # 初始化向量数据库客户端
        db_client = chromadb.HttpClient(
            host=os.getenv("CHROMADB_HOST"),
            port=os.getenv("CHROMADB_PORT"),
            settings=chromadb.Settings(anonymized_telemetry=False)
        )

        # 检查数据库连接
        db_client.heartbeat()
        logger.info("成功连接到向量数据库")

        # 初始化LLM客户端
        llm_client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("API_BASE_URL"),
            timeout=30  # 设置超时
        )

        return db_client, llm_client

    except Exception as e:
        logger.error(f"初始化客户端失败: {str(e)}")
        raise


def query_database(db_client, llm_client, query: str, n_results: int = 3):
    """查询向量数据库并返回结果"""
    try:
        # 获取集合
        collection = db_client.get_collection(
            name=os.getenv("CHROMADB_COLLECTION"))

        # 生成查询嵌入
        embed_response = llm_client.embeddings.create(
            model=os.getenv("EMBED_MODEL_NAME"),
            input=query,
            encoding_format="float"
        )
        embed = embed_response.data[0].embedding

        # 执行查询
        results = collection.query(
            query_embeddings=[embed],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        return results

    except Exception as e:
        logger.error(f"查询数据库时出错: {str(e)}")
        raise


def display_results(results):
    """格式化显示查询结果"""
    if not results or not results["documents"]:
        print("未找到相关结果")
        return

    print("\n" + "=" * 80)
    print(f"找到 {len(results['documents'][0])} 个相关结果:")
    print("=" * 80)

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ), 1):
        print(f"\n结果 #{i} (相似度: {1-dist:.2f})")
        print(f"来源: {meta['source'] if meta and 'source' in meta else '未知'}")
        print("-" * 60)
        print(doc)
        print("-" * 60)


def main():
    try:
        # 初始化客户端
        db_client, llm_client = initialize_clients()

        print("\n" + "=" * 80)
        print("RAG 查询系统 (输入 'quit' 或 'exit' 退出)")
        print("=" * 80)

        while True:
            try:
                query = input("\n请输入查询语句: ").strip()

                if query.lower() in ("quit", "exit"):
                    break

                if not query:
                    print("查询不能为空")
                    continue

                # 执行查询
                logger.info(f"正在处理查询: {query}")
                results = query_database(db_client, llm_client, query)

                # 显示结果
                display_results(results)

            except KeyboardInterrupt:
                print("\n检测到中断，准备退出...")
                break
            except Exception as e:
                logger.error(f"处理查询时出错: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"系统初始化失败: {str(e)}")
    finally:
        print("\n感谢使用RAG查询系统，再见！")


if __name__ == "__main__":
    main()
