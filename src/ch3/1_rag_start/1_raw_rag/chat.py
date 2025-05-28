import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def main():
    try:
        # 初始化客户端
        db_client = chromadb.HttpClient(
            host=os.getenv("CHROMADB_HOST", "localhost"),
            port=os.getenv("CHROMADB_PORT", "8000"),
            settings=chromadb.Settings(anonymized_telemetry=False)
        )

        # 检查连接
        try:
            db_client.heartbeat()
        except Exception as e:
            print(f"无法连接到ChromaDB: {e}")
            return

        # 获取集合
        collection_name = os.getenv(
            "CHROMADB_COLLECTION", "default_collection")
        collection = db_client.get_or_create_collection(name=collection_name)

        # 初始化LLM客户端
        llm_client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("API_BASE_URL"),
            timeout=30
        )

        # 主循环
        while True:
            query = input("\n请输入查询语句(输入quit退出): ").strip()
            if query.lower() == "quit":
                break
            if not query:
                print("查询不能为空")
                continue

            try:
                # 生成嵌入向量
                embed_response = llm_client.embeddings.create(
                    model=os.getenv("EMBED_MODEL_NAME",
                                    "text-embedding-ada-002"),
                    input=query,
                    encoding_format="float"
                )
                query_embed = embed_response.data[0].embedding

                # 检索上下文
                results = collection.query(
                    query_embeddings=[query_embed],
                    n_results=3  # 限制返回结果数量
                )
                docs = "\n\n".join(
                    results["documents"][0]) if results["documents"][0] else "无相关上下文"

                # 生成Prompt
                model_query = f"""请基于以下的上下文回答问题:
                
                上下文:
                ======
                {docs}
                ======
                
                问题: {query}
                
                如果上下文不足以回答问题，请回答'根据现有信息无法回答该问题'。"""

                # 生成回答
                response = llm_client.chat.completions.create(
                    model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
                    messages=[
                        {"role": "system", "content": "你是一个有帮助的助手，基于提供的上下文回答问题。"},
                        {"role": "user", "content": model_query},
                    ],
                    stream=True,
                    temperature=0.7  # 控制创造性
                )

                # 流式输出
                print("\n" + "="*50)
                print("回答: ", end="", flush=True)
                for chunk in response:
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
                print("\n" + "="*50)

            except Exception as e:
                print(f"\n处理查询时出错: {e}")

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序发生错误: {e}")


if __name__ == "__main__":
    main()
