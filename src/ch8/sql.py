import os

from dotenv import load_dotenv
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, SQLDatabase

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

from sqlalchemy import (
    create_engine,
)

engine = create_engine(f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@localhost:3306/ai")
sql_database = SQLDatabase(engine, include_tables=["customers", "orders"])

""" 
with engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM customers"))
    for row in result:
        print(row) 
"""

from llama_index.core.query_engine import NLSQLTableQueryEngine

# 创建SQLTable查询引擎
query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database, tables=["customers", "orders"], llm=llm
)

response = query_engine.query("一共有多少个订单")
print(response)
