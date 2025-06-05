import os
from typing import Any, Tuple, Dict

from dotenv import load_dotenv
from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from llama_index.readers.web import BeautifulSoupWebReader

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


def _baidu_reader(
        soup: Any, url: str, include_url_in_text: bool = True
) -> Tuple[str, Dict[str, Any]]:
    main_content = soup.find(class_='main')
    if main_content:
        text = main_content.get_text()
    else:
        text = ''
    return text, {"title": soup.find(class_="post__title").get_text()}


web_loader = BeautifulSoupWebReader(website_extractor={"cloud.baidu.com": _baidu_reader})

tool_xiaomai = OnDemandLoaderTool.from_defaults(
    web_loader,
    name="tool_xiaomai",
    description="用来查询本地文件中的小麦手机信息",
)
output = tool_xiaomai.call(urls=["https://cloud.baidu.com/doc/AppBuilder/s/6lq7s8lli"],
                           query_str='百度云千帆appbuilder是什么？请详细介绍')

print(output)
