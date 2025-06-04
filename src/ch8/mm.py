import os

from dotenv import load_dotenv
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.multi_modal_llms.dashscope import (
    DashScopeMultiModal,
    DashScopeMultiModalModels,
)
from llama_index.multi_modal_llms.dashscope.utils import load_local_images

load_dotenv()

image_urls = [
    "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
]

image_documents = load_image_urls(image_urls)

# 视觉模型
dashscope_multi_modal_llm = DashScopeMultiModal(
    model_name=DashScopeMultiModalModels.QWEN_VL_MAX,
    api_key=os.getenv("API_KEY")
)

complete_response = dashscope_multi_modal_llm.complete(
    prompt="What's in the image?",
    image_documents=image_documents,
)
print(complete_response)
