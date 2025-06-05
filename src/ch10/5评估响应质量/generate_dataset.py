import os

from dotenv import load_dotenv
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, SimpleDirectoryReader

load_dotenv()

llm = OpenAILike(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("API_BASE_URL"),
    is_chat_model=True,
    timeout=120  # 设置超时
)
embed_model = DashScopeEmbedding(
    model_name=os.getenv("EMBED_MODEL_NAME"),
    api_base=os.getenv("API_BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

Settings.llm = llm
Settings.embed_model = embed_model

# build documents
docs = SimpleDirectoryReader(input_files=['../../../data/citys/南京市.txt']).load_data()

# define generator, generate questions
dataset_generator = RagDatasetGenerator.from_documents(
    documents=docs,
    llm=llm,
    num_questions_per_chunk=1,  # 设置每个节点的问题数量
    show_progress=True,
    question_gen_query="您是一位老师。您的任务是为即将到来的考试设置{num_questions_per_chunk}个问题。这些问题必须基于提供的上下文信息生成，并确保上下文信息能够回答这些问题。确保每一行只有一个独立的问题。不要有多余解释。不要问题编号。"
)

import myprompts

dataset_generator.update_prompts({
    "text_question_template": myprompts.MY_QUESTION_GENERATION_PROMPT,
    "text_qa_template": myprompts.MY_TEXT_QA_PROMPT,
})

print('Generating questions from nodes...\n')
rag_dataset = dataset_generator.generate_dataset_from_nodes()
rag_dataset.save_json('./rag_eval_dataset.json')

print('Loading dataset...\n')
rag_dataset = LabelledRagDataset.from_json('./rag_eval_dataset.json')
for example in rag_dataset.examples:
    print(f'query: {example.query}')
    print(f'answer: {example.reference_answer}')
