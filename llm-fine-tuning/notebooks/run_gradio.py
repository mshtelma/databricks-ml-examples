# Databricks notebook source

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %pip install vllm

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os

os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

# COMMAND ----------
from huggingface_hub import notebook_login

notebook_login()

# COMMAND ----------

import logging


logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)

# COMMAND ----------

from databricks_llm.notebook_utils import get_dbutils
from databricks_llm.gradio_utils import *

# COMMAND ----------
DEFAULT_INPUT_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
SUPPORTED_INPUT_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mosaicml/mpt-30b-instruct",
    "mosaicml/mpt-7b-instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "codellama/CodeLlama-7b-hf",
    "codellama/CodeLlama-13b-hf",
    "codellama/CodeLlama-34b-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-40b-instruct",
    "HuggingFaceH4/starchat-beta",
]
# COMMAND ----------

# get_dbutils().widgets.text("num_gpus", "4", "num_gpus")
get_dbutils().widgets.combobox(
    "pretrained_name_or_path",
    DEFAULT_INPUT_MODEL,
    SUPPORTED_INPUT_MODELS,
    "pretrained_name_or_path",
)
get_dbutils().widgets.text("num_gpus", "4", "num_gpus")

# COMMAND ----------

pretrained_name_or_path = get_dbutils().widgets.get("pretrained_name_or_path")
num_gpus = int(get_dbutils().widgets.get("num_gpus"))

from vllm import LLM, SamplingParams

llm = LLM(model=pretrained_name_or_path, dtype="float16", tensor_parallel_size=num_gpus)


# COMMAND ----------
def generate_text_vllm(prompts, **kwargs):
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 1024

    print(kwargs)
    print(prompts)

    outputs = llm.generate(prompts, sampling_params=SamplingParams(**kwargs))
    texts = [out.outputs[0].text for out in outputs]

    return texts


# print(generate_text_vllm("What is AI?"))

# COMMAND ----------


def get_prompt_recreate_finding_llamav2(query: str, context: str) -> str:
    return f"""<s>[INST] <<SYS>>
    You are a helpful assistant. If you do not know, just say so.
    <</SYS>> 

    {context}

    Please answer the question:  
    {query} [/INST]"""


def generate_answer(query: str, context: str) -> str:
    if context and len(context) > 5:
        context = f"Additional Context:\n{context}\n"
    else:
        context = ""
    prompt = get_prompt_recreate_finding_llamav2(query, context)
    res = generate_text_vllm(prompt, temperature=0.15, top_p=0.9, top_k=10)

    return res[0]


# COMMAND ----------

import gradio as gr


with gr.Blocks() as demo:
    gr_state = gr.State(value={})
    with gr.Row():
        intro_md_str = """
        # Databricks LLM Interface
        ### Welcome to the QA Interface!
        """
        intro = gr.Markdown(intro_md_str)
    with gr.Row():
        c_box = gr.Textbox(label="Additional context", lines=5, value="")
    with gr.Row():
        q_box = gr.Textbox(
            label="Please provide your question", lines=10, value="What is ML?"
        )
    with gr.Row():
        output = gr.Textbox(label="Answer:", lines=4)

    with gr.Row():

        def ask_question(question, context):
            return generate_answer(question, context)

        submit_btn = gr.Button("Get Answer")
        submit_btn.click(fn=ask_question, inputs=[q_box, c_box], outputs=output)


# COMMAND ----------

dbx_app = DatabricksApp(8765)
dbx_app.mount_gradio_app(demo)
dbx_app.get_gradio_url()

# COMMAND ----------

import nest_asyncio

nest_asyncio.apply()
dbx_app.run()
