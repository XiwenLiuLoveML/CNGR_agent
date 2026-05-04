"""
llm.py

这个文件负责封装大模型调用。

它做三件事：

1. 从 .env 文件中读取模型相关配置
2. 创建 OpenAI 兼容格式的 client
3. 提供两个通用函数：
   - call_llm()：调用聊天模型
   - get_embedding()：调用 embedding 模型

本项目统一使用 .env 中的这些变量名：

- API_KEY
- BASE_URL
- MODEL_NAME
- EMBEDDING_MODEL_NAME
- WEBHOOK_URL

其中 WEBHOOK_URL 主要给 tool.py 里的消息推送工具使用，
llm.py 只负责模型调用。
"""

import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI


# 读取 .env 文件
load_dotenv()


# 从 .env 中读取模型配置
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")


# 做一个最基本的配置检查
if not API_KEY:
    raise ValueError("没有找到 API_KEY。请检查 .env 文件。")

if not BASE_URL:
    raise ValueError("没有找到 BASE_URL。请检查 .env 文件。")

if not MODEL_NAME:
    raise ValueError("没有找到 MODEL_NAME。请检查 .env 文件。")

if not EMBEDDING_MODEL_NAME:
    raise ValueError("没有找到 EMBEDDING_MODEL_NAME。请检查 .env 文件。")


# 创建 OpenAI 兼容 client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


def call_llm(
    user_prompt: str,
    system_prompt: str = "你是一个专业、可靠、表达清晰的 AI 助手。",
    model: str = MODEL_NAME,
    temperature: float = 0.3
) -> str:
    """
    调用聊天模型，返回模型生成的文本。

    参数：
    - user_prompt：用户输入，或已经拼好的完整 prompt
    - system_prompt：系统提示词，用来设定 AI 的角色和规则
    - model：使用哪个聊天模型，默认使用 .env 里的 MODEL_NAME
    - temperature：控制回答随机性，越低越稳定

    返回：
    - 模型生成的文本
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=temperature
    )

    return response.choices[0].message.content


def get_embedding(
    text: str,
    model: str = EMBEDDING_MODEL_NAME
) -> List[float]:
    """
    调用 embedding 模型，把文本转换成向量。

    这个函数主要给 Day4 的 RAG 使用。

    参数：
    - text：需要转换成向量的文本
    - model：使用哪个 embedding 模型，默认使用 .env 里的 EMBEDDING_MODEL_NAME

    返回：
    - 一个由数字组成的向量
    """

    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response.data[0].embedding


if __name__ == "__main__":
    # 简单测试：只有直接运行这个文件时才会执行
    result = call_llm("请用一句话介绍什么是职业数字人。")
    print(result)