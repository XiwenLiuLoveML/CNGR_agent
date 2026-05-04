"""
tool.py

这个文件用于封装 Day 3 的工具函数。

Day 3 的核心是：

模型负责思考和表达，
工具负责读取、搜索、分析、通知和执行。

本文件提供三个最小工具：

1. full_data_analysis_tool()
   读取 CSV / Excel 表格，做基础分析，并让模型总结重点。

2. web_qa_tool()
   读取网页内容，并根据用户问题生成回答。

3. send_message_tool()
   把消息推送到测试群。默认先预览，不直接发送。

注意：
- 任务分类不放在这里，放在 task_router.py。
- 模型连接不放在这里，放在 llm.py。
- 上下文 / RAG 不放在这里，放在 context.py。
- 主流程调度不放在这里，放在 agent_core.py。
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from src.llm import call_llm


# 读取 .env 文件，主要是为了拿 WEBHOOK_URL
load_dotenv()

WEBHOOK_URL = os.getenv("WEBHOOK_URL")


# =========================
# 工具 1：表格分析工具
# =========================

def read_table(file_path: str) -> pd.DataFrame:
    """
    读取 CSV 或 Excel 表格。

    参数：
    - file_path：表格文件路径，例如 data/sales_data.csv

    返回：
    - pandas DataFrame
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"没有找到文件：{file_path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError("目前只支持 CSV 和 Excel 文件。")


def summarize_table_basic(df: pd.DataFrame) -> str:
    """
    对表格做最基础的结构化摘要。

    包括：
    - 行数
    - 列数
    - 字段名
    - 缺失值
    - 数值列基础统计
    """

    row_count, col_count = df.shape
    columns = list(df.columns)

    missing_values = df.isna().sum()
    missing_summary = missing_values[missing_values > 0]

    numeric_summary = df.describe(include="number").to_string()

    summary = f"""
    表格基础信息：

    - 行数：{row_count}
    - 列数：{col_count}
    - 字段名：{columns}

    缺失值情况：
    {missing_summary.to_string() if not missing_summary.empty else "没有明显缺失值。"}

    数值列基础统计：
    {numeric_summary}
    """

    return summary


def full_data_analysis_tool(file_path: str = "data/sales_data.csv") -> str:
    """
    表格分析工具。

    它会：
    1. 读取表格
    2. 做基础统计
    3. 调用大模型，总结 3 个重点

    参数：
    - file_path：表格文件路径，默认读取 data/sales_data.csv

    返回：
    - 一段自然语言分析结果
    """

    try:
        df = read_table(file_path)
        table_summary = summarize_table_basic(df)

        # 为了避免把整张表都塞给模型，只取前 5 行作为样例
        preview = df.head(5).to_string(index=False)

        user_prompt = f"""
    下面是一份表格的基础统计信息和前几行样例。

    请你用简洁中文总结这份表格的 3 个重点。
    如果你发现明显异常、缺失值或值得关注的趋势，也请指出来。
    不要编造表格里没有的信息。

    【基础统计】
    {table_summary}

    【前 5 行样例】
    {preview}
    """

        result = call_llm(
            user_prompt=user_prompt,
            system_prompt="你是一个专业的数据分析助手，擅长把表格信息总结成清晰、实用的中文结论。",
            temperature=0.2
        )

        return result

    except Exception as e:
        return f"表格分析工具执行失败：{e}"


# =========================
# 工具 2：网页读取工具
# =========================

def read_webpage(url: str, max_chars: int = 4000) -> str:
    """
    读取网页正文内容。

    参数：
    - url：网页地址
    - max_chars：最多保留多少字符，避免内容太长

    返回：
    - 网页文本
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # 去掉脚本、样式等无关内容
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # 清理空行
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    return clean_text[:max_chars]


def web_qa_tool(url: str, question: str) -> str:
    """
    网页问答工具。

    它会：
    1. 读取网页文本
    2. 根据用户问题，从网页内容中提取答案
    3. 调用大模型生成简洁回答

    参数：
    - url：网页地址
    - question：用户想问的问题

    返回：
    - 基于网页内容的回答
    """

    try:
        webpage_text = read_webpage(url)

        user_prompt = f"""
    请根据下面的网页内容回答用户问题。

    要求：
    1. 只基于网页内容回答。
    2. 如果网页内容里没有答案，请明确说“网页中没有找到相关信息”。
    3. 回答要简洁、清楚。

    【用户问题】
    {question}

    【网页内容】
    {webpage_text}
    """

        result = call_llm(
            user_prompt=user_prompt,
            system_prompt="你是一个网页信息整理助手，擅长从网页文本中提取和总结关键信息。",
            temperature=0.2
        )

        return result

    except Exception as e:
        return f"网页读取工具执行失败：{e}"


# =========================
# 工具 3：群消息推送工具
# =========================

def send_message_tool(message: str, confirm: bool = False) -> str:
    """
    消息推送工具。

    默认不直接发送，只返回预览。
    当 confirm=True 时，才会真正发送到 WEBHOOK_URL。

    参数：
    - message：要发送的消息
    - confirm：是否确认发送

    返回：
    - 执行结果说明
    """

    if not message.strip():
        return "消息内容为空，不能发送。"

    if not confirm:
        return f"""
    【消息预览】

    {message}

    当前 confirm=False，所以还没有真正发送。
    如果确认要发送，请把 confirm 改成 True。
    """

    if not WEBHOOK_URL:
        return "没有找到 WEBHOOK_URL。请检查 .env 文件。"

    try:
        payload = {
            "msgtype": "text",
            "text": {
                "content": message
            }
        }

        response = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()

        return "消息已发送到测试群。"

    except Exception as e:
        return f"消息推送失败：{e}"


# =========================
# 简单测试
# =========================

if __name__ == "__main__":
    print("正在测试表格分析工具...")
    print(full_data_analysis_tool("data/sales_data.csv"))

    print("\n正在测试消息预览工具...")
    print(send_message_tool("这是一条测试消息。", confirm=False))