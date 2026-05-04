"""
context.py

这个文件用于封装 Day 4 的上下文能力。

Day 4 的核心是：

context = 模型这一次看到的全部信息。

在本项目里，context 主要包括：

1. 当前问题 current question
2. 身份资料 profile context
   - data/cv.txt
   - data/work_log.txt
3. 对话历史 history
   - data/context/history/
4. 长期记忆 memory
   - data/context/memory/
5. 外部资料 resource
   - data/context/resource/

本文件会提供一组函数，用来：

- 读取文本文件
- 读取 history / memory / resource 文件夹
- 构建 profile context
- 构建 basic context
- 对 resource 做最小 RAG
- 拼出 final context

注意：
- 模型连接不放在这里，放在 llm.py。
- 工具函数不放在这里，放在 tool.py。
- 主流程调度不放在这里，放在 agent_core.py。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.llm import get_embedding


# =========================
# 默认路径配置
# =========================

DATA_DIR = Path("data")

CV_PATH = DATA_DIR / "cv.txt"
WORK_LOG_PATH = DATA_DIR / "work_log.txt"

CONTEXT_DIR = DATA_DIR / "context"
HISTORY_DIR = CONTEXT_DIR / "history"
MEMORY_DIR = CONTEXT_DIR / "memory"
RESOURCE_DIR = CONTEXT_DIR / "resource"


# =========================
# 基础文件读取函数
# =========================

def load_text_file(file_path: str | Path) -> str:
    """
    读取一个文本文件。

    参数：
    - file_path：文件路径

    返回：
    - 文件内容字符串
    """

    path = Path(file_path)

    if not path.exists():
        return ""

    return path.read_text(encoding="utf-8")


def load_json_file(file_path: str | Path) -> Any:
    """
    读取一个 JSON 文件。

    参数：
    - file_path：文件路径

    返回：
    - JSON 解析后的 Python 对象
    """

    path = Path(file_path)

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_context_folder(folder_path: str | Path) -> List[Dict[str, str]]:
    """
    读取某个上下文文件夹里的所有文本类文件。

    支持：
    - .txt
    - .md
    - .json
    - .jsonl

    参数：
    - folder_path：文件夹路径

    返回：
    - 一个列表，每个元素包含 source 和 content
    """

    folder = Path(folder_path)

    if not folder.exists():
        return []

    results = []

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix in [".txt", ".md"]:
            content = load_text_file(file_path)

        elif suffix == ".json":
            data = load_json_file(file_path)
            content = json.dumps(data, ensure_ascii=False, indent=2)

        elif suffix == ".jsonl":
            lines = []
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)
            content = "\n".join(lines)

        else:
            continue

        if content.strip():
            results.append(
                {
                    "source": file_path.name,
                    "content": content.strip()
                }
            )

    return results


# =========================
# Profile Context
# =========================

def build_profile_context(
    cv_path: str | Path = CV_PATH,
    work_log_path: str | Path = WORK_LOG_PATH
) -> str:
    """
    构建身份资料 context。

    这个函数主要服务 profile 类问题。
    比如：
    - 你是谁？
    - 你做过什么？
    - 帮我写一段职业介绍。

    默认读取：
    - data/cv.txt
    - data/work_log.txt
    """

    cv_text = load_text_file(cv_path)
    work_log_text = load_text_file(work_log_path)

    profile_context = f"""
    【个人简历资料 cv.txt】
    {cv_text if cv_text.strip() else "未找到 cv.txt 或文件内容为空。"}

    【工作记录 work_log.txt】
    {work_log_text if work_log_text.strip() else "未找到 work_log.txt 或文件内容为空。"}
    """

    return profile_context.strip()


# =========================
# History / Memory
# =========================

def load_history(history_dir: str | Path = HISTORY_DIR) -> List[Dict[str, str]]:
    """
    读取 history 文件夹。

    history 表示当前对话或近期任务中已经发生过的信息。
    """

    return load_context_folder(history_dir)


def load_memory(memory_dir: str | Path = MEMORY_DIR) -> List[Dict[str, str]]:
    """
    读取 memory 文件夹。

    memory 表示长期稳定的信息。
    比如：
    - 用户角色
    - 表达偏好
    - 工作边界
    - 常用格式
    """

    return load_context_folder(memory_dir)


def format_context_items(items: List[Dict[str, str]], title: str) -> str:
    """
    把一组 context 文件格式化成 prompt 里容易阅读的文本。
    """

    if not items:
        return f"【{title}】\n暂无相关内容。"

    blocks = [f"【{title}】"]

    for item in items:
        blocks.append(
            f"""
    来源：{item["source"]}
    内容：
    {item["content"]}
    """.strip()
        )

    return "\n\n".join(blocks)


def build_basic_context(
    current_question: str,
    include_profile: bool = True,
    history_dir: str | Path = HISTORY_DIR,
    memory_dir: str | Path = MEMORY_DIR
) -> str:
    """
    构建基础 context。

    这里先不做 RAG。
    只拼：
    - 当前问题
    - profile
    - history
    - memory
    """

    profile_context = build_profile_context() if include_profile else ""

    history_items = load_history(history_dir)
    memory_items = load_memory(memory_dir)

    history_context = format_context_items(history_items, "History：最近对话和任务记录")
    memory_context = format_context_items(memory_items, "Memory：长期记忆和工作偏好")

    basic_context = f"""
    【当前问题】
    {current_question}

    {profile_context}

    {history_context}

    {memory_context}
    """

    return basic_context.strip()


# =========================
# Resource / RAG
# =========================

def load_resource_files(resource_dir: str | Path = RESOURCE_DIR) -> List[Dict[str, str]]:
    """
    读取 resource 文件夹。

    resource 表示外部资料。
    比如：
    - 模板
    - FAQ
    - 项目说明
    - 指标解释
    - 沟通规范
    - 保密规则
    """

    return load_context_folder(resource_dir)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """
    把长文本切成小段。

    参数：
    - text：原始文本
    - chunk_size：每段大约多少字符
    - overlap：相邻片段之间保留多少重叠内容

    返回：
    - 文本片段列表
    """

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0

        if start >= len(text):
            break

    return chunks


def build_chunks_from_resources(
    resource_items: List[Dict[str, str]],
    chunk_size: int = 500,
    overlap: int = 80
) -> List[Dict[str, str]]:
    """
    把 resource 文件切成 chunks。

    返回：
    - 每个 chunk 包含 source、chunk_id、content
    """

    all_chunks = []

    for item in resource_items:
        source = item["source"]
        content = item["content"]

        chunks = chunk_text(
            text=content,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for index, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "source": source,
                    "chunk_id": str(index),
                    "content": chunk
                }
            )

    return all_chunks


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算两个向量的余弦相似度。
    """

    a = np.array(vec1)
    b = np.array(vec2)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def build_resource_index(
    resource_dir: str | Path = RESOURCE_DIR,
    chunk_size: int = 500,
    overlap: int = 80
) -> List[Dict[str, Any]]:
    """
    给 resource 文件夹建立一个最小向量索引。

    这个函数会：
    1. 读取 resource 文件
    2. 切成 chunks
    3. 对每个 chunk 做 embedding
    4. 返回带 embedding 的索引列表

    注意：
    这是教学版最小 RAG。
    数据量很小时可以每次临时建立索引。
    真实项目里通常会把索引保存到数据库或向量库里。
    """

    resource_items = load_resource_files(resource_dir)
    chunks = build_chunks_from_resources(
        resource_items=resource_items,
        chunk_size=chunk_size,
        overlap=overlap
    )

    index = []

    for chunk in chunks:
        embedding = get_embedding(chunk["content"])

        index.append(
            {
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "embedding": embedding
            }
        )

    return index


def retrieve_relevant_chunks(
    question: str,
    resource_index: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    根据用户问题，从 resource index 里检索最相关的 chunks。

    参数：
    - question：用户问题
    - resource_index：build_resource_index() 生成的索引
    - top_k：返回最相关的几个片段

    返回：
    - 相关片段列表
    """

    if not resource_index:
        return []

    question_embedding = get_embedding(question)

    scored_chunks = []

    for item in resource_index:
        score = cosine_similarity(
            question_embedding,
            item["embedding"]
        )

        scored_chunks.append(
            {
                "source": item["source"],
                "chunk_id": item["chunk_id"],
                "content": item["content"],
                "score": score
            }
        )

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    return scored_chunks[:top_k]


def format_retrieved_chunks(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    把检索到的 resource chunks 格式化成 prompt 文本。
    """

    if not retrieved_chunks:
        return "【Retrieved Resource：检索到的相关资料】\n没有检索到相关资料。"

    blocks = ["【Retrieved Resource：检索到的相关资料】"]

    for item in retrieved_chunks:
        blocks.append(
            f"""
    来源：{item["source"]}
    片段编号：{item["chunk_id"]}
    相似度：{item["score"]:.4f}
    内容：
    {item["content"]}
    """.strip()
        )

    return "\n\n".join(blocks)


def build_final_context(
    current_question: str,
    resource_index: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 3,
    include_profile: bool = True,
    history_dir: str | Path = HISTORY_DIR,
    memory_dir: str | Path = MEMORY_DIR
) -> str:
    """
    构建最终 context。

    最终 context 包括：
    - 当前问题
    - profile
    - history
    - memory
    - retrieved resource

    这就是 Day4 的完整闭环：
    current question + history + memory + retrieved resource
    """

    basic_context = build_basic_context(
        current_question=current_question,
        include_profile=include_profile,
        history_dir=history_dir,
        memory_dir=memory_dir
    )

    if resource_index is None:
        resource_index = build_resource_index()

    retrieved_chunks = retrieve_relevant_chunks(
        question=current_question,
        resource_index=resource_index,
        top_k=top_k
    )

    retrieved_context = format_retrieved_chunks(retrieved_chunks)

    final_context = f"""
    {basic_context}

    {retrieved_context}
    """

    return final_context.strip()


# =========================
# History 写入函数
# =========================

def append_to_history(
    user_input: str,
    assistant_output: str,
    history_file: str | Path = HISTORY_DIR / "history.jsonl"
) -> None:
    """
    把一轮对话追加到 history.jsonl。

    参数：
    - user_input：用户输入
    - assistant_output：AI 回复
    - history_file：保存位置
    """

    history_path = Path(history_file)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "user": user_input,
        "assistant": assistant_output
    }

    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_to_history_with_time(
    user_input: str,
    assistant_output: str,
    history_file: str | Path = HISTORY_DIR / "history.jsonl"
) -> None:
    """
    把一轮对话追加到 history.jsonl，并记录时间。
    """

    from datetime import datetime

    history_path = Path(history_file)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "user": user_input,
        "assistant": assistant_output
    }

    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================
# 简单测试
# =========================

if __name__ == "__main__":
    test_question = "请根据我的资料，介绍一下我是谁。"

    print("正在测试 profile context：")
    print(build_profile_context())

    print("\n正在测试 basic context：")
    print(build_basic_context(test_question))

    print("\n正在测试 final context：")
    print(build_final_context(test_question))