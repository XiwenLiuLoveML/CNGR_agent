"""
context.py

这个文件用于复用 Day4 的 context 能力。

它主要做四件事：

1. 读取 history
2. 读取 memory
3. 读取 resource，并做最小 RAG 检索
4. 组装 basic context 和 final context

注意：
- 本文件不负责自动更新 history / memory / resource。
- Day4 先使用提前准备好的 context 文件。
- 后续 Lab5、Lab6 可以在此基础上继续扩展自动写入、guardrail 和 Web App。
"""

import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


# =========================
# 1. 读取文件
# =========================

def load_text_file(file_path: str | Path) -> str:
    """
    读取一个 txt 文件，并返回文本内容。

    参数：
    - file_path：文件路径

    返回：
    - 文件里的文本
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件：{file_path}")

    return file_path.read_text(encoding="utf-8")


def load_json_file(file_path: str | Path) -> Dict[str, Any]:
    """
    读取一个 JSON 文件，并返回 Python 字典。

    参数：
    - file_path：JSON 文件路径

    返回：
    - Python 字典
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件：{file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_resource_files(resource_files: List[str | Path]) -> List[Dict[str, str]]:
    """
    读取多个 resource 文件。

    参数：
    - resource_files：resource 文件路径列表

    返回：
    [
        {
            "source": 文件名,
            "content": 文件内容
        }
    ]
    """
    resources = []

    for file_path in resource_files:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"找不到文件：{file_path}")

        resources.append({
            "source": file_path.name,
            "content": file_path.read_text(encoding="utf-8")
        })

    return resources


# =========================
# 2. Embedding
# =========================

def get_embedding(text: str, client, embedding_model_name: str) -> list:
    """
    可选函数：把文字转换成 embedding。

    需要从 Notebook 传入：
    - client：OpenAI 兼容客户端
    - embedding_model_name：embedding 模型名

    示例：
    get_embedding(
        text="请把刚才的数据分析结果整理成项目群消息。",
        client=client,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )
    """
    response = client.embeddings.create(
        model=embedding_model_name,
        input=text
    )

    return response.data[0].embedding


# =========================
# 3. 格式化 history / memory
# =========================

def format_memory(memory: Dict[str, Any]) -> str:
    """
    把 memory 字典整理成适合放进 prompt 的文本。

    参数：
    - memory：用户长期信息、偏好、规则和边界

    返回：
    - 格式化后的文本
    """
    lines = []

    for key, value in memory.items():
        if isinstance(value, list):
            value_text = "；".join(str(item) for item in value)
        elif isinstance(value, dict):
            value_text = "；".join(f"{k}: {v}" for k, v in value.items())
        else:
            value_text = str(value)

        lines.append(f"- {key}: {value_text}")

    return "\n".join(lines)


def build_basic_context(
    current_question: str,
    history_text: str,
    memory: Dict[str, Any]
) -> str:
    """
    构建 basic context。

    basic context 包含：
    - 当前问题
    - history
    - memory

    还不包含 retrieved resource。
    """
    memory_text = format_memory(memory)

    basic_context = f"""
你是一个职业数字人助手。

请你根据下面的 context 回答用户问题。

【当前问题】
{current_question}

【History：当前对话前面发生过什么】
{history_text}

【Memory：长期需要记住的信息】
{memory_text}

请注意：
1. 回答要基于上面的 context。
2. 如果涉及群消息，请注意用户角色、表达风格和信息边界。
3. 不要展开敏感细节，不要编造 context 里没有的信息。
"""

    return basic_context.strip()


# =========================
# 4. Resource 处理：chunk
# =========================

def chunk_text(text: str, max_chars: int = 300) -> List[str]:
    """
    把一段长文本切成多个 chunk。

    参数：
    - text：原始文本
    - max_chars：每个 chunk 最大字符数

    返回：
    - chunk 文本列表
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chars:
            current_chunk += paragraph + "\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def build_chunks_from_resources(
    resources: List[Dict[str, str]],
    max_chars: int = 300
) -> List[Dict[str, Any]]:
    """
    把多个 resource 文件切成 chunks，并保留来源信息。

    参数：
    - resources：load_resource_files() 读取出来的资源列表
    - max_chars：每个 chunk 最大字符数

    返回：
    [
        {
            "source": 文件名,
            "chunk_id": chunk 编号,
            "text": chunk 文本
        }
    ]
    """
    all_chunks = []

    for item in resources:
        source = item["source"]
        content = item["content"]

        chunks = chunk_text(content, max_chars=max_chars)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": source,
                "chunk_id": i + 1,
                "text": chunk
            })

    return all_chunks


# =========================
# 5. Resource 处理：embedding & retrieval
# =========================

def build_resource_index(
    chunks: List[Dict[str, Any]],
    get_embedding_func
) -> List[Dict[str, Any]]:
    """
    给每个 chunk 生成 embedding，建立最小 resource index。

    参数：
    - chunks：切好的 resource chunks
    - get_embedding_func：外部传入的 embedding 函数

    注意：
    如果你想使用本文件里的 get_embedding()，可以先在 Notebook 里这样包装：

    embedding_func = lambda text: get_embedding(
        text=text,
        client=client,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )

    然后传入：
    build_resource_index(chunks, get_embedding_func=embedding_func)

    返回：
    [
        {
            "source": 文件名,
            "chunk_id": chunk 编号,
            "text": chunk 文本,
            "embedding": 向量
        }
    ]
    """
    index = []

    for item in chunks:
        embedding = get_embedding_func(item["text"])

        index.append({
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "text": item["text"],
            "embedding": embedding
        })

    return index


def cosine_similarity(vec1, vec2) -> float:
    """
    计算两个向量的余弦相似度。

    返回值越大，说明两个向量越相似。
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if denominator == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / denominator)


def retrieve_relevant_chunks(
    question: str,
    resource_index: List[Dict[str, Any]],
    get_embedding_func,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    根据用户问题，从 resource index 中检索最相关的 chunks。

    参数：
    - question：用户当前问题
    - resource_index：build_resource_index() 生成的索引
    - get_embedding_func：外部传入的 embedding 函数
    - top_k：取最相关的前几段

    返回：
    [
        {
            "source": 文件名,
            "chunk_id": chunk 编号,
            "text": chunk 文本,
            "score": 相似度分数
        }
    ]
    """
    question_embedding = get_embedding_func(question)

    scored_chunks = []

    for item in resource_index:
        score = cosine_similarity(question_embedding, item["embedding"])

        scored_chunks.append({
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "text": item["text"],
            "score": score
        })

    scored_chunks = sorted(
        scored_chunks,
        key=lambda x: x["score"],
        reverse=True
    )

    return scored_chunks[:top_k]


# =========================
# 6. 组装 final context
# =========================

def format_retrieved_chunks(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    把检索到的 chunks 整理成适合放进 prompt 的文本。
    """
    lines = []

    for item in retrieved_chunks:
        lines.append(
            f"来源：{item['source']}，chunk {item['chunk_id']}\n{item['text']}"
        )

    return "\n\n".join(lines)


def build_final_context(
    current_question: str,
    history_text: str,
    memory: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]]
) -> str:
    """
    构建 final context。

    final context 包含：
    - 当前问题
    - history
    - memory
    - retrieved resource
    """
    memory_text = format_memory(memory)
    retrieved_resource_text = format_retrieved_chunks(retrieved_chunks)

    final_context = f"""
    你是一个职业数字人助手。

    请你根据下面的 context 回答用户问题。

    【当前问题】
    {current_question}

    【History：当前对话前面发生过什么】
    {history_text}

    【Memory：长期需要记住的信息】
    {memory_text}

    【Retrieved Resource：这次任务检索到的相关资料】
    {retrieved_resource_text}

    请注意：
    1. 回答必须基于上面的 context。
    2. 如果要生成群消息，请遵守 retrieved resource 里的格式和沟通规则。
    3. 请结合 memory 中的用户角色、表达风格和信息边界。
    4. 不要展开客户名称、内部成本、利润或异常明细。
    5. 不要编造 context 里没有的信息。
    """

    return final_context.strip()


# =========================
# 7. 一站式封装：从文件到 final context
# =========================

def build_context_from_files(
    current_question: str,
    history_file: str | Path,
    memory_file: str | Path,
    resource_files: List[str | Path],
    get_embedding_func,
    top_k: int = 3,
    max_chars: int = 300
) -> Dict[str, Any]:
    """
    从文件直接构建 final context。

    这个函数适合在 Lab5、Lab6 里快速复用。

    参数：
    - current_question：用户当前问题
    - history_file：history 文件路径
    - memory_file：memory 文件路径
    - resource_files：resource 文件路径列表
    - get_embedding_func：外部传入的 embedding 函数
    - top_k：检索最相关的前几段资料
    - max_chars：每个 chunk 最大字符数

    返回：
    {
        "history_text": ...,
        "memory": ...,
        "resources": ...,
        "chunks": ...,
        "resource_index": ...,
        "retrieved_chunks": ...,
        "basic_context": ...,
        "final_context": ...
    }
    """
    history_text = load_text_file(history_file)
    memory = load_json_file(memory_file)

    basic_context = build_basic_context(
        current_question=current_question,
        history_text=history_text,
        memory=memory
    )

    resources = load_resource_files(resource_files)
    chunks = build_chunks_from_resources(resources, max_chars=max_chars)

    resource_index = build_resource_index(
        chunks=chunks,
        get_embedding_func=get_embedding_func
    )

    retrieved_chunks = retrieve_relevant_chunks(
        question=current_question,
        resource_index=resource_index,
        get_embedding_func=get_embedding_func,
        top_k=top_k
    )

    final_context = build_final_context(
        current_question=current_question,
        history_text=history_text,
        memory=memory,
        retrieved_chunks=retrieved_chunks
    )

    return {
        "history_text": history_text,
        "memory": memory,
        "resources": resources,
        "chunks": chunks,
        "resource_index": resource_index,
        "retrieved_chunks": retrieved_chunks,
        "basic_context": basic_context,
        "final_context": final_context
    }


# =========================
# 8. 可选扩展：追加 history
# =========================

def append_to_history(
    history_file: str | Path,
    user_question: str,
    ai_answer: str
) -> None:
    """
    可选扩展函数：把一轮对话追加到 history 文件里。

    注意：
    - 这个函数先作为进阶练习保留。
    - 真实系统里还需要时间、用户 ID、会话 ID 等信息。
    """
    history_file = Path(history_file)

    if not history_file.exists():
        raise FileNotFoundError(f"找不到文件：{history_file}")

    new_record = f"""

    【新一轮对话】
    用户问题：
    {user_question}

    AI 回答：
    {ai_answer}
    """

        with history_file.open("a", encoding="utf-8") as f:
            f.write(new_record)


# =========================
# 9. 可选扩展：快速追加带时间的 history
# =========================

def append_to_history_with_time(
    history_file: str | Path,
    user_question: str,
    ai_answer: str,
    timestamp: str
) -> None:
    """
    可选扩展函数：把一轮对话和时间一起追加到 history 文件里。

    参数：
    - history_file：history 文件路径
    - user_question：用户问题
    - ai_answer：AI 回答
    - timestamp：时间字符串，例如 "2026-04-30 15:30"
    """
    history_file = Path(history_file)

    if not history_file.exists():
        raise FileNotFoundError(f"找不到文件：{history_file}")

    new_record = f"""

    【新一轮对话】
    时间：
    {timestamp}

    用户问题：
    {user_question}

    AI 回答：
    {ai_answer}
    """

        with history_file.open("a", encoding="utf-8") as f:
            f.write(new_record)
