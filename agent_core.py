"""
agent_core.py

这个文件是职业数字人的“总控流程”。

它不负责具体干活。
具体干活交给 tool.py 里的工具。

它负责：
1. 接收用户输入
2. 调用 task_router.py 判断任务类型
3. 根据任务类型调用对应工具
4. 在需要时读取 context
5. 返回最终结果

这是 Day 4 升级版本：
在 Day 3 的工具调用基础上，加入 history、memory、resource 和 context。

核心变化：
- Day 3：职业数字人会判断任务类型，并调用工具。
- Day 4：职业数字人可以带着 history、memory、retrieved resource 来回答。
"""

from pathlib import Path
from typing import Callable, List, Optional, Dict, Any

from task_router import classify_task
from tool import (
    profile_tool,
    writing_tool,
    work_planning_tool,
    clarify_tool,
    full_data_analysis_tool,
    web_qa_tool,
    send_message_tool
)

from context import build_context_from_files


clarify_count = 0


# =========================
# 1. Context 构建
# =========================

def build_context_if_available(
    user_input: str,
    history_file: Optional[str | Path] = None,
    memory_file: Optional[str | Path] = None,
    resource_files: Optional[List[str | Path]] = None,
    get_embedding_func: Optional[Callable[[str], list]] = None,
    top_k: int = 3,
    max_chars: int = 300
) -> Optional[Dict[str, Any]]:
    """
    如果 context 文件和 embedding 函数都准备好了，
    就构建 final context。

    如果没有准备好，就返回 None。

    这样做的好处是：
    - Day3 的旧逻辑还能继续跑
    - Day4 以后可以逐步接入 context

    参数：
    - user_input：用户当前输入
    - history_file：history 文件路径
    - memory_file：memory 文件路径
    - resource_files：resource 文件路径列表
    - get_embedding_func：embedding 函数
    - top_k：从 resource 里检索最相关的前几段
    - max_chars：每个 chunk 的最大字符数

    返回：
    - context_result 字典，包含 final_context / basic_context / retrieved_chunks 等
    - 如果信息不完整，返回 None
    """

    if not history_file:
        return None

    if not memory_file:
        return None

    if not resource_files:
        return None

    if not get_embedding_func:
        return None

    context_result = build_context_from_files(
        current_question=user_input,
        history_file=history_file,
        memory_file=memory_file,
        resource_files=resource_files,
        get_embedding_func=get_embedding_func,
        top_k=top_k,
        max_chars=max_chars
    )

    return context_result


def answer_with_context(
    user_input: str,
    call_llm: Callable,
    final_context: str
) -> str:
    """
    使用 final context 调用大模型回答问题。

    参数：
    - user_input：用户当前问题
    - call_llm：Notebook 里封装好的 LLM 调用函数
    - final_context：由 context.py 构建出来的最终上下文

    返回：
    - 模型回答
    """

    return call_llm(
        user_question=user_input,
        system_prompt=final_context
    )


# =========================
# 2. 各类任务处理
# =========================

def handle_profile_task(
    user_input: str,
    call_llm: Callable,
    context_result: Optional[Dict[str, Any]] = None
) -> str:
    """
    处理 profile 类任务。

    如果有 context，就优先让模型基于 context 回答。
    如果没有 context，就使用 Day3 的 profile_tool。
    """
    if context_result:
        return answer_with_context(
            user_input=user_input,
            call_llm=call_llm,
            final_context=context_result["final_context"]
        )

    return profile_tool(user_input, call_llm)


def handle_writing_task(
    user_input: str,
    call_llm: Callable,
    context_result: Optional[Dict[str, Any]] = None
) -> str:
    """
    处理 writing 类任务。

    如果有 context，就优先让模型基于 context 写作。
    如果没有 context，就使用 Day3 的 writing_tool。
    """
    if context_result:
        return answer_with_context(
            user_input=user_input,
            call_llm=call_llm,
            final_context=context_result["final_context"]
        )

    return writing_tool(user_input)


def handle_work_planning_task(
    user_input: str,
    call_llm: Callable,
    context_result: Optional[Dict[str, Any]] = None
) -> str:
    """
    处理 work_planning 类任务。

    如果有 context，就优先让模型基于 context 做工作规划。
    如果没有 context，就使用 Day3 的 work_planning_tool。
    """
    if context_result:
        return answer_with_context(
            user_input=user_input,
            call_llm=call_llm,
            final_context=context_result["final_context"]
        )

    return work_planning_tool(user_input)


def handle_data_analysis_task(
    file_path: Optional[str],
    call_llm: Callable
) -> str:
    """
    处理 data_analysis 类任务。
    """
    if not file_path:
        return "请先提供 CSV 文件路径，例如：data/sales_data.csv"

    return full_data_analysis_tool(file_path, call_llm)


def handle_web_search_task(
    user_input: str,
    call_llm: Callable,
    url: Optional[str] = None,
    question: Optional[str] = None
) -> str:
    """
    处理 web_search 类任务。
    """
    if not url:
        return "请先提供网页 URL。"

    if not question:
        question = user_input

    return web_qa_tool(url, question, call_llm)


def handle_send_message_task(
    user_input: str,
    call_llm: Callable,
    message: Optional[str] = None,
    confirm: bool = False,
    context_result: Optional[Dict[str, Any]] = None
) -> str:
    """
    处理 send_message 类任务。

    Day4 之后，发消息前可以先用 context 生成一版更合适的消息：
    - 接住 history：刚才分析结果是什么
    - 参考 memory：用户角色、表达边界
    - 参考 resource：消息格式、沟通规范

    如果传入了 message，就直接发送传入的 message。
    如果没有 message，但有 context，就先生成消息文本，再交给 send_message_tool。
    如果都没有，就使用 user_input 作为消息。
    """
    final_message = message

    if not final_message and context_result:
        final_message = answer_with_context(
            user_input=user_input,
            call_llm=call_llm,
            final_context=context_result["final_context"]
        )

    if not final_message:
        final_message = user_input

    return send_message_tool(final_message, confirm=confirm)


def handle_clarify_task(user_input: str) -> str:
    """
    处理不明确任务。
    """
    global clarify_count

    clarify_count += 1
    return clarify_tool(user_input, clarify_count)


# =========================
# 3. 总控函数
# =========================

def handle_user_task(
    user_input: str,
    call_llm: Callable,
    file_path: Optional[str] = None,
    url: Optional[str] = None,
    question: Optional[str] = None,
    message: Optional[str] = None,
    confirm: bool = False,
    history_file: Optional[str | Path] = None,
    memory_file: Optional[str | Path] = None,
    resource_files: Optional[List[str | Path]] = None,
    get_embedding_func: Optional[Callable[[str], list]] = None,
    use_context: bool = False,
    top_k: int = 3,
    max_chars: int = 300,
    return_debug: bool = False
):
    """
    职业数字人的总控函数。

    参数说明：
    - user_input：用户原始输入
    - call_llm：Notebook 里已经封装好的大模型调用函数
    - file_path：数据分析时使用的文件路径
    - url：网页问答时使用的网址
    - question：网页问答时带着去网页里寻找答案的问题
    - message：需要发送到群里的消息
    - confirm：是否确认发送消息

    Day4 新增：
    - history_file：history 文件路径
    - memory_file：memory 文件路径
    - resource_files：resource 文件路径列表
    - get_embedding_func：Notebook 里封装好的 embedding 函数
    - use_context：是否启用 context
    - top_k：从 resource 里检索最相关的前几段
    - max_chars：每个 chunk 的最大字符数
    - return_debug：是否返回调试信息，包括 task_type 和 context_result

    返回：
    - 默认返回对应工具或模型的处理结果
    - 如果 return_debug=True，返回字典：
      {
          "task_type": ...,
          "context_result": ...,
          "result": ...
      }
    """

    task_type = classify_task(user_input, call_llm)

    context_result = None

    if use_context:
        context_result = build_context_if_available(
            user_input=user_input,
            history_file=history_file,
            memory_file=memory_file,
            resource_files=resource_files,
            get_embedding_func=get_embedding_func,
            top_k=top_k,
            max_chars=max_chars
        )

    # 1. profile 类任务
    if task_type == "profile":
        result = handle_profile_task(
            user_input=user_input,
            call_llm=call_llm,
            context_result=context_result
        )

    # 2. writing 类任务
    elif task_type == "writing":
        result = handle_writing_task(
            user_input=user_input,
            call_llm=call_llm,
            context_result=context_result
        )

    # 3. work_planning 类任务
    elif task_type == "work_planning":
        result = handle_work_planning_task(
            user_input=user_input,
            call_llm=call_llm,
            context_result=context_result
        )

    # 4. data_analysis 类任务
    elif task_type == "data_analysis":
        result = handle_data_analysis_task(
            file_path=file_path,
            call_llm=call_llm
        )

    # 5. web_search 类任务
    elif task_type == "web_search":
        result = handle_web_search_task(
            user_input=user_input,
            call_llm=call_llm,
            url=url,
            question=question
        )

    # 6. send_message 类任务
    elif task_type == "send_message":
        result = handle_send_message_task(
            user_input=user_input,
            call_llm=call_llm,
            message=message,
            confirm=confirm,
            context_result=context_result
        )

    # 7. 不明确任务
    else:
        result = handle_clarify_task(user_input)

    if return_debug:
        return {
            "task_type": task_type,
            "context_result": context_result,
            "result": result
        }

    return result


# =========================
# 4. 工具函数
# =========================

def reset_clarify_count() -> str:
    """
    重置澄清次数。
    适合开始一轮新任务时使用。
    """
    global clarify_count

    clarify_count = 0
    return "澄清次数已重置。"


def build_embedding_func(client, embedding_model_name: str):
    """
    快速构建 embedding_func。

    这个函数是为了让 Notebook 调用更简洁。

    用法：
    from context import get_embedding
    from agent_core import build_embedding_func

    embedding_func = build_embedding_func(
        client=client,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )
    """
    from context import get_embedding

    return lambda text: get_embedding(
        text=text,
        client=client,
        embedding_model_name=embedding_model_name
    )
