"""
agent_core.py

这个文件是职业数字人的“总控流程”。

它不负责具体干活。
具体干活交给 tool.py 里的工具。

它负责：
1. 接收用户输入
2. 调用 task_router.py 判断任务类型
3. 根据任务类型调用对应工具
4. 返回最终结果

现在这是 Day 3 版本。
Day 4 会继续升级：加入资料、history 和 context。
"""

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


clarify_count = 0


def handle_user_task(
    user_input,
    call_llm,
    file_path=None,
    url=None,
    question=None,
    message=None,
    confirm=False
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

    返回：
    - 对应工具的处理结果
    """

    global clarify_count

    task_type = classify_task(user_input, call_llm)

    if task_type == "profile":
        return profile_tool(user_input, call_llm)

    elif task_type == "writing":
        return writing_tool(user_input)

    elif task_type == "work_planning":
        return work_planning_tool(user_input)

    elif task_type == "data_analysis":
        if not file_path:
            return "请先提供 CSV 文件路径，例如：data/sales_data.csv"
        return full_data_analysis_tool(file_path, call_llm)

    elif task_type == "web_search":
        if not url:
            return "请先提供网页 URL。"

        if not question:
            question = user_input

        return web_qa_tool(url, question, call_llm)

    elif task_type == "send_message":
        if not message:
            message = user_input

        return send_message_tool(message, confirm=confirm)

    else:
        clarify_count += 1
        return clarify_tool(user_input, clarify_count)


def reset_clarify_count():
    """
    重置澄清次数。
    适合开始一轮新任务时使用。
    """
    global clarify_count
    clarify_count = 0
    return "澄清次数已重置。"
