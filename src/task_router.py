"""
task_router.py

这个文件用于复用职业数字人的任务判断能力。

它做一件事：

用户输入一句话，
系统先判断这是哪类任务，
再决定下一步应该走哪条路。

这个文件会使用 llm.py 里已经封装好的 call_llm()。
"""

from src.llm import call_llm


TASK_TYPES = [
    "profile",
    "writing",
    "data_analysis",
    "web_search",
    "send_message",
    "work_planning",
    "clarify"
]


def classify_task(user_input: str) -> str:
    """
    判断用户输入属于哪类任务。

    参数：
    - user_input：用户输入的一句话

    返回：
    - 一个任务类型字符串
    """

    system_prompt = """
    你是一个任务分类助手。

    请判断用户输入属于哪一类任务。

    只能从下面几类中选择一个：

    - profile：用户想了解职业数字人是谁、能做什么、个人介绍、项目经历等
    - writing：用户想写作、改写、润色、生成文案
    - data_analysis：用户想分析表格、CSV、Excel、数据
    - web_search：用户想查网页、查公司、查新闻、查公开信息
    - send_message：用户想把内容发送到群里、推送消息、通知别人
    - work_planning：用户想做计划、安排任务、总结工作重点
    - clarify：用户的问题不清楚，需要先追问

    只返回任务类型本身。
    不要解释。
    不要输出多余内容。
    """

    user_prompt = f"""
    用户输入：

    {user_input}

    请返回最合适的任务类型。
    """

    result = call_llm(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0
    )

    result = result.strip().lower()

    if result not in TASK_TYPES:
        return "clarify"

    return result