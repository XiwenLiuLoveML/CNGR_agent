"""
task_router.py

这个文件用于复用职业数字人的任务判断能力。

它做一件事：

用户输入一句话，
系统先判断这是哪类任务，
再决定下一步应该走哪条路。

这个文件不需要额外工具包。
它会使用 Notebook 里已经封装好的 call_llm() 函数。
"""


TASK_TYPES = [
    "profile",
    "writing",
    "data_analysis",
    "web_search",
    "send_message",
    "work_planning",
    "clarify"
]


def classify_task(user_input, call_llm):
    """
    判断用户输入属于哪类任务。
    """
    system_prompt = """
    你是一个任务分类助手。

    请判断用户输入属于哪一类任务。

    只能从下面几类中选择一个：

    profile
    用户在询问职业数字人是谁、能做什么、个人介绍或职业介绍。

    writing
    用户需要写作、润色、改写、总结文字。

    data_analysis
    用户需要分析表格、CSV、Excel、数据、画图、找异常。

    web_search
    用户需要读取网页、查询公开网站、查外部信息。

    send_message
    用户需要把内容发送到群里、推送消息、通知别人。

    work_planning
    用户需要安排工作、拆任务、做计划、排优先级。

    clarify
    用户的问题不够清楚，需要先追问。

    请只返回一个类别名称。
    不要解释。
    不要加标点。
    """

    user_question = f"""
    用户输入：

    {user_input}

    请判断任务类型。
    """

    result = call_llm(user_question, system_prompt)

    task_type = result.strip().lower()
    task_type = task_type.replace("`", "")
    task_type = task_type.replace(".", "")
    task_type = task_type.replace("。", "")

    if task_type not in TASK_TYPES:
        task_type = "clarify"

    return task_type


def route_task(task_type):
    """
    根据任务类型，返回下一步应该做什么。
    """
    if task_type == "profile":
        return "调用 profile_tool。"

    elif task_type == "writing":
        return "调用 writing_tool。"

    elif task_type == "data_analysis":
        return "调用 full_data_analysis_tool。"

    elif task_type == "web_search":
        return "调用 web_qa_tool。"

    elif task_type == "send_message":
        return "调用 send_message_tool。"

    elif task_type == "work_planning":
        return "调用 work_planning_tool。"

    else:
        return "调用 clarify_tool。"


def explain_route(user_input, call_llm):
    """
    一步完成：
    1. 判断任务类型
    2. 给出推荐路径
    """
    task_type = classify_task(user_input, call_llm)
    route = route_task(task_type)

    return {
        "user_input": user_input,
        "task_type": task_type,
        "route": route
    }