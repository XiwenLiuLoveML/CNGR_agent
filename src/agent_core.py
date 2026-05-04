"""
agent_core.py

这个文件用于组装职业数字人的主流程。

它负责把前四天的能力串起来：

Day 1：llm.py
- 调用大模型

Day 2：task_router.py
- 判断用户输入属于哪类任务

Day 3：tool.py
- 表格分析
- 网页读取
- 消息推送

Day 4：context.py
- 读取 cv.txt / work_log.txt
- 读取 history / memory / resource
- 构建 context
- 做最小 RAG

agent_core.py 做的事是：

用户输入
↓
判断任务类型
↓
检查这类任务需要哪些参数
↓
参数够了就执行
参数不够就追问
↓
返回结果

注意：
- guardrail 不放在这里，放在 guardrail.py。
- Gradio 页面不放在这里，放在 Notebook 或 app.py。
"""

import re
from pathlib import Path
from typing import Optional

import src.llm as llm
import src.task_router as task_router
import src.tool as tool
import src.context as context


# 用一个简单变量保存最近一次结果。
# 这样用户说“把刚才结果发到群里”时，系统有东西可以引用。
LAST_RESULT = ""
# 上一轮已经准备好、等待确认发送的消息。
PENDING_MESSAGE = ""

# =========================
# 参数提取辅助函数
# =========================

def extract_url(user_input: str) -> Optional[str]:
    """
    从用户输入中提取 URL。
    如果没有找到 URL，就返回 None。
    """

    pattern = r"https?://[^\s，。；、]+"
    match = re.search(pattern, user_input)

    if match:
        return match.group(0)

    return None


def remove_url(text: str, url: str) -> str:
    """
    从文本里移除 URL，剩下的部分通常可以当作 question。
    """

    return text.replace(url, "").strip()


def extract_file_path(user_input: str) -> Optional[str]:
    """
    从用户输入中提取可能的文件路径。

    支持简单识别：
    - .csv
    - .xlsx
    - .xls

    示例：
    请分析 data/sales_data.csv
    """

    pattern = r"[\w\-/\\\.]+(?:\.csv|\.xlsx|\.xls)"
    match = re.search(pattern, user_input)

    if match:
        return match.group(0)

    return None


def is_confirmed(user_input: str) -> bool:
    """
    判断用户是否明确确认发送。
    """

    confirm_words = [
        "确认",
        "确认发送",
        "可以",
        "可以了",
        "可以发送",
        "发吧",
        "发送吧",
        "发出去",
        "直接发送",
        "没问题",
        "好",
        "好的",
        "行",
        "可以的",
        "Ok",
        "ok了"
    ]

    return any(word in user_input for word in confirm_words)

# =========================
# 各类任务处理函数
# =========================

def handle_profile(user_input: str) -> str:
    """
    处理 profile 类问题。

    这类问题需要读取：
    - data/cv.txt
    - data/work_log.txt
    """

    profile_context = context.build_profile_context()

    user_prompt = f"""
    请基于下面的个人资料回答用户问题。

    要求：
    1. 只基于资料回答，不要编造。
    2. 如果资料里没有，就明确说明资料中没有相关信息。
    3. 回答要清楚、自然，适合职业场景。

    【用户问题】
    {user_input}

    【个人资料】
    {profile_context}
    """

    result = llm.call_llm(
        user_prompt=user_prompt,
        system_prompt="你是一个专业的职业数字人助手，擅长基于个人资料进行职业介绍和项目总结。",
        temperature=0.3
    )

    return result


def handle_writing(user_input: str) -> str:
    """
    处理写作类任务。

    写作任务通常需要结合用户身份、工作背景和表达偏好。
    """

    final_context = context.build_final_context(
        current_question=user_input,
        include_profile=True,
        top_k=3
    )

    user_prompt = f"""
    请根据下面的 context，完成用户的写作任务。

    要求：
    1. 符合用户的职业身份和表达风格。
    2. 如果涉及工作群、汇报、客户沟通，语气要专业、清楚。
    3. 不要编造 context 里没有的事实。

    【Context】
    {final_context}

    【写作任务】
    {user_input}
    """

    result = llm.call_llm(
        user_prompt=user_prompt,
        system_prompt="你是一个专业的职场写作助手，擅长把复杂信息整理成清楚、得体的表达。",
        temperature=0.4
    )

    return result


def handle_data_analysis(user_input: str) -> str:
    """
    处理数据分析任务。

    数据分析工具需要 file_path。
    如果用户没有给文件路径，课程里先默认使用 data/sales_data.csv。
    """

    file_path = extract_file_path(user_input)

    if not file_path:
        default_file = Path("data/sales_data.csv")

        if default_file.exists():
            file_path = str(default_file)
        else:
            return """
    我可以帮你分析 CSV 或 Excel 表格。

    请告诉我表格文件路径，例如：

    data/sales_data.csv
    """

    result = tool.full_data_analysis_tool(file_path=file_path)

    return result


def handle_web_search(user_input: str) -> str:
    """
    处理网页读取 / 网页问答任务。

    web_qa_tool 需要两个参数：
    - url
    - question

    如果缺 URL，就追问。
    如果有 URL 但没有具体问题，就追问用户想看什么。
    """

    url = extract_url(user_input)

    if not url:
        return """
    我可以帮你读取网页并回答问题。

    请把网页链接发给我。
    也可以一起告诉我你最关心什么问题。

    例如：
    请帮我看这个网页里有没有联系方式：https://example.com
    """

    question = remove_url(user_input, url)

    if not question:
        return """
    我已经收到网页链接了。

    你希望我重点看什么？

    比如：
    - 这个网页主要讲什么？
    - 有没有联系方式？
    - 这家公司是做什么的？
    - 有没有产品、地址或公开邮箱？
    """

    result = tool.web_qa_tool(
        url=url,
        question=question
    )

    return result


def handle_send_message(user_input: str) -> str:
    """
    处理消息推送任务。

    第一次请求发送时：
    - 如果没有明确确认，只生成预览，并把消息存入 PENDING_MESSAGE。

    第二次用户确认时：
    - 如果识别到确认词，就把 PENDING_MESSAGE 真正发送出去。
    """

    global LAST_RESULT
    global PENDING_MESSAGE

    # 如果用户是在确认上一条待发送消息
    if PENDING_MESSAGE.strip() and is_confirmed(user_input):
        result = tool.send_message_tool(
            message=PENDING_MESSAGE,
            confirm=True
        )

        # 发送后清空待确认消息
        PENDING_MESSAGE = ""

        return result

    # 如果还没有可发送内容
    if not LAST_RESULT.strip():
        return """
    目前还没有可发送的上一轮结果。

    你可以先让我完成一个任务，
    比如分析表格、总结网页、生成一段文字，
    然后再说：把刚才结果发到群里。
    """

    # 第一次请求发送：先预览，不直接发
    PENDING_MESSAGE = LAST_RESULT

    result = tool.send_message_tool(
        message=PENDING_MESSAGE,
        confirm=False
    )

    return result


def handle_work_planning(user_input: str) -> str:
    """
    处理工作计划 / 工作总结类任务。

    这类任务通常需要结合：
    - 用户身份
    - work_log
    - memory
    - resource
    """

    final_context = context.build_final_context(
        current_question=user_input,
        include_profile=True,
        top_k=3
    )

    user_prompt = f"""
    请根据下面的 context，帮助用户完成工作计划或工作总结。

    要求：
    1. 输出要清晰、可执行。
    2. 适合真实职场场景。
    3. 不要编造 context 里没有的事实。
    4. 如果信息不足，请指出还需要补充什么。

    【Context】
    {final_context}

    【用户任务】
    {user_input}
    """

    result = llm.call_llm(
        user_prompt=user_prompt,
        system_prompt="你是一个专业的工作规划助手，擅长把任务拆清楚、排优先级、形成可执行计划。",
        temperature=0.3
    )

    return result


def handle_clarify(user_input: str) -> str:
    """
    处理不明确任务。

    当 task_router 判断为 clarify 时，先追问用户。
    """

    return """
    我还不太确定你想让我做哪一类任务。

    你可以补充一下：

    1. 是想让我帮你写东西？
    2. 是想让我分析表格？
    3. 是想让我读取网页？
    4. 是想让我总结你的资料？
    5. 还是想让我把某个结果发出去？
    """


# =========================
# 主入口函数
# =========================

def handle_user_task(user_input: str) -> str:
    """
    职业数字人的主流程入口。
    """

    global LAST_RESULT
    global PENDING_MESSAGE

    if not user_input or not user_input.strip():
        return "请输入一个任务。"

    # 如果当前有待确认消息，并且用户表达了确认，
    # 就直接走消息发送，不再交给 task_router 判断。
    if PENDING_MESSAGE.strip() and is_confirmed(user_input):
        result = handle_send_message(user_input)
        return result

    task_type = task_router.classify_task(user_input)

    if task_type == "profile":
        result = handle_profile(user_input)

    elif task_type == "writing":
        result = handle_writing(user_input)

    elif task_type == "data_analysis":
        result = handle_data_analysis(user_input)

    elif task_type == "web_search":
        result = handle_web_search(user_input)

    elif task_type == "send_message":
        result = handle_send_message(user_input)

    elif task_type == "work_planning":
        result = handle_work_planning(user_input)

    else:
        result = handle_clarify(user_input)

    # 保存最近一次结果。
    # 注意：消息预览 / 发送结果本身不覆盖 LAST_RESULT，避免把“已发送”这种提示当成下一次要发送的内容。
    if task_type != "send_message":
        LAST_RESULT = result

    # 把这一轮写入 history
    try:
        context.append_to_history_with_time(
            user_input=user_input,
            assistant_output=result
        )
    except Exception:
        # history 写入失败不应该影响主流程
        pass

    return result


# =========================
# 简单测试
# =========================

if __name__ == "__main__":
    test_inputs = [
        "你是谁？你做过什么？",
        "帮我写一段周报开头。",
        "请分析 data/sales_data.csv，告诉我三个重点。",
        "请帮我看这个网页主要讲什么：https://www.python.org/",
        "我下周的工作重点应该是什么？",
        "把刚才结果发到群里。"
    ]

    for text in test_inputs:
        print("=" * 60)
        print("用户输入：", text)
        print("AI 回复：")
        print(handle_user_task(text))