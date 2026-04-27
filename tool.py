import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from IPython.display import Markdown, display


load_dotenv()

WEBHOOK_URL = os.getenv("WEBHOOK_URL")

TOOLS = {}


def show_markdown(text):
    """
    用 Markdown 方式展示内容。
    """
    display(Markdown(text))


def tool(func):
    """
    教学版 @tool 装饰器。
    把函数注册到 TOOLS 工具箱里。
    """
    TOOLS[func.__name__] = func
    return func


def read_text_file(file_path):
    """
    读取 txt 文件。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


# =========================
# Tool 1：Profile 工具
# =========================

@tool
def profile_tool(user_input, call_llm):
    """
    基于 data/CV.txt 和 data/work_log.txt 回答职业相关问题。
    """
    cv_text = read_text_file("data/CV.txt")
    work_log_text = read_text_file("data/work_log.txt")

    system_prompt = f"""
    你是一个职业数字人助手。

    请根据下面资料回答用户问题。

    资料一：CV
    {cv_text}

    资料二：Work Log
    {work_log_text}

    要求：
    1. 只基于资料回答
    2. 如果资料里没有，就说“资料中暂时没有相关信息”
    3. 用中文回答
    4. 简洁、友好、专业
    """

    return call_llm(user_input, system_prompt)


# =========================
# Tool 2：Writing 工具
# =========================

@tool
def writing_tool(user_input):
    """
    写作任务先询问需求，不直接起草。
    """
    return """
    我可以帮你写。

    为了写得更准确，请你补充三点：

    1. 这段文字用在什么场景？
    2. 你希望语气正式一点、自然一点，还是更有感染力一点？
    3. 有没有必须包含的信息？
    """


# =========================
# Tool 3：Work Planning 工具
# =========================

@tool
def work_planning_tool(user_input):
    """
    工作规划任务先询问任务清单、时间范围和优先级。
    """
    return """
    我可以帮你做工作规划。

    请你先补充三点：

    1. 你要规划的是哪段时间？比如今天、本周、下周。
    2. 目前手上有哪些任务？
    3. 哪些任务最紧急或最重要？
    """


# =========================
# Tool 4：Clarify 工具
# =========================

@tool
def clarify_tool(user_input, clarify_count=0):
    """
    用户需求不清楚时，最多追问 3 次。
    """
    clarify_count += 1

    if clarify_count <= 3:
        return f"""
    我还需要再确认一下你的需求。

    请你补充一点具体信息：

    你是想让我写东西、分析数据、查网页，还是发送消息？

    这是第 {clarify_count} 次澄清。
    """

    return """
    我先不继续追问了。

    你可以按下面这个格式重新告诉我：

    1. 你想完成什么任务？
    2. 有没有文件、网址或消息内容？
    3. 希望最后输出成什么形式？
    """


# =========================
# Tool 5：数据分析工具
# =========================


def read_csv(file_path):
    """
    读取 CSV 文件，并展示前几行。
    """
    df = pd.read_csv(file_path)

    show_markdown("""
    ### 📊 表格预览
    """)
    display(df.head())

    return df


def choose_chart_with_ai(df, call_llm):
    """
    让 AI 根据数据结构选择图表。
    """
    chart_prompt = """
    你是一个数据可视化助理。
    请根据数据内容，选择最适合的一种图表。

    只能选择：
    line, bar, pie, scatter

    请只返回 JSON，不要解释。

    JSON 格式：
    {
    "chart_type": "line",
    "x_column": "月份",
    "y_column": "销售额",
    "reason": "选择原因"
    }
    """

    data_preview = df.head().to_csv(index=False)

    return call_llm(
        f"下面是数据预览：\n{data_preview}",
        chart_prompt
    )


@tool
def plot_chart_tool(file_path, chart_type, x_column, y_column):
    """
    根据 AI 选择的图表类型画图。
    支持 line、bar、pie、scatter。
    """
    df = pd.read_csv(file_path)

    plt.figure(figsize=(8, 4))

    if chart_type == "line":
        plt.plot(df[x_column], df[y_column], marker="o")
        plt.xlabel(x_column)
        plt.ylabel(y_column)

    elif chart_type == "bar":
        plt.bar(df[x_column], df[y_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)

    elif chart_type == "pie":
        plt.pie(df[y_column], labels=df[x_column], autopct="%1.1f%%")

    elif chart_type == "scatter":
        plt.scatter(df[x_column], df[y_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)

    else:
        return "暂不支持这种图表类型。"

    plt.title(f"{y_column} 图表")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return "图表已生成。"


@tool
def full_data_analysis_tool(file_path, call_llm):
    """
    完整数据分析工具：
    1. 读取 CSV
    2. 展示前几行
    3. 让 AI 总结数据重点
    4. 让 AI 决定画什么图
    5. 调用画图工具
    """
    try:
        df = read_csv(file_path)
        data_text = df.to_csv(index=False)

        system_prompt = """
        你是一个数据分析助理。
        请根据用户提供的 CSV 数据，用中文做简洁分析。

        要求：
        1. 总结 3 个最重要的发现
        2. 说明是否有明显异常
        3. 给出 1 条业务建议
        4. 不要编造数据里没有的信息
        5. 回答要简洁，适合普通职场人阅读
        """

        analysis_result = call_llm(
            f"请分析下面这份数据：\n{data_text}",
            system_prompt
        )

        chart_plan_text = choose_chart_with_ai(df, call_llm)

        try:
            chart_plan = json.loads(chart_plan_text)

            plot_result = plot_chart_tool(
                file_path=file_path,
                chart_type=chart_plan["chart_type"],
                x_column=chart_plan["x_column"],
                y_column=chart_plan["y_column"]
            )

            chart_info = f"""
            ### 📈 AI 推荐图表

            - 图表类型：{chart_plan["chart_type"]}
            - 横轴：{chart_plan["x_column"]}
            - 纵轴：{chart_plan["y_column"]}
            - 推荐原因：{chart_plan["reason"]}

            {plot_result}
            """

        except Exception:
            chart_info = """
            ### 📈 图表提醒

            AI 没有返回标准图表格式，所以这次先不自动画图。
            """

            return f"""
            ### 🤖 数据分析结果

            {analysis_result}

            ---

            {chart_info}
            """

    except Exception as e:
        return f"数据分析失败：{str(e)}"


# =========================
# Tool 6：网页问答工具
# =========================

@tool
def read_webpage_tool(url):
    """
    读取网页，并提取文本。
    """
    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return "网页读取失败，请检查网址。"

        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n")

        lines = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                lines.append(line)

        return "\n".join(lines)[:5000]

    except Exception as e:
        return f"网页读取失败：{str(e)}"


@tool
def web_qa_tool(url, question, call_llm):
    """
    用户提供网址和问题，AI 基于网页内容回答。
    """
    content = read_webpage_tool(url)

    system_prompt = """
    你是一个网页信息分析助手。

    请根据网页内容回答用户问题。

    要求：
    1. 只基于网页内容回答
    2. 如果网页中找不到，就说“网页中未找到相关信息”
    3. 用中文回答
    4. 简洁清晰
    """

    user_prompt = f"""
    用户问题：
    {question}

    网页内容：
    {content}
    """

    return call_llm(user_prompt, system_prompt)


# =========================
# Tool 7：消息推送工具
# =========================


def preview_message(message):
    """
    预览即将发送的消息。
    """
    show_markdown(f"""
    ### 👀 消息预览

    下面这段内容将被发送到测试群：

    ---

    {message}

    ---
    """)


@tool
def send_message_tool(message, confirm=False):
    """
    发送消息到测试群。

    confirm=False：只预览，不发送
    confirm=True：确认发送
    """
    if not confirm:
        preview_message(message)
        return "消息已预览。请确认后再发送。"

    if not WEBHOOK_URL:
        return "发送失败：未检测到 WEBHOOK_URL，请检查 .env。"

    payload = {
        "msgtype": "text",
        "text": {
            "content": message
        }
    }

    try:
        response = requests.post(WEBHOOK_URL, json=payload, timeout=10)

        if response.status_code == 200:
            return "消息发送成功 ✅"
        else:
            return "消息发送失败，请检查 Webhook。"

    except Exception as e:
        return f"消息发送失败：{str(e)}"


# =========================
# 总工具箱查看函数
# =========================


def list_tools():
    """
    查看当前已经注册的工具。
    """
    return list(TOOLS.keys())
