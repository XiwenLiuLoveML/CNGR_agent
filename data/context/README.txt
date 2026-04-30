Lab4 Context 示例数据说明

这个文件夹用于 Day4 Lab4。

结构说明：
history/
  保存当前对话前面已经发生过的内容。

memory/
  保存长期需要记住的信息，比如用户角色、表达偏好、权限边界。

resource/
  保存外部资料。RAG 主要处理这一部分，从这里检索和当前问题最相关的内容。

教学提醒：
Lab4 只演示如何读取和使用 context。
暂时不演示如何自动更新 history、memory 和 resource。
