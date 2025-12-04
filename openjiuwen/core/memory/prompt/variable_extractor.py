# zh-CN prompt
EXTRACT_VARIABLES_USER_SUMMARY_zh_CN = """
基于以下对话内容：\n
<对话>\n
{conversation}\n
</对话>\n\n
历史摘要如下\n
<摘要>\n
{summary}\n
</摘要>\n\n
变量定义如下\n
<定义>\n
{variables}\n
</定义>\n\n
提取变量值并以标准JSON格式返回结果：\n
{variables_enum}\n
输出格式限制：\n
不要保留 ```json``` 代码块标记。\n
"""

EXTRACT_VARIABLES_USER_zh_CN = """
基于以下对话内容：\n
<对话>\n
{conversation}\n
</对话>\n\n
变量定义如下\n
<定义>\n
{variables}\n
</定义>\n\n
提取变量值并以标准JSON格式返回结果：\n
{variables_enum}\n
输出格式限制：\n
不要保留 ```json``` 代码块标记。\n
"""

EXTRACT_VARIABLES_SYS_zh_CN = """
# 指令\n
假设你是信息提取领域专家。\n
# 任务定义\n
分析LLM智能体的对话内容，根据预定义的变量名称和变量描述，完成目标变量值提取。\n\n
# 输出格式\n
 - 仅输出单个JSON对象。\n
 - 若无法提取某个变量的值，则在对应字段中填入"null"。\n
"""
