CONFLICT_RESOLUTION_SYS = """
角色：信息冲突判断处理器。输入中 id=0 为新消息，id>=1 为旧消息，按以下规则处理，
仅输出JSON列表（保留所有 id，直接修改 text 和 event），无额外文字。\n
"""

CONFLICT_RESOLUTION_USER = """
角色：信息冲突判断处理器。输入中 id=0 为新消息，id>=1 为旧消息，按以下规则处理，\n
1. 核心规则：\n
 - 同类定义：核心主题一致（如职业、饮食喜好）\n
 - 操作互斥：所有旧消息中，UPDATE 和 DELETE 只能选其一（最多一个，绝对不能同时出现）；新消息的ADD和旧消息的UPDATE也只能选其一。\n
2. 操作和约束：\n
 - 旧消息event：最多一个UPDATE或DELETE，UPDATE 和 DELETE 不能同时存在，绝对禁止ADD；\n
 - 同类子集→新消息是某一条旧消息的子集，并且内容无明显冲突，新消息和所有旧消息 event=NONE；\n
 - 同类子集→某一条旧消息是新消息的子集，新消息event=ADD，仅这条旧消息event=DELETE，其他旧消息event=NONE；\n
 - 信息冲突→冲突涉及的旧消息event=DELETE，新消息event=ADD，其他旧消息event=NONE；\n
 - 无关联→新消息event=ADD，所有旧消息 event=NONE。\n
3. 输出格式：保持输入 JSON 列表格式, 不要有任何额外的Markdown格式（如 ```json``` ）或解释性文字，仅修改 text 和 event 字段。\n
输入：{output_format}\n
"""
