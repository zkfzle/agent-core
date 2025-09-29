#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from enum import Enum


class StatusCode(Enum):
    """状态码枚举类"""

    SUCCESS = (0, "success")
    ERROR = (-1, "error")

    # 组件 100000 - 109999
    # 组件 组件开发 100000 - 100999
    # 组件 预制组件 101000 - 101999

    # 工作流 110000 - 119999
    # 工作流 工作流编排和执行 110000 - 110999
    # 工作流 异常处理 111000 - 111999

    # Agent编排 120000 - 129999
    # Agent编排 ReAct Agent编排和执行 120000 - 120999
    # Agent编排 Workflow Agent编排和执行 121000 - 121999
    # Agent编排 自定义Agent接口定义 122000 - 122999

    # Multi-Agent编排 130000 - 139999
    # Multi-Agent编排 多Agent通信机制 130000 - 130999
    # Multi-Agent编排 单机运行时 131000 - 131999
    # Multi-Agent编排 AgentGroup 132000 - 132999
    # Multi-Agent编排 多Agent调测能力 133000 - 133999
    # Multi-Agent编排 分布式运行时 134000 - 134999

    # 图执行引擎 140000 - 149999
    # 图执行引擎 图的编排和执行 140000 - 140999

    # 上下文引擎 150000 - 159999
    # 上下文引擎 上下文结构化存取 150000 - 150999
    # 上下文引擎 上下文动态组装 151000 - 151999
    # 上下文引擎 上下文异步加工 152000 - 152999

    # 开发工具链 160000 - 169999
    # 开发工具链 提示词生成 160000 - 160999
    # 开发工具链 Agent DL convertor 161000 - 161999
    # 开发工具链 NL2Agent 162000 - 162999

    # 调优工具链 170000 - 179999
    # 调优工具链 提示词自优化 170000 - 170999
    # 调优工具链 全链路优化 171000 - 171999
    # 调优工具链 AgentRL 172000 - 172999

    # 公共能力 180000 - 189999
    # 公共能力 提示词填充 180000 - 180999
    # 公共能力 大模型接口 181000 - 181999
    # 公共能力 工具定义和执行 182000 - 182999
    PLUGIN_UNEXPECTED_ERROR = (182000, "Plugin unexpected error")
    PLUGIN_REQUEST_TIMEOUT_ERROR = (182001, "Plugin restful api request timed out")
    PLUGIN_PROXY_CONNECT_ERROR = (182002, "Plugin restful api proxy connection error")
    PLUGIN_RESPONSE_TOO_BIG_ERROR = (182003, "Plugin restful api  response too big")
    PLUGIN_RESPONSE_HTTP_CODE_ERROR = (182004, "Plugin restful api http code error")
    PLUGIN_PARAMS_CHECK_FAILED = (182005, "Plugin params check failed")

    # 公共能力 日志Logger 183000 - 183999
    # 公共能力 异常处理 184000 - 184999
    # 公共能力 支持mcp插件 185000 - 185999

    # Runtime 190000 - 199999
    # Runtime 资源管理 190000 - 190999
    # Runtime 调测能力 191000 - 191999
    # Runtime 状态管理 192000 - 192999
    # Runtime 流式输出StreamWriter 193000 - 193999
    # Runtime Config管理 194000 - 194999
    # Runtime callback 195000 - 195999

    WORKFLOW_START_MISSING_GLOBAL_VARIABLE_VALUE = (101501, "start component: global variable(s) defined with no value assigned:  {variable_name}")
    WORKFLOW_START_CREATE_VALUE = (101502, "start component create error:  {reason}")

    WORKFLOW_LLM_INIT_ERROR = (101561, "LLM component initialization error, msg = {msg}")
    WORKFLOW_LLM_TEMPLATE_ASSEMBLE_ERROR = (101562, "LLM component template assemble error")
    WORKFLOW_LLM_STREAMING_OUTPUT_ERROR = (101563, "Get model streaming output error, msg = {msg}")

    WORKFLOW_INTENT_DETECTION_USER_INPUT_ERROR = (101695, "User input pre-processing failed with error"
                                                          "message = {error_msg}")
    WORKFLOW_INTENT_DETECTION_LLM_INVOKE_ERROR = (101696, "Model invoke failed with error message = {error_msg}")
    WORKFLOW_INTENT_DETECTION_PROMPT_INVOKE_ERROR = (101698, "Prompt invoke failed with error message = {error_msg}")

    WORKFLOW_QUESTIONER_EXCEED_LOOP = (101713, "Exceeded the maximum number of conversation")
    WORKFLOW_QUESTIONER_QUESTION_EMPTY_DIRECT_COLLECTION_ERROR = (
        101715, "The question cannot be empty in direct user response collection mode")
    WORKFLOW_QUESTIONER_INIT_STATE_ERROR = (101729, "Failed to initialize questioner state")

    TOOL_COMPONENT_PARAM_CHECK_ERROR = (101742, 'Tool component parameter check error')
    TOOL_COMPONENT_INPUTS_ERROR = (101743, 'Tool component inputs not defined')
    TOOL_COMPONENT_EXECUTE_ERROR = (101745, "Tool component execute error")

    WORKFLOW_MESSAGE_QUEUE_MANAGER_ERROR = (101771, "Message queue manager error: {error_msg}")

    PROMPT_ASSEMBLER_VARIABLE_INIT_ERROR = (102050, "Wrong arguments for initializing the variable")
    PROMPT_ASSEMBLER_INIT_ERROR = (102051, "Wrong arguments for initializing the assembler")
    PROMPT_ASSEMBLER_INPUT_KEY_ERROR = (
        102052,
        "Missing or unexpected key-value pairs passed in as arguments for the assembler or variable when updating"
    )
    PROMPT_ASSEMBLER_TEMPLATE_FORMAT_ERROR = (
        102053,
        "Errors occur when formatting the template content due to wrong format")
    PROMPT_JSON_SCHEMA_ERROR = (102056, "Invalid json schema, root cause = {error_msg}.")

    PROMPT_TEMPLATE_DUPLICATED_ERROR = (102101, "Template duplicated")
    PROMPT_TEMPLATE_NOT_FOUND_ERROR = (102102, "Template not found")
    PROMPT_TEMPLATE_INCORRECT_ERROR = (102103, "Template data incorrect")

    INVOKE_LLM_FAILED = (103004, "Failed to call model")
    CONTROLLER_INTERRUPTED_ERROR = (10312, "controller interrupted error")

    AGENT_SUB_TASK_TYPE_ERROR = (103032, "SubTask type {msg} is not supported")

    CONTEXT_ENGINE_MESSAGE_PROCESS_ERROR = (106000, "Message process error: {error_msg}")

    AGENT_BUILDER_PARAM_CHECK_FAILED_ERROR = (110000, "Error occur when input parameter varification failed")
    AGENT_BUILDER_LLM_CONFIG_MISS_ERROR = (110001, "LLM service configuration is missing: {error_msg}")
    AGENT_BUILDER_LLM_FALSE_RESULT_ERROR = (110002, "LLM service return false result due to {error_msg}")

    AGENT_BUILDER_PROMPT_OPTIMIZE_REFINE_INSTRUCTION_ERROR = (
        110003, "Prompt optimization failed to refine instruction, root cause: {error_msg}"
    )

    AGENT_BUILDER_PROMPT_OPTIMIZE_RESTART_TASK_ERROR = (110004, "Prompt optimization restart task error: {error_msg}")
    AGENT_BUILDER_PROMPT_OPTIMIZE_EVALUATE_ERROR = (110005, "Prompt optimization evaluate failed, root cause: {error_msg}")
    AGENT_BUILDER_PROMPT_OPTIMIZE_INVALID_PARAMS_ERROR = (
        110006, "Prompt optimization parameters are invalid, root cause = {error_msg}")
    AGENT_BUILDER_PROMPT_OPTIMIZE_CASE_VALIDATION_ERROR = (
        110007, "Prompt optimization validate input case failed, root cause = {error_msg}"
    )
    AGENT_BUILDER_CREATE_OPTIMIZER_ERROR = (
        110008, "Create optimizer failed, root cause = {error_msg}"
    )
    AGENT_BUILDER_PROMPT_OPTIMIZE_ERROR = (
        110009, "Do optimize failed, root cause = {error_msg}"
    )

    @property
    def code(self):
        return self.value[0]

    @property
    def errmsg(self):
        return self.value[1]
