#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from enum import Enum


class StatusCode(Enum):
    # Agent模块 103025~103050
    AGENT_SUB_TASK_TYPE_ERROR = (103032, "SubTask type {msg} is not supported")

    CONTROLLER_INTERRUPTED_ERROR = (10312, "controller interrupted error")
    PROMPT_JSON_SCHEMA_ERROR = (102056, "Invalid json schema, root cause = {error_msg}.")

    # Prompt 模板填充 102050 - 102099
    PROMPT_ASSEMBLER_VARIABLE_INIT_ERROR = (102050, "Wrong arguments for initializing the variable")
    PROMPT_ASSEMBLER_INIT_ERROR = (102051, "Wrong arguments for initializing the assembler")
    PROMPT_ASSEMBLER_INPUT_KEY_ERROR = (
        102052,
        "Missing or unexpected key-value pairs passed in as arguments for the assembler or variable when updating"
    )
    PROMPT_ASSEMBLER_TEMPLATE_FORMAT_ERROR = (
        102053,
        "Errors occur when formatting the template content due to wrong format")

    # 控制器 103000 - 103999
    INVOKE_LLM_FAILED = (103004, "Failed to call model")

    # Tool组件 101741-101770
    TOOL_COMPONENT_PARAM_CHECK_ERROR = (101742, 'Tool component parameter check error')
    TOOL_COMPONENT_INPUTS_ERROR = (101743, 'Tool component inputs not defined')
    TOOL_COMPONENT_EXECUTE_ERROR = (101745, "Tool component execute error")

    # Prompt 模板管理 102100 - 102149
    PROMPT_TEMPLATE_DUPLICATED_ERROR = (102101, "Template duplicated")
    PROMPT_TEMPLATE_NOT_FOUND_ERROR = (102102, "Template not found")
    PROMPT_TEMPLATE_INCORRECT_ERROR = (102103, "Template data incorrect")

    # 插件管理  105000~105999
    PLUGIN_UNEXPECTED_ERROR = (105001, "Plugin unexpected error")
    PLUGIN_REQUEST_TIMEOUT_ERROR = (105002, "Plugin restful api request timed out")
    PLUGIN_PROXY_CONNECT_ERROR = (105003, "Plugin restful api proxy connection error")
    PLUGIN_RESPONSE_TOO_BIG_ERROR = (105004, "Plugin restful api  response too big")
    PLUGIN_RESPONSE_HTTP_CODE_ERROR = (105005, "Plugin restful api http code error")
    PLUGIN_PARAMS_CHECK_FAILED = (105006, "Plugin params check failed")
    # start组件 101561-101590
    WORKFLOW_START_MISSING_GLOBAL_VARIABLE_VALUE = (101501, "start component: global variable(s) defined with no value assigned:  {variable_name}")

    # LLM组件 101561-101590
    WORKFLOW_LLM_INIT_ERROR = (101561, "LLM component initialization error, msg = {msg}")
    WORKFLOW_LLM_TEMPLATE_ASSEMBLE_ERROR = (101562, "LLM component template assemble error")
    WORKFLOW_LLM_STREAMING_OUTPUT_ERROR = (101563, "Get model streaming output error, msg = {msg}")

    # questioner组件 101040 - 101059
    WORKFLOW_QUESTIONER_EXCEED_LOOP = (101043, "Exceeded the maximum number of conversation")
    WORKFLOW_QUESTIONER_QUESTION_EMPTY_DIRECT_COLLECTION_ERROR = (
        101045, "The question cannot be empty in direct user response collection mode")
    WORKFLOW_QUESTIONER_INIT_STATE_ERROR = (101059, "Failed to initialize questioner state")

    # intent detection组件 101681-101710
    WORKFLOW_INTENT_DETECTION_USER_INPUT_ERROR = (101095, "User input pre-processing failed with error"
                                                          "message = {error_msg}")
    WORKFLOW_INTENT_DETECTION_LLM_INVOKE_ERROR = (101096, "Model invoke failed with error message = {error_msg}")
    WORKFLOW_INTENT_DETECTION_PROMPT_INVOKE_ERROR = (101098, "Prompt invoke failed with error message = {error_msg}")

    # message queue manager 101,711-101,719
    WORKFLOW_MESSAGE_QUEUE_MANAGER_ERROR = (101711, "Message queue manager error: {error_msg}")

    # context engine 102000 - 102500
    CONTEXT_ENGINE_MESSAGE_PROCESS_ERROR = (102000, "Message process error: {error_msg}")

    @property
    def code(self):
        return self.value[0]

    @property
    def errmsg(self):
        return self.value[1]
