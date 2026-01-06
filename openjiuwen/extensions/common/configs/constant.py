# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.


DEFAULT_INNER_LOG_CONFIG = {
                'level': 'WARNING',  # 修改为 WARNING 减少日志输出
                'output': ['file'],  # 只输出到文件，不输出到控制台
                'log_path': './logs/',
                'log_file': 'run/jiuwen.log',
                'interface_log_file': 'interface/jiuwen_interface.log',
                'interface_output': ['file'],  # 只输出到文件
                'prompt_builder_interface_log_file': 'interface/jiuwen_prompt_builder_interface.log',
                'performance_log_file': 'performance/jiuwen_performance.log',
                'performance_output': ['file'],  # 只输出到文件
                'backup_count': 20,
                'max_bytes': 20971520,
                'format': ('%(asctime)s | %(log_type)s | %(filename)s | %(lineno)d | '
                          '%(funcName)s | %(trace_id)s | %(levelname)s | %(message)s')
            }

DEFAULT_LOG_CONFIG = {
    'logging': DEFAULT_INNER_LOG_CONFIG
}
