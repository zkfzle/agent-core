# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import glob
import os
import re
import threading
from typing import Any, Dict, Iterable, List, Optional, Set

from openjiuwen.core.common.logging import logger
from openjiuwen.core.runtime.resources_manager.prompt_manager import PromptMgr
from openjiuwen.core.utils.prompt.template.template import Template

PR_PATTERN = re.compile(r"(?s)`#((?:user)|(?:system)|(?:assistant)|(?:tool))#`")


class ReadOnlyPromptManager:
    __slots__ = ("_initialized", "__all_prompt_names", "__mgr")
    __instance: Optional["ReadOnlyPromptManager"] = None
    __thread_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ReadOnlyPromptManager":
        with cls.__thread_lock:
            if cls.__instance is None:
                cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.__all_prompt_names: Set[str] = set()
        self.__mgr = PromptMgr()
        prompt_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "**", "*.pr"))
        language_dirs = set()
        for t_path in glob.glob(prompt_root_dir, recursive=True):
            language_dirs.add(os.path.dirname(t_path))
        for t_dir in language_dirs:
            self.register_in_bulk(t_dir, name=os.path.split(t_dir)[1].strip(os.path.sep))
        self._initialized = True

    def __contains__(self, key: Any) -> bool:
        return key in self.__all_prompt_names

    @staticmethod
    def load_pr_content(content: str) -> List[Dict[str, str]]:
        roles = {"user", "system", "assistant", "tool"}
        matching_role = True
        current_msg: Optional[Dict[str, str]] = None
        messages = []
        for line in PR_PATTERN.split(content):
            if not line:
                continue
            if matching_role:
                if line in roles:
                    current_msg = dict(role=line, content="")
                    matching_role = False
            else:
                current_msg["content"] = line
                messages.append(current_msg)
                current_msg = None
                matching_role = True
        return messages

    def get(self, name: str) -> Optional[Template]:
        return self.__mgr.get_prompt(name)

    def register_in_bulk(self, prompt_dir: str, name: str = ""):
        prompt_root_dir = os.path.abspath(os.path.join(prompt_dir, "*.pr"))
        prompt_paths = glob.glob(prompt_root_dir, recursive=True)
        if not prompt_paths:
            raise FileNotFoundError(f"No .pr prompt files found in {prompt_dir}")
        self._register_templates(prompt_paths)
        if name:
            name += "："
        logger.info(f"[图谱记忆]成功加载提示词模版%s%d个", name, len(prompt_paths))

    def _register_templates(self, template_paths: Iterable[str]):
        prompts = []
        for t_path in template_paths:
            t_name = os.path.split(t_path)[1].removesuffix(".pr")
            with open(t_path, "r", encoding="utf-8") as prompt_file:
                t_content = self.load_pr_content(prompt_file.read())
            self.__all_prompt_names.add(t_name)
            prompts.append((t_name, Template(name=t_name, content=t_content)))
        self.__mgr.add_prompts(prompts)
