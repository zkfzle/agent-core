from typing import Dict, Optional, Union, List, Literal

import yaml
from pydantic import BaseModel
from pathlib import Path

from openjiuwen.core.sys_operation.base import SysOperation


class Skill(BaseModel):
    name: str
    description: str = None
    directory: Path

    def __str__(self):
        return f"Skill: {self.name}\nDescription: {self.description}\nDirectory: {self.directory}"

    def __repr__(self):
        return (f"[Skill: {self.name} / Description: {self.description[:min(len(self.description), 30)] + '...'} "
                f"/ Directory: {self.directory}]")


class SkillManager:
    """Skill 管理器
    """

    def __init__(
            self,
            env: Literal["local", "sandbox"] = "sandbox"
    ):
        """初始化 Skill 注册表"""
        self._registry: Dict[str, Skill] = {}
        self._env: Literal["local", "sandbox"] = env

    @staticmethod
    def _load_yaml(path: Path, session_id: str):
        text = SysOperation().read_file(session_id, path, "text")
        if text.startswith("---"):
            _, yaml_block, body = text.split("---", 2)
            return yaml.safe_load(yaml_block), body.lstrip()
        return None, text

    def _load_description(self, path: Path, session_id: str) -> str:
        self.description = ""
        yaml_data, _ = self._load_yaml(path, session_id)
        if yaml_data is None or "description" not in yaml_data:
            raise KeyError("Skill.md file does not contain a description field")
        return yaml_data['description']

    def _create_skill_from_path(self, path: Path, session_id: str) -> Optional[Skill]:
        description = self._load_description(path, session_id)
        if description is not None:
            return Skill(name=path.name, description=description, directory=path)
        return None

    def register(
            self,
            skill_path: Union[Path, List[Path]],
            session_id: str = None,
            overwrite: bool = False
    ):
        """注册 Skill 元信息

        Args:
            skill_path: skill 路径
            session_id: session id
            overwrite: 如果为 True，当 skill 已存在时覆盖；否则抛出异常

        Raises:
            ValueError: 如果 skill 已存在且 overwrite 为 False
        """
        if skill_path is not None and isinstance(skill_path, Path):
            skill = self._create_skill_from_path(skill_path, session_id)
            self._registry[skill.name] = skill
        if skill_path is not None and isinstance(skill_path, list):
            for p in skill_path:
                skill = self._create_skill_from_path(p, session_id)
                self._registry[skill.name] = skill

    def unregister(self, name: str):
        """取消注册 Skill

        Args:
            name: Skill 名称

        Returns:
            bool: 如果成功取消注册返回 True，否则返回 False
        """
        if name in self._registry:
            del self._registry[name]

    def get(self, name: str) -> Optional[Skill]:
        """获取 Skill 元信息

        Args:
            name: Skill 名称

        Returns:
            Optional[SkillMeta]: Skill 元信息对象，如果不存在返回 None
        """
        if name in self._registry:
            return self._registry[name]
        return None

    def get_all(self) -> List[Skill]:
        """获取所有已注册的 Skill 元信息

        Returns:
            List[SkillMeta]: 所有 Skill 元信息列表
        """
        return list(self._registry.values())

    def get_names(self) -> List[str]:
        """获取所有已注册的 Skill 名称

        Returns:
            List[str]: Skill 名称列表
        """
        return list(self._registry.keys())

    def has(self, name: str) -> bool:
        """检查 Skill 是否已注册

        Args:
            name: Skill 名称

        Returns:
            bool: 如果已注册返回 True，否则返回 False
        """
        return name in self._registry

    def clear(self) -> None:
        """清空注册表"""
        self._registry.clear()

    def count(self) -> int:
        """获取注册的 Skill 数量

        Returns:
            int: Skill 数量
        """
        return len(self._registry)
        