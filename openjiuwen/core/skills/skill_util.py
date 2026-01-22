from pathlib import Path

from openjiuwen.core.foundation.prompt import PromptTemplate
from openjiuwen.core.skills.skill_manager import SkillManager
from openjiuwen.core.skills.tools.run_command import run_command_tool_card, run_command
from openjiuwen.core.skills.tools.execute_python_code import execute_python_code_tool_card, execute_python_code
from openjiuwen.core.skills.tools.view_file import view_file, view_file_tool_card
from openjiuwen.core.runner import Runner
from openjiuwen.core.single_agent import BaseAgent


skill_prompt = PromptTemplate(
    content='''
    为了帮助你更好地完成任务，为你提供了以下技能知识：
    {{skills}}
    你可以使用 view_file 工具读取对应的 Skill.md 文件获取相应的技能知识。
    '''
)


class SkillUtil:
    def __init__(self):
        self._skill_manager = SkillManager()

    @property
    def skill_manager(self):
        return self._skill_manager

    def register_skills(self, skill_path: str, agent: BaseAgent, session_id: str = None) -> bool:
        # 为 Agent 添加内置工具
        if agent.ability_kit.get(view_file_tool_card.name) is None:
            Runner().resource_mgr.add_tool(view_file)
            agent.ability_kit.add(view_file_tool_card)
        if agent.ability_kit.get(execute_python_code_tool_card.name) is None:
            Runner().resource_mgr.add_tool(execute_python_code)
            agent.ability_kit.add(execute_python_code_tool_card)
        if agent.ability_kit.get(run_command_tool_card.name) is None:
            Runner().resource_mgr.add_tool(run_command)
            agent.ability_kit.add(run_command_tool_card)
        self._skill_manager.register(Path(skill_path), session_id)

    def has_skill(self):
        return True if self._skill_manager.count() > 0 else False

    def get_skill_prompt(self) -> str:
        skills = self._skill_manager.get_all()
        skills_info = []
        for index, skill in enumerate(skills):
            skills_info.append(f"{index}.技能名：skill.name；技能描述：{skill.description}；技能文件路径：{skill.directory}")
        return skill_prompt.format({"skill_info": "\n".join(skills_info)}).content
