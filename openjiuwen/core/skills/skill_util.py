from pathlib import Path

from openjiuwen.core.foundation.prompt import PromptTemplate
from openjiuwen.core.single_agent.agent import BaseAgent
from openjiuwen.core.skills.skill_manager import SkillManager
from openjiuwen.core.skills.skill_tool_kit import SkillToolKit

skill_prompt = PromptTemplate(
    content='''
	     为了帮助你更好地完成任务，为你提供了以下技能知识：
	     {{skills}}
	     你可以使用 view_file 工具读取对应的 Skill.md 文件获取相应的技能知识。
	     '''
)


class SkillUtil:
    def __init__(self, sys_operation_id: str):
        self._skill_manager = SkillManager()
        self._skill_tool_kit = SkillToolKit(sys_operation_id)

    @property
    def skill_manager(self):
        return self._skill_manager

    def register_skills(self, skill_path: str, agent: BaseAgent, session_id: str = None) -> bool:
        self._skill_tool_kit.add_skill_tools(agent)
        self._skill_manager.register(Path(skill_path), session_id)

    def has_skill(self):
        return True if self._skill_manager.count() > 0 else False

    def get_skill_prompt(self) -> str:
        skills = self._skill_manager.get_all()
        skills_info = []
        for index, skill in enumerate(skills):
            skills_info.append(f"{index}.技能名：skill.name；技能描述：{skill.description}；技能文件路径：{skill.directory}")
        return skill_prompt.format({"skill_info": "\n".join(skills_info)}).content