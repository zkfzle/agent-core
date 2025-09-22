from typing import List, Tuple, Optional
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.core.context.controller_context.thread_safe_dict import ThreadSafeDict

class PromptMgr:
    def __init__(self) -> None:
        self._repo: ThreadSafeDict[str, Template] = ThreadSafeDict()

    def add_prompt(self, template_id: str, template: Template) -> None:
        self._repo[template_id] = template

    def add_prompts(self, templates: List[Tuple[str, Template]]) -> None:
        self._repo.update(templates)

    def remove_prompt(self, template_id: str) -> bool:
        return self._repo.pop(template_id, None) is not None

    def get_prompt(self, template_id: str) -> Optional[Template]:
        return self._repo.get(template_id)