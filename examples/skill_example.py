from openjiuwen.core.runner import Runner
from openjiuwen.core.single_agent import ReActAgent, AgentCard
from openjiuwen.core.sys_operation.base import SysOperation


def create_agent(card):
    agent = ReActAgent()
    agent.register_skill("skill在沙箱中的路径")
    return agent


if __name__ == "__main__":
    SysOperation().upload_file(
        local_file_path="skill本地路径",
        sandbox_file_path="skill在沙箱中的路径"
    )

    card = AgentCard(
        id = "skill 测试 agent",
        description="skill 测试 agent"
    )
    Runner().resource_mgr.add_agent(card, create_agent)
    Runner().run_agent(
        agent=card.id,
        inputs="帮我制作一个和詹姆斯25赛季表现有关的PPT"
    )