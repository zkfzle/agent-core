# -*- coding: utf-8 -*-
import os
import sys
import shutil
import asyncio
from pathlib import Path

from dotenv import load_dotenv


def _create_session(session_id: str):
    """
    Create a Session
    """
    try:
        from openjiuwen.core.single_agent import create_agent_session
        return create_agent_session(session_id=session_id)
    except Exception:
        from openjiuwen.core.session.session import Session
        return Session(session_id=session_id)

async def _invoke_agent(agent, session, query: str):
    await agent.context_engine.create_context(session=session)
    res = await agent.invoke(inputs={"query": query}, session=session)
    print(res.get("output", res))


async def main():
    project_root = Path(__file__).resolve().parent

    load_dotenv()

    QUERY = ("Analyze SuperStoreUS-2015.xlsx file. Write the analysis results in different worksheets of the same Excel file. Do not create new Excel file,\n"
             " and ensure all numbers are dynamically generated (no hard-coded values): What is the store's total revenue?\nWhich product category contributes the most to sales?\n"
             "What is the sales trend over the past year?\nWhich region has the highest sales and which region has the lowest?\nWhat is the store's average profit margin?")

    SKILLS_DIR = Path(os.getenv("SKILLS_DIR")).expanduser().resolve()
    FILES_BASE_DIR = os.getenv("FILES_BASE_DIR")

    SESSION_ID = "skill_session"
    SYS_OPERATION_ID = "default_sysop"
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "40"))

    api_base = os.getenv("API_BASE", "https://api.openai.com/v1")
    api_key = os.getenv("API_KEY", "")
    model_name = os.getenv("MODEL_NAME", "")
    model_provider = (os.getenv("MODEL_PROVIDER", "OpenAI") or "OpenAI").strip()

    system_prompt = (
        "You are an agent equipped with various skills to solve problems.\n"
        "Before attempting any task, read the relevant skill document (SKILL.md) using view_file and follow its workflow.\n"
        f"All user-provided files are located at '{FILES_BASE_DIR}\n"
    )

    from openjiuwen.core.runner.runner import Runner
    runner = Runner

    from openjiuwen.core.sys_operation.sys_operation import SysOperationCard, OperationMode
    from openjiuwen.core.sys_operation.local.config import LocalWorkConfig


    sysop_card = SysOperationCard(
        id=SYS_OPERATION_ID,
        mode=OperationMode.LOCAL,
        work_config=LocalWorkConfig(work_dir=None),
    )

    runner.resource_mgr.add_sys_operation(sysop_card)

    from openjiuwen.core.single_agent.schema.agent_card import AgentCard
    from openjiuwen.core.single_agent.agents.react_agent import ReActAgent, ReActAgentConfig

    agent = ReActAgent(card=AgentCard(name="skill_agent", description="Skill Agent"))

    cfg = ReActAgentConfig()
    cfg.sys_operation_id = SYS_OPERATION_ID
    cfg = (
        cfg.configure_model_client(
            provider=model_provider,
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            verify_ssl=(os.getenv("LLM_SSL_VERIFY", "true").lower() != "false"),
        )
        .configure_prompt_template([{"role": "system", "content": system_prompt}])
        .configure_max_iterations(MAX_ITERATIONS)
        .configure_context_limit(None)
    )
    agent.configure(cfg)
    agent._skill_util.skill_manager._sys_operation_id = SYS_OPERATION_ID
    agent._skill_util.skill_tool_kit.sys_operation_id = SYS_OPERATION_ID

    from openjiuwen.core.skills.skill_tool_kit import SkillToolKit

    toolkit = SkillToolKit(SYS_OPERATION_ID)

    if hasattr(toolkit, "_runner"):
        toolkit._runner = runner

    toolkit.add_skill_tools(agent)

    session = _create_session(SESSION_ID)

    await agent.register_skill(str(SKILLS_DIR))
    await _invoke_agent(agent, session, QUERY)


if __name__ == "__main__":
    asyncio.run(main())
