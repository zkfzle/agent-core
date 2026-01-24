# -*- coding: utf-8 -*-
import os
import sys
import shutil
import asyncio
from pathlib import Path

from dotenv import load_dotenv

# python_dir = r"D:\agent-core\.venv\Scripts"          # 里面通常有 python.exe
# os.environ["PYTHONPATH"] = python_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
def _maybe_add_repo_to_syspath(project_root: Path) -> None:
    """
    Make imports work when you run main.py directly in PyCharm.

    We prefer your active interpreter/venv, but also support a repo layout like:
      project_root/
        agent-core/   (contains openjiuwen package)
        main.py
    """
    agent_core = project_root / "agent-core"
    if agent_core.exists():
        sys.path.insert(0, str(agent_core))


def _create_session(session_id: str):
    """
    Create a Session with the library's helper if available.
    """
    try:
        from openjiuwen.core.single_agent import create_agent_session
        return create_agent_session(session_id=session_id)
    except Exception:
        from openjiuwen.core.session.session import Session
        return Session(session_id=session_id)


def _normalize_provider(provider: str) -> str:
    if not provider:
        return "OpenAI"
    p = provider.strip()
    if p.lower() == "openai":
        return "OpenAI"
    if p.lower() in ("siliconflow", "silicon_flow"):
        return "SiliconFlow"
    return p


async def _invoke_agent(agent, session, query: str):
    await agent.context_engine.create_context(session=session)
    res = await agent.invoke(inputs={"query": query}, session=session)
    print(res.get("output", res))


async def main():
    project_root = Path(__file__).resolve().parent
    _maybe_add_repo_to_syspath(project_root)

    load_dotenv()

    QUERY = "Analyze SuperStoreUS-2015.xlsx file. Please write the analysis results in different worksheets of the same Excel file. Do not create a new Excel file, and ensure all numbers are dynamically generated (no hard-coded values): What is the store's total revenue?\nWhich product category contributes the most to sales?\nWhat is the sales trend over the past year?\nWhich region has the highest sales and which region has the lowest?\nWhat is the store's average profit margin?"

    # 使用你指定的绝对路径
    SKILLS_DIR = Path(r"D:\agent-core\openjiuwen\skills-dir").resolve()

    FILES = [
        {
            "local_filepath": (project_root / "SuperStoreUS-2015.xlsx").resolve(),
            "sandbox_filepath": "D:/agent-core/openjiuwen/SuperStoreUS-2015.xlsx",
        },
    ]

    WORK_DIR = (project_root / "workspace").resolve()
    SESSION_ID = "skill_session"
    SYS_OPERATION_ID = "default_sysop"
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "40"))

    api_base = os.getenv("API_BASE", "https://api.openai.com/v1")
    api_key = os.getenv("API_KEY", "")
    model_name = os.getenv("MODEL_NAME", "")
    model_provider = _normalize_provider(os.getenv("MODEL_PROVIDER", "OpenAI"))

    system_prompt = (
        "You are an agent equipped with various skills to solve problems.\n"
        "Before attempting any task, read the relevant skill document (SKILL.md) using view_file and follow its workflow.\n"
        "All user-provided files are located at 'D:/agent-core/openjiuwen/\n"
    )

    from openjiuwen.core.runner.runner import Runner
    runner = Runner

    from openjiuwen.core.sys_operation.sys_operation import SysOperationCard, OperationMode
    from openjiuwen.core.sys_operation.local.config import LocalWorkConfig

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # 关键改动：work_dir=None，让 fs 支持直接读取 Windows 绝对路径（D:\...）
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

    skills_dst = WORK_DIR / "home" / "user" / "skills"
    if SKILLS_DIR.exists():
        if skills_dst.exists():
            shutil.rmtree(skills_dst)
        skills_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SKILLS_DIR, skills_dst)

    await agent.register_skill(str(SKILLS_DIR))
    sm = agent._skill_util.skill_manager
    print(f"[skills] count={sm.count()}")
    print(f"[skills] names={sm.get_names()}")
    for s in sm.get_all():
        print(f"[skills] {s.name} -> {s.directory}")

    for f in FILES:
        local_path = Path(f["local_filepath"]).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        target = f["sandbox_filepath"]
        target_rel = target.lstrip("/").replace("\\", "/")
        out_path = WORK_DIR / target_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(local_path.read_bytes())

    await _invoke_agent(agent, session, QUERY)


if __name__ == "__main__":
    asyncio.run(main())
