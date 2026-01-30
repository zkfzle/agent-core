from datetime import datetime

from openjiuwen.core.runner.runner import Runner

from openjiuwen.core.utils.llm.messages import HumanMessage

from src.agents.base import BaseSpace
from src.common.agent_config import AgentConfig, AgentRuntimeConfig
from src.common.clients.openai_client import OpenAIClient
from src.common.react_utils import ActionResult, async_run_react_loop
from src.tools import ToolResult
from src.utils.prompt_loader import get_prompt_loader
from src.utils.logger import get_logger


logger = get_logger()


class DeepSearchSpace(BaseSpace):
    def __init__(self, task_inputs, *args, **kwargs):
        super(DeepSearchSpace, self).__init__(*args, **kwargs)

        self.current_round = 0
        self.link2name = {}
        self.valid_links = {}
        self.used_sources = {}

        self.task_inputs = task_inputs

        self.tools = ["Google Search Engine", "Web page content fetcher"]

        self.actions.update(
            {
                "search": self._handle_search_action,
                "click": self._handle_click_action,
                "report": self._handle_report_action
            }
        )

    async def _handle_search_action(self, action):
        action_content = action.content
        search_engine = [item for item in self.tools if 'search' in item.lower()][0]
        try:
            search_result = await Runner.run_tool(search_engine, inputs={"query": action_content})
            search_result_list = []
            if len(search_result) == 0:
                result = f"Query `{action_content}` returned no results; please try again."
            else:
                result = f"Search results for `{action_content}`\n"

                for idx, item in enumerate(search_result):
                    title = item.name
                    link = item.link
                    description = item.description
                    search_result_list.append({
                        'query': action_content,
                        'title': title,
                        'link': link,
                        'description': description
                    })
                    self.link2name[link] = title
                    # Track this as a valid link for later validation
                    self.valid_links[link] = {
                        'title': title,
                        'description': description,
                        'query': action_content
                    }
                    result += 'Result ' + str(idx + 1) + ':\n'
                    result += f"Title: {title}\n"
                    result += f"Link: {link}\n"
                    result += f"Summary: {description}\n\n"

            for search_item in search_result:
                self.cache.append(search_item)

        except Exception as e:
            result = f"Query `{action_content}` failed with error: {str(e)}. Please retry."

        return ActionResult(
            is_finished=False,
            message=HumanMessage(content=result),
        )

    async def _handle_click_action(self, action):
        click_engine = [item for item in self.tools if 'content fetcher' in item.lower()][0]
        current_task = self.task_inputs.get('task')
        query = self.task_inputs.get('query')
        action_content = action.content
        # Validate that the URL was from search results
        if action_content not in self.valid_links:
            logger.warning(f"Click rejected: URL not found in search results: {action_content}")
            # Provide available links as guidance
            available_links_hint = ""
            if self.valid_links:
                available_links_hint = "\n\nAvailable links from search results:\n"
                for idx, (url, info) in enumerate(list(self.valid_links.items())[:10], 1):
                    available_links_hint += f"{idx}. {info['title']}\n   URL: {url}\n"

            result = (
                f"ERROR: The URL '{action_content}' was not found in your search results. "
                f"You can ONLY click URLs that appeared in previous search results. "
                f"Please use one of the URLs from your search results, or perform a new search."
                f"{available_links_hint}"
            )
            return ActionResult(
                message=HumanMessage(content=result),
            )

        try:
            logger.info(f"Click action started: url={action_content}")
            click_result = await Runner.run_tool(click_engine, inputs={"urls": [action_content],
                                                                       "task": f'Research goal: {current_task}; query: {query}'})
            # click_result = await click_engine.api_function([action_content], f'Research goal: {current_task}; query: {query}')
            if len(click_result) == 0:
                result = "Failed to fetch content for url: " + action_content
            else:
                result = click_result[0].data
                # Track this as a used source with content summary
                source_title = self.link2name.get(action_content,
                                                  self.valid_links.get(action_content, {}).get('title', 'Unknown'))
                self.used_sources[action_content] = {
                    'title': source_title,
                    'content_preview': result[:500] if len(result) > 500 else result
                }
            # add to memory
            if click_result[0].link in self.link2name:
                click_result[0].name = self.link2name[click_result[0].link]
            if not ('error' in click_result[0].name.lower()):
                self.cache.append(click_result[0])
            logger.info(f"Click action done: url={action_content}")

        except Exception as e:
            result = "Failed to fetch url: " + action_content + "\n"
            result += f'Error: {e}'
            logger.error(f"Click action failed: url={action_content}, error={e}", exc_info=True)

        return ActionResult(
            message=HumanMessage(content=result),
        )

    @staticmethod
    async def _handle_report_action(action):
        return ActionResult(
            is_finished=True,
            message=HumanMessage(content=action.content),
        )


class DeepSearchAgent:
    def __init__(self, config: AgentRuntimeConfig, llm_client):
        self._config = config
        self.llm_client = llm_client

        self.prompt_loader = get_prompt_loader('search_agent', report_type='general')
        self.DEEP_SEARCH_PROMPT = self.prompt_loader.get_prompt('deep_search')

        self.max_iterations = self._config.get_agent_config().metadata.get('max_iterations', 5)

    def config(self):
        return self._config

    def prepare_init_prompt(self, inputs):
        basic_task = inputs.get('task', '')
        query = inputs.get('query', None)
        target_language_name = self._config.get_agent_config().metadata.get('target_language', 'zh')

        return self.DEEP_SEARCH_PROMPT.format(
            basic_task=basic_task,
            question=query,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            max_iterations=self.max_iterations,
            target_language=target_language_name
        )

    async def invoke(self, inputs, runtime):
        action_space = DeepSearchSpace(
            task_inputs=inputs,
            working_dir=self._config.get_agent_config().metadata.get('working_dir'),
        )
        messages = [
            HumanMessage(content=self.prepare_init_prompt(inputs)),
        ]
        messages, reach_max_iter = await async_run_react_loop(self.llm_client, messages, max_iterations=self.max_iterations, action_space=action_space)

        if reach_max_iter:
            sources_text = self._build_available_sources_list(action_space.valid_links, action_space.used_sources)
            messages[-1].content += sources_text

            conversation_history = [item.content for item in messages]
            analysis_info = "\n\n".join(conversation_history)
            prompt = f"You have reached the maximum number of running iterations. Directly give the summary of your search process based on the conversation history.\n\nConversation history: {analysis_info}\n\n"
            response = await self.llm_client.ainvoke(
                messages=[
                    HumanMessage(content=prompt),
                ],
                # prompt specification required
                # response_format={"type": "json_object"}
            )
            final_result = response.content
        else:
            final_result = messages[-1].content

        action_space.cache.append(
            DeepSearchResult(
                query=inputs['query'],
                name=f"Summary of the search process for {inputs['query']}",
                description=final_result,
                data=final_result,
                source="deepsearch_agent"
            )
        )

        return {
            "final_result": final_result,
            "cache": action_space.cache,
        }

    @staticmethod
    def _build_available_sources_list(valid_links, used_sources) -> str:
        """Build a formatted list of all available sources from search results and browsed pages."""
        if not valid_links and not used_sources:
            return ""

        sources_text = "\n\n---\n**VERIFIED SOURCES AVAILABLE FOR CITATION:**\n"
        sources_text += "(You may ONLY use URLs from this list in your References section)\n\n"

        # First list browsed/used sources (highest quality)
        if used_sources:
            sources_text += "**Sources you have browsed (recommended for citation):**\n"
            for idx, (url, info) in enumerate(used_sources.items(), 1):
                sources_text += f"  {idx}. {info['title']}\n"
                sources_text += f"     URL: {url}\n"

        # Then list search results that weren't clicked
        unclicked_sources = {url: info for url, info in valid_links.items()
                             if url not in used_sources}
        if unclicked_sources:
            sources_text += "\n**Additional sources from search results (snippets only):**\n"
            for idx, (url, info) in enumerate(unclicked_sources.items(), 1):
                sources_text += f"  {idx}. {info['title']}\n"
                sources_text += f"     URL: {url}\n"
                sources_text += f"     Summary: {info['description'][:200]}...\n" if len(
                    info['description']) > 200 else f"     Summary: {info['description']}\n"

        sources_text += "\n---\n"
        return sources_text

    @classmethod
    def get_provider(cls, config):
        def provider():
            agent_config = AgentConfig(
                id="deepsearch_agent",
                name="deepsearch_agent",
                description=(
                    "Tool: Deep Search\n"
                    "Description: run comprehensive web searches (news, filings, research, etc.) "
                    "to gather evidence for a given task.\n"
                    "Parameters: query:str (describe exactly what information is needed; "
                    "avoid loose keyword lists).\n"
                ),
                metadata={
                    "target_language": config.config["language"],
                    "max_iterations": 30,
                    "working_dir": config.config["working_dir"],
                }
            )

            llm_client = OpenAIClient(**config.config["llm_config_list"][0])
            return cls(config=AgentRuntimeConfig(agent_config), llm_client=llm_client)
        return provider


class DeepSearchResult(ToolResult):
    def __init__(self, query, name, description, data, source=""):
        super().__init__(name, description, data, source)
        self.query = query

    def __str__(self):
        format_output = f'Summary Search Result for {self.query}\n'
        format_output += f"Summary: {self.description}\n"
        return format_output

    def __repr__(self):
        return self.__str__()