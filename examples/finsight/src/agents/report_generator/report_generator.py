import os
import pickle
import re
import subprocess

import docx2pdf
import numpy as np
from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.utils.llm.messages import HumanMessage

from src.agents.report_generator.report_class import Section
from src.agents.search_agent.search_agent import DeepSearchResult
from src.common.clients.async_embedding_client import EmbeddingClient
from src.common.clients.openai_client import OpenAIClient
from src.tools import ClickResult, SearchResult
from src.utils import get_md_img, IndexBuilder, extract_markdown
from src.utils.figure_helper import draw_kline_chart
from src.utils.logger import get_logger
from src.utils.prompt_loader import get_prompt_loader

logger = get_logger()


class ReportGenerator:
    def __init__(self, config):
        self._state = {}
        self.config = config
        self.working_dir = config["working_dir"]

        self.llm_client = OpenAIClient(**config["llm_config_list"][0])

        embedding_client = EmbeddingClient(**config["llm_config_list"][1])

        self.index = IndexBuilder(
            embedding_client=embedding_client,
            working_dir=config["working_dir"]
        )

        self.prompt_loader = get_prompt_loader('report_generator', report_type=config["target_type"])
        self.TITLE_PROMPT = self.prompt_loader.get_prompt('title_generation')
        self.ABSTRACT_PROMPT = self.prompt_loader.get_prompt('abstract')
        self.TABLE_BEAUTIFY_PROMPT = self.prompt_loader.get_prompt('table_beautify')

        self.target_language = config["language"]

        self.add_introduction = True
        self.enable_chart = True

    def save(self):
        with open(os.path.join(self.working_dir, "data_analyzer.pkl"), "wb") as f:
            pickle.dump(self._state, f)

    def load(self):
        try:
            with open(os.path.join(self.working_dir, "data_analyzer.pkl"), "rb") as f:
                self._state = pickle.load(f)
        except FileNotFoundError:
            pass

    async def invoke(self, task):
        inputs = task.inputs

        self.load()
        report = self._state.get("report")
        if not report or not report.done:
            result = await Runner.run_agent("outline_generator", inputs)
            report = result["report"]
            self._state["report"] = report
            self._state["cache"] = result["cache"]
            self.save()

            start_index = 0
            for idx, section in enumerate(report.sections):
                if idx < start_index:
                    continue
                section_input_data = inputs.copy()
                section_input_data['section_outline'] = section.outline
                section_input_data["collected_data_list"].extend(self._state["cache"])
                logger.info(f"[Phase1] Section {idx + 1}/{len(report.sections)} start")
                result = await Runner.run_agent("section_writer", section_input_data)
                self._state["cache"].extend(result["cache"])

                final_section = result["final_section"]
                logger.debug(f"[Phase1] Final section length={len(final_section)}")

                section.set_content(final_section)

                # Save global progress after each section to resume later
                # Update in-memory progress pointer
                section_index_done = idx + 1
                logger.info(
                    f"[Phase1] Section {idx + 1} done, checkpoint saved (section_index={section_index_done})")

            report.done = True
            self._state["report"] = report
            self._state["cache"].extend(inputs["collected_data_list"])
            self.save()

        collect_data_list = self._state["cache"]
        await self._post_process(report, inputs['analysis_result_list'], collect_data_list)

        return

    async def _post_process(self, report, analysis_result_list, collect_data_list):
        report = await self._replace_image_path(report, self.enable_chart, analysis_result_list)
        if self.add_introduction:
            report = await self._add_abstract(report)
        report = await self._add_title(report)
        report = await self._add_cover_page(report, collect_data_list)
        report = await self._add_reference(report, collect_data_list)

        working_dir = self.config["working_dir"]
        def clean_filename(filename, replace_with="_"):
            illegal_chars = r'[<>:"/\\|?*]'
            filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
            filename = re.sub(illegal_chars, replace_with, filename)
            filename = filename.strip('.')
            if not filename:
                filename = "untitled"
            if len(filename) > 255:
                filename = filename[:255]
            return filename

        file_name = clean_filename(report.title)
        md_path = os.path.join(working_dir, f'{file_name}.md')
        docx_path = os.path.join(working_dir, f'{report.title}.docx')
        content = report.content
        content = content.replace("```markdown", "").replace("```", "")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(content)
        media_dir = os.path.join(working_dir, "media")
        reference_doc = self.config['reference_doc_path']
        pandoc_cmd = [
            "pandoc",
            md_path,
            "-o",
            docx_path,
            "--standalone",
            "--toc",
            "--toc-depth=3",
            f"--resource-path={working_dir}",
            f"--reference-doc={reference_doc}"
        ]
        if os.path.exists(media_dir):
            pandoc_cmd.append(f"--extract-media={media_dir}")
        print(" ".join(pandoc_cmd))
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        subprocess.run(pandoc_cmd, check=True, capture_output=True, text=True, encoding='utf-8', env=env)

        pdf_path = docx_path.replace(".docx", ".pdf")
        try:
            docx2pdf.convert(docx_path, pdf_path)
        except Exception as e:
            logger.error(f"Failed to convert docx to pdf: {e}", exc_info=True)

    async def _add_reference(self, report, collect_data_list):
        collect_data_list = [d for d in collect_data_list if not isinstance(d, DeepSearchResult)]
        all_data = []

        url_title = {}
        for d in [d for d in collect_data_list if isinstance(d, SearchResult)]:
            url_title[d.link] = d.name

        for item in collect_data_list:
            # TODO: directly set these keys in ToolResult
            name = item.name + '\n' + item.description  # used for index
            content = item.source  # used for display citation
            # content = item.name + '\n' + item.link  # used for display citation

            if isinstance(item, ClickResult):
                url = item.link
                title = url_title.get(url, None)
                if not title:
                    title = item.name
                content = f"{title}\n{url}"

            if content not in [ii['content'] for ii in all_data]:
                all_data.append({
                    'name': name,
                    'content': content
                })
        logger.info(f"Total data for reference: {len(all_data)}")

        total_corpus = [item['name'] for item in all_data]
        # index = IndexBuilder(embedding_model=self., working_dir=runtime.get("working_dir"))
        await self.index._build_index(total_corpus, batch_size=8)

        total_cited_dict = {}
        for section in report.sections:
            # Optional: log section length
            try:
                logger.debug(f"Processing section, content length={len(section.content)}")
            except Exception:
                pass
            section_new_content = []
            for p_paragraph in section._content:
                content = p_paragraph
                print(content)
                # Locate citation placeholders
                match_list = re.findall(r'\[[Ss]ource[：:]\s*(.*?)\]', content)
                logger.debug(f"Match list: {match_list}")
                for match_item in match_list:
                    # Use BM25/embedding search
                    search_result = await self.index.search(match_item, top_k=5)
                    score_list = [item['score'] for item in search_result]
                    id_list = [item['id'] for item in search_result]  # Get actual data indices
                    logger.debug(f"Score list: {score_list}")
                    logger.debug(f"ID list: {id_list}")
                    print(search_result)
                    # Sort by score (descending) and get corresponding indices
                    sorted_idx = np.argsort(score_list)[::-1]
                    score_list = np.array(score_list)
                    score_list = np.exp(score_list) / np.sum(np.exp(score_list))

                    cite_list = []
                    for pos in sorted_idx:
                        pos = int(pos)
                        actual_idx = id_list[pos]  # Get the actual data index
                        if score_list[pos] > 0.2 and len(cite_list) < 5:
                            cite_list.append(actual_idx)
                    if len(cite_list) == 0:
                        # If no item meets threshold, use the top result
                        cite_list.append(id_list[sorted_idx[0]])
                    new_cite_list = []
                    for idx in cite_list:
                        if idx not in total_cited_dict:
                            total_cited_dict[idx] = len(total_cited_dict) + 1
                    new_cite_list = [total_cited_dict[idx] for idx in cite_list]
                    # Build the regex for replacement
                    pattern_to_replace = r'\[[Ss]ource[：:]\s*' + re.escape(match_item) + r'\]'
                    content = re.sub(pattern_to_replace, f'[{",".join([str(item) for item in new_cite_list])}]',
                                     content)

                section_new_content.append(content)
            section._content = section_new_content

        reference_str = "## Reference Data Sources\n\n"
        for old_index, new_index in total_cited_dict.items():
            content = all_data[old_index]['content']
            content = content.replace("\n", " ").replace("[PDF]", "")
            reference_str += f"{new_index}. {content}\n"
        new_section = Section('Reference Data Sources', reference_str)
        new_section.set_content(reference_str)
        report.sections.append(new_section)
        return report

    async def _add_cover_page(self, report, collect_data_list):
        pipeline_type = self.config["target_type"]
        if pipeline_type != 'company':
            return report
        stock_code = self.config['stock_code']
        if stock_code == "":
            return report

        output_str = "\n\n## Company Fundamentals\n\n"
        # Three statements + shareholder profile
        collect_data_list = [d for d in collect_data_list if not isinstance(d, DeepSearchResult)]
        table_configs = [
            ("Income statement", "Income Statement"),
            ("Balance sheet", "Balance Sheet"),
            ("Cash-flow statement", "Cash-Flow Statement"),
            ("Shareholding structure", "Shareholder Structure"),
        ]
        for keyword, display_name in table_configs:
            target_item_list = [item for item in collect_data_list if keyword in item.name and stock_code in item.name]
            if len(target_item_list) == 0:
                print(f"No {display_name} data found")
                continue
            else:
                table_data = target_item_list[0].data
                if table_data is None:
                    print(f"{display_name} data is empty, skip formatting")
                    continue

                if keyword in ["Income statement", "Balance sheet", "Cash-flow statement"]:
                    if 'Category' in table_data.columns:
                        table_data.rename(columns={'Category': 'Line item (RMB mn)'}, inplace=True)
                prompt = self.TABLE_BEAUTIFY_PROMPT.format(table_name=display_name,
                                                           table_data=table_data.to_markdown(index=False))
                response = await self.llm_client.ainvoke(
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                table_string = "\n".join([line for line in response.content.split("\n") if line.strip() != ""])

                output_str += f'\n\n### {display_name}\n\n'
                output_str += table_string

                output_str += '\n\n'

        # Render stock-price chart
        try:
            logger.info("Rendering stock-price chart for cover page")
            target_item_list = [item for item in collect_data_list if
                                'candlestick' in item.name.lower() and stock_code in item.name]
            if len(target_item_list) != 0:
                kline_data = target_item_list[0].data
                if kline_data is None:
                    logger.warning("Candlestick data is empty; skip price visualization")
                else:
                    if isinstance(kline_data, list) and len(kline_data) == 1:
                        kline_data = kline_data[0]
                    if 'date' not in kline_data.columns:
                        if '\u65e5\u671f' in kline_data.columns:
                            kline_data.rename(columns={'\u65e5\u671f': 'date'}, inplace=True)
                        if '\u6536\u76d8' in kline_data.columns:
                            kline_data.rename(columns={'\u6536\u76d8': 'close'}, inplace=True)
                    fig_path = draw_kline_chart(kline_data, self.config["working_dir"])
                    output_str += f'\n\n### Share Price Trend\n\n'
                    output_str += f'![Trailing price performance]({fig_path})\n\n'
        except Exception as e:
            logger.error(f"Failed to draw price trend: {e}", exc_info=True)
            pass

        first_section = Section('Company Fundamentals', output_str)
        first_section.set_content(output_str)
        report.sections = [first_section] + report.sections

        return report

    async def _add_title(self, report):
        new_title = await self.llm_client.ainvoke(
            messages=[
                HumanMessage(content=self.TITLE_PROMPT.format(
                    target_language=self.target_language,
                    report_content=report.content
                ))
            ]
        )
        new_title = new_title.content.replace("#", "").strip()
        report._content = f"# {new_title}\n\n"
        return report

    async def _add_abstract(self, report):
        response_content = await self.llm_client.ainvoke(
            messages=[
                {
                    'role': 'user',
                    'content': self.ABSTRACT_PROMPT.format(
                        target_language=self.target_language,
                        report_content=report.content
                    )
                }
            ])
        response_content = extract_markdown(response_content.content)
        report.abstract = response_content

        return report

    async def _replace_image_path(self, report, enable_chart, analysis_result_list):
        if not enable_chart:
            for section in report.sections:
                section_new_content = []
                for p_paragraph in section._content:
                    # Replace @import.* with empty string
                    p_paragraph = re.sub(r'@import.*', '', p_paragraph, flags=re.DOTALL)
                    section_new_content.append(p_paragraph)
                section._content = section_new_content
            return report

        def remove_suffix(name: str):
            return name.replace(".png", "").replace(".jpg", "").replace(".jpeg", "").replace(".md", "")

        def is_image_file(name: str):
            return name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".md")

        all_analysis_result = analysis_result_list
        img_captions = []
        img_paths = []
        for analysis_result in all_analysis_result:
            short2long = {}
            img_dicts = {}  # caption: abs_path
            chart_name_mapping = analysis_result.chart_name_mapping
            for long_name, short_name in chart_name_mapping.items():
                short2long[remove_suffix(short_name)] = remove_suffix(long_name)
            image_save_dir = analysis_result.image_save_dir
            for image_name in os.listdir(image_save_dir):
                if is_image_file(image_name):
                    img_path = os.path.join(image_save_dir, image_name)
                    img_name = remove_suffix(image_name)
                    long_image_name = short2long.get(img_name, "")
                    if long_image_name != "":
                        img_dicts[long_image_name] = img_path
            img_captions.extend(list(img_dicts.keys()))
            img_paths.extend(list(img_dicts.values()))
        if len(img_captions) == 0:
            logger.warning("No image captions found, skip image path replacement")
            return report
        logger.info(f"Building index for {len(img_captions)} images")

        await self.index._build_index(img_captions, batch_size=8)

        used_img_list = []
        figure_idx = 1
        for section in report.sections:
            section_new_content = []
            for p_paragraph in section._content:
                match = re.findall(r'@import.*', p_paragraph, flags=re.DOTALL)
                try:
                    logger.debug(f"Section image placeholders: {len(match)}")
                except Exception:
                    pass
                if match and len(match) > 0:
                    for img_name in match:
                        # img_name is the short placeholder string
                        most_similar_idx = (await self.index.search(img_name))[0]['id']
                        detect_img_name = img_captions[most_similar_idx]
                        detect_img_path = img_paths[most_similar_idx]

                        if len(img_captions) == 1:
                            # No images left to map
                            logger.warning("Available images are exhausted; stop replacing images.")
                            # directly delete the image placeholder
                            p_paragraph = p_paragraph.replace(img_name, "")
                            continue
                        del img_captions[most_similar_idx]
                        del img_paths[most_similar_idx]
                        # Rebuild the index after consuming this caption
                        await self.index._build_index(img_captions)

                        new_string = get_md_img(detect_img_path, remove_suffix(os.path.basename(detect_img_path)),
                                                figure_idx)
                        figure_idx += 1
                        used_img_list.append(detect_img_name)
                        p_paragraph = p_paragraph.replace(img_name, new_string)

                section_new_content.append(p_paragraph)
            section._content = section_new_content
        return report