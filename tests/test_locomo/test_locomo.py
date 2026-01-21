import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import uuid
from filelock import FileLock
from openjiuwen.core.utils.llm.model_utils.default_model import RequestChatModel
from pandas.core.window.doc import kwargs_scipy
from sqlalchemy.ext.asyncio import create_async_engine
from tqdm import tqdm
import sys
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

# 加载环境变量
load_dotenv(dotenv_path=r"./tests/test_locomo/.env")

# 导入核心模块
from openjiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
from openjiuwen.core.memory.store.message import create_tables
from openjiuwen.core.common.logging import logger
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.memory.config.config import SysMemConfig
from openjiuwen.core.memory.embed_models.api import APIEmbedModel
from openjiuwen.core.memory.engine.memory_engine import MemoryEngine
from openjiuwen.core.memory.store.impl.dbm_kv_store import DbmKVStore
from openjiuwen.core.memory.store.impl.default_db_store import DefaultDbStore
from openjiuwen.core.memory.store.impl.chroma_semantic_store import ChromaSemanticStore
from openjiuwen.core.utils.llm.base import BaseModelClient, BaseModelInfo
from openjiuwen.core.utils.llm.messages import AIMessage, BaseMessage, HumanMessage
from tests.test_locomo.test_locomo_prompt import validation_prompt, ANSWER_PROMPT

# 配置项
API_BASE = os.getenv("API_BASE", "mock://api.openai.com/v1")
API_KEY = os.getenv("API_KEY", "sk-fake")
MODEL_NAME = os.getenv("MODEL_NAME", "")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "")
VALIDATE_API_BASE = os.getenv("VALIDATE_API_BASE", "mock://api.openai.com/v1")
VALIDATE_API_KEY = os.getenv("VALIDATE_API_KEY", "sk-fake")
VALIDATE_MODEL_NAME = os.getenv("VALIDATE_MODEL_NAME", "")
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.05"))
RERANK_API_BASE = os.getenv("RERANK_API_BASE", "mock://api.openai.com/v1")
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "sk-fake")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "fake-model")
RERANK_MAX_RETRIES = int(os.getenv("RERANK_MAX_RETRIES", 1))
RERANK_TIMEOUT = int(os.getenv("RERANK_TIMEOUT", 30))
os.environ.setdefault("LLM_SSL_VERIFY", "false")
data_path = os.getenv("INPUT_DATA_FILE", "")
response_path = os.getenv("RESPONSE_PATH", "./test_response")
result_path = os.getenv("RESULT_PATH", "./test_result.json")

# 全局计数器
processed_conversations = 0
progress_bar = None


import json
from typing import List, Dict, Any
from datetime import datetime
from openjiuwen.core.common.logging import logger
from openjiuwen.core.common.security.user_config import UserConfig
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.messages import AIMessage



class RequestRerankerModel(RequestChatModel):
    """
    Abstract Request Reranker

    """
    api_endpoint_suffix = "/services/rerank/text-rerank/text-rerank"

    def _parse_response(self, model_name: str, response_data: Dict) -> list[float]:
        return [r["relevance_score"] for r in response_data["output"]["results"]]

    def rerank(self, query: str, documents: list[str], model_name: str = None):
        response = self.invoke(
            model_name=model_name or self.model_name,
            messages=[{"query": query, "documents": documents}],
        )
        return response

    async def arerank(self, query: str, documents: list[str], model_name: str = None):
        response = await self.ainvoke(
            model_name=model_name or self.model_name,
            messages=[{"query": query, "documents": documents}],
        )
        return response


class APIRequestReranker(RequestRerankerModel):

    def __init__(self, api_key: str = RERANK_API_KEY, api_base: str = RERANK_API_KEY, max_retries: int = RERANK_MAX_RETRIES, timeout: int = RERANK_TIMEOUT, **kwargs):
        super().__init__(api_base=api_base, api_key=api_key, max_retries=max_retries, timeout=timeout, **kwargs)
        self.model_name = RERANK_MODEL_NAME

    def _request_params(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                        **kwargs: Any) -> Dict:
        documents = messages[0]["documents"]
        query = messages[0]["query"]
        params = {
            "model": model_name,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "top_n": len(documents),
                "return_documents": False,
            },
            **kwargs
        }

        if tools:
            params["tools"] = tools

        if UserConfig.is_sensitive():
            logger.info("Before request chat model, request params is ready.")
        else:
            logger.info(f"Before request chat model, request params is ready. "
                        f"params: {params}, timeout: {self.timeout}")

        return params



class ContextFilter:
    """
    Reranks retrieved memory entries and filters them based on reranking scores, in order to make more
    efficient usage of the context.
    """

    def __init__(self):
        self.reranker = APIRequestReranker()
        self.min_similarity_score = 0.25
        self.min_rerank_score = 0.5
        self.max_ratio = 0.5

    async def afilter(self, query: str, documents: list[dict]):
        documents = [d for d in documents if d["score"] >= self.min_similarity_score]
        scores = await self.reranker.arerank(query=query, documents=[d["mem"] for d in documents])
        document_scores =  sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        top_score = document_scores[0][1]
        min_score = max(top_score * self.max_ratio, self.min_rerank_score)
        document_scores = [(doc, score) for doc, score in document_scores if score >= min_score]
        return document_scores


class ConversationProcessor:
    """单个Conversation的处理器（完全独立的MemoryEngine实例）"""
    def __init__(self, conv_id: int):
        self.conv_id = conv_id
        self.memory_engine: MemoryEngine = None  # 私有化实例
        self.llm_base: BaseModelClient = None
        self.resource_dir = None
        self.init_done = False
        self.init_lock = asyncio.Lock()
        self.global_chroma_lock = FileLock(f".chroma_init_lock_{conv_id}.lock")
        self.context_filter = ContextFilter()


    async def init_resources(self):
        """初始化当前conversation的独立资源（核心：不使用全局单例）"""
        async with self.init_lock:
            if self.init_done:
                return
            with self.global_chroma_lock:
                # 1. 创建独立的资源目录
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.resource_dir = os.path.join(project_root, f'resources_conv_{self.conv_id}')
                os.makedirs(self.resource_dir, exist_ok=True)
                logger.info(f"📁 Conversation {self.conv_id} 资源目录: {self.resource_dir}")

                # 2. 初始化Embed模型
                embed_model = APIEmbedModel(
                    base_url=os.getenv("EMBED_API_BASE"),
                    model_name=os.getenv("EMBED_MODEL_NAME"),
                    api_key=os.getenv("EMBED_API_KEY"),
                    timeout=int(os.getenv("EMBED_TIMEOUT")),
                    max_retries=int(os.getenv("EMBED_MAX_RETRIES")),
                )

                # 3. 创建独立的存储实例（不注册到全局）
                # Chroma语义存储
                semantic_store = ChromaSemanticStore(self.resource_dir, embed_model)
                # KV存储
                kv_db_path = os.path.join(self.resource_dir, f'dbmstore_{self.conv_id}')
                kv_store = DbmKVStore(kv_db_path)
                # SQL存储
                utc_now = datetime.now(timezone.utc)
                time_str = utc_now.strftime("%Y%m%d%H%M%S")
                uuid_str = uuid.uuid4().hex[:6]
                db_path = Path(f"{self.resource_dir}/test_sql_db_{time_str}_{uuid_str}_{self.conv_id}.db").resolve()
                db_store = DefaultDbStore(create_async_engine(f"sqlite+aiosqlite:///{db_path}"))
                await create_tables(db_store)
                # 4. 手动创建MemoryEngine实例（关键：不使用全局单例）
                # 重置全局状态（防止残留）
                if hasattr(MemoryEngine, '_instance'):
                    delattr(MemoryEngine, '_instance')

                # 创建新实例并手动赋值存储
                self.memory_engine = MemoryEngine(SysMemConfig(), kv_store, semantic_store, db_store)
                # 手动设置存储（绕过全局register_store）
                self.memory_engine._kv_store_instance = kv_store
                self.memory_engine._db_store_instance = db_store
                self.memory_engine._semantic_store_instance = semantic_store

                # 6. 设置LLM配置（绑定到当前实例）
                self.memory_engine.set_group_llm_config(
                    "default",
                    ModelConfig(
                        MODEL_PROVIDER,
                        BaseModelInfo(
                            api_key=API_KEY,
                            api_base=API_BASE,
                            model=MODEL_NAME,
                            temperature=TEMPERATURE
                        )
                    )
                )

                # 7. 初始化LLM客户端
                self.llm_base = ModelFactory().get_model(MODEL_PROVIDER, API_KEY, API_BASE, temperature=TEMPERATURE or 0.05)

                self.init_done = True
                logger.info(f"✅ Conversation {self.conv_id} 资源初始化完成（独立MemoryEngine）")

    async def add_memory(self, user_id: str, app_id: str, messages: list[BaseMessage], timestamp: datetime, session_id: str, retries=3) -> None:
        """异步添加内存（使用私有MemoryEngine）"""
        for retry in range(retries):
            try:
                # 直接使用当前实例的memory_engine，而非全局
                await self.memory_engine.add_conversation_messages(
                    user_id=user_id,
                    group_id=app_id,
                    messages=messages,
                    timestamp=timestamp,
                    session_id=session_id
                )
                logger.debug(f"📝 Conversation {self.conv_id} 成功添加 {len(messages)} 条内存")
                break
            except Exception as e:
                if retry < retries - 1:
                    await asyncio.sleep(30)
                    logger.warning(f"⚠️ Conversation {self.conv_id} 添加内存失败，重试 {retry+1}/3: {str(e)}")
                    continue
                else:
                    logger.error(f"❌ Conversation {self.conv_id} 添加内存失败: {str(e)}")
                    raise e

    async def process_locomo_data(self, speaker_a: str, speaker_b: str, data: dict):
        """异步处理对话数据（写入私有目录）"""
        conversation_data = data['conversation']
        user_id = "default"
        app_id = "default"

        # for key in conversation_data.keys():
        for key in list(conversation_data.keys())[:8]:
            messages = []
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            session_id = f"{self.conv_id}-{str(key).replace('_', '-')}"
            data_time_key = key + "_date_time"
            timestamp = conversation_data[data_time_key]
            timestamp = datetime.strptime(timestamp, "%I:%M %p on %d %B, %Y")
            chats = conversation_data[key]

            for chat in chats:
                message = f"{chat['text']}"
                if chat.get('blip_caption'):
                    message += f"(The conversation is accompanied by an image, and the description of the image is '{chat['blip_caption']}')"
                message = message

                if chat['speaker'] == speaker_a:
                    message = HumanMessage(content=message, name=chat['speaker'])
                elif chat['speaker'] == speaker_b:
                    message = AIMessage(content=message, name=chat['speaker'])

                messages.append(message)

                if len(messages) == 4:
                    await self.add_memory(user_id, app_id, messages, timestamp, session_id)
                    messages = []
                    timestamp += timedelta(seconds=1)

            if messages:
                await self.add_memory(user_id, app_id, messages, timestamp, session_id)

    async def llm_answer(self, user_id: str, app_id: str, query: str, retrieve_num: int = 50) -> str:
        """异步调用LLM生成回答（读取私有目录数据）"""
        # 使用私有memory_engine检索
        user_memory = await self.memory_engine.search_user_mem(
            user_id=user_id,
            group_id=app_id,
            query=query,
            num=retrieve_num
        )

        filtered_memory = await self.context_filter.afilter(query=query, documents=user_memory)
        user_memory = [m for m, _ in filtered_memory]

        memory_msg_list = []
        for memory in user_memory:
            date_str = datetime.fromtimestamp(memory["timestamp"]).strftime("%d %B %Y")
            memory_context_entry = f"${date_str}: ${memory['mem']}"
            memory_msg_list.append((memory["timestamp"], memory_context_entry))
            logger.warning(memory_context_entry)

        memory_msg_list = [m for _, m in sorted(memory_msg_list, key=lambda x: x[0], reverse=False)]

        memory_msg = "\n".join(memory_msg_list)

        llm_prompt = ANSWER_PROMPT.substitute(question=query, memory=memory_msg)
        logger.debug(f"📝 Conversation {self.conv_id} LLM Prompt: {llm_prompt[:100]}...")

        message = HumanMessage(content=llm_prompt)
        response = await self.llm_base.ainvoke(model_name=MODEL_NAME, messages=[message])
        return response.content

    async def generate_response(self, qa_data: list, user_name1: str, user_name2: str):
        """异步生成QA响应"""
        response_path_qa = f"{response_path}{self.conv_id}.json"
        # 清空文件（避免追加旧数据）
        open(response_path_qa, 'w', encoding='utf-8').close()

        user_id = "default"
        app_id = "default"

        for idx, qa_enum in enumerate(qa_data):
            category = qa_enum['category']
            if category > 4:
                continue

            # question = qa_enum['question'].replace(user_name1, 'user').replace(user_name2, 'assistant')
            # answer = str(qa_enum['answer']).replace(user_name1, 'user').replace(user_name2, 'assistant')
            question = qa_enum['question']
            answer = str(qa_enum['answer'])

            for retry in range(3):
                try:
                    response = await self.llm_answer(user_id, app_id, question)
                    break
                except Exception as e:
                    if retry < 2:
                        await asyncio.sleep(30)
                        logger.warning(f"⚠️ Conversation {self.conv_id} QA生成失败，重试 {retry+1}/3: {str(e)}")
                        continue
                    else:
                        logger.error(f"❌ Conversation {self.conv_id} QA生成失败: {str(e)}")
                        raise e

            data_dict = {
                "question": question,
                "answer": answer,
                "response": response,
                "category": category
            }
            with open(response_path_qa, 'a', encoding='utf-8') as file:
                json.dump(data_dict, file, ensure_ascii=False)
                file.write('\n')

    async def llm_service(self, user_message: str) -> AIMessage:
        """异步调用LLM进行验证"""
        messages = [
            BaseMessage(**{
                "role": "system",
                "content": "You are an expert grader that determines if answers to questions match a gold standard answer"
            }),
            BaseMessage(**{
                "role": "user",
                "content": user_message
            })
        ]
        for retry in range(3):
            try:
                response = await self.llm_base.ainvoke(VALIDATE_MODEL_NAME, messages)
                break
            except Exception as e:
                if retry < 2:
                    await asyncio.sleep(30)
                    logger.warning(f"⚠️ Conversation {self.conv_id} 判断失败，重试 {retry+1}/3: {str(e)}")
                    continue
                else:
                    logger.error(f"❌ Conversation {self.conv_id} 判断失败: {str(e)}")
                    raise e
        return response

    async def validate_locomo_data(self, speaker_a: str, speaker_b: str):
        """异步验证回答正确性"""
        response_path_enum = f"{response_path}{self.conv_id}.json"
        qa_result_dict = defaultdict(int)
        correct_result_dict = defaultdict(int)

        with open(response_path_enum, 'r', encoding='utf-8') as f:
            for qa_enum_str in f:
                if not qa_enum_str.strip():
                    continue

                qa_enum = json.loads(qa_enum_str)
                # question = qa_enum['question'].replace(speaker_a, 'user').replace(speaker_b, 'assistant')
                # gold_answer = str(qa_enum['answer']).replace(speaker_a, 'user').replace(speaker_b, 'assistant')
                question = qa_enum['question']
                gold_answer = str(qa_enum['answer'])
                category = qa_enum['category']
                response = qa_enum['response']

                qa_result_dict[category] += 1
                if str(response).strip() == "":
                    continue

                test_prompt = validation_prompt.format(
                    question=question,
                    gold_answer=gold_answer,
                    response=response
                )

                result = await self.llm_service(test_prompt)
                logger.info(f"🔍 Conversation {self.conv_id} 验证结果: {result.content[:50]}...")

                if "CORRECT" in result.content and "WRONG" not in result.content:
                    correct_result_dict[category] += 1

        accuracy_per_class = {}
        for cls in [1, 2, 3, 4]:
            total = qa_result_dict[cls]
            correct = correct_result_dict[cls]
            accuracy = correct / total if total != 0 else 0.0
            accuracy_per_class[cls] = round(accuracy, 4)

        with open(result_path, 'a', encoding='utf-8') as file:
            total_result = {
                "conversation_id": self.conv_id,
                "accuracy_per_class": accuracy_per_class,
                "qa_result_dict": qa_result_dict,
                "correct_result_dict": correct_result_dict
            }
            json.dump(total_result, file, ensure_ascii=False)
            file.write('\n')

    async def process_full(self, data_enum: dict, build_memory=True):
        """处理单个conversation的完整流程"""
        global processed_conversations, progress_bar

        try:
            # 1. 初始化独立资源（关键：创建私有MemoryEngine）
            await self.init_resources()

            # 2. 获取speaker信息
            speaker_a = data_enum['conversation']['speaker_a']
            speaker_b = data_enum['conversation']['speaker_b']

            # 3. 处理对话数据（写入私有目录）
            logger.info(f"🚀 开始处理 Conversation {self.conv_id}")
            if build_memory:
                await self.process_locomo_data(speaker_a, speaker_b, data_enum)

            # 4. 生成QA响应（读取私有目录数据）
            self.memory_engine.set_group_llm_config(
                    "default",
                    ModelConfig(
                        MODEL_PROVIDER,
                        BaseModelInfo(
                            api_key=VALIDATE_API_KEY,
                            api_base=VALIDATE_API_BASE,
                            model=VALIDATE_MODEL_NAME,
                            temperature=TEMPERATURE
                        )
                    )
                )
            self.llm_base = ModelFactory().get_model(MODEL_PROVIDER, VALIDATE_API_KEY, VALIDATE_API_BASE, temperature=TEMPERATURE)
            await self.generate_response(data_enum['qa'], speaker_a, speaker_b)

            # 5. 验证结果

            await self.validate_locomo_data(speaker_a, speaker_b)

            # 验证：检查当前目录是否有数据写入
            if os.path.exists(self.resource_dir):
                file_count = len(os.listdir(self.resource_dir))
                logger.info(f"📊 Conversation {self.conv_id} 目录文件数: {file_count}")

            logger.info(f"✅ 完成处理 Conversation {self.conv_id}")

        except Exception as e:
            logger.error(f"❌ Conversation {self.conv_id} 处理失败: {str(e)}", exc_info=True)
            raise
        finally:
            # 更新进度
            processed_conversations += 1
            if progress_bar:
                progress_bar.update(1)

class TESTLOCOMO:
    @staticmethod
    def get_locomo_data(data_path: str):
        """加载数据"""
        with open(data_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
            return data

    @staticmethod
    def overall_compute(result_path_enum: str) -> None:
        """计算总体结果"""
        key_mapping = {
            "4": "Single-hop",
            "1": "Multi-hop",
            "2": "Temporal reasoning",
            "3": "Open domain"
        }
        total_qa = defaultdict(int)
        total_correct = defaultdict(int)

        with open(result_path_enum, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                if "qa_result_dict" not in data:
                    continue

                for k, v in data["qa_result_dict"].items():
                    total_qa[key_mapping.get(str(k), str(k))] += v
                for k, v in data["correct_result_dict"].items():
                    total_correct[key_mapping.get(str(k), str(k))] += v

        total_accuracy = {}
        total_correct_all = 0
        total_samples = 0

        for key in key_mapping.values():
            correct = total_correct.get(key, 0)
            total = total_qa.get(key, 0)
            total_accuracy[key] = round(correct / total if total != 0 else 0, 4)
            total_correct_all += correct
            total_samples += total

        total_accuracy["overall"] = round(
            total_correct_all / total_samples if total_samples != 0 else 0,
            4
        )

        logger.info("📊 总样本量统计：%s", dict(total_qa))
        logger.info("📊 总正确量统计：%s", dict(total_correct))
        logger.info("📊 总准确率：%s", total_accuracy)

        with open(result_path_enum, "a", encoding="utf-8") as f:
            json.dump({
                "total_samples": dict(total_qa),
                "total_correct": dict(total_correct),
                "Overall": total_accuracy
            }, f, ensure_ascii=False, indent=4)
            f.write('\n')

async def main(args):
    global progress_bar
    logger.set_level(logging.INFO)

    # 清空结果文件（避免追加旧数据）
    open(result_path, 'w', encoding='utf-8').close()

    # 1. 加载数据
    test = TESTLOCOMO()
    data = test.get_locomo_data(args.data_path)
    total_convs = len(data)
    logger.info(f"📥 加载到 {total_convs} 个conversation")

    # 2. 初始化进度条
    progress_bar = tqdm(total=total_convs, desc="处理所有Conversation")

    # 3. 逐个创建处理器并执行（确保实例不被覆盖）
    #    注：这里改为逐个创建+立即执行，而非批量创建后gather，进一步避免单例冲突
    tasks = []
    for conv_id in args.conversation_ids:
        data_enum = data[conv_id]
        processor = ConversationProcessor(conv_id)
        # 添加到任务列表
        task = processor.process_full(data_enum, build_memory=args.build_memory)
        tasks.append(task)

    # 4. 执行所有任务（协程交替执行）
    await asyncio.gather(*tasks, return_exceptions=False)

    # 5. 计算总体结果
    test.overall_compute(result_path)

    # 6. 完成
    progress_bar.close()
    logger.info("🎉 所有Conversation处理完成！")
    logger.info("🔍 验证：每个conversation的数据已写入各自的resources_conv_*目录")

if __name__ == '__main__':
    # Windows事件循环修复
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, default=data_path)
    parser.add_argument("-c", "--conversation-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument('-rr', '--rerank', action="store_true")
    parser.add_argument('-fc', '--filter-context', action="store_true")
    parser.add_argument('-lm', '--build-memory', action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args))