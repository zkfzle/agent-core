import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
import os
import time
from sqlalchemy.ext.asyncio import create_async_engine
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"./tests/test_locomo/.env")

from openjiuwen.core.common.logging import logger
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.memory.config.config import SysMemConfig
from openjiuwen.core.memory.embed_models.api import APIEmbedModel
from openjiuwen.core.memory.engine.memory_engine import MemoryEngine
from openjiuwen.core.memory.store.impl.dbm_kv_store import DbmKVStore
from openjiuwen.core.memory.store.impl.default_db_store import DefaultDbStore
from openjiuwen.core.memory.store.impl.chroma_semantic_store import ChromaSemanticStore
from openjiuwen.core.utils.llm.base import BaseModelInfo
from openjiuwen.core.utils.llm.messages import AIMessage, BaseMessage, HumanMessage
from openjiuwen.core.utils.llm.model_library.siliconflow import Siliconflow
from test_locomo_prompt import validation_prompt, ANSWER_PROMPT

API_BASE = os.getenv("API_BASE", "mock://api.openai.com/v1")
API_KEY = os.getenv("API_KEY", "sk-fake")
MODEL_NAME = os.getenv("MODEL_NAME", "")
os.environ.setdefault("LLM_SSL_VERIFY", "false")
data_path = os.getenv("INPUT_DATA_FILE", "")
work_dir = os.getenv("WORK_DIR", "")
utc_now = datetime.now(timezone.utc)
time_str = utc_now.strftime("%Y%m%d%H%M%S")
cur_work_dir = os.path.join(work_dir, time_str)
response_path = os.path.join(cur_work_dir, "response.json")
result_path = os.path.join(cur_work_dir, "result.json")


class TESTLOCOMO():
    def __init__(self):
        self.memory_engine: MemoryEngine = None
        self.llm_base: Siliconflow = None

    @classmethod
    async def create(cls):
        instance = cls()
        await instance._create_memory_engine()
        instance.memory_engine = MemoryEngine.get_mem_engine_instance()  # 假设这是同步方法
        instance.llm_base = Siliconflow(API_KEY, API_BASE)
        return instance

    @staticmethod
    def get_locomo_data(data_path: str):
        with open(data_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
            return data

    async def _create_memory_engine(self):
        if not os.path.exists(cur_work_dir):
            os.makedirs(cur_work_dir)
        kv_db_path = os.path.join(cur_work_dir, 'dbmstore')
        embed_model = APIEmbedModel(
            base_url=os.getenv("EMBED_API_BASE"),
            model_name=os.getenv("EMBED_MODEL_NAME"),
            api_key=os.getenv("EMBED_API_KEY"),
            timeout=int(os.getenv("EMBED_TIMEOUT")),
            max_retries=int(os.getenv("EMBED_MAX_RETRIES")),
        )
        semantic_store = ChromaSemanticStore(cur_work_dir, embed_model)
        db_path = os.path.join(cur_work_dir, "test_sql_db.db")
        db_store = DefaultDbStore(create_async_engine(f"sqlite+aiosqlite:///{db_path}"))
        MemoryEngine.register_store(kv_store=DbmKVStore(kv_db_path), db_store=db_store, semantic_store=semantic_store)
        await MemoryEngine.create_mem_engine_instance(SysMemConfig())
        MemoryEngine.get_mem_engine_instance().set_group_llm_config("default", ModelConfig("siliconflow", BaseModelInfo(
            api_key=API_KEY, api_base=API_BASE, model=MODEL_NAME)))
        print("✅ Memory engine created")

    async def add_conversation_data(self, speaker_a: str, speaker_b: str, data: dict, idx: int):
        conversation_data = data['conversation']
        user_id = "default"
        app_id = "default"
        for key in conversation_data.keys():
            messages = []
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue
            session_id = str(idx) + "-" + str(key).replace("_", "-")
            data_time_key = key + "_date_time"
            timestamp = conversation_data[data_time_key]
            timestamp = datetime.strptime(timestamp, "%I:%M %p on %d %B, %Y")
            chats = conversation_data[key]
            for chat in tqdm(chats, desc=f"Processing {session_id}"):
                message = f"{chat['text']}"
                message = message.replace(speaker_a, '').replace(speaker_b, '')
                if chat['speaker'] == speaker_a:
                    message = HumanMessage(content=message, name=chat['speaker'])
                elif chat['speaker'] == speaker_b:
                    message = AIMessage(content=message, name=chat['speaker'])
                messages.append(message)
                if len(messages) == 4:
                    await self.add_memory(user_id, app_id, messages, timestamp, session_id)
                    messages = []
            if messages:
                await self.add_memory(user_id, app_id, messages, timestamp, session_id)

    async def add_memory(self, user_id: str, app_id: str, messages: list[BaseMessage], timestamp: datetime,
                         session_id: str, retries=3) -> None:
        for retry in range(retries):
            try:
                await self.memory_engine.add_conversation_messages(user_id=user_id, group_id=app_id, messages=messages,
                                                                   timestamp=timestamp, session_id=session_id)
                break
            except Exception as e:
                if retry < retries - 1:
                    time.sleep(2)
                    continue
                else:
                    raise e

    async def llm_answer(self, user_id: str, app_id: str, query: str, retrieve_num: int = 5) -> str:
        user_memory = await self.memory_engine.search_user_mem(user_id=user_id, group_id=app_id, query=query,
                                                               num=retrieve_num)
        memory_msg = ""
        for memory in user_memory:
            memory_msg += f"${memory['timestamp']}: ${memory['mem']}\n"
        llm_prompt = ANSWER_PROMPT.substitute(question=query, memory=memory_msg)
        logger.info(f"llm_prompt:{llm_prompt}")
        message = HumanMessage(content=llm_prompt)
        response = await self.llm_base.ainvoke(model_name=MODEL_NAME, messages=[message])
        return response.content

    async def test_question_and_answer(self, qa_data: list, user_name1: str, user_name2: str, response_path_qa: str,
                                       retries=3) -> None:
        # Generate answer with memory
        user_id = "default"
        app_id = "default"
        for idx, qa_enum in enumerate(tqdm(qa_data, desc="Processing QA")):
            category = qa_enum['category']
            if category > 4:
                continue
            question = qa_enum['question']
            question = question.replace(user_name1, 'user').replace(user_name2, 'assistant')
            answer = qa_enum['answer']
            answer = str(answer)
            answer = answer.replace(user_name1, 'user').replace(user_name2, 'assistant')
            for retry in range(retries):
                try:
                    response = await self.llm_answer(user_id=user_id,
                                                     app_id=app_id, query=question, retrieve_num=10)
                    data_dict = {"question": question, "answer": answer, "response": response, "category": category}
                    if str(response).strip() != "":
                        test_prompt = validation_prompt.format(question=question, gold_answer=answer,
                                                               response=response)
                        result = await self.llm_service(test_prompt)
                        logger.info(f"result: {result}")
                        data_dict["correct"] = "CORRECT" in result.content and "WRONG" not in result.content

                    with open(response_path_qa, 'a', encoding='utf-8') as file:
                        json.dump(data_dict, file, ensure_ascii=False)
                        file.write('\n')
                    break
                except Exception as e:
                    if retry < retries - 1:
                        logger.error(f"llm answer error, go to retry: {e}")
                        time.sleep(2)
                        continue
                    else:
                        raise e

    def statistics_conversation(self, response_path_enum: str) -> None:
        qa_result_dict = defaultdict(int)
        correct_result_dict = defaultdict(int)
        with open(response_path_enum, 'r', encoding='utf-8') as f:
            # Validation
            for qa_enum in tqdm(f, desc="Processing validation"):
                if qa_enum.strip():
                    qa_enum = json.loads(qa_enum)
                    category = qa_enum['category']
                    correct = qa_enum['correct']
                    qa_result_dict[category] += 1
                    if correct:
                        correct_result_dict[category] += 1
        logger.info(f"qa_result_dict:{qa_result_dict}, correct_result_dict:{correct_result_dict}")
        accuracy_per_class = {}
        for cls in [1, 2, 3, 4]:
            total = qa_result_dict[cls]
            correct = correct_result_dict[cls]
            accuracy = correct / total if total != 0 else 0.0  # 避免除零
            accuracy_per_class[cls] = round(accuracy, 4)
        with open(result_path, 'a', encoding='utf-8') as file:
            total_result = {"accuracy_per_class": accuracy_per_class, "qa_result_dict": qa_result_dict,
                            "correct_result_dict": correct_result_dict}
            json.dump(total_result, file, ensure_ascii=False)
            file.write('\n')

    async def llm_service(self, user_message: str) -> AIMessage:
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
        response = await self.llm_base.ainvoke(MODEL_NAME, messages)
        return response

    @staticmethod
    def overall_compute(result_path_enum: str) -> None:
        # Compute overall results
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

                mapped_acc = {
                    key_mapping[k]: v
                    for k, v in data["accuracy_per_class"].items()
                }
                print(mapped_acc)
                logger.info("单行准确率：", mapped_acc)
                for k, v in data["qa_result_dict"].items():
                    total_qa[key_mapping[k]] += v
                for k, v in data["correct_result_dict"].items():
                    total_correct[key_mapping[k]] += v

        total_accuracy = {}
        total_correct_all = 0
        total_samples = 0

        for key in key_mapping.values():
            correct = total_correct[key]
            total = total_qa[key]
            total_accuracy[key] = round(correct / total if total != 0 else 0, 4)
            total_correct_all += correct
            total_samples += total

        total_accuracy["overall"] = round(
            total_correct_all / total_samples if total_samples != 0 else 0,
            4
        )

        logger.info("总样本量统计：", dict(total_qa))
        logger.info("总正确量统计：", dict(total_correct))
        logger.info("总准确率：", total_accuracy)

        with open(result_path_enum, "a", encoding="utf-8") as f:
            json.dump({
                "total_samples": dict(total_qa),
                "total_correct": dict(total_correct),
                "Overall": total_accuracy
            }, f, ensure_ascii=False, indent=4)
            f.write('\n')


async def main():
    logger.set_level(logging.DEBUG)
    test = await TESTLOCOMO.create()
    data = test.get_locomo_data(data_path)
    for idx, data_enum in enumerate(tqdm(data, desc="Processing total data")):
        try:
            await test.memory_engine.delete_mem_by_user_id("default", "default")  # delete memory
        except:
            pass
        speaker_a = data_enum['conversation']['speaker_a']
        speaker_b = data_enum['conversation']['speaker_b']
        response_path_enum = response_path + str(idx) + ".json"
        await test.add_conversation_data(speaker_a, speaker_b, data_enum, idx)
        await test.test_question_and_answer(data_enum['qa'], speaker_a, speaker_b, response_path_enum)
        test.statistics_conversation(response_path_enum)
    test.overall_compute(result_path)


if __name__ == '__main__':
    asyncio.run(main())
