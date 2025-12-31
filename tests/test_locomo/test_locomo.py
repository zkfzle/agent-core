import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid
from sqlalchemy.ext.asyncio import create_async_engine
from tqdm import tqdm
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path=r"C:\Users\12975\Desktop\git_huawei\agent-core-zhao\tests\test_locomo\.env")
sys.path.append(r"C:\Users\12975\Desktop\git_huawei\agent-core-zhao")
from openjiuwen.core.common.logging import logger
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.memory.config.config import SysMemConfig
from openjiuwen.core.memory.embed_models.api import APIEmbedModel
from openjiuwen.core.memory.engine.memory_engine import MemoryEngine
from openjiuwen.core.memory.store.impl.dbm_kv_store import DbmKVStore
from openjiuwen.core.memory.store.impl.default_db_store import DefaultDbStore
from openjiuwen.core.memory.store.impl.milvus_semantic_store import MilvusSemanticStore
from openjiuwen.core.utils.llm.base import BaseModelInfo
from openjiuwen.core.utils.llm.messages import AIMessage, BaseMessage, HumanMessage
from openjiuwen.core.utils.llm.model_library.siliconflow import Siliconflow
from test_locomo_prompt import validation_prompt, ANSWER_PROMPT, CHAR_PROMPT

API_BASE = os.getenv("API_BASE", "mock://api.openai.com/v1")
API_KEY = os.getenv("API_KEY", "sk-fake")
MODEL_NAME = os.getenv("MODEL_NAME", "")
os.environ.setdefault("LLM_SSL_VERIFY", "false")
data_path = "D://data/locomo10.json"
response_path = "./test_response"
result_path = "../test_result.json"

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
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        resource_dir = os.path.join(project_root, 'resources')
        if not os.path.exists(resource_dir):
            os.makedirs(resource_dir)
        message_path = os.path.join(resource_dir, 'message_db')
        path = Path(message_path)
        kv_db_path = os.path.join(resource_dir, 'dbmstore')
        embed_model = APIEmbedModel(
            base_url=os.getenv("EMBED_API_BASE"),
            model_name=os.getenv("EMBED_MODEL_NAME"),
            api_key=os.getenv("EMBED_API_KEY"),
            timeout=int(os.getenv("EMBED_TIMEOUT")),
            max_retries=int(os.getenv("EMBED_MAX_RETRIES")),
        )
        semantic_store = MilvusSemanticStore(
            milvus_host=os.getenv("MILVUS_HOST"),
            milvus_port=os.getenv("MILVUS_PORT"),
            collection_name=os.getenv("MILVUS_COLLECTION_NAME"),
            embedding_dims=int(os.getenv("EMBEDDING_MODEL_DIMENTION", 1024)),
            embed_model=embed_model,
            token=os.getenv("MILVUS_TOKEN", None)
        )
        utc_now = datetime.now(timezone.utc)
        time_str = utc_now.strftime("%Y%m%d%H%M%S")
        uuid_str = uuid.uuid4().hex[:6]
        path = Path(f"{resource_dir}/test_sql_db_{time_str}_{uuid_str}.db").resolve()
        db_store = DefaultDbStore(create_async_engine(f"sqlite+aiosqlite:///{path}"))
        MemoryEngine.register_store(kv_store=DbmKVStore(kv_db_path), db_store=db_store, semantic_store=semantic_store)
        await MemoryEngine.create_mem_engine_instance(SysMemConfig())
        MemoryEngine.set_group_llm_config(MemoryEngine.get_mem_engine_instance(), "default", ModelConfig("siliconflow", BaseModelInfo(api_key=API_KEY, api_base=API_BASE, model=MODEL_NAME)))
        print("✅ Memory engine created")
    
    async def process_locomo_data(self, data: dict, idx: int):
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
                message = f"{chat['speaker']}: {chat['text']}. Time: {timestamp}."
                message = HumanMessage(content=message, name=chat['speaker'])
                messages.append(message)
                if len(messages) == 4:
                    await self.add_memory(user_id, app_id, messages, timestamp, session_id)
                    messages = []
            if messages:
                await self.add_memory(user_id, app_id, messages, timestamp, session_id)
    async def add_memory(self, user_id: str, app_id: str, messages: list[BaseMessage], timestamp: datetime, session_id: str, retries=3) -> None:
        for retry in range(retries):
            try:
                await self.memory_engine.add_conversation_messages(user_id=user_id, group_id=app_id, messages=messages, timestamp=timestamp, session_id=session_id)
            except Exception as e:
                if retry < retries - 1:
                    time.sleep(2)
                    continue
                else:
                    raise e

    async def llm_answer(self, user_id: str, app_id: str, query: str, user_name1: str = "user",
                         user_name2: str = "agent", retrieve_num: int = 5) -> str:
        user_memory = await self.memory_engine.search_user_mem(user_id=user_id, group_id=app_id, query=query,
                                           num=retrieve_num)
        llm_prompt = ANSWER_PROMPT.substitute(question=query, user_name1=user_name1, user_name2=user_name2,
                                              memory=user_memory)
        message = HumanMessage(content=llm_prompt)
        response = self.llm_base.invoke(model_name=MODEL_NAME, messages=[message])
        return response.content

    async def generate_response(self, qa_data: list, user_name1: str, user_name2: str, response_path_qa: str, retries=3) -> None:
        # Generate answer with memory
        user_id = "default"
        app_id = "default"
        agent_id = "default"
        for idx, qa_enum in enumerate(tqdm(qa_data, desc="Processing QA")):
            category = qa_enum['category']
            if category > 4:
                continue
            question = qa_enum['question']
            answer = qa_enum['answer']
            for retry in range(retries):
                try:
                    response = await self.llm_answer(user_name1=user_name1, user_name2=user_name2, user_id=user_id,
                                        app_id=app_id, query=question, retrieve_num=10)
                    data_dict = {"question": question, "answer": answer, "response": response, "category": category}
                    with open(response_path_qa, 'a', encoding='utf-8') as file:
                        json.dump(data_dict, file, ensure_ascii=False)
                        file.write('\n')
                    break
                except Exception as e:
                    if retry < retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        raise e


    def validate_locomo_data(self, response_path_enum: str) -> None:
        # Verify whether the agent's response is correct through LLM.
        qa_result_dict = defaultdict(int)
        correct_result_dict = defaultdict(int)
        with open(response_path_enum, 'r', encoding='utf-8') as f:
            # Validation
            for qa_enum in tqdm(f, desc="Processing validation"):
                if qa_enum.strip():
                    qa_enum = json.loads(qa_enum)
                    question = qa_enum['question']
                    gold_answer = qa_enum['answer']
                    category = qa_enum['category']
                    response = qa_enum['response']
                    qa_result_dict[category] += 1
                    if str(response).strip() == "":
                        continue
                    test_prompt = validation_prompt.format(question=question, gold_answer=gold_answer, response=response)
                    result = self.llm_service(test_prompt)
                    logger.info(f"result:{result}")
                    if "CORRECT" in result.content and "WRONG" not in result.content:
                        print("Correct!")
                        correct_result_dict[category] += 1
        logger.info(f"qa_result_dict:{qa_result_dict}, correct_result_dict:{correct_result_dict}")
        accuracy_per_class = {}
        for cls in [1, 2, 3, 4]:
            total = qa_result_dict[cls]
            correct = correct_result_dict[cls]
            accuracy = correct / total if total != 0 else 0.0  # 避免除零
            accuracy_per_class[cls] = round(accuracy, 4)
        with open(result_path, 'a', encoding='utf-8') as file:
            total_result = {"accuracy_per_class": accuracy_per_class, "qa_result_dict": qa_result_dict, "correct_result_dict": correct_result_dict}
            json.dump(total_result, file, ensure_ascii=False)
            file.write('\n')


    def llm_service(self, user_message: str) -> AIMessage:
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
        response = self.llm_base.invoke(MODEL_NAME, messages)
        return response

    @staticmethod
    def overall_compute(result_path_enum: str) -> None:
        # Compute overall results
        key_mapping = {
            "1": "Single-hop",
            "2": "Multi-hop",
            "3": "Temporal reasoning",
            "4": "Open domain"
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

        with open(result_path_enum,"a", encoding="utf-8") as f:
            json.dump({
                "total_samples": dict(total_qa),
                "total_correct": dict(total_correct),
                "Overall": total_accuracy
            }, f, ensure_ascii=False, indent=4)
            f.write('\n')


async def main():
    test = await TESTLOCOMO.create()
    data = test.get_locomo_data(data_path)
    for idx, data_enum in enumerate(tqdm(data, desc="Processing total data")):
        try:
            await test.memory_engine.delete_mem_by_user_id("default", "default") # delete memory
        except:
            pass
        speaker_a = data_enum['conversation']['speaker_a']
        speaker_b = data_enum['conversation']['speaker_b']
        response_path_enum = response_path + str(idx) + ".json"
        await test.process_locomo_data(data_enum, idx)
        await test.generate_response(data_enum['qa'], speaker_a, speaker_b, response_path_enum)
        test.validate_locomo_data(response_path_enum)
    test.overall_compute(result_path)
    
if __name__ == '__main__':
    asyncio.run(main())
    