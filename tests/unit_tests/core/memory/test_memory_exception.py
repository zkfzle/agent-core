import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from openjiuwen.core.memory.long_term_memory import LongTermMemory
from openjiuwen.core.common.exception.errors import BaseError
from openjiuwen.core.common.exception.status_code import StatusCode

from openjiuwen.core.memory.store.base_kv_store import BaseKVStore
from openjiuwen.core.memory.store.base_semantic_store import BaseSemanticStore

@pytest.mark.asyncio
async def test_register_store_kv_store_none():
    mem = LongTermMemory()

    with pytest.raises(BaseError) as e:
        await mem.register_store(kv_store=None)

    err = e.value
    assert "kv_store is required" in err.message
    assert err.status == StatusCode.MEMORY_ENGINE_REGISTER_STORE_ERROR
    assert err.code == StatusCode.MEMORY_ENGINE_REGISTER_STORE_ERROR.code


@pytest.mark.asyncio
async def test_register_store_semantic_store_wrong_type():
    mem = LongTermMemory()

    fake_kv = MagicMock(spec=BaseKVStore)
    wrong_semantic = object()  # 不是 BaseSemanticStore

    with pytest.raises(BaseError) as e:
        await mem.register_store(
            kv_store=fake_kv,
            semantic_store=wrong_semantic
        )

    assert e.value.status == StatusCode.MEMORY_ENGINE_REGISTER_STORE_ERROR


@pytest.mark.asyncio
async def test_register_store_db_store_wrong_type():
    mem = LongTermMemory()

    fake_kv = MagicMock(spec=BaseKVStore)
    fake_semantic = MagicMock(spec=BaseSemanticStore)
    wrong_db = object()

    with pytest.raises(BaseError) as e:
        await mem.register_store(
            kv_store=fake_kv,
            semantic_store=fake_semantic,
            db_store=wrong_db
        )

    assert e.value.status == StatusCode.MEMORY_ENGINE_REGISTER_STORE_ERROR


def test_set_scope_config_llm_init_failed():
    mem = LongTermMemory()

    fake_scope_cfg = MagicMock()
    fake_scope_cfg.model_cfg = MagicMock()
    fake_scope_cfg.model_client_cfg = MagicMock()

    with patch(
        "openjiuwen.core.memory.long_term_memory.LongTermMemory._get_llm_from_config",
        side_effect=Exception("llm init failed")
    ):
        with pytest.raises(BaseError) as e:
            mem.set_scope_config("scope1", fake_scope_cfg)

    err = e.value
    assert err.status == StatusCode.MEMORY_ENGINE_SET_SCOPE_CONFIG_ERROR
    assert "llm init failed" in err.message


@pytest.mark.asyncio
async def test_delete_mem_by_id_write_manager_not_init():
    mem = LongTermMemory()
    mem.write_manager = None
    mem.kv_store = MagicMock()

    mem.kv_store.exclusive_set = AsyncMock(return_value=True)

    with pytest.raises(BaseError) as e:
        await mem.delete_mem_by_id(mem_id="mem123", user_id="u1", scope_id="s1")

    err = e.value
    assert err.status == StatusCode.MEMORY_ENGINE_DELETE_MEMORY_ERROR
    assert "Write manager is not initialized" in err.message
