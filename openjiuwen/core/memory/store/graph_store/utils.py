# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import uuid
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from openjiuwen.core.common.logging import logger
from openjiuwen.core.utils.llm.messages import BaseMessage


def get_uuid() -> str:
    """Generate UUID from uuid4.

    Returns:
        str: uuid with length of 32.
    """
    return uuid.uuid4().hex


def ensure_unique_uuids(backend: Any, ids: List[Any], collection: str, skip: bool = False) -> List[Any]:
    """De-duplicate given uuids in a specific collection of graph memory database.

    Args:
        backend (GraphBackend): graph memory database backend.
        ids (List[Any]): list of uuids to de-duplicate.
        collection (str): name of collection (entities/relations/episodes).
        skip (bool): skip de-duplication to avoid certain errors.

    Returns:
        List[Any]: list of unique uuids.
    """
    unique_ids = [_id or get_uuid() for _id in ids]
    if skip:
        return unique_ids
    kwargs = dict(collection=collection, output_fields=["uuid"])
    if not backend.is_empty(collection):
        dup_list = [dup["uuid"] for dup in backend.query(ids=unique_ids, **kwargs)]
        while dup_list:
            new_uuids = []
            for dup_uuid in dup_list:
                unique_ids[unique_ids.index(dup_uuid)] = tmp_uuid = get_uuid()
                new_uuids.append(tmp_uuid)
            dup_list = [dup["uuid"] for dup in backend.query(ids=new_uuids, **kwargs)]
    return unique_ids


def msg2dict(messages: List[Union[Dict, BaseMessage]], preserve_meta: bool = False, **kwargs) -> List[Dict[str, Any]]:
    """Convert a list of BaseMessage into a list of dict.

    Args:
        messages (List[Union[Dict, BaseMessage]]): list of BaseMessage to convert.
        preserve_meta (bool, optional): preserve the extra fields in BaseMessage. Defaults to False.

    Returns:
        List[Dict[str, Any]]: list of converted dict messages.
    """
    if not (isinstance(messages, list) and all(isinstance(msg, (dict, BaseMessage)) for msg in messages)):
        raise ValueError("Input is not a list of dict or BaseMessage")
    if preserve_meta:
        return [msg.model_dump(mode="python", **kwargs) if isinstance(msg, BaseMessage) else msg for msg in messages]
    return [dict(role=msg.role, content=msg.content) if isinstance(msg, BaseMessage) else msg for msg in messages]


def format_list_of_messages(
    messages: List[Dict], role_replace: Optional[Mapping[str, str]] = None, template: str = "{role}: {content}\n"
) -> str:
    """Format a list of messages into a string.

    Args:
        messages (List[Dict]): list of messages representing a conversation.
        role_replace (Optional[Mapping[str, str]]): mapping from roles to names (assistant -> Writing Assistant Agent)
        template (str, optional): formatting template for each message entry. Defaults to "{role}: {content}\n".

    Returns:
        str: formatted list of messages.
    """
    result = ""
    role_replace = role_replace or dict()
    for msg in messages:
        msg = msg.copy()
        role = msg.pop("role", "")
        result += template.format(role=role_replace.get(role, role), **msg)
    return result


def safe_timestamp(datetime_obj: datetime) -> float:
    """Some Operating System (i.e. Windows) cannot handle negative timestamps natively"""
    if datetime_obj.year < 1970:
        this_tzinfo = timezone.utc if datetime_obj.tzinfo else None
        return (datetime_obj - datetime(1970, 1, 1, tzinfo=this_tzinfo)).total_seconds()
    return datetime_obj.timestamp()


def get_current_utc_timestamp() -> int:
    """Get current UTC timestamp as integer.

    Returns:
        int: UTC timestamp.
    """
    return int(safe_timestamp(datetime.now(timezone.utc)))


def format_timestamp(t: Union[int, float], tz: tzinfo = timezone.utc, fmt: str = r"(%a) %Y/%b/%d %H:%M:%S") -> str:
    """Format a UNIX timestamp into readable representation like "(Wed) 2025/Sep/10 15:56:53".

    Args:
        t (Union[int, float]): timestamp.
        tz (tzinfo, optional): timezone. Defaults to timezone.utc.
        fmt (str, optional): format for datetime representation. Defaults to "(%a) %Y/%b/%d %H:%M:%S".

    Returns:
        str: formatted datetime.
    """
    if t != -1:
        datetime.fromtimestamp(t, tz).isoformat(timespec="seconds")
        return datetime.fromtimestamp(t, tz).strftime(fmt)
    return "Unknown Datetime"


def format_timestamp_iso(t: Union[int, float], tz: Optional[tzinfo] = timezone.utc) -> str:
    """Format a UNIX timestamp into ISO 8601 representation like "2025-09-10T15:56:53+08:00".
    The "+00:00" suffix would be omitted if tz=None.

    Args:
        t (Union[int, float]): timestamp.
        tz (Optional[tzinfo], optional): timezone. Defaults to timezone.utc.

    Returns:
        str: iso-formatted datetime.
    """
    if t != -1:
        return datetime.fromtimestamp(t, tz).isoformat(timespec="seconds")
    return "Unknown Datetime"


def iso2timestamp(iso_str: str) -> Tuple[int, int]:
    """Convert time from ISO 8601 representation like "2025-09-10T15:56:53+08:00" into UNIX timestamp.
    The "+00:00" suffix may be omitted.

    Args:
        iso_str (str): time in iso format.

    Returns:
        Tuple[int, int]: UNIX timestamp and offset (in units of 15 min), (-1, 0) if input is invalid.
    """
    try:
        iso_str = iso_str.replace("24:00:00", "23:59:59").removesuffix("+")
        datetime_obj = datetime.fromisoformat(iso_str)
        return int(safe_timestamp(datetime_obj)), _store_tz_offset(datetime_obj.tzname())
    except Exception as e:
        logger.error(f"Invalid iso -> timestamp conversion ({iso_str}): {e}")
        return -1, 0


def load_stored_time_from_db(timestamp: Union[int, float], offset: int) -> Optional[datetime]:
    """Load stored timestamp and offset from database into datetime object.

    Args:
        timestamp (Union[int, float]): timestamp.
        offset (int): timezone offset, in units of 15 min.

    Returns:
        Optional[datetime]: datetime object if timestamp is non-negative.
    """
    if timestamp != -1:
        tz = _load_tz_offset(offset)
        return datetime.fromtimestamp(timestamp, tz)
    return None


def _store_tz_offset(tz_str: str) -> int:
    """Parse timezone string and return integer offset (in unit of 15 minutes)"""
    if tz_str and tz_str.removeprefix("UTC"):
        offsets = tz_str.removeprefix("UTC+").split(":")
        hr = mi = 0
        if offsets:
            hr = offsets[0]
            if len(offsets) > 1:
                mi = offsets[1]
        return int(hr) * 4 + int(mi) // 15
    return 0


def _load_tz_offset(tz_offset: int) -> timezone:
    """Load timezone offset integer from database and convert to timezone"""
    min_offset = tz_offset * 15
    return timezone(timedelta(minutes=min_offset))
