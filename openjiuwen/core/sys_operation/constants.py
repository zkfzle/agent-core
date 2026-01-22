import inspect
import os

from e2b_code_interpreter import Sandbox


LOGS_DIR = os.getenv("LOGS_DIR")
E2B_API_KEY = os.getenv("E2B_API_KEY")


def _has_var_keyword(params) -> bool:
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    ) if params else False

def _supports_api_key(params) -> bool:
    """Return True when the inspected signature accepts api_key or arbitrary kwargs."""
    if not params:
        return True
    return "api_key" in params or _has_var_keyword(params)

def _first_matching_param(params, candidates):
    for name in candidates:
        if params and name in params:
            return name
    return None


try:
    _SANDBOX_INIT_PARAMS = inspect.signature(Sandbox.__init__).parameters
except (OSError, ValueError, TypeError):
    _SANDBOX_INIT_PARAMS = {}
_SANDBOX_CREATE_FUNC = getattr(Sandbox, "create", None)
_SANDBOX_TEMPLATE_PARAM = _first_matching_param(
    _SANDBOX_INIT_PARAMS, ("template", "template_id")
)
_SANDBOX_INIT_REQUIRES_DETAILS = all(
    key in _SANDBOX_INIT_PARAMS
    for key in (
        "sandbox_id",
        "envd_version",
        "envd_access_token",
        "sandbox_domain",
        "connection_config",
    )
)
_SANDBOX_SHOULD_USE_CREATE = bool(_SANDBOX_CREATE_FUNC) and (
    _SANDBOX_INIT_REQUIRES_DETAILS or not _SANDBOX_TEMPLATE_PARAM
)
_SANDBOX_SHOULD_USE_CREATE = bool(_SANDBOX_CREATE_FUNC) and (
    _SANDBOX_INIT_REQUIRES_DETAILS or not _SANDBOX_TEMPLATE_PARAM
)
if callable(_SANDBOX_CREATE_FUNC):
    try:
        _SANDBOX_CREATE_PARAMS = inspect.signature(_SANDBOX_CREATE_FUNC).parameters
    except (OSError, ValueError, TypeError):
        _SANDBOX_CREATE_PARAMS = {}
_SANDBOX_CREATE_SUPPORTS_API_KEY = _supports_api_key(_SANDBOX_CREATE_PARAMS)

_SANDBOX_CREATE_SUPPORTS_TIMEOUT = "timeout" in _SANDBOX_CREATE_PARAMS
_SANDBOX_INIT_SUPPORTS_TIMEOUT = "timeout" in _SANDBOX_INIT_PARAMS
DEFAULT_TIMEOUT = 120  # seconds
DEFAULT_TEMPLATE_ID = "7nj6zr8212e5zjcd8627"
_SANDBOX_CREATE_TEMPLATE_PARAM = _first_matching_param(
    _SANDBOX_CREATE_PARAMS, ("template", "template_id")
)
_SANDBOX_SUPPORTS_API_KEY = _supports_api_key(_SANDBOX_INIT_PARAMS)
try:
    _SANDBOX_CONNECT_PARAMS = inspect.signature(Sandbox.connect).parameters
except (OSError, ValueError, TypeError):
    _SANDBOX_CONNECT_PARAMS = {}
_SANDBOX_CONNECT_SUPPORTS_API_KEY = _supports_api_key(_SANDBOX_CONNECT_PARAMS)