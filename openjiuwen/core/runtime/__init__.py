import contextvars

workflow_runtime_vars: contextvars.ContextVar[dict] = contextvars.ContextVar("workflow_runtime_vars", default={})