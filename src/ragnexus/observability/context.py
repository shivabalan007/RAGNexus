from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("ragnexus_request_id", default="-")
trace_id_var: ContextVar[str] = ContextVar("ragnexus_trace_id", default="-")
