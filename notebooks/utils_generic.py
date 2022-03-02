import inspect
import json
import hashlib


def filter_params(params: dict, method: callable) -> dict:
    method_params = inspect.signature(method).parameters
    return {p: p_val for p, p_val in params.items() if p in method_params}


def filter_params_by_prefix(params: dict, prefix: str) -> dict:
    return {p: p_val for p, p_val in params.items() if p.startswith(prefix)}


def generate_uuid(content, indent: int=2) -> str:
    """Deterministic uuid generator of the `content`."""
    content = json\
        .dumps(content, sort_keys=True, indent=indent)\
        .encode("utf-8")
    return hashlib.md5(content).hexdigest()