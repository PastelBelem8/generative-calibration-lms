import inspect


def filter_params(params: dict, method: callable) -> dict:
    method_params = inspect.signature(method).parameters
    return {p: p_val for p, p_val in params.items() if p in method_params}
