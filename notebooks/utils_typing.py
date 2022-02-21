# utils.typing
is_str = lambda s: isinstance(s, str)
is_str_list = lambda ss: isinstance(ss, list) and all([is_str(s) for s in ss])
