"""Standard utils used during data processing"""


def drop_duplicates(data, col):
    unique_col_values = {k: True for k in data.unique(col)}
    return data.filter(lambda example: unique_col_values.pop(example[col], False))
