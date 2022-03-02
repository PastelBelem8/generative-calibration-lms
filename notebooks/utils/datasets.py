"""Standard utils used during data processing"""
from collections import defaultdict
from typing import Dict, List, Tuple, Union

from utils_generic import generate_uuid, filter_params

import datasets
import numpy as np
import pandas as pd
import pyarrow.lib as pylib


def drop_duplicates(data, col):
    unique_col_values = {k: True for k in data.unique(col)}
    return data.filter(lambda example: unique_col_values.pop(example[col], False))


def load_dataset(
    dataset: Union[str, Tuple[str]], split: str, local: bool, **kwargs
) -> datasets.Dataset:
    """Delegate loading of dataset split to the corresponding method.
    Currently, supported methods are Hugging Face's
    ``datasets.load_dataset`` and loading from a local TSV file.
    Check the corresponding documentation for more information
    about the keyword arguments you should pass.
    Parameters
    ----------
    dataset: Union[str, Tuple[str]]
        Name or name and subnames to use upon loading the dataset.
    split: str
        The split to load.
    local: bool
        Whether the dataset is local or online.
    See Also
    --------
    datasets.load_dataset: loads dataset for
    qalibration.utils.dataset.load_local_tsv_dataset: loads local tsv dataset.
    """
    if local:
        loading_fn = load_local_tsv_dataset
    else:
        loading_fn = datasets.load_dataset

    dataset = dataset if isinstance(dataset, (list, tuple)) else (dataset,)
    load_fn_kwargs = filter_params(kwargs, loading_fn)
    return loading_fn(*dataset, split=split, **load_fn_kwargs)


def load_local_tsv_dataset(
    dataset: str,
    split: str,
    local_dir: str = ".",
    sep: str = "\\n",
    ext: str = "tsv",
    **kwargs,
) -> datasets.Dataset:
    """Loads a TSV dataset from the local filesystem.
    It looks up ``local_dir`` for a filename ``{dataset}_{split}``.
    """
    # FIXME:
    # create the dataset to have a structure similar to the one retrieved by the hugging face transformers
    # load_dataset for SQuAD 1.1. However more thought should be put into this.
    filepath = f"{local_dir}/{dataset}_{split}.{ext}"

    with open(filepath, "r") as f:
        data = f.readlines()
        question_contexts, answers = zip(*[d.split("\t") for d in data])
        questions, contexts = zip(*[qc.split(sep) for qc in question_contexts])
        answers = map(lambda a: {"text": [a.rstrip()]}, answers)

        data = pd.DataFrame(
            {
                "id": np.arange(len(data)),
                "title": [None] * len(data),
                "question": questions,
                "context": contexts,
                "answers": list(answers),
            }
        )

        return datasets.Dataset(pylib.Table.from_pandas(data))


def create_metadata(data, col, features, **kwargs):
    """Loads the metadata."""

    def _get_uuid(row):
        metadata = {feature: row[feature] for feature in features}
        row[col] = generate_uuid(metadata)
        return row

    return data.map(_get_uuid, **kwargs)


def unfold_multiple_answers(data, answer_col: str = "answers") -> Dict[str, List]:
    # Set a column order (for deterministic results)
    cols = sorted(data.keys())
    cols = np.unique([answer_col] + cols)
    # ^Note: Setting answer in the first position makes implementation
    # easier. We do not need to know details about other columns, since
    # this method is focused in unfolding multiple answers.

    unfolded_results = defaultdict(list)

    for row in zip(*[data[c] for c in cols]):
        multiple_answers = row[0]
        unique_answers = np.unique(multiple_answers["text"]).tolist()

        # Create the output mapping with the unfolded answers
        for i, col in enumerate(cols[1:]):
            unfolded_results[col].extend([row[1 + i]] * len(unique_answers))
        unfolded_results[cols[0]].extend(unique_answers)
        # Add multi-way annotations in case they are needed for downstream tasks
        # (e.g., evaluation)
        # We have to add one entry per unique_answer, hence:
        unfolded_results[f"{answer_col}_multi_way"].extend(
            [str(unique_answers)] * len(unique_answers)
        )
        # ^Bug Fix: Since dumping the file to a csv file compromises
        # the commas in the `unique_answers` list due to implicit type
        # conversions we stringify the list prior to its use. This
        # facilitates the use of `eval` to retrieve the correct list.
        # ------------------------------------------------------------
    return unfolded_results


def normalize_qa_text(text: str, is_question: bool = False) -> str:
    """Normalizes the input ``text``.
    Transform into lower case, remove single-quote, and replace whitespaces
    from the input text. If ``is_question=True`` it suffixes the text with
    "?" if it is not yet present.
    Notes
    -----
    This normalization method mimicks the one followed in the
    UnifiedQA paper (https://github.com/allenai/unifiedqa/).
    References
    ----------
    - https://github.com/allenai/unifiedqa/blob/7bf0653c6fb68a51019924fd4c51615155acbebe/tasks.py
    - https://github.com/allenai/unifiedqa/blob/7bf0653c6fb68a51019924fd4c51615155acbebe/encode_datasets.py#L83
    """
    import re

    text = text.lower()
    text = text.strip()

    text = text.replace("\t", "")
    text = text.replace("   ", " ")
    text = text.replace("  ", " ")
    text = text.replace("\n", " ")

    if is_question and "?" not in text:
        text += "?"

    return re.sub(r"'(.*?)'", r"\1", text)
