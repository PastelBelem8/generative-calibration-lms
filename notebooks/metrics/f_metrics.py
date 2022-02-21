from collections import Counter
from typing import Dict, List, Union
import logging


Tokens = List[str]
Text = Union[str, Tokens]


def exact_match(y_true: Text, y_pred: Text) -> int:
    """Determine whether two texts (or sequences of tokens) are equal."""
    if isinstance(y_true, str) and isinstance(y_pred, str):
        return int(y_true == y_pred)

    elif isinstance(y_true, (list, tuple)) and isinstance(y_pred, (list, tuple)):
        if len(y_true) != len(y_pred):
            logging.debug(
                f"Dimension mismatch (default value is 0): {y_true} vs {y_pred}"
            )
            return 0
        return int(all(map(lambda t1, t2: t1 == t2, y_true, y_pred)))
    else:
        error_msg = f"y_true ({type(y_true)}) and y_pred ({type(y_pred)})"
        raise ValueError(
            f"Cannot compare `exact_match` for argument types: {error_msg}"
        )


def first_error_position(y_true: Tokens, y_pred: Tokens, no_err_val: int = None) -> int:
    """Determine the position in the predicted sequence of the first error.

    Notes
    -----
    If both text sequences are equivalent we return ``no_err_val`` as the position.
    Otherwise, we iterate for each token in ``y_pred`` and look for the first
    mismatch between ``y_pred`` and ``y_true`` tokens returning that position.

    Examples
    --------
    >>> y_true = ["The", "sky", "is", "blue"]
    >>> y_pred = ["A", "sky", "is", "blue"]
    >>> first_error_position(y_true, y_pred)
    1
    >>> y_pred = ["The", "sky", "IS", "blue"]
    >>> first_error_position(y_true, y_pred)
    3
    >>> first_error_position(y_true, y_true, no_err_val=-1)
    -1
    """
    assert isinstance(y_true, (list, tuple)) and len(y_true) != 0
    assert isinstance(y_pred, (list, tuple)) and len(y_pred) != 0

    # When no error occurs return the `no_err_val`
    if exact_match(y_true, y_pred):
        return no_err_val

    # If there are differences then they are one of two types:
    # 1. Token mismatch: which will occur in the common length of
    # the two sequences. Values can vary between 0 and min(lengths)
    # 2. Misnumber of tokens: one of the sequences is longer than the
    # other, causing them to be wrong.
    max_mismatch_ix = min(len(y_true), len(y_pred))

    for i in range(max_mismatch_ix):
        if y_true[i] != y_pred[i]:
            return i
    return max_mismatch_ix


def _precision(tp, fp, tn, fn) -> float:
    return 0 if tp == 0 else tp / (tp + fp)


def _recall(tp, fp, tn, fn) -> float:
    return 0 if tp == 0 else tp / (tp + fn)


def _critical_success_index(tp, fp, tn, fn):
    "Ratio of positives w.r.t. number of errors (also dubbed threat score)."
    return 0 if tp == 0 else tp / (tp + fn + fp)


def _f1_score(precision=None, recall=None, **kwargs) -> float:
    if precision is not None and recall is not None:
        p = precision
        r = recall
        # return if precision or recall are 0
        if p == 0 or r == 0:
            return 0
    else:
        p = _precision(**kwargs)
        r = _recall(**kwargs)

    return (2 * p * r) / (p + r)


def f_metrics(y_true: Tokens, y_pred: Tokens) -> Dict[str, float]:
    true_tokens, pred_tokens = Counter(y_true), Counter(y_pred)
    tp = sum((true_tokens & pred_tokens).values())
    fp = len(y_pred) - tp
    fn = len(y_true) - tp
    tn = 0
    assert tp + fp + fn == sum((true_tokens | pred_tokens).values())

    prec = _precision(tp=tp, fp=fp, tn=tn, fn=fn)
    rec = _recall(tp=tp, fp=fp, tn=tn, fn=fn)
    return {
        "precision": prec,
        "recall": rec,
        "f1_score": _f1_score(precision=prec, recall=rec),
        "csi": _critical_success_index(tp=tp, fp=fp, tn=tn, fn=fn),
    }
