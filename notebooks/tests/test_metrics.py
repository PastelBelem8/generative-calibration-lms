from metrics.f_metrics import *


def test_success_exact_match():
    assert exact_match("Hello john!", "Hello john!") == 1
    assert exact_match(["Hello john!"], ["Hello john!"]) == 1
    assert exact_match(["Hello", "john!"], ["Hello", "john!"]) == 1


def test_fail_exact_match():
    assert exact_match("Hello!", "Hello john!") == 0
    assert exact_match(["Hello!"], ["Hello", "john!"]) == 0
    assert exact_match(["Hello", "john!"], ["Hello", "john", "!"]) == 0

    # Multiple list of tokens, defaults to 0
    assert exact_match(["Hello john!"], [["Hello", "Haaaa"], ["aaaa"]]) == 0


def test_success_first_error_position():
    y_true = ["The", "sky", "is", "blue"]
    assert first_error_position(y_true, y_true) == None
    assert first_error_position(y_true, y_true, no_err_val=-1) == -1

    y_pred = ["A", "sky", "is", "blue"]
    assert first_error_position(y_true, y_pred) == 0
    assert first_error_position(y_true, y_true + ["a"]) == len(y_true)
    first_error_position(
        ["she", "is", "poisoned", "by", "barabas", "and", "ithamore"],
        ["she", "is", "poisoned", "by", "barabas"],
    ) == 5
    first_error_position(
        ["she", "is", "poisoned", "by", "barabas"],
        ["she", "is", "poisoned", "by", "barabas", "and", "ithamore"],
    ) == 5

    assert exact_match(["Hello", "john!"], ["Hello", "john", "!"]) == 0


def test_success_precision():
    assert _precision(0, 0, 0, 0) == 0
    assert _precision(0, 0, 1, 1) == 0
    assert _precision(0, 1, 1, 1) == 0
    assert _precision(1, 0, 1, 1) == 1
    assert _precision(1, 1, 1, 1) == 0.5
    assert _precision(1, 1, 0, 0) == 0.5


def test_success_recall():
    assert _recall(0, 0, 0, 0) == 0
    assert _recall(0, 0, 1, 1) == 0
    assert _recall(0, 1, 1, 1) == 0
    assert _recall(1, 0, 1, 1) == 0.5
    assert _recall(1, 1, 1, 1) == 0.5
    assert _recall(1, 1, 0, 0) == 1


def test_success_f1_score():
    raise NotImplementedError("not implemented yet")


def test_success_f_metrics():
    y_true = ["The", "sky", "is", "blue"]
    y_pred = ["A", "sky", "blue"]
    results = f_metrics(y_true, y_pred)
    assert results["precision"] == 2 / 3, "wrong precision"
    assert results["recall"] == 2 / 4, "wrong recall"
    assert results["f1_score"] == (2 * (2 / 3) * 2 / 4) / (
        2 / 3 + 2 / 4
    ), "wrong f1-score"
    assert results["csi"] == 2 / (2 + 1 + 2), "wrong csi score"

    y_true = ["The", "sky", "is", "blue"]
    y_pred = ["no"]
    results = f_metrics(y_true, y_pred)
    assert results["precision"] == 0, "wrong precision"
    assert results["recall"] == 0, "wrong recall"
    assert results["f1_score"] == 0, "wrong f1-score"
    assert results["csi"] == 0, "wrong csi score"

    y_true = ["The", "sky", "is", "blue"]
    y_pred = []
    results = f_metrics(y_true, y_pred)
    assert results["precision"] == 0, "wrong precision"
    assert results["recall"] == 0, "wrong recall"
    assert results["f1_score"] == 0, "wrong f1-score"
    assert results["csi"] == 0, "wrong csi score"


test_success_exact_match()
test_fail_exact_match()
test_success_first_error_position()
test_success_precision()
test_success_recall()
test_success_f1_score()
test_success_f_metrics()
