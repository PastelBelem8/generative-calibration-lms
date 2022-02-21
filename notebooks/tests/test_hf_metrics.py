from datasets import load_metric
from metrics.hf_metrics import *


def test_success_parse_hf_methods():
    rouge_example = {"rouge1": 0.5, "rouge2": 0.0, "rougeL": 0.5, "rougeLsum": 0.5}
    rouge = load_metric("rouge")
    assert (
        parse_hf_rouge(
            rouge.compute(predictions=["Hello ohn"], references=["Hello john"])
        )
        == rouge_example
    )


test_success_parse_hf_methods()
