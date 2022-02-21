from utils_generic import filter_params

import datasets


def parse_hf_bleu(results):
    results_precisions = results.pop("precisions")
    for i, p in enumerate(results_precisions):
        results[f"bleu_{i+1}"] = p
    return results


def parse_hf_rouge(results):
    """Parses the results obtained via hugging face metrics.

    It simplifies the results into a more human-readable and
    decluttered fashion, so that we can re-use these in our application.

    Example
    -------
    >>> rouge = load_metric("rouge")
    >>> results = rouge.compute(predictions=["Hello ohn"],references=["Hello john"])
    >>> results
    {'rouge1': AggregateScore(low=Score(precision=0.5, recall=0.5, fmeasure=0.5),
    mid=Score(precision=0.5, recall=0.5, fmeasure=0.5), high=Score(precision=0.5,
    recall=0.5, fmeasure=0.5)),
    'rouge2': AggregateScore(low=Score(precision=0.0, recall=0.0, fmeasure=0.0),
    mid=Score(precision=0.0, recall=0.0, fmeasure=0.0), high=Score(precision=0.0,
    recall=0.0, fmeasure=0.0)),
    'rougeL': AggregateScore(low=Score(precision=0.5, recall=0.5, fmeasure=0.5),
    mid=Score(precision=0.5, recall=0.5, fmeasure=0.5), high=Score(precision=0.5,
    recall=0.5, fmeasure=0.5)),
    'rougeLsum': AggregateScore(low=Score(precision=0.5, recall=0.5, fmeasure=0.5),
    mid=Score(precision=0.5, recall=0.5, fmeasure=0.5), high=Score(precision=0.5,
    recall=0.5, fmeasure=0.5))}
    >>> _postproc_hf_rouge(results)
    {'rouge1': 0.5, 'rouge2': 0.0, 'rougeL': 0.5, 'rougeLsum': 0.5}
    """
    for metric_name, metric_score in results.items():
        results[metric_name] = metric_score.mid.fmeasure
    return results


def hf_metric(name, parser_fn: callable = None, **kwargs) -> callable:
    load_metric_kwargs = filter_params(kwargs, datasets.load_metric)
    metric = datasets.load_metric(name, **load_metric_kwargs)

    compute_kwargs = filter_params(kwargs, metric.compute)
    parser_fn = parser_fn if parser_fn else lambda s: s

    def _apply_and_parse(predictions, references, **kwargs):
        results = metric.compute(
            predictions=predictions, references=references, **compute_kwargs
        )
        return parser_fn(results)

    return _apply_and_parse
