from dataclasses import dataclass, field
from models.predictions import GreedyGenerator
from utils_generic import filter_params, filter_params_by_prefix

import logging
import transformers


@dataclass
class T5Model:
    model_name: str
    model_hyperparameters: dict

    model_hf_kwargs: dict = field(default_factory=dict)
    _tokenizer = None
    _model = None

    def __post_init__(self):
        super().__init__()
        # TODO
        # - Add greedygenerator attribute

    def _format_row(self, row, features):
        prefixes = [f"{f}: {row[f]}" for f in features]
        sep = f" {self._tokenizer.eos_token} "
        return {"encoded": sep.join(prefixes)}

    def encode(self, data, target_label, prefix: str = None):
        if prefix is None:
            hyperparams = self.model_hyperparameters
        else:
            hyperparams = filter_params_by_prefix(self.model_hyperparameters, prefix)

        hyperparams = filter_params(hyperparams, self._tokenizer)
        logging.warning(
            f"Using {hyperparams} to encode (target={target_label}, prefix={prefix}): {hyperparams}"
        )
        return self._tokenizer(data[target_label], **hyperparams)

    def load(self):
        # Configuration (defines vocab size, model dimensions, ...)
        # config = transformers.T5Config.from_pretrained(
        #    model_name, **kwargs)
        # ^ Note:
        # Changes in T5 configurations should be done here:
        # ...
        tokenizer_fn = transformers.T5TokenizerFast.from_pretrained
        tokenizer_params = filter_params(self.model_hf_kwargs, tokenizer_fn)
        self._tokenizer = tokenizer_fn(self.model_name, **tokenizer_params)

        model_fn = transformers.T5ForConditionalGeneration.from_pretrained
        model_params = filter_params(self.model_hf_kwargs, model_fn)
        self._model = model_fn(self.model_name, **model_params)

    def generate(self, data, id_cols, batch_size: int = 100, **kwargs):
        generator_kwargs = {
            "data": data,
            "id_cols": id_cols,
            "model": self._model,
            "tokenizer": self._tokenizer,
            "batch_size": batch_size,
        }
        raise NotImplementedError
