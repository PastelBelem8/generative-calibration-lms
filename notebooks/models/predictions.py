"""Predictions classes


Similarly to the implementation of
[lm-calibration](https://github.com/jzbjyb/lm-calibration/blob/887e3e13df0462842ce288fffe588e549a3360ee/model/gpt2.py#L67)
we apply log-softmax to the log-probabilities before summing them.
In the case you're using a T5-like or BART-like model, the output of the call below will be a
[GreedySearchEncoderDecoderOutput](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.generation_utils.GreedySearchEncoderDecoderOutput).
"""
from itertools import filterfalse
from utils_generic import generate_uuid

import logging
import pandas as pd
import torch
import torch.nn
import torch.nn.functional as F


class GreedyGenerator:
    def __init__(self):
        super().__init__()
        self._num_beams = 1
        self._do_sample = False
        self._clean_up_tokenization_spaces = True
        self._skip_special_tokens = True

    @property
    def generate_hyperparams(self) -> dict:
        return {
            "num_beams": self._num_beams,
            "do_sample": self._do_sample,
        }

    @property
    def decoding_hyperparams(self) -> dict:
        return {
            "clean_up_tokenization_spaces": self._clean_up_tokenization_spaces,
            "skip_special_tokens": self._skip_special_tokens,
        }

    def generate(
        self, data, id_cols, tokenizer, model, batch_size: int = None, **kwargs
    ) -> dict:
        """
        - `score_proba`: score associated with the generated sentence. computed as the multiplication of the individual raw_scores. the score is within $[0, 1]$.
        - `preds`: textual representation of the generated instance
        - `preds_raw_int`: tokens id
        - `preds_raw_str`: tokens str
        - `preds_raw_scores`: scores for each of the tokens, lie in the range $[0, 1]$.
        - `len`: length of the sentence
        - `truncated`: whether the sequence was truncated (i.e., actually had the eos token).
        """
        if batch_size is None:
            batch_size = len(data)
        else:
            batch_size = min(batch_size, len(data))

        n = len(data)
        logging.info(f"Processing {n} examples in total")
        for b_start in range(0, n, batch_size):
            metadata = {}
            # Batch indexing
            # ---------------------------------------------------------------
            b_end = b_start + batch_size
            b_end = min(b_end, n)
            batch = data.select(range(b_start, b_end))
            # logging.info(f"Processing examples {b_start}-{b_end} (out of {n})")
            print(f"Processing examples {b_start}-{b_end} (out of {n})")

            metadata.update({id_col: batch[id_col] for id_col in id_cols})

            # Generate
            # ---------------------------------------------------------------
            results = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                # We're interested in returning information about the scores
                output_scores=True,
                return_dict_in_generate=True,
                # Force truncation (ensure the last token is always the EOS)
                forced_eos_token_id=tokenizer.eos_token_id,
                **self.generate_hyperparams,
                **kwargs,  # max_length
            )

            # Textual representation of the predicted sequence
            metadata["preds"] = tokenizer.batch_decode(
                results.sequences, **self.decoding_hyperparams
            )

            # Compute unique identifiers for each prediction
            # Ideally the identifier will depend on the model's,
            # the tokenizer's and the matrix's uuid but for now
            # we will simplify and only consider the generated text.
            #
            # Note: This assumes the name of the prediction file is being
            # handled by some component that has access to all this information
            # and is, therefore, able to avoid name clashes.
            uuid_metadata = {}
            uuid = lambda pred: generate_uuid(dict(text=pred, **uuid_metadata))
            metadata["preds_id"] = [uuid(pred) for pred in metadata["preds"]]

            # Individual tokens raw representation
            def skip_tokens(seq, token):
                predicate = filterfalse(lambda t: t == token, seq)
                return list(predicate)

            metadata["preds_raw_int"] = results.sequences.tolist()
            metadata["preds_raw_int"] = [
                skip_tokens(s, tokenizer.pad_token_id)
                for s in metadata["preds_raw_int"]
            ]

            # Individual tokens raw textual representation
            # metadata["preds_raw_str"] = [[tokenizer.decode(t, skip_special_tokens=True) for t in seq]
            metadata["preds_raw_str"] = [
                tokenizer.convert_ids_to_tokens(seq, skip_special_tokens=True)
                for seq in metadata["preds_raw_int"]
            ]

            # Individual tokens count (does not include special tokens like EOS or pad)
            metadata["preds_raw_count"] = list(map(len, metadata["preds_raw_str"]))

            # Whether the sentence was truncated or not (i.e., it has an EOS token)
            is_truncated = lambda s: int(any(s == tokenizer.eos_token_id))
            metadata["truncated"] = [is_truncated(seq) for seq in results.sequences]

            # ---------------------------------------------------------------------------------
            # Compute score_proba
            # ---------------------------------------------------------------------------------
            # Pair each timestep logits `score_t` with corresponding generated token
            # *Note*: since greedy_results.scores is a T sized tuple with B * V matrices
            # representing the logits for the different instances in the batch at each timestep
            # we can couple the actual logit score at each timestep with the corresponding token.
            scores, seq_tokens = results.scores, results.sequences[:, 1:]
            # ^Note: The sequences are considering an initial pad token whose score is not outputted
            # by the greedy decoder.
            assert (
                len(scores) == seq_tokens.shape[-1]
            ), "Dimension mismatch: Sequences vs scores"

            n_pred_timesteps = len(scores)
            scores_tokens = [
                (F.log_softmax(scores[t], dim=-1), seq_tokens[:, t])
                for t in range(n_pred_timesteps)
            ]
            # ^Note:
            # - `scores` is a |B| X |V| matrix with all the logits per batch per vocabulary
            # at prediction timestep t. Like Jiang et. al
            # (https://github.com/jzbjyb/lm-calibration/blob/887e3e13df0462842ce288fffe588e549a3360ee/model/gpt2.py#L67)
            # we apply F.log_softmax to ensure the logprobabilities are comparable amongst
            # the different batches
            # - `seq_tokens[:, t]` is a |B| X 1 matrix with the predicted token types at
            # timestep t
            greedy_scores = [
                scores_t.gather(-1, token_t.unsqueeze(-1))
                for scores_t, token_t in scores_tokens
            ]
            greedy_scores = torch.cat(greedy_scores, dim=1)
            # Must mask the greedy scores corresponding to the padding
            pad_mask = seq_tokens == tokenizer.pad_token_id
            greedy_scores[pad_mask] = 0

            metadata["score_proba"] = torch.exp(
                torch.sum(greedy_scores, dim=1)
            ).tolist()
            metadata["preds_raw_scores"] = torch.exp(greedy_scores).tolist()

            # Drop the tokens that are not important
            metadata["preds_raw_scores"] = [
                skip_tokens(s, 1) for s in metadata["preds_raw_scores"]
            ]

            del results
            del scores
            del seq_tokens
            del greedy_scores
            del pad_mask
            yield pd.DataFrame(metadata)
