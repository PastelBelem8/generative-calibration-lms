from dataclasses import dataclass, field
from typing import Iterable
from utils_generic import import_method, method_name

import os
import yaml
import logging


@dataclass
class OutputResult:
    filename: str

    output_fn_classpath: str = field(default="pandas.DataFrame.to_csv")
    output_fn_kwargs: dict = field(default_factory=dict)
    output_dir: str = field(default="./outputs")
    out_extension: str = None

    def __post_init__(self):
        super().__init__()

        self.output_fn_kwargs = {
            "compression": "gzip",
            "index": False,
            "header": True,
            "encoding": "utf-8",
        }
        os.makedirs(self.output_dir, exist_ok=True)

        # Dynamically load function
        if isinstance(self.output_fn_classpath, str):
            self.output_fn = import_method(self.output_fn_classpath)

        elif isinstance(self.output_fn_classpath, callable):
            self.output_fn = self.output_fn_classpath
            self.output_fn_classpath = method_name(self.output_fn_classpath)

    @property
    def filepath(self):
        filepath = f"{self.output_dir}/{self.filename}"
        if self.out_extension:
            filepath = f"{filepath}.{self.out_extension}"
        return filepath

    @property
    def configpath(self):
        return f"{self.filepath}.out_config"

    def dump_configs(self):
        with open(self.configpath, "w") as f:
            yaml.dump(self.output_fn_kwargs, f)

    def write(self, batches: Iterable, exists_new: bool = True):
        batches = iter(batches)

        out_kwargs = self.output_fn_kwargs.copy()

        if exists_new is True:
            first_batch = next(batches)
            out_kwargs["mode"] = "w"
            logging.info(
                f"Creating file {self.filepath} w/ {self.output_fn_classpath} and arguments: {out_kwargs}"
            )
            self.output_fn(first_batch, self.filepath, **out_kwargs)
            out_kwargs["header"] = False

        out_kwargs["mode"] = "a"
        for batch in batches:
            self.output_fn(batch, self.filepath, **out_kwargs)

        self.dump_configs()
