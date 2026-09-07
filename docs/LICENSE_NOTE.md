# License note

IRx-1 is fine-tuned from `Qwen/Qwen3.5-2B`, distributed under
[Apache 2.0](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE).

Apache 2.0 has no field-of-use or redistribution restrictions. Still good practice
before publishing IRx-1 anywhere (Hugging Face, OpenRouter, an app listing):

- Include the Apache 2.0 license text (`LICENSE`) with the distribution.
- Retain the standard Apache 2.0 `NOTICE`/attribution: state that IRx-1 is derived from
  Qwen3.5-2B (Alibaba).
- Don't imply Alibaba created, endorses, or is responsible for IRx-1.
- Double-check current terms before release — license terms can change between
  versions: https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE

## Training data: databricks-dolly-15k

A 400-example balanced sample (`data/processed/distilled_dolly.jsonl`) is drawn from
[databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k),
CC-BY-SA-3.0, human-written (not AI-generated), owned by Databricks, Inc. CC-BY-SA is a
share-alike license: attribution to Databricks should be retained in the model's
training-data documentation (as here), and Databricks does not endorse IRx-1.
