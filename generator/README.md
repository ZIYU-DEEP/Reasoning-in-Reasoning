# README
[TOC]

## `Raw Data`
Here we show one example entry from the `test.json` in the `random` split of the LeanDojo Benchmark from the [Mathlib](https://github.com/leanprover-community/mathlib4/blob/fe4454af900584467d21f4fd4fe951d29d9332a7/Mathlib/Data/Complex/Abs.lean#L327) repository.

The original theorem is:
```bash
theorem lim_im (f : CauSeq ℂ Complex.abs) : lim (cauSeqIm f) = (lim f).im := by
  rw [lim_eq_lim_im_add_lim_re]; simp [ofReal']
```
Basically, for each theorem, the following information is recorded:
- `url` / `commit` / `file_path` / `full_name` / `start` / `end`
- `traced_tactics`
    - `tactic` / `annotated tactic` (an annotation and a list) / `state_before` / `state_after`
```bash
{
"url": "https://github.com/leanprover-community/mathlib4", 
"commit": "fe4454af900584467d21f4fd4fe951d29d9332a7", 
"file_path": "Mathlib/Data/Complex/Abs.lean", 
"full_name": "Complex.lim_im", 
"start": [327, 1], 
"end": [328, 48], 
"traced_tactics": [
    {
        "tactic": "rw [lim_eq_lim_im_add_lim_re]",
    
        "annotated_tactic": [
            "rw [<a>lim_eq_lim_im_add_lim_re</a>]", 
            [{"full_name": "Complex.lim_eq_lim_im_add_lim_re", 
              "def_path": "Mathlib/Data/Complex/Abs.lean", 
              "def_pos": [313, 9], 
              "def_end_pos": [313, 33]}]], 
             
        "state_before": "f : CauSeq \u2102 \u21d1abs'\n\u22a2 CauSeq.lim (cauSeqIm f) = (CauSeq.lim f).im", 
           
        "state_after": "f : CauSeq \u2102 \u21d1abs'\n\u22a2 CauSeq.lim (cauSeqIm f) = (\u2191(CauSeq.lim (cauSeqRe f)) + \u2191(CauSeq.lim (cauSeqIm f)) * I).im"
    }, 
           
    {
        "tactic": "simp [ofReal']", 
      
        "annotated_tactic": [
            "simp [<a>ofReal'</a>]", 
            [{"full_name": "Complex.ofReal'", 
              "def_path": "Mathlib/Data/Complex/Basic.lean", 
              "def_pos": [89, 5], 
              "def_end_pos": [89, 12]}]], 
          
        "state_before":"f : CauSeq \u2102 \u21d1abs'\n\u22a2 CauSeq.lim (cauSeqIm f) = (\u2191(CauSeq.lim (cauSeqRe f)) + \u2191(CauSeq.lim (cauSeqIm f)) * I).im"}, 

        "state_after": "no goals"
    }]
}
```

## `datamodule.py`

### Overview
It defines the loading, preprocessing, and batching of data for training a model that generates tactics as part of a theorem proving process. An example of the data looks like:
```bash
{
    "url": thm["url"],
    "commit": thm["commit"],
    "file_path": thm["file_path"],
    "full_name": thm["full_name"],
    "state": format_state(tac["state_before"]), 
    "tactic": tactic,                            
}
```
With the batch collate function, the batch will look like:
```bash
{
    "url": ..,
    "commit": ..,
    "file_path": ..,
    "full_name": ...,

    "state": ..., 
    "state_ids": ...,
    "state_mask": ...,

    "tactic": ...,
    "tactic_ids": ...,
    "tactic_mask": ...,   
}
```
In the `module.py`, the loss will be directly computed using `loss = self(batch["state_ids", "state_mask", "tactic_ids"])`.

We already have a byT5 model trained on that, and we want to finetune the model. For simplicity we will be using the same keys, but adding some post-processing.

For **next-goal prediction**: 
`[CURRENT GOAL]{state_before}[NEXT GOAL]{state_after}`
- `input`: We attach `[CURRENT GOAL]\n` as the prefix and `\n[TARGET GOAL]\n` as a suffix to the original input of `format_state(tac["state_before"])`, for it to get distinguished from the default proof step prediction in the original model.
- `output`: We use `format_state(tac["state_after"])` as the output.

For **goal-conditioned tactic prediction**: 
`[CURRENT GOAL]{state_before}[TARGET GOAL]{state_after}[PROOFSTEP]{tactic}`
- `input`:  We attach `[CURRENT GOAL]\n` as the prefix and add `\n[TARGET GOAL]\n{format_state(tac["state_after"])}[PROOFSTEP]` as the suffix to the original input of `format_state(tac["state_before"])`.
- `output`: We use the `tactic` as the output.

We may need to modify the `evaluate.py` for the model to extract states correctly.

### Key Components

#### `GeneratorDataset`
- **Purpose**: Manages individual data items, preparing them for processing by the model. It formats the theorem proving data into a usable format for machine learning models.
- **Functionality**:
  - `_load_data`: Load the data from the data path and format it as the above.
  - `collate`: This will be used in the `collate_fn` of the `dataloader()` in the `GeneratorDataModule`. Getting a list of data together, transform strings into token ids, and make a dictionary to contain them.
  - `__len__` and `__getitem__`: as usual.

#### `GeneratorDataModule`
- **Purpose**: Setup the overall data pipeline for both training and validation phases, using the `GeneratorDataset`.
- **Functionality**:
  - Sets up training and validation datasets.
  - Provides DataLoader configurations that handle the efficient loading and batching of data.

### Methodologies

- **Data Loading**: Data is loaded from structured JSON files, ensuring each theorem and its tactics are correctly parsed and formatted.
- **Tokenization**: Uses the ByT5 tokenizer to handle a wide range of tokens, including those specific to mathematical and programming contexts.
- **Batch Processing**: Implements a custom collate function to prepare data batches in a format compatible with the model's input requirements.



## `model.py`

### Overview
The `model.py` file contains the implementation of a PyTorch Lightning module named `RetrievalAugmentedGenerator`, designed to generate tactics for theorem proving. This module integrates a T5 model for generating tactic predictions, potentially augmented with premises retrieved by a `PremiseRetriever`.

### Key Components

- **`TopkAccuracy` (Class):** A custom metric class for evaluating the top-k accuracy of model predictions.
- **`TacticGenerator` (Abstract Base Class):** Defines the interface for tactic generators, requiring implementations for `generate` and `batch_generate` methods.
- **`RetrievalAugmentedGenerator` (Class):** Inherits from `TacticGenerator` and `pytorch_lightning.LightningModule`, encapsulating the main functionality of the tactic generation model.

    - **`__init__`:** Initializes the model, setting up the tokenizer and the generator, optional premise retriever, training arguments.
    - **`forward`:** The default setting is `self.generator(input_ids=state_ids, attention_mask=state_mask, labels=tactic_ids)`.
    - **`training_step`:** Conducts a single step of model training, calculating loss and logging key information. The default setting for the loss is `loss = self(batch['state_ids'], batch['state_mask'], batch['tactic_ids'])`.
    - **`configure_optimizers`:** Set the optimizer with learning rate and warmup steps.
    - **`validation_step`:** Similar to `training_step`, but also includes generation of tactics using beam search to evaluate the model during validation.
    - **`generate` and `batch_generate`:** Methods for generating tactics based on input state; `generate` handles single instances, while `batch_generate` is designed for batch processing.
    - **load**: Loading the model checkpoint.

### Utilities

- **`_log_io_texts`:** Logs the input and output texts to wandb for monitoring during training or validation phases.
- **`configure_optimizers`:** Sets up optimizers and learning rate schedules based on the model configuration.

### Usage
This module is typically used in a training script where instances of `RetrievalAugmentedGenerator` are created, trained on batches of theorem proving tasks, and evaluated on their ability to generate appropriate tactics.

### Files and Dependencies
- Relies on `common.py` for utility functions and `retrieval.model` for optional premise retrieval functionality.
- Utilizes libraries such as `torch`, `pytorch_lightning`, `transformers`, and `torchmetrics`.

