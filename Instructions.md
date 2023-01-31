# INSTRUCTIONS

These are instructions to run the fine-tuning code:

# 1. Preliminary settings

We use:

- Weight & Biases to track logs and reports during training.
- HuggingFace (HF) Hub to host the fine-tuned (best) models when training finishes.

Therefore you need:

- A wandb account and an API Key. After signing up get the API key here: [https://wandb.ai/authorize](https://wandb.ai/authorize).
- A HF account and a write-access token. How to get a write-access token see [here](https://huggingface.co/docs/hub/security-tokens#how-to-manage-user-access-tokens).

Next
- To run the fine-tuning script on Colab see  [here](notebooks/run_fine_tuning.ipynb).
- Else, read below.
# 2. Get project repo

```python
$ git clone https://github.com/Joqsan/bert-vs-fnet.git
```

# 3. Set a `.env` file with necessary environment variables

```python
$ cd bert-vs-fnet
$ vim .env
```

In `.env` include the following environment variables:

- `write_hub_token`: your HF write-access token.
- `WANDB_SILENT`: whether you want to print wandb logs in the terminal (we do want that).
- `WANDB_API_KEY`: your wandb API key.
- `WANDB_PROJECT`: a project name the reports will be under.
- `WANDB_LOG_MODEL`: whether to save the models to wandb as artifacts

For example:

```bash
# .env
write_hub_token=hf_aCsG...kjF
WANDB_SILENT=false # set to true if silent
WANDB_API_KEY=6eff...fgF
WANDB_PROJECT=comparison-bert-fnet
WANDB_LOG_MODEL=true
```

# 4. Create venv and install requirements

- Create and activate a venv.
- Install requirements:
    
    ```bash
    $ pip install -r requirements.txt
    ```
    

# 5. Run a fine-tuning task

On the terminal run with the synopsys command:

```bash
$ python models/fine_tuning.py model_name task
```

where:

- `model_name`: either `bert-base-uncased` or `Joqsan/custom-fnet`.
- `task`: any of the GLUE task (in our case that will be `cola`, `qnli` and `rte`.

Examples:

```bash
$ python models/fine_tuning.py bert-base-uncased cola
$ python models/fine_tuning.py Joqsan/custom-fne qnli
```