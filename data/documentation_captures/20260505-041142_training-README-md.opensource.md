### What changed
The `training/README.md` file was updated to provide more detailed information about the training flow, including the paths and commands used for dataset capture and training. The file now includes links to other documentation files for canonical paths, current-vs-historical adapters and runs, and CLI examples.

### Why it matters
This change improves the documentation by providing more context and links to other relevant files, making it easier for developers to understand the training process and how to use the provided scripts and configurations.

### Commands/tests run
To capture the dataset and train the model, the following commands were run:
```bash
source .venv/bin/activate
export SOURCE_REPO=/path/to/fallen-empire
python scripts/export_repo_for_training.py
python scripts/build_lora_dataset.py --out-dir data/lora/qwen25-coder-7b/game_text
mlx_lm.lora --train -c training/lora_qwen25_coder_7b.yaml
```

### Next validation steps
After running the training, the next validation steps include:
1. Checking the training logs for any errors or issues.
2. Evaluating the model's performance on the validation set.
3. Comparing the base model and fine-tuned model using the `scripts/chat_gradio.py` or `scripts/run_game_benchmark.py` scripts.
4. Updating the `docs/DATA_LAYOUT.md` and `docs/RUNS.md` files with the new paths and adapter information.
