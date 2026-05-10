### What Changed
The script `build_game_task_pairwise_dataset.py` was updated to include functionality for handling apply contract documentation and filtering records based on apply failure tags. It now reads a contract from a specified file and includes it in the generated messages. Additionally, it introduces a new function to filter records where the loser's error indicates an apply failure.

### Why It Matters
This change enhances the dataset generation process by providing context from the apply contract and filtering out records where the apply attempt failed, which can improve the quality and relevance of the training data.

### Commands/Tests Run
1. **Run the script with default parameters:**
   ```bash
   python scripts/build_game_task_pairwise_dataset.py
   ```

2. **Run the script with the `--focus-apply-failures` flag to filter records:**
   ```bash
   python scripts/build_game_task_pairwise_dataset.py --focus-apply-failures
   ```

3. **Verify the output files are generated correctly:**
   ```bash
   ls data/lora/game_task_pairwise
   ls data/lora/compare_feedback/pairwise_feedback.jsonl
   ```

4. **Check the contract file is read correctly:**
   ```bash
   cat docs/GAME_ARENA_APPLY_CONTRACT.md
   ```

### Next Validation Steps
1. **Review the generated dataset files (`game_task_pairwise` and `pairwise_feedback.jsonl`) to ensure they contain the expected data and formatting.**
2. **Run a small training pass using the generated dataset to verify that the model learns effectively from the apply contract and filtered records.**
3. **Update the documentation to reflect the new functionality and usage of the `--focus-apply-failures` flag.**
