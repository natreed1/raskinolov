# Router Classifier Artifact Contract (v1)

Router V2 uses an OSS linear classifier persisted to:

- `training/router_classifier_v1/manifest.json`
- `training/router_classifier_v1/vocab.json`
- `training/router_classifier_v1/weights.npy`

Produced by `scripts/train_routing_classifier.py`.

## Artifact schema

`manifest.json` (`router_classifier_artifact_v1`) includes:

- `model_version`: `router_classifier_linear_v1`
- `labels`: ordered adapter label list aligned with logits row order
- `feature_type`: `bow_log_tf_linear_softmax`
- `vocab_size`
- `weights_shape`: `[num_labels, vocab_size_plus_bias]`
- `train_summary` (train/valid/test accuracy and hyperparameters)
- `dataset_manifest` (copied from source dataset build)

## Inference contract

`scripts/router/classifier.py` exposes `LinearRouterClassifier.predict(text)` returning:

- `adapter_id`
- `confidence` in `[0,1]`
- `ambiguity = 1 - confidence`
- full probability map (`probs`) keyed by adapter id

## Router integration behavior

`scripts/model_router.py`:

- loads classifier artifacts from `training/router_classifier_v1/` when available
- uses classifier prediction when confidence is above threshold (default `0.62`)
- falls back to deterministic lexical specialist matching for low-confidence cases
- derives final route (`local|hybrid|frontier`) after adapter prediction (adapter-first policy)

