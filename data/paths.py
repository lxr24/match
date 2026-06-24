"""Default dataset root paths and datalist split aliases."""

# Competition directory layout on disk
DATASET_TRAIN_ROOT = "dataset_train"
DATASET_TEST_NOISY_ROOT = "dataset_test_noisy"
DATALIST_DIR = "datalist"

# Canonical split -> possible datalist file stems
DATALIST_SPLIT_ALIASES = {
    "train": ["train"],
    "validate": ["validate", "validation", "val"],
    "validation": ["validate", "validation", "val"],
    "test": ["test"],
}
