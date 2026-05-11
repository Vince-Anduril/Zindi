import re
import pandas as pd
from functools import lru_cache
from config import TRAIN_CSV, VAL_CSV, TEST_CSV


def extract_hash(row_id: str) -> str:
    """Extract topic hash from ID like ID_TR_Aka_Gha_A3B1799D → A3B1799D"""
    return row_id.split("_")[-1]


def extract_subset(row_id: str) -> str:
    """Extract language subset from ID like ID_TR_Aka_Gha_A3B1799D → Aka_Gha"""
    parts = row_id.split("_")
    return f"{parts[2]}_{parts[3]}"


@lru_cache(maxsize=1)
def load_train() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV)
    df["hash"] = df["ID"].apply(extract_hash)
    df["subset"] = df["ID"].apply(extract_subset)
    return df


@lru_cache(maxsize=1)
def load_val() -> pd.DataFrame:
    df = pd.read_csv(VAL_CSV)
    df["hash"] = df["ID"].apply(extract_hash)
    df["subset"] = df["ID"].apply(extract_subset)
    return df


@lru_cache(maxsize=1)
def load_test() -> pd.DataFrame:
    df = pd.read_csv(TEST_CSV)
    df["hash"] = df["ID"].apply(extract_hash)
    df["subset"] = df["ID"].apply(extract_subset)
    return df


@lru_cache(maxsize=1)
def build_train_index() -> dict:
    """Build hash → {question, answer, subset} index from train set."""
    df = load_train()
    index = {}
    for _, row in df.iterrows():
        index[row["hash"]] = {
            "question": row["input"],
            "answer": row["output"],
            "subset": row["subset"],
        }
    return index


def get_language_name(subset: str) -> str:
    from config import LANGUAGE_MAP
    return LANGUAGE_MAP.get(subset, subset)


if __name__ == "__main__":
    train = load_train()
    val = load_val()
    test = load_test()
    idx = build_train_index()

    print(f"Train: {len(train)} rows")
    print(f"Val:   {len(val)} rows")
    print(f"Test:  {len(test)} rows")
    print(f"Train index: {len(idx)} unique hashes")

    # Check coverage: how many test hashes exist in train
    test_hashes = set(test["hash"])
    train_hashes = set(idx.keys())
    coverage = len(test_hashes & train_hashes)
    print(f"Test hashes covered by train: {coverage}/{len(test_hashes)} ({100*coverage/len(test_hashes):.1f}%)")

    print("\nSubsets in train:")
    print(train["subset"].value_counts())
