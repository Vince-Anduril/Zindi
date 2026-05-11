from src.data import build_train_index, extract_hash, extract_subset


def get_context(row_id: str) -> dict | None:
    """
    Given a test/val row ID, return the matching train example by topic hash.
    Returns None if no match found (shouldn't happen in this dataset).
    """
    hash_key = extract_hash(row_id)
    index = build_train_index()
    return index.get(hash_key)


def has_context(row_id: str) -> bool:
    return get_context(row_id) is not None


if __name__ == "__main__":
    from src.data import load_test

    test = load_test()
    covered = sum(1 for rid in test["ID"] if has_context(rid))
    print(f"Test coverage: {covered}/{len(test)} ({100*covered/len(test):.1f}%)")

    # Show a sample
    sample_id = test["ID"].iloc[0]
    ctx = get_context(sample_id)
    print(f"\nTest ID: {sample_id}")
    print(f"Context found: {ctx is not None}")
    if ctx:
        print(f"Train Q: {ctx['question'][:100]}...")
        print(f"Train A: {ctx['answer'][:100]}...")
