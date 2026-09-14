"""
evaluate.py
-----------
Evaluates LLM location prediction against frequency baseline and random baseline.

Train/test split: last 7 days of each user's stay history held out as test set.
"""

import random
from collections import Counter


TEST_DAYS = 7


def split_train_test(stays):
    """
    Split stays into train and test by date (last TEST_DAYS days = test).

    Parameters
    ----------
    stays : list[dict]
        Each dict must have 'arrival' (datetime) and 'name' fields.

    Returns
    -------
    (train, test) : tuple[list, list]
    """
    if not stays:
        return [], []

    sorted_stays = sorted(stays, key=lambda s: s["arrival"])
    last_date = sorted_stays[-1]["arrival"].date()
    cutoff = last_date - __import__("datetime").timedelta(days=TEST_DAYS)

    train = [s for s in sorted_stays if s["arrival"].date() <= cutoff]
    test  = [s for s in sorted_stays if s["arrival"].date() > cutoff]
    return train, test


def frequency_baseline(train, test):
    """
    Predict the most-frequent location in training data for every test stay.

    Returns
    -------
    float : accuracy (0.0 – 1.0)
    """
    if not train or not test:
        return 0.0
    freq = Counter(s["name"] for s in train)
    most_common = freq.most_common(1)[0][0]
    correct = sum(1 for s in test if s["name"] == most_common)
    return correct / len(test)


def random_baseline(train, test):
    """
    Predict a random location from the training set for every test stay.

    Returns
    -------
    float : accuracy (0.0 – 1.0)
    """
    if not train or not test:
        return 0.0
    unique_locs = list(set(s["name"] for s in train))
    correct = sum(1 for s in test if random.choice(unique_locs) == s["name"])
    return correct / len(test)


def slot_accuracy(predictions, test):
    """
    Break down accuracy by 2-hour time slot.

    Parameters
    ----------
    predictions : list[str]
        LLM predicted location names, same length as test.
    test : list[dict]

    Returns
    -------
    dict[str, dict]  slot -> {correct, total, accuracy}
    """
    slot_stats = {}
    for pred, stay in zip(predictions, test):
        slot = stay["hour_slot"]
        if slot not in slot_stats:
            slot_stats[slot] = {"correct": 0, "total": 0}
        slot_stats[slot]["total"] += 1
        if pred == stay["name"]:
            slot_stats[slot]["correct"] += 1

    for slot, stats in slot_stats.items():
        stats["accuracy"] = stats["correct"] / stats["total"]

    return dict(sorted(slot_stats.items()))


def print_results(user_id, train, test, llm_preds, verbose=True):
    """
    Print a summary table for one user.

    Parameters
    ----------
    user_id : str
    train, test : list[dict]
    llm_preds : list[str]

    Returns
    -------
    dict  with keys: llm_acc, freq_acc, rand_acc, n_test, n_unique
    """
    freq_acc  = frequency_baseline(train, test)
    rand_acc  = random_baseline(train, test)
    llm_acc   = sum(p == s["name"] for p, s in zip(llm_preds, test)) / len(test) if test else 0.0
    n_unique  = len(set(s["name"] for s in train + test))
    delta     = llm_acc - freq_acc

    if verbose:
        sign = "✓" if delta >= 0 else "✗"
        print(f"\n{'='*55}")
        print(f"User: {user_id}")
        print(f"  Train stays : {len(train):>4} | Test stays: {len(test):>4}")
        print(f"  Unique locs : {n_unique:>4}")
        print(f"  LLM  acc    : {llm_acc*100:>6.1f}%")
        print(f"  Freq acc    : {freq_acc*100:>6.1f}%")
        print(f"  Rand acc    : {rand_acc*100:>6.1f}%")
        print(f"  LLM vs Freq : {delta*100:>+6.1f}pp {sign}")

    return {
        "user": user_id,
        "llm_acc": llm_acc,
        "freq_acc": freq_acc,
        "rand_acc": rand_acc,
        "n_train": len(train),
        "n_test": len(test),
        "n_unique": n_unique,
        "delta_pp": delta * 100,
    }
