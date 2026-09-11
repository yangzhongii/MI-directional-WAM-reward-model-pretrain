"""Order-invariant strict preference summaries, including ties explicitly."""

from collections import Counter


def summarize_margins(margins):
    margins = list(margins)
    count = len(margins)
    wins = sum(m > 0 for m in margins)
    ties = sum(m == 0 for m in margins)
    return {
        "pairs": count, "wins": wins, "ties": ties, "losses": count - wins - ties,
        "strict_accuracy": wins / count if count else None,
        "tie_rate": ties / count if count else None,
        "tie_adjusted_accuracy": (wins + 0.5 * ties) / count if count else None,
        "reward_margin": sum(margins) / count if count else None,
    }


def quality_pair_summary(records):
    from itertools import combinations
    order = {"failure": 0, "suboptimal": 1, "successful": 2}
    margins, success_failure = [], []
    for a, b in combinations(records, 2):
        if a["task"] != b["task"]:
            continue
        qa, qb = order[a["quality_label"]], order[b["quality_label"]]
        if qa == qb:
            continue
        margin = (a["progress_pred"][-1] - b["progress_pred"][-1]) * (1 if qa > qb else -1)
        margins.append(margin)
        if {qa, qb} == {0, 2}:
            success_failure.append(margin)
    return {
        "quality_ranking": summarize_margins(margins),
        "success_failure": summarize_margins(success_failure),
        "reward_counts": dict(Counter(str(r["progress_pred"][-1]) for r in records)),
    }
