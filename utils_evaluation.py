import numpy as np


# Calculate Gini coefficient
def gini_inefficient(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x) ** 2 * np.mean(x))


def gini(x, w=None):
    # https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (
            cumxw[-1] * cumw[-1]
        )
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def normalize_min_max(x):
    return (x - min(x)) / (max(x) - min(x))


""" 
Code from https://github.com/MilkaLichtblau/Multinomial_FA-IR

Copyright is held by the authors

Created on Jun 12, 2020

@author: meike
"""


def dcg_score(y_score, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    if gains == "exponential":
        gains = 2**y_score - 1
    elif gains == "linear":
        gains = y_score
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_score)) + 2)
    return np.sum(gains / discounts)


""" 
Code from https://github.com/MilkaLichtblau/Multinomial_FA-IR

Copyright is held by the authors

Created on Jun 12, 2020

@author: meike
"""


def ndcg_score(y_true, y_score, k=10, gains="linear"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" or "linear" (default).
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true[:k], gains)
    actual = dcg_score(y_score[:k], gains)
    return actual / best


def evaluate_ranking_scores(original_score, mitigated_score, K=1000):
    from sklearn.metrics import mean_squared_error
    from scipy import stats
    from sklearn.metrics import r2_score

    from utils_evaluation import gini, ndcg_score, normalize_min_max
    from divrank_mitigation import discounted_cumulative_gain

    positions_original = (
        len(original_score) - stats.rankdata(original_score, method="ordinal") + 1
    )
    dcg_original = discounted_cumulative_gain(original_score, positions_original, k=K)

    positions_mit = (
        len(mitigated_score) - stats.rankdata(mitigated_score, method="ordinal") + 1
    )

    dcg_mit = discounted_cumulative_gain(mitigated_score, positions_mit, k=K)

    # Maximum individual change
    # Score
    min_score_change = np.min(mitigated_score - original_score)
    max_score_change = np.max(mitigated_score - original_score)

    gini_index = gini(normalize_min_max(mitigated_score))
    gini_index_original = gini(normalize_min_max(original_score))
    # Rank
    min_rank_change = -np.min(positions_mit - positions_original)
    max_rank_change = -np.max(positions_mit - positions_original)

    id_sort_new = np.argsort(-mitigated_score, kind="mergesort")

    ndcgLoss = 1 - ndcg_score(original_score, original_score[id_sort_new], k=K)

    res = {
        "MSE": mean_squared_error(original_score, mitigated_score),
        "spearmanr_score": stats.spearmanr(original_score, mitigated_score)[0],
        "spearmanr_rank": stats.spearmanr(positions_original, positions_mit)[0],
        "kendalltau_score": stats.kendalltau(original_score, mitigated_score)[0],
        "kendalltau_rank": stats.kendalltau(positions_original, positions_mit)[0],
        "kendalltau_score2": stats.kendalltau(
            original_score, original_score[id_sort_new]
        )[0],
        "dcg_mit": dcg_mit,
        "delta_dcg": dcg_mit - dcg_original,
        "r2": r2_score(original_score, mitigated_score),
        "min_score_change": min_score_change,
        "max_score_change": max_score_change,
        "max_gained_rank_positions": min_rank_change,
        "max_lost_rank_positions": max_rank_change,
        "gini": gini_index,
        "gini_diff": gini_index_original - gini_index,
        "ndcgLoss": ndcgLoss,
    }
    return res


from utils_mitigation import get_fpdiv, get_divergent, get_negatively_divergent


def divergence_stats(
    df, target, min_sup=0.05, metric_i="d_outcome", t_col="t_value_outcome"
):
    """
    Get divergence stats
    """
    fp_i = get_fpdiv(df, target, min_sup=min_sup)
    fp_div = get_divergent(fp_i, t=t_col)
    fp_neg_div = get_negatively_divergent(fp_i, t=t_col, metric_i=metric_i)
    fp_neg_all = get_negatively_divergent(fp_i, metric_i=metric_i, significant=False)

    return {
        "#div": len(fp_div),
        "#negdiv": len(fp_neg_div),
        "#FP": fp_i.freq_metrics.shape[0],
        "#negdiv_all": len(fp_neg_all),
        "avg_abs_div": fp_div[metric_i].abs().mean() if fp_div.shape[0] > 0 else 0,
        "avg_abs_neg_div": fp_neg_div[metric_i].abs().mean()
        if fp_neg_div.shape[0] > 0
        else 0,
        "avg_abs_neg_div_all": fp_i.freq_metrics[metric_i].abs().mean(),
        "min_div": fp_i.freq_metrics[metric_i].min(),
        "max_div": fp_i.freq_metrics[metric_i].max(),
    }


def print_table_ranking_res(df):
    r = df.copy()
    r["#posdiv"] = r["#div"] - r["#negdiv"]
    cols = [
        "method",
        "protected attributes",
        "protected values",
        "#posdiv",
        "#negdiv",
        "min_div",
        "max_div",
    ] + [
        "gini",
        "kendalltau_score",
        "max_gained_rank_positions",
        "max_lost_rank_positions",
        "ndcgLoss",
    ]
    r = r[cols]
    cols_r = [
        "min_div",
        "max_div",
        "gini",
        "kendalltau_score",
        "ndcgLoss",
    ]
    cols_int = [
        "#posdiv",
        "#negdiv",
        "max_gained_rank_positions",
        "max_lost_rank_positions",
    ]
    r[cols_int] = r[cols_int].astype(int)
    r[cols_r] = r[cols_r].astype(float).round(2)
    r.rename(
        columns={
            "max_gained_rank_positions": "rank gain",
            "max_lost_rank_positions": "rank drop",
            "kendalltau_score": "kendall tau",
            "min_div": "minΔ",
            "max_div": "maxΔ",
        },
        inplace=True,
    )

    return r
