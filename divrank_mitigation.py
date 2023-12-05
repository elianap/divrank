from calendar import c
from tabnanny import verbose
from utils_mitigation import (
    get_fpdiv,
    get_negatively_divergent,
    slice_by_itemset,
    plot_stats,
    get_divergent,
)

from typing import Tuple

from copy import deepcopy
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import numpy as np

from IPython.display import display


def _check_equal_as_computed(
    patterns_after_mitigation, patterns_after_mitigation_from_de, metric_target
):
    sum_value_difference = abs(
        sum(
            round(
                patterns_after_mitigation[metric_target]
                - patterns_after_mitigation_from_de[metric_target],
                7,
            )
        )
    )

    if sum_value_difference > 0:
        raise ValueError(sum_value_difference)


def get_quality_stats(df_input, df_mit_score, target="score", metric="d_outcome"):
    from scipy import stats

    original_scores = df_input[target].values
    mitigated_scores = df_mit_score[target].values

    tau_kendall, p_value = stats.kendalltau(original_scores, mitigated_scores)
    tau_spearmanr, p_value = stats.spearmanr(original_scores, mitigated_scores)

    positions_original = (
        len(original_scores) - stats.rankdata(original_scores, method="ordinal") + 1
    )

    positions_mitigated = (
        len(mitigated_scores) - stats.rankdata(mitigated_scores, method="ordinal") + 1
    )

    lost_positions = np.min(positions_original - positions_mitigated)
    gained_positions = np.max(positions_original - positions_mitigated)

    return {
        "kendalltau_score": tau_kendall,
        "spearmanr_score": tau_spearmanr,
        "lost_positions": lost_positions,
        "gained_positions": gained_positions,
    }


def check_mitigation_stats_results(
    df_input,
    df_mit_score,
    last_result,
    target="score",
    metric="d_outcome",
    min_sup=0.05,
):
    fp_divergence_mit_score = get_fpdiv(df_mit_score, target, min_sup=min_sup)

    assert last_result["min_div"] == min(
        fp_divergence_mit_score.freq_metrics[metric]
    ), "Min divergence differ"

    from scipy import stats
    from sklearn.metrics import r2_score

    original_scores = df_input[target].values
    mitigated_scores = df_mit_score[target].values

    tau_kendall, p_value = stats.kendalltau(original_scores, mitigated_scores)
    tau_spearmanr, p_value = stats.spearmanr(original_scores, mitigated_scores)

    positions_original = (
        len(original_scores) - stats.rankdata(original_scores, method="ordinal") + 1
    )

    positions_mitigated = (
        len(mitigated_scores) - stats.rankdata(mitigated_scores, method="ordinal") + 1
    )

    lost_positions = np.min(positions_original - positions_mitigated)
    gained_positions = np.max(positions_original - positions_mitigated)

    pos_min, pos_max = np.argmin(positions_original - positions_mitigated), np.argmax(
        positions_original - positions_mitigated
    )

    assert last_result["kendalltau_score"] == tau_kendall, "Kendall tau differs"
    assert last_result["spearmanr_score"] == tau_spearmanr, "Spearmanr tau differs"
    assert last_result["r2"] == r2_score(
        original_scores, mitigated_scores
    ), "Spearmanr tau differs"

    assert (
        last_result["max_lost_rank_positions"] == lost_positions
    ), "Lost positions differs"
    assert (
        last_result["max_gained_rank_positions"] == gained_positions
    ), "Gained positions differs"


def define_tau_cap_minmax(
    current_itemset,
    current_support_cnt,
    current_divergence,
    mitigated_itemset,
    mitigated_support_cnt,
    pattern_support_dict,
    dfb_mit_test,
    min_alpha=0,
):
    len_dataset = dfb_mit_test.shape[0]
    union_itemset_support = get_union_support(
        mitigated_itemset, current_itemset, pattern_support_dict, dfb_mit_test
    )

    # update_value = tau * (union_itemset_support)/current_support_cnt - ((tau * mitigated_support_cnt)/(len_dataset - mitigated_support_cnt))*((current_support_cnt-union_itemset_support)/current_support_cnt)
    # new_divergence = current_divergence + update_value

    # update_value_cap = - min_alpha - current_divergence

    tau_cap = None

    if current_support_cnt == len_dataset:
        return tau_cap

    if (current_itemset.issuperset(mitigated_itemset) == False) and (
        current_itemset.issubset(mitigated_itemset) == False
    ):
        # tau_cap =  (update_value_cap * current_support_cnt)/ ((union_itemset_support) - ( mitigated_support_cnt/(len_dataset - mitigated_support_cnt))*((current_support_cnt-union_itemset_support)))

        tau_mul = (union_itemset_support) / current_support_cnt - (
            (mitigated_support_cnt) / (len_dataset - mitigated_support_cnt)
        ) * ((current_support_cnt - union_itemset_support) / current_support_cnt)

        if tau_mul == 0:
            print(
                "union_itemset_support",
                (union_itemset_support),
                current_support_cnt,
                "-",
                (mitigated_support_cnt),
                (len_dataset, mitigated_support_cnt),
            )

        tau_cap = -(current_divergence - min_alpha) / tau_mul

    return tau_cap


def get_union_support(itemset_a, itemset_b, pattern_support_dict, df_data):
    union_itemset = itemset_a.union(itemset_b)

    if len(set([item[0] for item in union_itemset])) != len(union_itemset):
        union_itemset_support = 0
    else:
        # Check if we already have its info (above support threshold)
        if union_itemset not in pattern_support_dict:
            # We have to slice the dataset searching for union_itemset_info
            from utils_mitigation import slice_by_itemset

            union_itemset_support = len(slice_by_itemset(df_data, union_itemset))
            # We store the support of the union_itemset
            pattern_support_dict[union_itemset] = union_itemset_support

        else:
            union_itemset_support = pattern_support_dict[union_itemset]

    return union_itemset_support


def discounted_cumulative_gain(relevances, positions, k=None):
    import math

    if k is not None:
        relevances = relevances[np.where(positions <= k)]
        positions = positions[np.where(positions <= k)]
    return sum(
        [(relevances[i] / math.log(positions[i] + 1)) for i in range(len(positions))]
    )


class Mitigation:
    """
    Model wrapper
    """

    def __init__(self):
        pass

    def _ordered_divergent_patterns(
        self, divergent_patterns_df, sort_by_value, ascending
    ):
        # To avoid ties, sort by t value first
        if "t_value_outcome" in divergent_patterns_df.columns:
            ordered_fp_divergent_df = divergent_patterns_df.sort_values(
                "t_value_outcome", key=abs, ascending=False
            )
        else:
            # We sort the patterns by suppot
            ordered_fp_divergent_df = divergent_patterns_df.sort_values(
                "support", ascending=False
            )

        if sort_by_value not in ordered_fp_divergent_df:
            raise ValueError(
                f"{sort_by_value} not available. Specify another term in {list(ordered_fp_divergent_df.columns)}"
            )

        ordered_fp_divergent_df = ordered_fp_divergent_df.sort_values(
            sort_by_value, key=abs, ascending=ascending
        )
        return ordered_fp_divergent_df

    def get_cap_value(
        self,
        mitigated_itemset,
        mitigated_support_cnt,
        patterns_info,
        df_data,
        pattern_support_dict,
        min_alpha=0,
        small_v=0.99,
    ):
        tau_cap_values = [
            define_tau_cap_minmax(
                current_itemset,
                current_support_cnt,
                current_divergence,
                mitigated_itemset,
                mitigated_support_cnt,
                pattern_support_dict,
                df_data,
                min_alpha=min_alpha,
            )
            for current_itemset, current_support_cnt, current_divergence in zip(
                patterns_info.itemsets,
                patterns_info.support_count,
                patterns_info["d_outcome"],
            )
        ]
        if tau_cap_values == []:
            print(f"{tau_cap_values} IS NONE! set to 0")
            return 0
        if [v for v in tau_cap_values if v != None and v >= 0] == []:
            print(
                f"{[v for v in tau_cap_values if v != None and v >= 0]} IS NONE! set to 0"
            )
            return 0
        tau_cap = min([v for v in tau_cap_values if v != None and v >= 0])
        return tau_cap * small_v

    def mitigate_pattern(
        self, df_mit, itemset, delta_increment, target="score", lower_others=True
    ):
        indexes_sel = slice_by_itemset(df_mit, itemset).index
        df_mit.loc[indexes_sel, target] = (
            df_mit.loc[indexes_sel][target] + delta_increment
        )
        if lower_others:
            tot_increment = delta_increment * len(indexes_sel)
            decrease_other_value = tot_increment / (len(df_mit) - len(indexes_sel))

            df_mit.loc[df_mit.index.difference(indexes_sel), target] = (
                df_mit.loc[df_mit.index.difference(indexes_sel), target]
                - decrease_other_value
            )

        return df_mit

    def stats_mitigation_minmax(
        self,
        fp_divergence_i=None,
        fp_divergence_mit=None,
        metric_i="d_outcome",
        t_col="t_value_outcome",
        dfb=None,
        df_mit=None,
        target="score",
        original_score=None,
        mitigated_score=None,
        avoid_recompute=False,
        frequent_patterns_after_mitigation=None,
        K=100,
        min_sup=0.05,
    ):
        from utils_evaluation import gini, ndcg_score, normalize_min_max

        """
        Iteratively mitigate divergent pattern. Monotone process

        Args:
            fp_divergence_i (FP_divergence): divergence object for the input data. If not available, it is computed from dfb
            fp_divergence_mit (FP_divergence): divergence object for the mitigated data. If not available, it is computed from df_mit
            metric_i (str):  divergence metric. "d_outcome" as default
            t_col (str): column name for the statistical significance. Default "t_value_outcome"

            dfb (pd.DataFrame) : input data, instances to mitigate with their score or rank
            df_mit (pd.DataFrame: the mitigated instance. The instances and score/rank after mitigation
            target (str): target column for the mitigation process, i.e., name of the value/score column. "score" as default

            original_score (np.array): original scores. If not available, it is computed from dfb and "target" column
            mitigated_score (np.array): mitigated scores. If not available, it is computed from df_mit and "target" column

            K (int)

            avoid_recompute: if True, it avoids the re-computation of the divergence object. In this case, we might have incomplete results
        Returns:

        """

        if fp_divergence_i is None:
            if dfb is None:
                raise ValueError()
            fp_divergence_i = get_fpdiv(df_mit, target, min_sup=min_sup)

        if fp_divergence_mit is None:
            if avoid_recompute is False:
                if df_mit is None:
                    raise ValueError()
                fp_divergence_mit = get_fpdiv(df_mit, target, min_sup=min_sup)

        if original_score is None:
            if dfb is None:
                raise ValueError()
            original_score = dfb[target].values

        if mitigated_score is None:
            if df_mit is None:
                raise ValueError()
            mitigated_score = df_mit[target].values

        # TODO - improve - do it only one time

        if fp_divergence_mit is not None:
            fp_div_mit = get_divergent(fp_divergence_mit, t=t_col, th_redundancy=None)
            fp_neg_div_mit = get_negatively_divergent(
                fp_divergence_mit, t=t_col, metric_i=metric_i, th_redundancy=None
            )
            fp_neg_div_mit_all = get_negatively_divergent(
                fp_divergence_mit,
                metric_i=metric_i,
                significant=False,
                th_redundancy=None,
            )
            frequent_patterns_after_mitigation = fp_divergence_mit.freq_metrics

            number_of_divergent_patterns = fp_div_mit.shape[0]
            number_of_negatively_divergent_patterns = fp_neg_div_mit.shape[0]
            number_of_negatively_divergent_patterns_all = fp_neg_div_mit_all.shape[0]
        else:
            number_of_divergent_patterns = np.nan
            number_of_negatively_divergent_patterns = np.nan

            if frequent_patterns_after_mitigation is None:
                raise ValueError(
                    "frequent_patterns_after_mitigation and fp_divergence_mit are None. Specify at least one of them"
                )
            number_of_negatively_divergent_patterns_all = (
                frequent_patterns_after_mitigation.loc[
                    frequent_patterns_after_mitigation["d_outcome"] < 0
                ].shape[0]
            )

        from sklearn.metrics import mean_squared_error
        from scipy import stats

        positions_original = (
            len(original_score) - stats.rankdata(original_score, method="ordinal") + 1
        )
        dcg_original = discounted_cumulative_gain(
            original_score.values, positions_original, k=K
        )

        positions_mit = (
            len(mitigated_score) - stats.rankdata(mitigated_score, method="ordinal") + 1
        )

        dcg_mit = discounted_cumulative_gain(mitigated_score.values, positions_mit, k=K)

        # Maximum individual change
        # Score
        min_score_change = np.min(mitigated_score - original_score)
        max_score_change = np.max(mitigated_score - original_score)

        gini_index = gini(normalize_min_max(mitigated_score))
        gini_index_original = gini(normalize_min_max(original_score))
        # Rank
        min_rank_change = -np.min(positions_mit - positions_original)
        max_rank_change = -np.max(positions_mit - positions_original)

        ndcgLoss = 1 - ndcg_score(original_score.values, mitigated_score.values, k=K)
        if number_of_divergent_patterns is np.nan:
            avg_abs_div = np.nan
        else:
            if number_of_divergent_patterns > 0:
                avg_abs_div = fp_div_mit["d_outcome"].abs().mean()
            else:
                avg_abs_div = 0

        if number_of_negatively_divergent_patterns is np.nan:
            avg_abs_neg_div = np.nan
        else:
            if number_of_negatively_divergent_patterns > 0:
                avg_abs_neg_div = fp_neg_div_mit["d_outcome"].abs().mean()
            else:
                avg_abs_neg_div = 0

        res = {
            "#div": number_of_divergent_patterns,
            "#negdiv": number_of_negatively_divergent_patterns,
            "#FP": frequent_patterns_after_mitigation.shape[0],
            "#negdiv_all": number_of_negatively_divergent_patterns_all,
            "avg_abs_div": avg_abs_div,
            "avg_abs_neg_div": avg_abs_neg_div,
            "avg_abs_neg_div_all": frequent_patterns_after_mitigation["d_outcome"]
            .abs()
            .mean(),
            "min_div": frequent_patterns_after_mitigation["d_outcome"].min(),
            "max_div": frequent_patterns_after_mitigation["d_outcome"].max(),
            "MSE": mean_squared_error(original_score, mitigated_score),
            "spearmanr_score": stats.spearmanr(original_score, mitigated_score)[0],
            "spearmanr_rank": stats.spearmanr(positions_original, positions_mit)[0],
            "kendalltau_score": stats.kendalltau(original_score, mitigated_score)[0],
            "kendalltau_rank": stats.kendalltau(positions_original, positions_mit)[0],
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

    def iterative_mitigate_minmax(
        self,
        dfb,
        target="score",
        metric_i="d_outcome",
        sort_by_value="d_outcome",
        ascending=False,
        t_col="t_value_outcome",
        break_if_no_significant=False,
        break_if_greater_than=None,
        lower_others=True,
        n_iterations=None,
        K=100,
        verbose=False,
        small_v=0.99,
        round_v=10,
        ratio_mitigation: float = 1,
        min_sup=0.05,
    ):
        """
        Iteratively mitigate divergent pattern. Monotone process

        Args:
            dfb (pd.DataFrame) : input data, instances to mitigate with their score or rank
            target (str): target column for the mitigation process, i.e., name of the value/score column. "score" as default
            metric_i (str): divergence metric to mitigate. "d_outcome" as default
            sort_by_value (str): sorting value for the mitigation process. Divergence metric ("d_outcome") as default
            ascending (bool): order of the mitigation. Default 'False', sorted for desceding order.

            break_if_no_significant (bool): if True, the process terminates if there are no more significant negatively divergent patterns.
            break_if_greater_than (int): if set, the process terminates if the minimum divergence is greater than this value

            # TODO
            t_col (str): column name for the statistical significance. Default "t_value_outcome"


            lower_others (bool): if True, the score/value of the instances not in the subgroups is decreased accordingly to presere the average

            round_v (int): round to the round_v

            ratio_mitigation (float): ratio of mitigation. 1 --> try to mitigate as much as it can. The lower the value, the lower the mitigation.

        Returns:
            (   pd.DataFrame: the stats of the mitigation process
                pd.DataFrame: the mitigated instance. The instances and score/rank after mitigation
            )
        """

        # Check parameters
        if ratio_mitigation <= 0 or ratio_mitigation > 1:
            raise ValueError(f"{ratio_mitigation} must be between 0 and 1")

        # df_mit will store the (output) mitigated DataFrame
        df_mit = deepcopy(dfb)

        # Get the frequent patterns and their divergence
        fp_divergence_i = get_fpdiv(df_mit, target, min_sup=min_sup)

        # Sort the patterns by metric_i (and round them TODO)
        divergent_fp_neg_df = fp_divergence_i.freq_metrics.sort_values(metric_i).round(
            round_v
        )
        # Get the negatively divergent patterns
        divergent_fp_neg_df = divergent_fp_neg_df.loc[divergent_fp_neg_df[metric_i] < 0]

        # Order the negatively divergent patterns by sort_by_value in "ascending value" order
        ordered_fp_divergent_df = self._ordered_divergent_patterns(
            divergent_fp_neg_df, sort_by_value, ascending
        )

        from tqdm import tqdm

        # Stats
        start_time = time.time()

        all_results = {}

        all_mitigated_itemsets = []

        # Init
        patterns_info = deepcopy(fp_divergence_i.freq_metrics)

        # Dictionary {FPattern: support count}
        pattern_support_dict = patterns_info.set_index("itemsets")[
            ["support_count"]
        ].to_dict()["support_count"]

        # min_alpha is the minimum divergence
        min_alpha = min(fp_divergence_i.freq_metrics[metric_i].values)

        # idx is the index of the pattern to mitigate
        idx = 0
        n_iter = 0
        j_all_iterations = 0

        if n_iterations is None:
            n_iterations = divergent_fp_neg_df.shape[0]

        pbar = tqdm(
            total=n_iterations, desc="Mitigated patterns", position=0, leave=True
        )

        res = self.stats_mitigation_minmax(
            target=target,
            fp_divergence_i=fp_divergence_i,
            fp_divergence_mit=fp_divergence_i,
            original_score=dfb[target],
            mitigated_score=dfb[target],
            K=K,
            t_col=t_col,
            metric_i=metric_i,
            min_sup=min_sup,
        )

        # Minimum alpha
        res["min_alpha"] = min_alpha

        # Number of iterations --> it corresponds to the mitigated patterns
        res["#iteration"] = 0

        # Complete number of iterations --> it corresponds to all tests (so also when the mitigation was not possible)
        res["#iteration_all"] = 0
        # Mitigated pattern at iteration n_iter+1
        res["mitigated"] = None

        # Number of distinc mitigated patterns
        res["#distinct_mitigated"] = 0

        res["time"] = 0

        all_results[0] = res

        # patterns_after_mitigation = deepcopy(fp_divergence_i.freq_metrics)

        # We iterate til n_iterations
        while n_iter < n_iterations:
            # TODO
            if len(ordered_fp_divergent_df) <= idx:
                break
            j_all_iterations += 1
            if verbose:
                print("j_all_iterations", j_all_iterations)
            if verbose:
                print("i", idx)

            # We take the pattern to mitigate.
            # idx is initialized at 0
            itemset_info = ordered_fp_divergent_df.iloc[idx].to_dict()
            itemset_mitigate = itemset_info["itemsets"]
            itemset_support_cnt = itemset_info["support_count"]
            itemset_divergence = itemset_info[metric_i]

            # We get the tau_cap: maximum mitigation we can apply
            tau_cap = self.get_cap_value(
                itemset_mitigate,
                itemset_support_cnt,
                patterns_info,
                df_mit,
                pattern_support_dict,
                min_alpha=min_alpha,
                small_v=small_v,
            )

            # The applied mitigation is the minimum between the itemset divergence (we would like to fully mitigate it)
            # and the maximum mitigation we can apply, tau_cap
            delta_increment = min(abs(itemset_divergence), tau_cap)

            # Smoother mitigation if ratio_mitigation<0
            delta_increment = ratio_mitigation * delta_increment

            # We check if we can apply the mitigation.
            if delta_increment == 0:
                # TODO --> tau_cap is 0 when no mitigation can be applied
                # If we cannot apply a mitigation, i.e., delta_increment is 0
                # We pass to the next pattern to mitigate.
                idx += 1
                if idx >= ordered_fp_divergent_df.shape[0]:
                    # If there are no other patterns to mitigate, we stop the iteration process
                    break
                else:
                    continue
            else:
                # If we can apply a mitigation, in the next step we start again from the first pattern (we always start from the first)
                idx = 0

            # Mitigate an individual pattern based on the computed delta_increment
            df_mit = self.mitigate_pattern(
                df_mit,
                itemset_mitigate,
                delta_increment,
                target=target,
                lower_others=lower_others,
            )

            # We update the number of iterations --> it represents the number of mitigated patterns
            n_iter += 1

            pbar.update(1)

            # We extract again the frequent pattern and we re-compute again their divergence
            # TODO --> AVOID THE RECOMPUTATION! We can compute it analytically
            # -----------------

            fp_divergence_mit = get_fpdiv(df_mit, target, min_sup=min_sup)
            new_negative_fp_divergent = get_negatively_divergent(fp_divergence_mit)

            # -----------------
            """
            Not included since it not yet support statistial significance
            patterns_after_mitigation = self.update_pattern_divergence(
                patterns_after_mitigation,
                itemset_info,
                itemset_mitigate,
                delta_increment,
                df_mit,
                lower_others=lower_others,
            )
            patterns_mit = fp_divergence_mit.freq_metrics
            display(patterns_after_mitigation.head())
            display(patterns_mit.head())
            _check_equal_as_computed(patterns_after_mitigation, patterns_mit, metric_i)
            """
            # -----------------

            divergent_fp_neg_df = fp_divergence_mit.freq_metrics.sort_values(
                metric_i
            ).round(round_v)
            divergent_fp_neg_df = deepcopy(
                divergent_fp_neg_df.loc[divergent_fp_neg_df[metric_i] < 0]
            )
            patterns_info = deepcopy(fp_divergence_mit.freq_metrics)
            ordered_fp_divergent_df = self._ordered_divergent_patterns(
                divergent_fp_neg_df, sort_by_value, ascending
            )
            # -----------------

            # Current min_alpha
            current_min_alpha = min(fp_divergence_mit.freq_metrics[metric_i].values)

            assert round(current_min_alpha, 5) >= round(
                min_alpha, 5
            ), f"Error. Non-monotone process! {current_min_alpha} > {min_alpha}. {round(current_min_alpha, 5)} > {round(min_alpha, 5)}"

            if current_min_alpha == min_alpha:
                print(f"{current_min_alpha} == {min_alpha}")

            # Set current min_alpha as min_alpha
            min_alpha = current_min_alpha

            # For stats purposes: we keep the list of mitigated pattern
            all_mitigated_itemsets.append((itemset_mitigate, n_iter))

            res = self.stats_mitigation_minmax(
                target=target,
                fp_divergence_i=fp_divergence_i,
                fp_divergence_mit=fp_divergence_mit,
                original_score=dfb[target],
                mitigated_score=df_mit[target],
                K=K,
                t_col=t_col,
                metric_i=metric_i,
                min_sup=min_sup,
            )

            # Minimum alpha
            res["min_alpha"] = min_alpha

            # Number of iterations --> it corresponds to the mitigated patterns
            res["#iteration"] = n_iter

            # Complete number of iterations --> it corresponds to all tests (so also when the mitigation was not possible)
            res["#iteration_all"] = j_all_iterations + 1
            # Mitigated pattern at iteration n_iter+1
            res["mitigated"] = itemset_mitigate

            # Number of distinc mitigated patterns
            res["#distinct_mitigated"] = len(
                set([i[0] for i in all_mitigated_itemsets])
            )

            res["time"] = time.time() - start_time

            all_results[n_iter] = res

            # If there are no significant divergent pattern to mitigate
            # and we want to break the iteration process when there are no more statistical significant pattern with negative divergence
            if (new_negative_fp_divergent.shape[0] == 0) and break_if_no_significant:
                break

            # If set, the process terminates if the minimum negative divergence is greater than the input value
            if break_if_greater_than:
                if min_alpha > break_if_greater_than:
                    break

        pbar.close()

        all_results_df = pd.DataFrame(all_results).T

        return all_results_df, df_mit

    def update_pattern_divergence(
        self,
        patterns_divergence,
        mitigated_itemset_info,
        mitigated_itemset,
        delta_increment,
        df_data,
        fp_divergence_support_orig,
        lower_others=True,
        target="outcome",
        metric_i="d_outcome",
        t_value_col="t_value_outcome",
    ):
        """ " Update pattern divergence analitically
        patterns_divergence (pd.DataFrame): patterns with their divergence info
        mitigated_itemset_info (pd.Series): info of the mitigated pattern
        mitigated_itemset (frozenset): mitigated pattern
        delta_increment (float): delta increment applied to mitigate the pattern
        """

        if metric_i != "d_outcome":
            raise ValueError()

        mitigated_support_cnt = mitigated_itemset_info["support_count"]

        import time

        ss = time.time()

        # TODO Maybe not necessary
        patterns_after_mitigation = deepcopy(patterns_divergence)

        patterns_after_mitigation[target] = [
            self.update_after_mitigation_by_row(
                current_itemset,
                current_support_cnt,
                current_outcome,
                mitigated_itemset,
                mitigated_support_cnt,
                delta_increment,
                fp_divergence_support_orig,
                df_data,
                lower_others=lower_others,
            )
            for current_itemset, current_support_cnt, current_outcome in zip(
                patterns_after_mitigation.itemsets,
                patterns_after_mitigation.support_count,
                patterns_after_mitigation[target],
            )
        ]
        if patterns_after_mitigation.loc[0].itemsets != frozenset():
            raise ValueError()
        patterns_after_mitigation[metric_i] = (
            patterns_after_mitigation[target] - patterns_after_mitigation.loc[0][target]
        )
        if t_value_col in patterns_after_mitigation.columns:
            patterns_after_mitigation.drop(columns=[t_value_col], inplace=True)

        return patterns_after_mitigation

    def update_after_mitigation_by_row(
        self,
        current_itemset,
        current_support_cnt,
        current_outcome,
        mitigated_itemset,
        mitigated_support_cnt,
        delta_increment,
        fp_divergence_support_orig,
        df_data,
        lower_others=True,
    ):
        len_dataset = len(df_data)
        if lower_others:
            tot_increment = delta_increment * mitigated_support_cnt
            other_decrement = tot_increment / (len_dataset - mitigated_support_cnt)

        else:
            other_decrement = 0

        if current_itemset == frozenset():
            if lower_others:
                new_outcome = current_outcome
            else:
                new_outcome = current_outcome + (
                    delta_increment * mitigated_support_cnt
                ) / len(df_data)
        elif current_itemset.issuperset(mitigated_itemset):
            # It is a superset. Hence if we mitigate mitigated_itemset, we mitigate also ALL its supersets
            new_outcome = current_outcome + delta_increment
        elif current_itemset.issubset(mitigated_itemset):
            # It is a subset.
            new_outcome = (
                current_outcome
                + delta_increment * (mitigated_support_cnt) / current_support_cnt
                - other_decrement
                * (current_support_cnt - mitigated_support_cnt)
                / current_support_cnt
            )
        elif (current_itemset.issuperset(mitigated_itemset) == False) and (
            current_itemset.issubset(mitigated_itemset) == False
        ):
            union_itemset = mitigated_itemset.union(current_itemset)

            # if [item_i.split("=")[0] for item_i in current_itemset if item_i.split("=")[0] in [item_j.split("=")[0] for item_j in mitigated_itemset]]:
            if len(set([item[0] for item in union_itemset])) != len(union_itemset):
                union_itemset_support = 0
                new_outcome = current_outcome - other_decrement
            else:
                union_itemset = mitigated_itemset.union(current_itemset)
                if union_itemset not in fp_divergence_support_orig:
                    # if union_itemset not in fp_divergence_support_orig: # If was a dictionary
                    union_itemset_support = len(
                        slice_by_itemset(df_data, union_itemset)
                    )
                    # We have to slice the dataset searching for union_itemset_info
                    # raise ValueError(union_itemset)
                    # We update the support of the union_itemset
                    fp_divergence_support_orig[union_itemset] = union_itemset_support
                else:
                    # Not clear with loc not working..
                    union_itemset_support = fp_divergence_support_orig[union_itemset]
                    # union_itemset_support = fp_divergence_support_orig[union_itemset][
                    #    "support_count"
                    # ]
            new_outcome = (
                current_outcome
                + delta_increment * (union_itemset_support) / current_support_cnt
                - other_decrement
                * (current_support_cnt - union_itemset_support)
                / current_support_cnt
            )
        else:
            new_outcome = None

        # print(current_itemset_info["itemsets"], current_itemset_info["outcome"], current_itemset_info["support_count"], "A", mitigated_itemset_info["itemsets"], mitigated_itemset_info["support_count"], new_outcome)
        return new_outcome

    def iterative_mitigate_minmax_TEST(
        self,
        dfb,
        target="score",
        metric_i="d_outcome",
        sort_by_value="d_outcome",
        ascending=False,
        t_col="t_value_outcome",
        break_if_no_significant=False,
        break_if_greater_than=None,
        lower_others=True,
        n_iterations=None,
        K=100,
        verbose=False,
        small_v=0.99,
        round_v=10,
        ratio_mitigation: float = 1,
        min_sup=0.05,
    ):
        """
        Iteratively mitigate divergent pattern. Monotone process

        Args:
            dfb (pd.DataFrame) : input data, instances to mitigate with their score or rank
            target (str): target column for the mitigation process, i.e., name of the value/score column. "score" as default
            metric_i (str): divergence metric to mitigate. "d_outcome" as default
            sort_by_value (str): sorting value for the mitigation process. Divergence metric ("d_outcome") as default
            ascending (bool): order of the mitigation. Default 'False', sorted for desceding order.

            break_if_no_significant (bool): if True, the process terminates if there are no more significant negatively divergent patterns.
            break_if_greater_than (int): if set, the process terminates if the minimum divergence is greater than this value

            # TODO
            t_col (str): column name for the statistical significance. Default "t_value_outcome"


            lower_others (bool): if True, the score/value of the instances not in the subgroups is decreased accordingly to presere the average

            round_v (int): round to the round_v

            ratio_mitigation (float): ratio of mitigation. 1 --> try to mitigate as much as it can. The lower the value, the lower the mitigation.

        Returns:
            (   pd.DataFrame: the stats of the mitigation process
                pd.DataFrame: the mitigated instance. The instances and score/rank after mitigation
            )
        """

        # Check parameters
        if ratio_mitigation <= 0 or ratio_mitigation > 1:
            raise ValueError(f"{ratio_mitigation} must be between 0 and 1")

        # df_mit will store the (output) mitigated DataFrame
        df_mit = deepcopy(dfb)

        # Get the frequent patterns and their divergence

        import time

        start_time_divergence_evaluation = time.time()
        fp_divergence_i = get_fpdiv(df_mit, target, min_sup=min_sup)
        end_time_divergence_evaluation = time.time()

        # Execution time of the divergence evaluation
        execution_time_divergence_evaluation = (
            end_time_divergence_evaluation - start_time_divergence_evaluation
        )

        # Sort the patterns by metric_i (and round them TODO)
        divergent_fp_neg_df = fp_divergence_i.freq_metrics.sort_values(metric_i).round(
            round_v
        )
        # Get the negatively divergent patterns
        divergent_fp_neg_df = divergent_fp_neg_df.loc[divergent_fp_neg_df[metric_i] < 0]

        # Order the negatively divergent patterns by sort_by_value in "ascending value" order
        ordered_fp_divergent_df = self._ordered_divergent_patterns(
            divergent_fp_neg_df, sort_by_value, ascending
        )

        from tqdm import tqdm

        # Stats
        start_time = time.time()

        all_results = {}

        all_mitigated_itemsets = []

        # Init
        patterns_info = deepcopy(fp_divergence_i.freq_metrics)

        # Dictionary {FPattern: support count}

        sss = time.time()
        pattern_support_dict = patterns_info.set_index("itemsets")[
            ["support_count"]
        ].to_dict()["support_count"]

        # min_alpha is the minimum divergence
        min_alpha = min(fp_divergence_i.freq_metrics[metric_i].values)

        # idx is the index of the pattern to mitigate
        idx = 0
        n_iter = 0
        j_all_iterations = 0

        if n_iterations is None:
            n_iterations = divergent_fp_neg_df.shape[0]

        pbar = tqdm(
            total=n_iterations, desc="Mitigated patterns", position=0, leave=True
        )

        res = self.stats_mitigation_minmax(
            target=target,
            fp_divergence_i=fp_divergence_i,
            fp_divergence_mit=fp_divergence_i,
            original_score=dfb[target],
            mitigated_score=dfb[target],
            K=K,
            t_col=t_col,
            metric_i=metric_i,
            min_sup=min_sup,
        )

        # Minimum alpha
        res["min_alpha"] = min_alpha

        # Number of iterations --> it corresponds to the mitigated patterns
        res["#iteration"] = 0

        # Complete number of iterations --> it corresponds to all tests (so also when the mitigation was not possible)
        res["#iteration_all"] = 0
        # Mitigated pattern at iteration n_iter+1
        res["mitigated"] = None

        # Number of distinc mitigated patterns
        res["#distinct_mitigated"] = 0

        res["time"] = 0

        all_results[0] = res

        # We use it to update the patterns after mitigation
        patterns_after_mitigation = deepcopy(fp_divergence_i.freq_metrics)

        # We iterate til n_iterations
        while n_iter < n_iterations:
            # TODO
            if len(ordered_fp_divergent_df) <= idx:
                break
            j_all_iterations += 1
            if verbose:
                print("j_all_iterations", j_all_iterations)
            if verbose:
                print("i", idx)

            # We take the pattern to mitigate.
            # idx is initialized at 0
            itemset_info = ordered_fp_divergent_df.iloc[idx].to_dict()
            itemset_mitigate = itemset_info["itemsets"]
            itemset_support_cnt = itemset_info["support_count"]
            itemset_divergence = itemset_info[metric_i]

            # We get the tau_cap: maximum mitigation we can apply
            tau_cap = self.get_cap_value(
                itemset_mitigate,
                itemset_support_cnt,
                patterns_info,
                df_mit,
                pattern_support_dict,
                min_alpha=min_alpha,
                small_v=small_v,
            )

            # The applied mitigation is the minimum between the itemset divergence (we would like to fully mitigate it)
            # and the maximum mitigation we can apply, tau_cap
            delta_increment = min(abs(itemset_divergence), tau_cap)

            # Smoother mitigation if ratio_mitigation<0
            delta_increment = ratio_mitigation * delta_increment

            # We check if we can apply the mitigation.
            if delta_increment == 0:
                # TODO --> tau_cap is 0 when no mitigation can be applied
                # If we cannot apply a mitigation, i.e., delta_increment is 0
                # We pass to the next pattern to mitigate.
                idx += 1
                if idx >= ordered_fp_divergent_df.shape[0]:
                    # If there are no other patterns to mitigate, we stop the iteration process
                    break
                else:
                    continue
            else:
                # If we can apply a mitigation, in the next step we start again from the first pattern (we always start from the first)
                idx = 0

            # Mitigate an individual pattern based on the computed delta_increment
            df_mit = self.mitigate_pattern(
                df_mit,
                itemset_mitigate,
                delta_increment,
                target=target,
                lower_others=lower_others,
            )

            # We update the number of iterations --> it represents the number of mitigated patterns
            n_iter += 1

            pbar.update(1)
            avoid_recompute = False

            YY_TIME = 1

            # If it requires more than YY=1 seconds to recompute the divergence, we avoid the re-computation
            # Note that after X=50 iterations, we recompute the divergence anyway
            if (
                execution_time_divergence_evaluation > YY_TIME
                and j_all_iterations % 50 != 0
            ):
                avoid_recompute = True
                # If the execution time of the divergence evaluation is greater than YY seconds, we avoid recompute the divergence
                # WE AVOID THE RECOMPUTATION! We can compute it analytically

                patterns_after_mitigation = deepcopy(
                    self.update_pattern_divergence(
                        patterns_after_mitigation,
                        itemset_info,
                        itemset_mitigate,
                        delta_increment,
                        df_mit,
                        pattern_support_dict,  # IMPORTANT MODIFICATION PROVIDED AS INPUT
                        lower_others=lower_others,
                    )
                )
                patterns_info = deepcopy(patterns_after_mitigation)

                # FOR TESTING PURPOSES
                # We check if the update  is indeed correct

                """
                fp_divergence_mit_test = get_fpdiv(df_mit, target, min_sup=min_sup)
                patterns_mit_test = deepcopy(fp_divergence_mit_test.freq_metrics)
                _check_equal_as_computed(
                    deepcopy(patterns_after_mitigation), patterns_mit_test, metric_i
                )
                """

            else:
                avoid_recompute = False
                # If the execution time is lower than YY seconds, we recompute the divergence
                fp_divergence_mit = get_fpdiv(df_mit, target, min_sup=min_sup)
                new_negative_fp_divergent = get_negatively_divergent(
                    fp_divergence_mit, th_redundancy=None
                )

                patterns_after_mitigation = deepcopy(fp_divergence_mit.freq_metrics)
                patterns_after_mitigation.drop(columns=[t_col], inplace=True)
                patterns_info = deepcopy(fp_divergence_mit.freq_metrics)

            divergent_fp_neg_df = patterns_after_mitigation.sort_values(metric_i).round(
                round_v
            )

            divergent_fp_neg_df = deepcopy(
                divergent_fp_neg_df.loc[divergent_fp_neg_df[metric_i] < 0]
            )

            ordered_fp_divergent_df = self._ordered_divergent_patterns(
                divergent_fp_neg_df, sort_by_value, ascending
            )
            # -----------------

            # Current min_alpha
            current_min_alpha = min(patterns_after_mitigation[metric_i].values)

            assert round(current_min_alpha, 5) >= round(
                min_alpha, 5
            ), f"Error. Non-monotone process! {current_min_alpha} > {min_alpha}. {round(current_min_alpha, 5)} > {round(min_alpha, 5)}"

            if current_min_alpha == min_alpha:
                print(f"{current_min_alpha} == {min_alpha}")

            # Set current min_alpha as min_alpha
            min_alpha = current_min_alpha

            # For stats purposes: we keep the list of mitigated pattern
            all_mitigated_itemsets.append((itemset_mitigate, n_iter))

            res = self.stats_mitigation_minmax(
                target=target,
                fp_divergence_i=fp_divergence_i,
                fp_divergence_mit=fp_divergence_mit
                if avoid_recompute is False
                else None,
                original_score=dfb[target],
                mitigated_score=df_mit[target],
                K=K,
                t_col=t_col,
                metric_i=metric_i,
                min_sup=min_sup,
                avoid_recompute=avoid_recompute,
                frequent_patterns_after_mitigation=patterns_after_mitigation,
            )

            # Minimum alpha
            res["min_alpha"] = min_alpha

            # Number of iterations --> it corresponds to the mitigated patterns
            res["#iteration"] = n_iter

            # Complete number of iterations --> it corresponds to all tests (so also when the mitigation was not possible)
            res["#iteration_all"] = j_all_iterations + 1
            # Mitigated pattern at iteration n_iter+1
            res["mitigated"] = itemset_mitigate

            # Number of distinc mitigated patterns
            res["#distinct_mitigated"] = len(
                set([i[0] for i in all_mitigated_itemsets])
            )

            res["time"] = time.time() - start_time

            all_results[n_iter] = res

            # If there are no significant divergent pattern to mitigate
            # and we want to break the iteration process when there are no more statistical significant pattern with negative divergence
            # Note that we can do it only if we have compute the statistical significance
            if avoid_recompute is False:
                if (
                    new_negative_fp_divergent.shape[0] == 0
                ) and break_if_no_significant:
                    break

            # If set, the process terminates if the minimum negative divergence is greater than the input value
            if break_if_greater_than:
                if min_alpha > break_if_greater_than:
                    break

        pbar.close()

        all_results_df = pd.DataFrame(all_results).T

        return all_results_df, df_mit


def score_to_outcome(
    df: pd.DataFrame, target_name: str, outcome_type: str = "score", alpha: float = 0.03
) -> Tuple[pd.DataFrame, str]:
    from scipy import stats

    """ Generate outcome function
    Args:
        df: the input dataset
        target_name: the score/rank columns. Higher score = higher positions in the ranking
        outcome_type: ['score', 'inv_rank]
                    'score': the score in the target_name column itself is adopted
                    'inv_rank': the target is the inversed rank (higher positions, the better). It is derived from target_name
                    'utility': rank(x)**(-alpha), the rank is derived from target_name
                    'utility_norm': normalized rank(x)**(-alpha) with min-max normalization. In the range 0-100 the rank is derived from target_name.
        alpha: for the utility outcome

    Returns:
        pd.DataFrame: the dataframe with the target outcome
        str: the target column name

    """
    from copy import deepcopy

    if outcome_type == "score":
        df_analysis = deepcopy(df)
        return df_analysis, target_name
    elif outcome_type == "inv_rank":
        target = "inv_rank"
        dfbr = deepcopy(df)
        dfbr[target] = stats.rankdata(np.array(dfbr[target_name]), "ordinal")
        dfbr.drop(columns=[target_name], inplace=True)

        return dfbr, target

    elif outcome_type == "utility" or outcome_type == "utility_norm":
        target = outcome_type

        dfu = deepcopy(df)

        dfu["rank"] = (
            len(dfu) - stats.rankdata(np.array(dfu[target_name]), "ordinal") + 1
        )
        dfu[target] = dfu["rank"] ** (-alpha)

        if target == "utility_norm":
            dfu[target] = (
                (dfu[target] - min(dfu[target]))
                * 100
                / (max(dfu[target]) - min(dfu[target]))
            )
        dfu.drop(columns=[target_name, "rank"], inplace=True)
        return dfu, target
    else:
        raise ValueError(f"{outcome_type} not admitted")
