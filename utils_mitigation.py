from copy import deepcopy
from tkinter import N
from divexplorer.FP_Divergence import FP_Divergence
from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils_plot import compareShapleyValues
import time


def slice_by_itemset(df: pd.DataFrame, itemset) -> pd.DataFrame:
    """Slice the dataFrame to select the instances satisfying the itemset
    Args:
        df (pd.DataFrame): the input table
        itemset (frozenset): the itemset

    Returns:
        pd.DataFrame: the slice of the data satifying the itemset
    """

    indexes = df.index
    for item in itemset:
        s = item.split("=")
        attr, value = s[0], "=".join(s[1:])
        indexes = df.loc[indexes].loc[df[attr].astype(str) == value].index
    return df.loc[indexes]


def get_fpdiv(
    df,
    target,
    metric_i="d_outcome",
    min_sup=0.05,
):
    fp_diver = FP_DivergenceExplorer(df, target_name=target)
    FP_fm = fp_diver.getFrequentPatternDivergence(
        metrics=[metric_i], min_support=min_sup
    )
    fp_divergence_i = FP_Divergence(FP_fm, metric_i)
    return fp_divergence_i


def delta_increment_f(
    itemset_info, metric_i, alpha=1, len_attributes=None, type_f="div"
):
    if type_f == "div":
        delta_increment = -1 * (itemset_info[metric_i])
    elif type_f == "div_len":
        delta_increment = -1 * (itemset_info[metric_i]) + (
            len_attributes / itemset_info["length"]
        )  # (itemset_info["support"])
    else:
        raise ValueError(f"{type_f} is not implemented")
    delta_increment = alpha * delta_increment
    return delta_increment


def get_negatively_divergent(
    fp_divergence_i,
    metric_i="d_outcome",
    th_redundancy=0,
    perc=True,
    t="t_value_outcome",
    significant=True,
    sort_by="signif",
):
    if th_redundancy is not None and th_redundancy > 0:
        if perc:
            if fp_divergence_i.freq_metrics.iloc[0]["itemsets"] != frozenset():
                raise ValueError()
            mean_outcome = fp_divergence_i.freq_metrics.iloc[0]["outcome"]
            th_redundancy = mean_outcome * th_redundancy
        print(f"Redundancy threshold: {th_redundancy:.2f}")
    fp = fp_divergence_i.getDivergence(th_redundancy=th_redundancy)
    fp = fp.loc[fp[metric_i] < 0]
    if significant:
        fp = fp.loc[fp[t] > 2][::-1]
    if sort_by == "signif":
        fp = fp.sort_values(t, key=abs, ascending=False)
    if sort_by == "div":
        fp = fp.sort_values(metric_i, key=abs, ascending=False)
    return fp


def get_divergent(
    fp_divergence_i,
    # metric_i="d_outcome",
    th_redundancy=0,
    significant=True,
    t="t_value_outcome",
):
    if th_redundancy is not None and th_redundancy > 0:
        if fp_divergence_i.freq_metrics.iloc[0]["itemsets"] != frozenset():
            raise ValueError()
        mean_outcome = fp_divergence_i.freq_metrics.iloc[0]["outcome"]
        th_redundancy = mean_outcome * th_redundancy
        print(f"Redundancy threshold: {th_redundancy:.2f}")
    fp = fp_divergence_i.getDivergence(th_redundancy=th_redundancy)
    if significant:
        fp = fp.loc[fp[t] > 2][::-1]
    return fp


def greedy_mitigation(
    df,
    target,
    len_attributes,
    metric_i="d_outcome",
    n_iterations=5,
    alpha=0.4,
    type_f="div",
    min_sup=0.05,
):
    df_mit = deepcopy(df)
    all_mitigated_itemsets = []
    if n_iterations is None:
        fp_divergence_to_mit = get_fpdiv(df_mit, target, min_sup=min_sup)
        divergent_fp_neg_df = get_negatively_divergent(fp_divergence_to_mit)
        n_iterations = divergent_fp_neg_df.shape[0]

    for iteration in range(0, n_iterations):

        fp_divergence_to_mit = get_fpdiv(df_mit, target, min_sup=min_sup)
        divergent_fp_neg_df = get_negatively_divergent(fp_divergence_to_mit)
        if divergent_fp_neg_df.shape[0] == 0:
            break

        mitigated_itemsets = []
        for i in range(len(divergent_fp_neg_df)):

            itemset_info = divergent_fp_neg_df.iloc[i]
            itemset = itemset_info.itemsets

            # subset_mitigated_detail = [(mitigated_itemset, mitigated_itemset.issubset(itemset)) for mitigated_itemset in  mitigated_itemsets]
            subset_mitigated = [
                mitigated_itemset
                for mitigated_itemset in mitigated_itemsets
                if mitigated_itemset.issubset(itemset)
            ]
            superset_mitigated = [
                mitigated_itemset
                for mitigated_itemset in mitigated_itemsets
                if mitigated_itemset.issuperset(itemset)
            ]

            if len(subset_mitigated) == 0 and len(superset_mitigated) == 0:

                mitigated_itemsets.append(itemset)

                # display(divergent_fp_neg_df.iloc[i:i+1])
                df_sel = slice_by_itemset(df_mit, itemset)

                delta_increment = delta_increment_f(
                    itemset_info,
                    metric_i,
                    len_attributes=len_attributes,
                    alpha=alpha,
                    type_f=type_f,
                )

                all_mitigated_itemsets.append((itemset, delta_increment, iteration))

                # print(f"Delta increment: {delta_increment}")
                df_mit.loc[df_sel.index, target] = (
                    df_mit.loc[df_sel.index][target] + delta_increment
                )

    return df_mit, all_mitigated_itemsets


def get_stats_mitigation(df_mit, target, min_sup=0.05):
    fp_divergence_mit = get_fpdiv(df_mit, target, min_sup=min_sup)

    fp_div = get_divergent(fp_divergence_mit)
    fp_mit = get_negatively_divergent(fp_divergence_mit)

    info_mitigation = {}
    if fp_div.shape[0] == 0:
        info_mitigation["#div"] = 0
    else:
        info_mitigation["#div"] = len(fp_div)

    if fp_mit.shape[0] == 0:
        info_mitigation["#div_neg"] = 0
    else:
        info_mitigation["#div_neg"] = len(fp_mit)
    return info_mitigation


def get_pattern_mitigation(df_mit, target, min_sup=0.05):
    fp_divergence_mit = get_fpdiv(df_mit, target, min_sup=min_sup)

    # fp_div = get_divergent(fp_divergence_mit)
    fp_mit = get_negatively_divergent(fp_divergence_mit)
    return fp_mit


def print_comparison_mitigation(
    dfb,
    df_mit,
    target,
    fp_divergence_i=None,
    plot_global_shapley=False,
    display_pos=False,
    display_res=False,
    min_sup=0.05
):

    fp_divergence_mit = get_fpdiv(df_mit, target, min_sup=min_sup)

    fp_div = get_divergent(fp_divergence_mit)
    fp_mit = get_negatively_divergent(fp_divergence_mit)

    if fp_div.shape[0] != 0:
        if display_res:
            print(f"#Divergent patterns: {len(fp_div)}")
        if display_pos:
            display(fp_div)

    if fp_mit.shape[0] != 0:

        if display_res:
            print(f"#Negatively divergent patterns: {len(fp_mit)}")
            display(fp_mit)
    else:
        if display_res:
            print("There is no negatively divergent pattern.")

    if plot_global_shapley:
        if fp_divergence_i is None:
            fp_divergence_i = get_fpdiv(dfb, target)

        sym_appr = r"$\tilde{\Delta}^g$"

        compareShapleyValues(
            (fp_divergence_i.computeGlobalShapleyValue()),
            (fp_divergence_mit.computeGlobalShapleyValue()),
            title=[f"{sym_appr} - original", f"{sym_appr} - mitigation"],
            labelsize=8,
            height=0.5,
            sizeFig=(3, 2),
            shared_x=True,
        )

    from sklearn.metrics import mean_squared_error
    from scipy import stats

    if display_res:
        print(f"MSE: {mean_squared_error(dfb[target], df_mit[target]):.3f}")
        print(
            f"Spearman rank corr: {stats.spearmanr(dfb[target].values, df_mit[target])[0]:.3f}"
        )

    return {
        "#div": len(fp_div),
        "#negdiv": len(fp_mit),
        "MSE": mean_squared_error(dfb[target], df_mit[target]),
        "rankcorr": stats.spearmanr(dfb[target].values, df_mit[target])[0],
    }


def compare_mitigate_itemset(
    itemset,
    target,
    metric_i,
    fp_divergence_i=None,
    fp_divergence_mit=None,
    dfb=None,
    df_mit=None,
    t="t_value_outcome",
    min_sup=0.05,
):
    if fp_divergence_i is None and dfb is None:
        raise ValueError("Specify dfb or fp_divergence_i")
    if fp_divergence_i is None:
        fp_divergence_i = get_fpdiv(dfb, target)

    if fp_divergence_mit is None and df_mit is None:
        raise ValueError("Specify df_mit or fp_divergence_mit")
    if fp_divergence_mit is None:
        fp_divergence_mit = get_fpdiv(df_mit, target, min_sup=min_sup)

    info_itemset_before = dict(fp_divergence_i.getInfoItemset(itemset).iloc[0])
    info_itemset_mitigated = dict(fp_divergence_mit.getInfoItemset(itemset).iloc[0])

    print(
        f"{itemset} - Divergence from {info_itemset_before[metric_i]:.3f} to {info_itemset_mitigated[metric_i]:.3f}, t_value from {info_itemset_before[t]:.2f} to {info_itemset_mitigated[t]:.2f}"
    )


def get_min_for_configuration(result_df_dict, parameters_grid):

    vs = []

    for sort_by_value, ascending_by_value in parameters_grid:

        d = result_df_dict[(sort_by_value, ascending_by_value)]
        min_n_negdiv = min(d["#negdiv"])
        min_mse = min(d.loc[d["#negdiv"] == min_n_negdiv]["MSE"])
        best = d.loc[(d["#negdiv"] == min_n_negdiv) & (d["MSE"] == min_mse)].iloc[0]
        vs.append([best.name, sort_by_value, ascending_by_value] + list(best.values))
    res = pd.DataFrame(
        vs, columns=["eps", "sort_by_value", "ascending_by_value"] + list(d.columns)
    )
    return res.sort_values(["#negdiv", "MSE"])


def compare_results_grid(
    result_df_dict,
    parameters_grid,
    max_show=3,
    figsize=(7, 7),
    save_fig=False,
    name_fig=None,
    show_figure=False,
    show_points=False,
):
    n_rows = len(result_df_dict) / 2
    fig, axs = plt.subplots(3, 2, figsize=figsize, sharex=True, sharey=True)
    for e, k_values in enumerate(parameters_grid):
        i, j = int(e / 2), e % 2
        d = result_df_dict[k_values]
        min_n_negdiv = min(d["#negdiv"])
        min_mse = min(d.loc[d["#negdiv"] == min_n_negdiv]["MSE"])
        best = d.loc[(d["#negdiv"] == min_n_negdiv) & (d["MSE"] == min_mse)].iloc[0]
        if best.name >= max_show:
            raise ValueError()
        d = d.loc[d.index <= max_show]
        marker = "." if show_points else None
        axs[i][j].plot(d["#div"], label="#div")
        axs[i][j].plot(d["#negdiv"], label="#neg", marker=marker)
        axs[i][j].plot(d["MSE"], label="MSE")
        str_params = " ".join(map(str, k_values))
        epsilon_value = best.name if "epsilon_value" not in best else best.epsilon_value
        axs[i][j].set_title(
            f"min#: {int(min_n_negdiv)} eps: {epsilon_value:.2f} MSE:  {min_mse:.2f} \n {str_params}"
        )
        axs[i][j].legend()
    plt.tight_layout()
    if save_fig:
        if name_fig is None:
            raise ValueError("Specify the path: name_fig")
        plt.savefig(name_fig, bbox_inches="tight")
    if show_figure:
        plt.show()
        plt.close()


"""
def print_info(df_fp, info_col = None,  col_names_replace = None):
    if info_col is None:
        info_col = ["support", "itemsets", "d_outcome", "t_value_outcome"]
    if col_names_replace is None:
        col_names_replace = {"itemsets": "itemset", "d_outcome": "Δ", "t_value_outcome": "t"}
    return df_fp[info_col].rename(columns = col_names_replace)
"""


def variance(a):
    return sum((a - np.mean(a)) ** 2) / len(a)


def _compute_t_test_welch(a, b):
    return (abs(np.mean(a) - np.mean(b))) / (
        (variance(a) / len(a) + variance(b) / len(b)) ** 0.5
    )


def plot_stats(all_results_df, max_show=None, figsize=(3, 3), show_figure=True):

    min_n_negdiv = min(all_results_df["#negdiv"])
    min_mse = min(all_results_df.loc[all_results_df["#negdiv"] == min_n_negdiv]["MSE"])
    best = all_results_df.loc[
        (all_results_df["#negdiv"] == min_n_negdiv) & (all_results_df["MSE"] == min_mse)
    ].iloc[0]
    if max_show is None:
        max_show = max(all_results_df.index)
    if best.name >= max_show:
        raise ValueError()
    d = all_results_df.loc[all_results_df.index <= max_show]
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(d["#div"], label="#div")
    plt.plot(d["#negdiv"], label="#neg")
    plt.plot(d["MSE"], label="MSE")
    plt.title(f"Min #neg_div: {min_n_negdiv:.2f}, min MSE:  {min_mse:.2f}")
    plt.legend()
    if show_figure:
        plt.show()
        plt.close()

    return fig


from tqdm import tqdm


def iterative_mitigation(
    dfb,
    divergent_patterns_df,
    sort_by_value,
    ascending=False,
    only_top=False,
    break_if_found=True,
    metric_i="d_outcome",
    target="score",
    epsilons=None,
    show_fig=False,
    update_divergent_patterns=False,
    n_iterations=None,
    figsize=(4, 3),
    use_support=False,
    min_sup=0.05,
):
    fp_divergence_i = get_fpdiv(dfb, target)

    if epsilons is None:
        epsilons = np.arange(0, 2.05, 0.05)

    all_results = {}

    if sort_by_value not in divergent_patterns_df:
        raise ValueError(
            f"{sort_by_value} not available. Specify another term in {list(divergent_patterns_df.columns)}"
        )

    ordered_fp_divergent_df = divergent_patterns_df.sort_values(
        sort_by_value, key=abs, ascending=ascending
    )

    if only_top:
        # We mitigate only the top k
        n_iterations = 1
    else:
        # We mitigate all terms

        n_iterations = len(ordered_fp_divergent_df)

    for eps in tqdm(epsilons):
        df_mit = deepcopy(dfb)
        start_time = time.time()
        n_mitigated = 0
        all_mitigated_itemsets = []
        for n_iter in range(0, n_iterations):
            if update_divergent_patterns:
                i = 0

            else:
                i = n_iter

            itemset_info = ordered_fp_divergent_df.iloc[i]
            divergence_itemset = itemset_info[metric_i]

            itemset = itemset_info.itemsets

            df_sel = slice_by_itemset(df_mit, itemset)

            if use_support:
                delta_increment = -eps * divergence_itemset + itemset_info.support
            else:
                delta_increment = -eps * divergence_itemset

            df_mit.loc[df_sel.index, target] = (
                df_mit.loc[df_sel.index][target] + delta_increment
            )
            n_mitigated += 1
            res = print_comparison_mitigation(
                dfb,
                df_mit,
                target,
                fp_divergence_i=fp_divergence_i,
                plot_global_shapley=False,
            )
            all_results[eps] = res
            if break_if_found:
                if res["#negdiv"] == 0:
                    break
            if update_divergent_patterns:
                fp_divergence_mit = get_fpdiv(df_mit, target, min_sup=min_sup)
                ordered_fp_divergent_df = get_negatively_divergent(
                    fp_divergence_mit
                ).sort_values(sort_by_value, key=abs, ascending=ascending)
                all_mitigated_itemsets.append((itemset, delta_increment, n_iter))
        res["#iteration"] = n_iter + 1
        res["time"] = time.time() - start_time
        res["#mitigated"] = n_mitigated
        if update_divergent_patterns:
            res["#distinct_mitigated"] = len(
                set([i[0] for i in all_mitigated_itemsets])
            )
        else:
            res["#distinct_mitigated"] = n_mitigated
        all_results[eps] = res

    all_results_df = pd.DataFrame(all_results).T
    if show_fig:

        fig = plot_stats(all_results_df, figsize=figsize)
    return all_results_df, df_mit


def iterative_mitigation_one_at_the_time(
    dfb,
    divergent_patterns_df,
    sort_by_value,
    ascending=False,
    only_top=False,
    break_if_found=True,
    metric_i="d_outcome",
    target="score",
    epsilons=None,
    show_fig=False,
    update_divergent_patterns=False,
    n_iterations=None,
    figsize=(4, 3),
    use_support=False,
    min_sup=0.05
):
    fp_divergence_i = get_fpdiv(dfb, target)

    if epsilons is None:
        epsilons = np.arange(0, 2.05, 0.05)

    all_results = {}

    if sort_by_value not in divergent_patterns_df:
        raise ValueError(
            f"{sort_by_value} not available. Specify another term in {list(divergent_patterns_df.columns)}"
        )

    ordered_fp_divergent_df = divergent_patterns_df.sort_values(
        sort_by_value, key=abs, ascending=ascending
    )

    if only_top:
        # We mitigate only the top k
        n_iterations = 1
    else:
        # We mitigate all terms
        if n_iterations is None:
            n_iterations = len(ordered_fp_divergent_df)

    df_mit = deepcopy(dfb)
    start_time = time.time()
    n_mitigated = 0
    all_mitigated_itemsets = []
    for n_iter in range(0, n_iterations):
        found = False
        if update_divergent_patterns:
            i = 0
        else:
            i = n_iter

        itemset_info = ordered_fp_divergent_df.iloc[i]
        divergence_itemset = itemset_info[metric_i]

        itemset = itemset_info.itemsets

        indexes_sel = slice_by_itemset(df_mit, itemset).index

        for eps in epsilons:

            df_mit_try_modify = deepcopy(df_mit)
            if use_support:
                delta_increment = -eps * divergence_itemset + itemset_info.support
            else:
                delta_increment = -eps * divergence_itemset

            df_mit_try_modify.loc[indexes_sel, target] = (
                df_mit_try_modify.loc[indexes_sel][target] + delta_increment
            )
            sel_values = df_mit_try_modify.loc[indexes_sel, target].values
            all_values = df_mit_try_modify[target].values
            # f_I = np.mean(sel_values)
            # f_D = np.mean(all_values)
            t_value = _compute_t_test_welch(sel_values, all_values)

            if t_value < 2:
                df_mit = df_mit_try_modify
                found = True
                n_mitigated += 1
                res = print_comparison_mitigation(
                    dfb,
                    df_mit,
                    target,
                    fp_divergence_i=fp_divergence_i,
                    plot_global_shapley=False,
                )
                break

        if update_divergent_patterns:
            fp_divergence_mit = get_fpdiv(df_mit, target, min_sup=min_sup)
            ordered_fp_divergent_df = get_negatively_divergent(
                fp_divergence_mit
            ).sort_values(sort_by_value, key=abs, ascending=ascending)
            all_mitigated_itemsets.append((itemset, delta_increment, n_iter))

        if found == True:
            res["#iteration"] = n_iter + 1
            res["time"] = time.time() - start_time
            res["#mitigated"] = n_mitigated
            res["epsilon_v"] = eps
            res["mitigated"] = itemset
            if update_divergent_patterns:
                res["#distinct_mitigated"] = len(
                    set([i[0] for i in all_mitigated_itemsets])
                )
            else:
                res["#distinct_mitigated"] = n_mitigated
            all_results[n_iter] = res
            if break_if_found:
                if res["#negdiv"] == 0:
                    break

    all_results_df = pd.DataFrame(all_results).T
    if show_fig:

        fig = plot_stats(all_results_df, figsize=figsize)
    return all_results_df, df_mit


def greedy_mitigation_grid(
    dfb,
    # divergent_patterns_df,
    sort_by_value,
    ascending=False,
    target="score",
    metric_i="d_outcome",
    n_iterations=5,
    epsilons=None,
    show_fig=False,
    use_support=False,
    update_others=False,
    min_sup=0.05,
):

    fp_divergence_i = get_fpdiv(dfb, target)

    if n_iterations is None:
        n_iterations = get_negatively_divergent(fp_divergence_i).shape[0]

    if epsilons is None:
        epsilons = np.arange(0, 3.05, 0.05)

    all_results = {}

    for eps in tqdm(epsilons):
        df_mit = deepcopy(dfb)
        all_mitigated_itemsets = []
        start_time = time.time()
        n_mitigated = 0
        for iteration in range(0, n_iterations):

            fp_divergence_to_mit = get_fpdiv(df_mit, target, min_sup=min_sup)
            divergent_patterns_df = get_negatively_divergent(fp_divergence_to_mit)
            if divergent_patterns_df.shape[0] == 0:
                break

            if eps == epsilons[0] and iteration == 0:
                if sort_by_value not in divergent_patterns_df:
                    raise ValueError(
                        f"{sort_by_value} not available. Specify another term in {list(divergent_patterns_df.columns)}"
                    )
            ordered_fp_divergent_df = divergent_patterns_df.sort_values(
                sort_by_value, key=abs, ascending=ascending
            )

            mitigated_itemsets = []
            for i in range(len(ordered_fp_divergent_df)):

                itemset_info = ordered_fp_divergent_df.iloc[i]
                itemset = itemset_info.itemsets
                divergence_itemset = itemset_info[metric_i]

                subset_mitigated = [
                    mitigated_itemset
                    for mitigated_itemset in mitigated_itemsets
                    if mitigated_itemset.issubset(itemset)
                ]
                superset_mitigated = [
                    mitigated_itemset
                    for mitigated_itemset in mitigated_itemsets
                    if mitigated_itemset.issuperset(itemset)
                ]

                if len(subset_mitigated) == 0 and len(superset_mitigated) == 0:

                    mitigated_itemsets.append(itemset)

                    df_sel = slice_by_itemset(df_mit, itemset)

                    if use_support:
                        delta_increment = (
                            -eps * divergence_itemset + itemset_info.support
                        )
                    else:
                        delta_increment = -eps * divergence_itemset

                    all_mitigated_itemsets.append((itemset, delta_increment, iteration))

                    # print(f"Delta increment: {delta_increment}")
                    df_mit.loc[df_sel.index, target] = (
                        df_mit.loc[df_sel.index][target] + delta_increment
                    )
                    if update_others:
                        tot_increment = delta_increment * len(df_sel.index)
                        decrease_other_value = tot_increment / len(
                            df_mit.index.difference(df_sel.index)
                        )
                        df_mit.loc[df_mit.index.difference(df_sel.index), target] = (
                            df_mit.loc[df_mit.index.difference(df_sel.index), target]
                            - decrease_other_value
                        )

                    n_mitigated += 1
        res = print_comparison_mitigation(
            dfb,
            df_mit,
            target,
            fp_divergence_i=fp_divergence_i,
            plot_global_shapley=False,
        )
        res["#iteration"] = iteration + 1
        res["time"] = time.time() - start_time
        res["#mitigated"] = n_mitigated
        res["#distinct_mitigated"] = len(set([i[0] for i in all_mitigated_itemsets]))

        all_results[eps] = res
        from sklearn.metrics import mean_squared_error

        mse = mean_squared_error(dfb[target], df_mit[target])

    all_results_df = pd.DataFrame(all_results).T
    if show_fig:

        fig = plot_stats(all_results_df)
    return all_results_df, df_mit
    # return df_mit, all_mitigated_itemsets


def data_and_outcome(data, target_name, target, alpha = 0.05):
    from scipy import stats 

    if target == "inv_rank":
        df_input = deepcopy(data)

        df_input["inv_rank"]= stats.rankdata( np.array(df_input[target_name]), "ordinal")
        df_input.drop(columns = [target_name], inplace=True)

    elif target == "utility":
        df_input = deepcopy(data)

        
        df_input["rank"]= len(df_input) - stats.rankdata( np.array(df_input[target_name]), "ordinal") +1
        df_input[target]=  df_input["rank"]**(-alpha)

        df_input.drop(columns = [target_name, "rank"], inplace=True)

    elif target == "score":
        df_input = deepcopy(data)
    return df_input