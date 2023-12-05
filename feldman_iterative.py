from copy import deepcopy
import numpy as np
from utils_evaluation import divergence_stats


import time

DONTCARE = "-"


def feldman_iterative(
    df_input,
    target,
    target_name,
    attributes,
    protected_attributes_list,
    protected_attribute_values_list,
    min_sup=0.1,
):
    time_feldman = {}

    ######################

    from utils_comparison import (
        get_protected_unprotected,
        feldman_processing,
        merge_ranking_protected_unnprotected,
    )

    mitigation_name = "feldman"

    ranking_results_feldman = {}
    res_mit_feldman = {}

    df_input_iterative = deepcopy(df_input)

    for protected_attributes, protected_values in zip(
        protected_attributes_list, protected_attribute_values_list
    ):
        # print(f"\n\nprotected_attributes: {protected_attributes}")
        pa_idxs = np.argsort(np.array(protected_attributes))
        protected_attributes = [protected_attributes[pa_idx] for pa_idx in pa_idxs]
        protected_values = [protected_values[pa_idx] for pa_idx in pa_idxs]
        # print(f"protected_attributes conv: {protected_attributes}")
        # print(f"protected_values: {protected_values}")

        # for K in [df_input.shape[0]]:
        K = df_input_iterative.shape[0]
        protected_values_str = ""
        for p in protected_values:
            if type(p) == list:
                protected_value_i = f"[{', '.join(p)}]"
                protected_values_str = (
                    ", ".join([protected_values_str, protected_value_i])
                    if protected_values_str != ""
                    else str(protected_value_i)
                )
            else:
                protected_values_str = (
                    ", ".join([protected_values_str, str(p)])
                    if protected_values_str != ""
                    else str(p)
                )

        protected_config_str = (
            f"{'_'.join(protected_attributes)}_{protected_values_str}_{K}"
        )

        input_configuration = {
            "method": mitigation_name,
            "protected attributes": f'{", ".join(sorted(protected_attributes))}',
            "protected values": protected_values_str,
        }

        config = f"{mitigation_name} - {target} : {protected_config_str}"

        protected_attributes_f = [
            a for a, v in zip(protected_attributes, protected_values) if v != DONTCARE
        ]
        protected_values_f = [
            v for a, v in zip(protected_attributes, protected_values) if v != DONTCARE
        ]

        (
            original_protected_scores,
            original_non_protected_scores,
            original_protected_cid,
            original_non_protected_cid,
        ) = get_protected_unprotected(
            df_input_iterative,
            protected_attributes_f,
            protected_values_f,
            target_name,
            cid_name="cid",
        )
        # print(
        #    "Cardinalities protected, unprotected:",
        #    len(original_protected_scores),
        #    len(original_non_protected_scores),
        # )

        start_time = time.time()
        new_protected_scores = feldman_processing(
            original_protected_scores, original_non_protected_scores, K=K
        )
        mit_res = merge_ranking_protected_unnprotected(
            original_protected_scores,
            original_non_protected_scores,
            new_protected_scores,
            original_protected_cid,
            original_non_protected_cid,
            K,
            verbose=False,
        )
        time_feldman[config] = time.time() - start_time

        df_mit_with_ids = df_input_iterative[attributes].join(mit_res.set_index("cid"))

        df_mit_with_ids_original = df_input[attributes + [target]].join(
            mit_res.set_index("cid")
        )

        if (df_mit_with_ids.index != df_input_iterative.index).any() == True:
            raise ValueError("The order does not match with the original one")

        if (df_mit_with_ids.index != df_input.index).any() == True:
            raise ValueError("The order does not match with the original one")

        original_score = df_mit_with_ids_original[target].values
        mitigated_score = df_mit_with_ids["new_score"].values

        from utils_evaluation import evaluate_ranking_scores

        k_value = 300

        res = evaluate_ranking_scores(original_score, mitigated_score, K=k_value)

        df_selection = df_input[protected_attributes].copy()
        df_selection[target] = mitigated_score
        res_div_stats = divergence_stats(df_selection, target, min_sup=min_sup)
        ranking_results_feldman[config] = res_div_stats

        ranking_results_feldman[config].update(res)
        ranking_results_feldman[config].update(input_configuration)

        if (df_input["score"].values != original_score).all() == True:
            raise ValueError()
        df_selection[f"original_{target}"] = original_score

        res_mit_feldman[config] = df_selection

        df_input_iterative[target] = mitigated_score

    from utils_evaluation import evaluate_ranking_scores

    original_config = {
        "method": "original",
        "protected attributes": "UGPA, ZFYA, race, sex",
        "protected values": "-",
    }

    original_score = df_input[target].values
    res = evaluate_ranking_scores(original_score, original_score, K=k_value)
    res_div_stats = divergence_stats(
        df_input[attributes + [target]], target, min_sup=min_sup
    )

    ranking_results_feldman[f"original - {config}"] = res_div_stats
    ranking_results_feldman[f"original - {config}"].update(res)
    ranking_results_feldman[f"original - {config}"].update(original_config)

    return ranking_results_feldman, res_mit_feldman, time_feldman
