from argparse import ArgumentParser
from copy import deepcopy
import numpy as np
import pandas as pd
from pathlib import Path

import os


from utils_evaluation import divergence_stats
from utils_evaluation import print_table_ranking_res
from utils_mitigation import data_and_outcome

from process_dataset import (
    generate_artifical_injected,
    import_compas_score_processed,
    import_law_processed,
    import_german_processed,
    import_compas_scores,
    import_law_eth,
)
import time
import pickle


DONTCARE = "-"

if __name__ == "__main__":
    parser = ArgumentParser(description="Feldman mitigation")

    parser.add_argument(
        "--dataset_name",
        help="Dataset name",
        default="law_school",
        required=True,
        choices=["artificial_1", "german", "law_school", "compas-poc"],
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        help="Output directory in which results are stored",
        default="output_results/feldman_results_mitigation",
        required=False,
        type=str,
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name
    name_output_dir = args.output_dir

    output_dir = os.path.join(name_output_dir, dataset_name)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if dataset_name == "law_school-poc":
        data, target_name, sensitive_attributes = import_law_processed()
    elif dataset_name == "law_school":
        data, target_name, sensitive_attributes = import_law_eth()
    elif dataset_name == "compas-poc":
        data, target_name, sensitive_attributes = import_compas_score_processed()
    elif dataset_name == "compas":
        data, target_name, sensitive_attributes = import_compas_scores()
    elif dataset_name == "german":
        data, target_name, sensitive_attributes = import_german_processed()
    elif dataset_name == "german_foreigner":
        data, target_name, sensitive_attributes = import_german_processed(
            use_foreigner=True
        )
    elif dataset_name == "artificial_1":
        data, target_name, sensitive_attributes = generate_artifical_injected(n_attr=5)
    else:
        raise ValueError()

    protected_attributes_list = [[sa] for sa in sensitive_attributes] + [
        sensitive_attributes
    ]

    attributes = list(data.columns)
    attributes.remove(target_name)

    if dataset_name == "law_school":
        protected_attributes_list = [
            ["race"],
            ["sex"],
            ["race"],
            ["race", "sex"],
            ["race", "sex"],
        ]
        protected_attribute_values_list = [
            ["Black"],
            ["female"],
            [
                [
                    "Asian",
                    "Mexican",
                    "Hispanic",
                    "Other",
                    "Black",
                    "Puertorican",
                    "Amerindian",
                ]
            ],
            [["Black"], ["female"]],
            [
                [
                    "Asian",
                    "Mexican",
                    "Hispanic",
                    "Other",
                    "Black",
                    "Puertorican",
                    "Amerindian",
                ],
                ["female"],
            ],
        ]
    elif dataset_name == "compas-poc":
        protected_attributes_list = [
            ["sex"],
            ["age_cat"],
            ["age_cat"],
            ["age_cat"],
            ["race"],
            ["sex", "age_cat", "race"],
            ["sex", "age_cat", "race"],
            ["sex", "age_cat", "race"],
        ]
        protected_attribute_values_list = [
            ["Female"],
            ["25 - 45"],
            ["Less than 25"],
            [["25 - 45", "Less than 25"]],
            ["protected"],
            ["Female", "Less than 25", "protected"],
            ["Female", "Less than 25", "Caucasian"],
            ["Male", "Less than 25", "protected"],
        ]

    elif dataset_name == "german":
        protected_attributes_list = [
            ["sex"],
            ["age"],
            ["sex", "age"],
            ["sex", "age"],
            ["sex", "age"],
            ["sex", "age"],
            ["sex", "age"],
        ]
        # g1, g2, g3, g4, g5
        protected_attribute_values_list = [
            ["female"],
            ["young"],
            ["female", "young"],
            ["female", "adult"],
            ["female", "elder"],
            ["male", "young"],
            ["male", "elder"],
        ]

    elif dataset_name == "artificial_1":
        protected_attributes_list = [
            ["a"],
            ["b"],
            ["c"],
            ["d"],
            ["e"],
            ["a", "b"],
            ["a", "b", "c"],
            sensitive_attributes,
            sensitive_attributes,
        ]
        # g1, g2, g3, g4, g5
        protected_attribute_values_list = [
            [1],
            [1],
            [1],
            [1],
            [1],
            [1, 1],
            [1, 1, 1],
            [1, 1, 1, DONTCARE, DONTCARE],
            [1, 1, 1, 1, 1],
        ]

    time_feldman = {}

    ######################

    min_sup = 0.01

    ######################

    # target = "inv_rank"
    target = "score"
    # target = "utility"

    df_input = data_and_outcome(data, target_name, target)

    from utils_comparison import (
        get_protected_unprotected,
        feldman_processing,
        merge_ranking_protected_unnprotected,
    )

    mitigation_name = "feldman"

    ranking_results_feldman = {}
    res_mit_feldman = {}
    for protected_attributes, protected_values in zip(
        protected_attributes_list, protected_attribute_values_list
    ):
        print(f"\n\nprotected_attributes: {protected_attributes}")
        pa_idxs = np.argsort(np.array(protected_attributes))
        protected_attributes = [protected_attributes[pa_idx] for pa_idx in pa_idxs]
        protected_values = [protected_values[pa_idx] for pa_idx in pa_idxs]
        print(f"protected_attributes conv: {protected_attributes}")
        print(f"protected_values: {protected_values}")

        # for K in [df_input.shape[0]]:
        K = df_input.shape[0]
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
            df_input,
            protected_attributes_f,
            protected_values_f,
            target_name,
            cid_name="cid",
        )
        print(
            "Cardinalities protected, unprotected:",
            len(original_protected_scores),
            len(original_non_protected_scores),
        )

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

        df_mit_with_ids = df_input[attributes].join(mit_res.set_index("cid"))

        if (df_mit_with_ids.index != df_input.index).any() == True:
            raise ValueError("The order does not match with the original one")

        original_score = df_mit_with_ids["original_score"].values
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

    with open(
        os.path.join(output_dir, f"{dataset_name}_feldman_results_mitigation.pickle"),
        "wb",
    ) as fp:
        pickle.dump(ranking_results_feldman, fp)

    with open(
        os.path.join(
            output_dir, f"{dataset_name}_feldman_results_mitigation_time.pickle"
        ),
        "wb",
    ) as fp:
        pickle.dump(time_feldman, fp)


# python A_002_feldman_mitigation.py --dataset_name law_school --output_dir results/feldman_results_mitigation
# python A_002_feldman_mitigation.py --dataset_name german  --output_dir results/feldman_results_mitigation
# python A_002_feldman_mitigation.py --dataset_name artificial_1 --output_dir results/feldman_results_mitigation
# python A_002_feldman_mitigation.py --dataset_name compas-poc --output_dir results/feldman_results_mitigation
