from argparse import ArgumentParser
from copy import deepcopy
import numpy as np
import pandas as pd
from pathlib import Path

import os

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


if __name__ == "__main__":
    parser = ArgumentParser(description="DivRank mitigation analysis")

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
        default="output_results/divrank_results_mitigation",
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

    # target = "inv_rank"
    # target = "utility"

    target = "score"

    df_input = data_and_outcome(data, target_name, target)

    """
    Generate Div-mitigation
    """
    mitigation_name = "div-mit"
    from divrank_mitigation import Mitigation
    import os

    n_iterations = 100
    min_sup = 0.01

    res_div_mit = {}

    ranking_results = {}
    divrank_time = {}
    for selected_attributes in protected_attributes_list:
        df_selection = df_input[selected_attributes + [target]].copy()

        config = f'{mitigation_name} - {target} : {", ".join(selected_attributes)}'

        input_configuration = {
            "method": mitigation_name,
            "protected attributes": f'{", ".join(sorted(selected_attributes))}',
            "protected values": "-",
        }

        df_mit_score = deepcopy(df_selection)

        start_time = time.time()
        mit_obj = Mitigation()
        all_results, df_mit_score = mit_obj.iterative_mitigate_minmax(
            df_mit_score,
            n_iterations=n_iterations,
            break_if_no_significant=True,
            min_sup=min_sup,
            target=target,
        )
        computing_time = time.time() - start_time

        assert (
            len(
                [
                    i
                    for i in range(0, len(all_results) - 1)
                    if all_results["min_div"].values[i]
                    > all_results["min_div"].values[i + 1]
                ]
            )
            == 0
        ), "Monotonicity not satisfied"

        res_div_mit[config] = df_mit_score.copy()

        df_mit_score[f"original_{target}"] = df_input[target].values

        ### Add divergence_results first

        k_value = 300

        from utils_evaluation import divergence_stats

        df_original = df_selection.copy()

        ranking_results[f"original - {config}"] = divergence_stats(
            df_original, target, min_sup=min_sup
        )

        from utils_evaluation import evaluate_ranking_scores

        a = df_mit_score
        original_score = a[f"original_{target}"].values
        mitigated_score = a[target].values

        res = evaluate_ranking_scores(original_score, original_score, K=k_value)
        ranking_results[f"original - {config}"].update(res)
        ranking_results[f"original - {config}"].update(input_configuration)
        ranking_results[f"original - {config}"]["method"] = "original"

        df_mit_score.drop(columns=["original_score"], inplace=True)
        res_div_stats = divergence_stats(df_mit_score, target, min_sup=min_sup)

        res = evaluate_ranking_scores(original_score, mitigated_score, K=k_value)
        ranking_results[f"{config}"] = res_div_stats
        ranking_results[f"{config}"].update(res)
        ranking_results[f"{config}"].update(input_configuration)

        id_sort_new = np.argsort(-mitigated_score, kind="mergesort")
        divrank_time[config] = computing_time

    r = pd.DataFrame(ranking_results).T
    print_table_ranking_res(r)

    with open(
        os.path.join(output_dir, f"{dataset_name}_divrank_results_mitigation.pickle"),
        "wb",
    ) as fp:
        pickle.dump(ranking_results, fp)

    with open(
        os.path.join(
            output_dir, f"{dataset_name}_divrank_results_mitigation_time.pickle"
        ),
        "wb",
    ) as fp:
        pickle.dump(divrank_time, fp)


# python A_001_divrank_mitigation.py --dataset_name law_school --output_dir results/divrank_results_mitigation
# python A_001_divrank_mitigation.py --dataset_name german  --output_dir results/divrank_results_mitigation
# python A_001_divrank_mitigation.py --dataset_name artificial_1 --output_dir results/divrank_results_mitigation
# python A_001_divrank_mitigation.py --dataset_name compas-poc --output_dir results/divrank_results_mitigation
