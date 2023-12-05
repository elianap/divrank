from argparse import ArgumentParser
from copy import deepcopy
import numpy as np
import pandas as pd
from pathlib import Path

import os

from utils_evaluation import print_table_ranking_res
from utils_mitigation import data_and_outcome

from process_dataset import (
    import_german_processed_all_attributes,
    import_law_eth_all_attributes,
    import_compas_score_processed_all_attributes,
)

DATASET_DIR = os.path.join(os.path.curdir, "datasets")


import time
import pickle


if __name__ == "__main__":
    parser = ArgumentParser(description="DivRank mitigation analysis")

    parser.add_argument(
        "--dataset_name",
        help="Dataset name",
        default="law_school_all_attributes",
        required=True,
        choices=[
            "german_all_attributes",
            "law_school_all_attributes",
            "compas-poc_all_attributes",
        ],
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        help="Output directory in which results are stored",
        default="output_results/divrank_results_mitigation",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--min_sup",
        help="Specify the minumum support for the divrank algorithm",
        default=None,
        required=False,
        type=float,
    )

    parser.add_argument(
        "--min_support_count",
        help="Specify the minumum support count for the divrank algorithm",
        default=None,
        required=False,
        type=float,
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name
    name_output_dir = args.output_dir
    min_sup = args.min_sup
    min_support_count = args.min_support_count

    output_dir = os.path.join(name_output_dir, dataset_name)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if dataset_name == "law_school_all_attributes":
        data, target_name, sensitive_attributes = import_law_eth_all_attributes()
    elif dataset_name == "compas-poc_all_attributes":
        (
            data,
            target_name,
            sensitive_attributes,
        ) = import_compas_score_processed_all_attributes()
    elif dataset_name == "german_all_attributes":
        (
            data,
            target_name,
            sensitive_attributes,
        ) = import_german_processed_all_attributes()
    elif dataset_name == "artificial_1_all_attributes":
        from process_dataset import generate_artifical_injected

        data, target_name, sensitive_attributes = generate_artifical_injected(n_attr=5)
    else:
        raise ValueError()

    if min_support_count is None and min_sup is None:
        raise ValueError("Specify min_sup or min_support_count")
    if min_support_count is not None and min_sup is None:
        min_sup = min_support_count / len(data)

    print(f"Min support: {min_sup}")

    if min_sup * len(data) < 25:
        raise ValueError("Min support too low")

    attributes = [a for a in list(data.columns) if a != target_name]

    # Use all attributes as sensitive!
    protected_attributes_list = [attributes]

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

    n_iterations = 50000

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
        all_results, df_mit_score = mit_obj.iterative_mitigate_minmax_TEST(
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

    df_mit_score.to_csv(
        os.path.join(
            output_dir, f"{dataset_name}_divrank_results_mitigation_df_mit_score.csv"
        )
    )

    with open(
        os.path.join(
            output_dir,
            f"{dataset_name}_divrank_results_mitigation_all_resultscsv.pickle",
        ),
        "wb",
    ) as fp:
        pickle.dump(all_results, fp)

# python A_005_divrank_all_attributes.py --dataset_name law_school_all_attributes --output_dir results/divrank_results_mitigation_test_all_attributes_50 --min_sup 0.0025
# python A_005_divrank_all_attributes.py --dataset_name compas-poc_all_attributes --output_dir results/divrank_results_mitigation_test_all_attributes_50 --min_sup 0.008
# python A_005_divrank_all_attributes.py --dataset_name german_all_attributes  --output_dir results/divrank_results_mitigation_test_all_attributes_50 --min_sup 0.05

# python A_005_divrank_all_attributes.py --dataset_name law_school_all_attributes --output_dir results/divrank_results_mitigation_test_all_attributes_100 --min_sup 0.005
# python A_005_divrank_all_attributes.py --dataset_name compas-poc_all_attributes --output_dir results/divrank_results_mitigation_test_all_attributes_100 --min_sup 0.016
# python A_005_divrank_all_attributes.py --dataset_name german_all_attributes  --output_dir results/divrank_results_mitigation_test_all_attributes_100 --min_sup 0.1
