from copy import deepcopy
import numpy as np
import pandas as pd
from pathlib import Path

import os
import time
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
from argparse import ArgumentParser

from utils_subgroups import get_list_group_by_dataset
from copy import deepcopy


import pickle


OTHER = "Other-candidates"


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate explanations")

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
        default="output_results/cfa_results_mitigation",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--set_iterations",
        help="Specify iteration of interest",
        nargs="*",
        default=None,
        required=False,
        type=int,
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name
    name_output_dir = args.output_dir
    set_iterations = args.set_iterations

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

    attributes = list(data.columns)
    attributes.remove(target_name)

    output_dir = os.path.join(name_output_dir, dataset_name)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    import itertools

    sa_values = {sa: list(data[sa].unique()) for sa in sensitive_attributes}
    a = sa_values.values()
    groups = list(itertools.product(*a))
    all_sensitive_attr_values = [
        (sensitive_attributes, list(group)) for group in groups
    ]

    list_protected_attributes, list_attribute_group_values = get_list_group_by_dataset(
        dataset_name, data, sensitive_attributes, all_sensitive_attr_values
    )

    score_stepsize = 0.005
    regForOT = 5e-3

    # INIT

    ranking_result_cfa = {}

    df_results_cfa = {}

    cfa_time = {}

    print(dataset_name)

    if set_iterations is None:
        set_iterations = list(range(len(list_protected_attributes)))

    print("Set iterations:", set_iterations)
    print("Tot", len(list_protected_attributes))

    for i in range(len(list_protected_attributes)):
        print(i)
        if i not in set_iterations:
            print("continue")
            continue

        protected_attributes = list_protected_attributes[i]
        attribute_group_values = list_attribute_group_values[i]

        print(protected_attributes)
        print(attribute_group_values)

        target = target_name

        df_input = deepcopy(data[protected_attributes + [target]])
        df_groups = pd.DataFrame(range(len(attribute_group_values)), columns=["group"])

        names = {}
        for gid, (aa, vv) in enumerate(attribute_group_values):
            name = ""
            for a, v in zip(aa, vv):
                name = name + f"{a}={v}"
            names[gid] = name

        df_input_sorted = df_input.sort_values(
            target, kind="mergesort", ascending=False
        ).copy()

        # Get slice

        df_input_sorted["group"] = 0
        for gid, group in enumerate(attribute_group_values):
            print(gid, group)
            protected_attributes, protected_values = group
            print(protected_attributes, protected_values)
            if protected_values == OTHER:
                continue
            sel_indexes = df_input_sorted.index
            for attribute, value in zip(protected_attributes, protected_values):
                print(attribute, value)

                if type(value) == list:
                    sel_indexes = (
                        df_input_sorted.loc[sel_indexes]
                        .loc[df_input_sorted[attribute].isin(value)]
                        .index
                    )
                else:
                    sel_indexes = (
                        df_input_sorted.loc[sel_indexes]
                        .loc[df_input_sorted[attribute] == value]
                        .index
                    )
                sel_indexes = df_input_sorted.loc[sel_indexes].index
            df_input_sorted.loc[sel_indexes, "group"] = gid
            # display(df_input_sorted.head())
        assert -1 not in df_input_sorted["group"].unique(), "error"

        from cfa.cfa import ContinuousFairnessAlgorithm

        thetas = tuple([1] * len(df_groups))

        # Upd
        # config = f"cfa - {', '.join(sorted(protected_attributes))} - {', '.join(sorted([', '.join(sorted(v[1])) for v in attribute_group_values[1:]]))}"
        protected_attributes_str = f"{', '.join(sorted(protected_attributes))}"
        attribute_group_values_str = f"{', '.join(sorted([', '.join(sorted(list(map(str, v[1] if v[1]!=OTHER else [v[1]])))) for v in attribute_group_values]))}"
        config = f"cfa - {protected_attributes_str} - {attribute_group_values_str}"

        start_time = time.time()
        cfa = ContinuousFairnessAlgorithm(
            df_input_sorted,
            df_groups,
            names,
            target,
            score_stepsize,
            thetas,
            regForOT,
            plot=False,
        )
        result = cfa.run()

        cfa_time[config] = time.time() - start_time

        df_mit_cfa = result[protected_attributes + [target, "fairScore"]].copy()

        min_sup = 0.01

        original_score = df_mit_cfa[target].values
        mitigated_score = df_mit_cfa["fairScore"].values
        from utils_evaluation import evaluate_ranking_scores, divergence_stats

        rs = evaluate_ranking_scores(original_score, mitigated_score, K=300)
        rs.update(
            divergence_stats(
                df_mit_cfa[protected_attributes + ["fairScore"]],
                "fairScore",
                min_sup=min_sup,
            )
        )
        rs["protected attributes"] = protected_attributes_str
        rs["protected values"] = attribute_group_values_str
        rs["method"] = "cfa" + r"$\theta$"
        ranking_result_cfa[config] = rs
        df_results_cfa[config] = df_mit_cfa[
            protected_attributes + [target, "fairScore"]
        ].copy()

        print()
        r = pd.DataFrame(ranking_result_cfa).T
        s = print_table_ranking_res(r).drop(columns=["rank gain", "rank drop"])

        print()

    r = pd.DataFrame(ranking_result_cfa).T
    s = print_table_ranking_res(r).drop(columns=["rank gain", "rank drop"])

    print(os.path.join(output_dir, f"{dataset_name}_cfa.csv"))

    s.to_csv(os.path.join(output_dir, f"{dataset_name}_cfa.csv"), index=False)

    with open(
        os.path.join(output_dir, f"{dataset_name}_cfa_results.pickle"), "wb"
    ) as fp:
        pickle.dump(ranking_result_cfa, fp)

    with open(os.path.join(output_dir, f"{dataset_name}_cfa_time.pickle"), "wb") as fp:
        pickle.dump(cfa_time, fp)


# python A_003_cfa_mitigation.py --dataset_name law_school  --output_dir results/cfa_results_mitigation
# python A_003_cfa_mitigation.py --dataset_name german  --output_dir results/cfa_results_mitigation
# python A_003_cfa_mitigation.py --dataset_name compas-poc --output_dir results/cfa_results_mitigation
# python A_003_cfa_mitigation.py --dataset_name artificial_1 --output_dir results/cfa_results_mitigation
