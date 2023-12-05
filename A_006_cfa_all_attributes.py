from argparse import ArgumentParser
from copy import deepcopy
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import os

import time

from utils_evaluation import print_table_ranking_res
from utils_mitigation import data_and_outcome

from process_dataset import (
    import_law_eth_all_attributes,
    import_compas_score_processed_all_attributes,
    import_german_processed_all_attributes,
)


OTHER = "Other-candidates"


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate explanations")

    parser.add_argument(
        "--dataset_name",
        help="Dataset name",
        default="law_school_all_attributes",
        required=True,
        choices=[
            "law_school_all_attributes",
            "compas-poc_all_attributes",
            "german_all_attributes",
            "artificial_1_all_attributes",
        ],
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        help="Output directory in which results are stored",
        default="zz_all_attributes_tests_delete_all",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--min_sup",
        help="Specify the minimum support",
        default=0.0025,
        type=float,
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name
    name_output_dir = args.output_dir
    min_sup = args.min_sup

    print(f"Dataset name: {dataset_name}")
    print(f"Output directory: {name_output_dir}")
    print(f"Minimum support: {min_sup}")

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

    print("Loaded")

    score_stepsize = 0.005
    regForOT = 5e-3

    attributes = list(data.columns)
    attributes.remove(target_name)

    sensitive_attributes = attributes

    groups_sel = [list(i) for i in data[attributes].drop_duplicates(keep=False).values]
    all_sensitive_attr_values_sel = [
        (sensitive_attributes, list(group_sel)) for group_sel in groups_sel
    ]

    print("len(all_sensitive_attr_values_sel)", len(all_sensitive_attr_values_sel))

    existing_all_sensitive_attr_values_sel = []
    for attrs, vals in all_sensitive_attr_values_sel:
        indexes = data.index
        for attr, val in zip(attrs, vals):
            indexes = data.loc[indexes].loc[data[attr] == val].index
        if data.loc[indexes].shape[0] != 0:
            existing_all_sensitive_attr_values_sel.append((attrs, vals))
    all_sensitive_attr_values_sel = existing_all_sensitive_attr_values_sel

    print("len(all_sensitive_attr_values_sel)", len(all_sensitive_attr_values_sel))

    protected_attributes = attributes
    print("Protected attributes:", protected_attributes)
    attribute_group_values = all_sensitive_attr_values_sel

    print("Number of groups:", len(attribute_group_values))

    # INIT

    ranking_result_cfa = {}

    df_results_cfa = {}

    cfa_time = {}

    print(dataset_name)

    print("attribute_group_values", len(attribute_group_values))

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
        # print(gid, group)
        protected_attributes, protected_values = group
        print(gid, protected_attributes, protected_values)
        if protected_values == OTHER:
            continue
        sel_indexes = df_input_sorted.index
        for attribute, value in zip(protected_attributes, protected_values):
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

            print("sel_indexes", sel_indexes.shape[0])
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

    from utils_evaluation import divergence_stats

    df_original = df_input.copy()

    ranking_result_cfa["original"] = divergence_stats(
        df_original, target, min_sup=min_sup
    )

    from utils_evaluation import evaluate_ranking_scores

    input_configuration = {
        "method": "original",
        "protected attributes": "all",
        "protected values": "-",
    }

    original_score = df_original[target].values

    res = evaluate_ranking_scores(original_score, original_score, K=300)
    ranking_result_cfa["original"].update(res)
    ranking_result_cfa["original"].update(input_configuration)

    print()
    r = pd.DataFrame(ranking_result_cfa).T
    s = print_table_ranking_res(r).drop(columns=["rank gain", "rank drop"])

    print(s)

    import pickle

    output_dir = os.path.join(name_output_dir, dataset_name)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(os.path.join(output_dir, f"{dataset_name}_cfa_{min_sup}.csv"))

    s.to_csv(os.path.join(output_dir, f"{dataset_name}_cfa_{min_sup}.csv"), index=False)

    with open(
        os.path.join(output_dir, f"{dataset_name}_cfa_results_{min_sup}.pickle"), "wb"
    ) as fp:
        pickle.dump(ranking_result_cfa, fp)

    with open(
        os.path.join(output_dir, f"{dataset_name}_cfa_time_{min_sup}.pickle"), "wb"
    ) as fp:
        pickle.dump(cfa_time, fp)
