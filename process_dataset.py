import numpy as np

import pandas as pd
import os
import scipy.stats as stats


DATASET_DIR = os.path.join(os.path.curdir, "datasets")


def import_compas_score_processed(input_dir=DATASET_DIR):
    """
    Import the COMPAS dataset
    Returns:
        pd.DataFrame: the COMPAS dataset - columns = ['sex', 'age_cat', 'race', 'score']
        str: the name of the score column ('score')
        list: list of sensitive attributes (['sex', 'age_cat', 'race'])

    We use the dataset as in Multinomial_FA-IR, downloaded from https://github.com/MilkaLichtblau/Multinomial_FA-IR
    We preprocess the datas as in Multinomial_FA-IR

    The processed dataset is characterized as follows:
    'race' : ['protected', 'Caucausian'], where protected encodes all ethinicities that differ from the caucasian one
    'age_cat': ['Less than 25', '25 - 45', 'Greater than 45']
    'sex' : ['Male', 'Female']
    """
    # Downloaded from https://github.com/MilkaLichtblau/Multinomial_FA-IR
    filename = os.path.join(input_dir, "compas_sexAgeRace.csv")
    if os.path.isfile(filename):
        data = pd.read_csv(filename)
    else:
        # Read from original URL
        data = pd.read_csv(
            "https://raw.githubusercontent.com/MilkaLichtblau/Multinomial_FA-IR/5358ef2dff5f30964752e1e6c11654b37677b732/experiments/dataExperiments/data/COMPAS/compas_sexAgeRace.csv"
        )
    data.drop(columns="two_year_recid", inplace=True)
    data["race"] = data["race"].replace({1: "protected", 0: "Caucasian"})
    data["age_cat"] = data["age_cat"].replace(
        {1: "Less than 25", 2: "25 - 45", 0: "Greater than 45"}
    )
    data["sex"] = data["sex"].replace({1: "Male", 0: "Female"})
    target_name = "score"
    attributes = list(data.columns)
    attributes.remove(target_name)
    sensitive_attributes = ["sex", "age_cat", "race"]

    return data, target_name, sensitive_attributes


def generate_artifical_injected(
    inject_type=1, n=10000, n_attr=3, input_dir=DATASET_DIR
):
    import string

    target = "score"
    attributes = list(string.ascii_lowercase)[:n_attr]

    if n_attr == 5 and inject_type == 1 and n == 10000:
        filename = os.path.join(input_dir, "artificial_ab_c_5.csv")
        if os.path.isfile(filename):
            data = pd.read_csv(filename)
            return data, target, attributes

    a = np.random.randint(0, high=2, size=(n, n_attr), dtype=int)

    dfb = pd.DataFrame(a, columns=attributes)
    dfb[target] = np.random.randint(1, high=101, size=n)

    if inject_type == 0:
        print(f"[a=1] or [b=1]")
        dfb.loc[(dfb["a"] == 1) & (dfb[target] > 70), target] -= 25
        dfb.loc[(dfb["b"] == 1) & (dfb[target] > 70), target] -= 25

    else:
        print(f"[a=1, b=1] or [c=1]")
        dfb.loc[(dfb["a"] == 1) & (dfb["b"] == 1) & (dfb[target] > 70), target] -= 25
        dfb.loc[(dfb["c"] == 1) & (dfb[target] > 70), target] -= 25

    dfb = dfb.sort_values(target, ascending=False)

    return dfb, target, attributes


def import_law_processed(input_dir=DATASET_DIR):
    # Downloaded from  https://github.com/MilkaLichtblau/Multinomial_FA-IR
    filename = os.path.join(input_dir, "LSAT_sexRace_java.csv")
    if os.path.isfile(filename):
        data = pd.read_csv(filename)
    else:
        # Read from original URL
        data = pd.read_csv(
            "https://raw.githubusercontent.com/MilkaLichtblau/Multinomial_FA-IR/master/experiments/dataExperiments/data/LSAT/LSAT_sexRace_java.csv"
        )

    data["sex"] = -1
    data["race"] = -1
    data.loc[data["group"] == 0, ["sex", "race"]] = (0, 0)
    data.loc[data["group"] == 1, ["sex", "race"]] = (1, 0)
    data.loc[data["group"] == 2, ["sex", "race"]] = (0, 1)
    data.loc[data["group"] == 3, ["sex", "race"]] = (1, 1)
    data["race"].replace({1: "protected", 0: "White"}, inplace=True)
    data["sex"].replace({1: "female", 0: "male"}, inplace=True)
    data.drop(columns=["uuid", "group"], inplace=True)
    sensitive_attributes = ["sex", "race"]
    return data, "score", sensitive_attributes


def import_german_processed(input_dir=DATASET_DIR, use_foreigner=False):
    """
    Import the german dataset
    Args:
        input_dir: Directory where the "germanCredit_sexAgeForeigner.csv" is stored. DATASET_DIR as default (./datasets)
        use_foreigner: if True consider also the foreigner attrobite
    Returns:
        pd.DataFrame: the german dataset - columns =  ['sex', 'age', 'score'] or ['sex', 'age', 'foreigner', 'score'] if use_foreigner is True
        str: the name of the score column ('score')
        list: list of sensitive attributes (['sex', 'age'] or if ['sex', 'age', 'foreigner'] )

    We use the dataset as in Multinomial_FA-IR, downloaded from https://github.com/MilkaLichtblau/Multinomial_FA-IR
    We preprocess the datas as in Multinomial_FA-IR

    The processed dataset is characterized as follows:
    'sex' : ['male', 'female'], where protected encodes all ethinicities that differ from the caucasian one
    'age_cat': ['young', 'adult', 'elder']
    'foreigner' : [0, 1]
    """
    # Downloaded from  https://github.com/MilkaLichtblau/Multinomial_FA-IR
    filename = os.path.join(input_dir, "germanCredit_sexAgeForeigner.csv")
    if os.path.isfile(filename):
        data = pd.read_csv(filename)
    else:
        # Read from original URL
        data = pd.read_csv(
            "https://raw.githubusercontent.com/MilkaLichtblau/Multinomial_FA-IR/master/experiments/dataExperiments/data/GermanCredit/germanCredit_sexAgeForeigner.csv"
        )
    data["sex"] = data["sex"].replace({0: "male", 1: "female"})
    data["age"] = data["age"].replace({0: "young", 1: "adult", 2: "elder"})
    if use_foreigner:
        data.drop(columns=["decileRank"], inplace=True)
        sensitive_attributes = ["sex", "age", "foreigner"]
    else:
        data.drop(columns=["decileRank", "foreigner"], inplace=True)
        sensitive_attributes = ["sex", "age"]
    return data, "score", sensitive_attributes


# Code adapted from https://github.com/MilkaLichtblau/Multinomial_FA-IR
def import_compas_scores(input_dir=DATASET_DIR):
    """
    Returns
        pd.DataFrame
        str: column name of the target (scores)
    """
    data = pd.read_csv(os.path.join(input_dir, "compas-scores-two-years.txt"), header=0)
    # we do the same data cleaning as is done by ProPublica. See link for details
    # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    data = data[data["days_b_screening_arrest"] <= 30]
    data = data[data["days_b_screening_arrest"] >= -30]
    data = data[data["is_recid"] != -1]
    data = data[data["c_charge_degree"] != "O"]
    data = data[data["score_text"] != "N/A"]

    # drop irrelevant columns
    keep_cols = [
        "sex",
        "age_cat",
        "race",
        "decile_score",
        "v_decile_score",
        "priors_count",
        # "two_year_recid",
    ]
    data = data[keep_cols]

    # normalize numeric columns to interval [0,1]
    scaledDecile = (data["decile_score"] - np.min(data["decile_score"])) / np.ptp(
        data["decile_score"]
    )
    scaledVDecile = (data["v_decile_score"] - np.min(data["v_decile_score"])) / np.ptp(
        data["v_decile_score"]
    )
    scaledpriors = (data["priors_count"] - np.min(data["priors_count"])) / np.ptp(
        data["priors_count"]
    )

    # calculate score based on recidivism score, violent recidivism score and number of prior arrests
    # violent recidivism weighs highest.
    data["score"] = np.zeros(data.shape[0])
    for idx, _ in data.iterrows():
        recidivism = scaledDecile[idx]
        violentRecidivism = scaledVDecile[idx]
        priorArrests = scaledpriors[idx]

        score = 0.25 * recidivism + 0.5 * violentRecidivism + 0.25 * priorArrests
        data.loc[idx, "score"] = score

    # higher scores should be better scores
    data["score"] = data["score"].max() - data["score"]

    # add some random noise to break ties
    noise = np.random.normal(0, 0.000001, data.shape[0])
    data["score"] = data["score"] + noise

    # drop these columns
    data = data.drop(columns=["decile_score", "v_decile_score", "priors_count"])
    data.sort_values(by=["score"], ascending=False, inplace=True)

    data.to_csv(
        os.path.join(DATASET_DIR, "compas_sensitives.csv"), header=True, index=False
    )

    sensitive_attributes = ["sex", "age_cat", "race"]
    target_name = "score"
    return data, target_name, sensitive_attributes


def import_law_eth(input_dir=DATASET_DIR):
    """
    Import the law dataset
    Args:
        input_dir: Directory where the "law_data.csv" is stored. DATASET_DIR as default (./datasets)
    Returns:
        pd.DataFrame: the Law School dataset - columns = ['race', 'sex', 'score']
        str: the name of the score column ('score')
        list: list of sensitive attributes (['race', 'sex'])

    We use the dataset as in Multinomial_FA-IR and as in CFA
    We preprocess the datas as in https://github.com/MilkaLichtblau/ContinuousFairness/blob/master/src/data_preparation/LSAT.py

    """
    filename = os.path.join(input_dir, "law_data.csv")
    data = pd.read_csv(filename)
    data = data.drop(
        columns=["region_first", "sander_index", "first_pf", "UGPA", "ZFYA"]
    )

    data["sex"] = data["sex"].replace([2], 0)

    data["LSAT"] = data["LSAT"].apply(str)
    data["LSAT"] = data["LSAT"].str.replace(".7", ".75", regex=False)
    data["LSAT"] = data["LSAT"].str.replace(".3", ".25", regex=False)
    data["LSAT"] = pd.to_numeric(data["LSAT"])
    data["sex"].replace({1: "female", 0: "male"}, inplace=True)
    data = data.rename(columns={"LSAT": "score"})
    data = data.sort_values(by=["score"], kind="mergesort")[::-1]
    sensitive_attributes = ["sex", "race"]
    data.drop(columns=["id"], inplace=True)
    return data, "score", sensitive_attributes


DATASET_DIR = os.path.join(os.path.curdir, "datasets")


def import_compas_score_processed_all_attributes(
    input_dir=DATASET_DIR, discretize=True
):
    """
    Import the COMPAS dataset
        discretize: if True, discretize the dataset
    Returns:
        pd.DataFrame: the COMPAS dataset - columns = ['sex', 'age_cat', 'race', "c_charge_degree", "length_of_stay", 'score']
        str: the name of the score column ('score')
        list: list of sensitive attributes (['sex', 'age_cat', 'race'])

    We use the dataset as in Multinomial_FA-IR, downloaded from https://github.com/MilkaLichtblau/Multinomial_FA-IR
    We preprocess the datas as in Multinomial_FA-IR

    The processed dataset is characterized as follows:
    'race' : ['protected', 'Caucausian'], where protected encodes all ethinicities that differ from the caucasian one
    'age_cat': ['Less than 25', '25 - 45', 'Greater than 45']
    'sex' : ['Male', 'Female']
    """

    """
    First part of the code adapted from https://github.com/MilkaLichtblau/Multinomial_FA-IR/blob/5358ef2dff5f30964752e1e6c11654b37677b732/experiments/dataExperiments/src/cleanCompasData.py#L38 
    """

    data = pd.read_csv(
        "https://raw.githubusercontent.com/MilkaLichtblau/Multinomial_FA-IR/5358ef2dff5f30964752e1e6c11654b37677b732/experiments/dataExperiments/data/COMPAS/compas-scores-two-years.csv",
        header=0,
    )
    # we do the same data cleaning as is done by ProPublica. See link for details
    # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    data = data[data["days_b_screening_arrest"] <= 30]
    data = data[data["days_b_screening_arrest"] >= -30]
    data = data[data["is_recid"] != -1]
    data = data[data["c_charge_degree"] != "O"]
    data = data[data["score_text"] != "N/A"]

    # drop irrelevant columns
    # keep_cols = ["sex", "age_cat", "race", "decile_score", "v_decile_score", "priors_count", "two_year_recid"]
    # data = data[keep_cols]
    data["sex"] = data["sex"].replace({"Male": 1, "Female": 0})
    data["age_cat"] = data["age_cat"].replace(
        {"Less than 25": 1, "25 - 45": 2, "Greater than 45": 0}
    )
    data["race"] = data["race"].replace(
        {
            "Caucasian": 0,
            "African-American": 1,
            "Hispanic": 1,
            "Asian": 1,
            "Native American": 1,
            "Other": 1,
        }
    )
    # normalize numeric columns to interval [0,1]
    scaledDecile = (data["decile_score"] - np.min(data["decile_score"])) / np.ptp(
        data["decile_score"]
    )
    scaledVDecile = (data["v_decile_score"] - np.min(data["v_decile_score"])) / np.ptp(
        data["v_decile_score"]
    )
    scaledpriors = (data["priors_count"] - np.min(data["priors_count"])) / np.ptp(
        data["priors_count"]
    )

    # calculate score based on recidivism score, violent recidivism score and number of prior arrests
    # violent recidivism weighs highest.
    data["score"] = np.zeros(data.shape[0])
    for idx, _ in data.iterrows():
        recidivism = scaledDecile[idx]
        violentRecidivism = scaledVDecile[idx]
        priorArrests = scaledpriors[idx]

        score = 0.25 * recidivism + 0.5 * violentRecidivism + 0.25 * priorArrests
        data.loc[idx, "score"] = score

    # higher scores should be better scores
    data["score"] = data["score"].max() - data["score"]

    # add some random noise to break ties
    noise = np.random.normal(0, 0.000001, data.shape[0])
    data["score"] = data["score"] + noise

    # drop these columns
    # data = data.drop(columns=['decile_score', 'v_decile_score', 'priors_count'])
    data.sort_values(by=["score"], ascending=False, inplace=True)

    sensitive_attributes = ["sex", "age_cat", "race"]

    """"
    We keep 'length_of_stay', 'c_charge_degree', 
    We do not inclue "priors_count" since it is used to compute the score
    """

    data["length_of_stay"] = (
        pd.to_datetime(data["c_jail_out"]).dt.date
        - pd.to_datetime(data["c_jail_in"]).dt.date
    ).dt.days

    data = data.loc[data["c_charge_degree"] != "O"]  # F: felony, M: misconduct

    cols_propb = (
        sensitive_attributes + ["c_charge_degree", "length_of_stay"] + ["score"]
    )

    data = data[cols_propb]

    # Quantize length of stay
    def quantizeLOS(x):
        if x <= 7:
            return "<week"
        if 8 < x <= 93:
            return "1w-3M"
        else:
            return ">3Months"

    if discretize:
        data["length_of_stay"] = data["length_of_stay"].apply(lambda x: quantizeLOS(x))

    return data, "score", sensitive_attributes


def import_law_eth_all_attributes(input_dir=DATASET_DIR, discretize=True):
    """
    Import the law dataset
    Args:
        input_dir: Directory where the "law_data.csv" is stored. DATASET_DIR as default (./datasets)
        discretize: if True, discretize the dataset
    Returns:
        pd.DataFrame: the Law School dataset - columns = ['race', 'sex', 'score']
        str: the name of the score column ('score')
        list: list of sensitive attributes (['race', 'sex'])

    We use the dataset as in Multinomial_FA-IR and as in CFA
    We preprocess the datas as in https://github.com/MilkaLichtblau/ContinuousFairness/blob/master/src/data_preparation/LSAT.py

    """
    filename = os.path.join(input_dir, "law_data.csv")
    data = pd.read_csv(filename)
    data = data.drop(columns=["region_first", "sander_index", "first_pf"])

    data["sex"] = data["sex"].replace([2], 0)

    data["LSAT"] = data["LSAT"].apply(str)
    data["LSAT"] = data["LSAT"].str.replace(".7", ".75", regex=False)
    data["LSAT"] = data["LSAT"].str.replace(".3", ".25", regex=False)
    data["LSAT"] = pd.to_numeric(data["LSAT"])
    data["sex"].replace({1: "female", 0: "male"}, inplace=True)
    data = data.rename(columns={"LSAT": "score"})
    data = data.sort_values(by=["score"], kind="mergesort")[::-1]
    sensitive_attributes = ["sex", "race"]
    data.drop(columns=["id"], inplace=True)

    continuous_columns = ["UGPA", "ZFYA"]

    if discretize:
        from sklearn.preprocessing import KBinsDiscretizer

        est = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")
        est.fit(data[continuous_columns])
        data[continuous_columns] = est.transform(data[continuous_columns])
        interpretable_values_dict = {
            continuous_columns[i]: {
                0: f"<{est.bin_edges_[i][1]}",
                1: f"[{est.bin_edges_[i][1]}-{est.bin_edges_[i][2]})",
                2: f">={est.bin_edges_[i][2]}",
            }
            for i in range(est.bin_edges_.shape[0])
        }
        data = data.replace(interpretable_values_dict)

    features = sensitive_attributes + continuous_columns + ["score"]
    data = data[features]
    return data, "score", sensitive_attributes


def import_german_processed_all_attributes(
    input_dir=DATASET_DIR, use_foreigner=False, discretize=True
):
    """
    Import the german dataset
    Args:
        input_dir: Directory where the "germanCredit_sexAgeForeigner.csv" is stored. DATASET_DIR as default (./datasets)
        use_foreigner: if True consider also the foreigner attribute
        discretize: if True, discretize the dataset
    Returns:
        pd.DataFrame: the german dataset - columns =  ['sex', 'age', 'score'] or ['sex', 'age', 'foreigner', 'score'] if use_foreigner is True
        str: the name of the score column ('score')
        list: list of sensitive attributes (['sex', 'age'] or if ['sex', 'age', 'foreigner'] )

    We use the dataset as in Multinomial_FA-IR, downloaded from https://github.com/MilkaLichtblau/Multinomial_FA-IR
    We preprocess the datas as in Multinomial_FA-IR

    The processed dataset is characterized as follows:
    'sex' : ['male', 'female'], where protected encodes all ethinicities that differ from the caucasian one
    'age_cat': ['young', 'adult', 'elder']
    'foreigner' : [0, 1]
    """

    data = pd.read_csv(
        "https://raw.githubusercontent.com/MilkaLichtblau/Multinomial_FA-IR/5358ef2dff5f30964752e1e6c11654b37677b732/experiments/dataExperiments/data/GermanCredit/german.data",
        sep=" ",
        header=None,
    )

    # keep credit duration (A2), credit amount (A5), status of existing account (A1), employment length (A7)
    # plus protected attributes sex (A9), age (A13) and foreigner (A20)
    data = data.iloc[:, [0, 1, 4, 6, 8, 12, 19]]
    data.columns = [
        "accountStatus",
        "creditDuration",
        "creditAmount",
        "employmentLength",
        "sex",
        "age",
        "foreigner",
    ]
    # 0 means no account exists
    data["accountStatus"] = data["accountStatus"].replace(
        {"A11": 1, "A12": 2, "A13": 3, "A14": 0}
    )
    # 0 means unemployed
    data["employmentLength"] = data["employmentLength"].replace(
        {"A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4}
    )
    # 0 is male, 1 is female
    data["sex"] = data["sex"].replace(
        {"A91": 0, "A92": 1, "A93": 0, "A94": 0, "A95": 1}
    )
    # 0 is resident, 1 is foreigner
    data["foreigner"] = data["foreigner"].replace({"A201": 1, "A202": 0})

    # categorize age data, oldest (group 2) and youngest (group 1) decile are protected
    data["decileRank"] = pd.qcut(data["age"], 10, labels=False)

    data["age"] = np.where(data["decileRank"] == 0, 1, data["age"])
    data["age"] = np.where(data["decileRank"].between(1, 8), 0, data["age"])
    data["age"] = np.where(data["decileRank"] == 9, 2, data["age"])

    target_name = "score"

    data[target_name] = np.zeros(data.shape[0])
    for idx, row in data.iterrows():
        accountStatus = row.loc["accountStatus"]
        creditDuration = row.loc["creditDuration"]
        creditAmount = row.loc["creditAmount"]
        employmentLength = row.loc["employmentLength"]

        score = (
            0.25 * accountStatus
            + 0.25 * creditDuration
            + 0.25 * creditAmount
            + 0.25 * employmentLength
        )
        data.loc[idx, target_name] = score

    # data = data.drop(
    #    columns=["accountStatus", "creditDuration", "creditAmount", "employmentLength"]
    # )
    data[target_name] = stats.zscore(data[target_name])

    # We drop this columns since these are generated to create the score
    drop_cols = ["duration", "credit_amount", "checking_status", "employment"]

    if use_foreigner:
        sensitive_attributes = ["sex", "age", "foreigner"]
    else:
        sensitive_attributes = ["sex", "age"]

    # We load the interpretable dataset - it has already the attribute values converted
    filename = os.path.join(input_dir, "credit-g.csv")
    if os.path.isfile(filename):
        datai = pd.read_csv(filename)

    datai[sensitive_attributes] = data[sensitive_attributes]
    status_map = {
        "'male single'": "single",
        "'female div/dep/mar'": "married/wid/sep",
        "'male mar/wid'": "married/wid/sep",
        "'male div/sep'": "married/wid/sep",
    }
    datai["civil_status"] = datai["personal_status"].replace(status_map)
    datai.drop(columns=["personal_status", "class"] + drop_cols, inplace=True)
    datai[target_name] = data[target_name]

    continuous_columns = []

    # Try to discretize all columns with more than 10 values
    for c in datai.columns:
        if c != target_name:
            if len(datai[c].value_counts()) > 10:
                continuous_columns.append(c)

    if discretize and continuous_columns != []:
        from sklearn.preprocessing import KBinsDiscretizer

        est = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")
        est.fit(datai[continuous_columns])
        datai[continuous_columns] = est.transform(datai[continuous_columns])
        interpretable_values_dict = {
            continuous_columns[i]: {
                0: f"<{est.bin_edges_[i][1]:.2f}",
                1: f"[{est.bin_edges_[i][1]:.2f}-{est.bin_edges_[i][2]:.2f})",
                2: f">={est.bin_edges_[i][2]:.2f}",
            }
            for i in range(est.bin_edges_.shape[0])
        }
        datai = datai.replace(interpretable_values_dict)
    datai.sort_values(by=["score"], ascending=False, inplace=True)
    return datai, target_name, sensitive_attributes
