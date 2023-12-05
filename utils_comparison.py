''''
Code adapted from  https://github.com/MilkaLichtblau/FA-IR_Ranking/blob/FA-IR_CIKM_17/src/post_processing_methods/fair_ranker/create.py

@author: meike.zehlike

modification by Anonymous Authors
'''

def get_protected_unprotected(df_input, protected_attributes,  protected_values, target, cid_name = 'cid' ):

    if type(protected_attributes)==str:
        protected_attributes = [protected_attributes]

    if type(protected_values)==str:
        protected_attributes = [protected_values]

    
    df_input_sorted = df_input.sort_values(target, kind='mergesort', ascending=False).copy()

    # Get slice
    sel_indexes = df_input_sorted.index
    for protected_attribute, protected_value in zip(protected_attributes,  protected_values):
        if type(protected_value)==list:
            sel_indexes = df_input_sorted.loc[sel_indexes].loc[df_input_sorted[protected_attribute].isin(protected_value)].index
        else:
            sel_indexes = df_input_sorted.loc[sel_indexes].loc[df_input_sorted[protected_attribute]==protected_value].index

    df_protected = df_input_sorted.loc[sel_indexes][[target]]
    df_non_protected = df_input_sorted.loc[~ df_input_sorted.index.isin(sel_indexes)][[target]]

    # CID is the original candidate index
    df_protected = df_protected.reset_index(names = cid_name)
    df_non_protected = df_non_protected.reset_index(names = cid_name)
    
    original_protected_scores = df_protected[target].values
    original_non_protected_scores = df_non_protected[target].values

    original_protected_cid = df_protected[cid_name].values
    original_non_protected_cid = df_non_protected[cid_name].values
    return original_protected_scores, original_non_protected_scores, original_protected_cid, original_non_protected_cid
    
    
def feldman_processing(original_protected_scores, original_non_protected_scores, K=1000):
    from copy import deepcopy
    from tqdm import tqdm
    from scipy.stats import percentileofscore
    from scipy.stats import scoreatpercentile

    new_protected_scores = deepcopy(original_protected_scores)

    for candidate_index in range(len(original_protected_scores)):
        if candidate_index >= K:
            # only need to adapt the scores for protected candidates up to required length
            # the rest will not be considered anyway
            break

        # find percentile of protected candidate
        p = percentileofscore(original_protected_scores, original_protected_scores[candidate_index])
        if p>=100:
            # Round
            p = 100
        # find score of a non-protected in the same percentile
        score = scoreatpercentile(original_non_protected_scores, p)
        new_protected_scores[candidate_index] = score
    return new_protected_scores

def merge_ranking_protected_unnprotected(original_protected_scores, original_non_protected_scores, new_protected_scores, original_protected_cid, original_non_protected_cid, K = 1000, verbose = False):

    import numpy as np
    sorted_indexes_protected_mit = np.argsort(-new_protected_scores, kind='mergesort')
    sorted_indexes_non_protected_mit = np.argsort(-original_non_protected_scores, kind='mergesort')


    i_protected = 0
    i_non_protected = 0 

    merged_scores = []
    merged_protected = []
    merged_ids = []
    for i in range(K):

        if verbose:
            print(i, i_protected, i_non_protected, len(original_protected_scores), len(original_non_protected_scores))
        if i_protected >=len(original_protected_scores):
            
            inp = sorted_indexes_non_protected_mit[i_non_protected]
            if verbose:
                print(
                    i, inp,
                    "no more protected candidates available, take non-protected instead",
                )
            merged_scores.append([original_non_protected_cid[inp], original_non_protected_scores[inp], 0, original_non_protected_scores[inp]])
            i_non_protected += 1
        elif i_non_protected >= len(original_non_protected_scores):
            ip = sorted_indexes_protected_mit[i_protected]
            if verbose:
                print(
                    i, ip, 
                    "no more non-protected candidates available, take protected instead",
                )
            merged_scores.append([original_protected_cid[ip], new_protected_scores[ip], 1, original_protected_scores[ip]])
            i_protected += 1
        else:

            ip = sorted_indexes_protected_mit[i_protected]
            inp = sorted_indexes_non_protected_mit[i_non_protected]
            if ip >= len(new_protected_scores):
                if verbose:
                    print(
                        i,
                        "A no more protected candidates available, take non-protected instead",
                    )
                merged_scores.append([original_non_protected_cid[inp], original_non_protected_scores[inp], 0, original_non_protected_scores[inp]])
                i_non_protected += 1

            elif inp >= len(original_non_protected_scores):
                if verbose:
                    print(
                        i,
                        "A no more non-protected candidates available, take protected instead",
                    )
                merged_scores.append([original_protected_cid[ip], new_protected_scores[ip], 1, original_protected_scores[inp]])
                i_protected += 1
            else:
                if verbose:
                    print(i, "find the best candidate available")
                if (
                    new_protected_scores[ip]
                    >= original_non_protected_scores[inp]
                ):
                    if verbose:
                        print(ip, " the best is a protected one")
                    merged_scores.append([original_protected_cid[ip], new_protected_scores[ip], 1, original_protected_scores[ip]])
                    i_protected += 1
                else:
                    if verbose:
                        print(inp, " the best is a non-protected one")
                    merged_scores.append([original_non_protected_cid[inp], original_non_protected_scores[inp], 0, original_non_protected_scores[inp]])
                    i_non_protected += 1
    import pandas as pd
    mit_res = pd.DataFrame(merged_scores, columns = ['cid', 'new_score', 'protected', 'original_score'])
    return mit_res
        