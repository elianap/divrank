# DivRank mitigation algorithm

To replicate the results, users can launch A_00* scripts.

A_001, A_002 and A_003 executes the re-ranking algorithms for all the analyzed datasets.

- A_001 runs the mitigation with Div-rank
- A_002 runs the mitigation with Feldman
- A_003 runs the mitigation with CFA-\theta
- A_004 runs the sensitivity analysis of Div-rank
- A_005 runs the additional test with div-rank and all attributes as input
- A_006 runs the additional test with CFA$theta$ and all attributes as input

The notebook 'B_collect_results' collect the results of A_001 to A_004.


The notebook 'B_01_collect_results' collect the results of A_001 to A_004.

The notebook 'B_02_all_attributes_collect_results' collect the results of A_005 and A_006 and runs the additional test with Feldman and all attributes as input