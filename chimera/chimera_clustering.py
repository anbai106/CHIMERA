from chimera.algorithm import clustering_main

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen, Aoyan Dong"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Maintaining"

def clustering(feature_tsv, output_dir, k, covariate_tsv, weight_covariate=-1.0, weight_site=10, lambda_b=10.0,
               lambda_A=100.0, transformation_type='affine', tol=0.001, max_iteration=1000, num_initialization_run=3,
               save_model=True, standardization_method='zscore', saving_criterion='reproducibility', verbose=False):
    """
    Clustering heterogenous disease effects via distribution matching of imaging patterns.
    Ref: https://pubmed.ncbi.nlm.nih.gov/26452275/
    Args:
        feature_tsv:str, path to the tsv containing extracted feature, following the BIDS convention. The tsv contains
        the following headers: "
                                 "i) the first column is the participant_id;"
                                 "ii) the second column should be the session_id;"
                                 "iii) the third column should be the diagnosis;"
                                 "The following column should be the extracted features. e.g., the ROI features"
        output_dir: str, path to store the clustering results
        k: int, number of clusters
        covariate_tsv: str, path to the tsv containing the covariates, eg., age or sex. The header (first 3 columns) of
                     the tsv file is the same as the feature_tsv, following the BIDS convention.
        weight_covariate: float, weight of covariates distance (default -1.0)
        weight_site: float, weight of set distance (default 10)
        lambda_b: int, the value of lambda1, for b (non-negative, default 10)
        lambda_A: int, the value of lambda2, for A (non-negative, default 100)
        transformation_type: str, Transformation to be used, choices=["affine", "duo", "trans"], default is "affine".
        tol: float, stopping critarion tolerance (default 0.001)
        max_iteration: int, maximum iteration allowed (default 1000)
        num_initialization_run: int, number of Runs of optimization with different initialization (default 50)
        save_model: Bool, if save trained model, default is False.
        standardization_method: str, standardization method, choices=["minmax", "zscore"], default is "minmax".
        saving_criterion: str, saving model criterion, choices=["energy_min", "reproducibility"], default is "energy_min".
        verbose: Bool, default is False. If the output message is verbose.

    Returns: clustering outputs.

    """
    print('chimera for semi-supervised clustering...')

    ### generate the dictionary for the arguments
    config_arg = {'K': k, 'verbose': verbose, 'lambda1': lambda_b, 'lambda2': lambda_A, 'r': weight_covariate, 'rs': weight_site,
    'eps': tol, 'max_iter': max_iteration, 'numRun': num_initialization_run, 'modelFile': save_model, 'transform': transformation_type,
    'norm': standardization_method, 'mode': saving_criterion, 'quiet': True}

    ## go into the core function of chimera
    clustering_main(feature_tsv, covariate_tsv, output_dir, config_arg)

    print('Finish...')
