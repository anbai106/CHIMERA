from .algorithm import *

##NOTE: this script has not been adapted! Don't use it directly!

def train(dataFile, outFile, numClusters, lambda1=10.0, lambda2=100.0, r=-1.0, rs=10, eps=0.001, max_iter=100,\
        numRun=50, modelFile="", transform="affine", norm=1, mode=2, verbose=False, quiet=True):
    """
    CHIMERA--
    Clustering heterogenous disease effects via distribution matching of imaging patterns.
    
    *** For best performance, pick optimal lambda values based on cross-validation *** 

    Required:
    
    dataFile:    <string> File name of data in .csv format with header line, one subject per row 
                          Header line can have types of:
                            ID    : subject ID (optional)
                            Group : subject diagnosis(1: patient; 0: normal control) (required)
                            Set   : subject dataset ID (optional)
                            COVAR : covariates (optional)
                            IMG   : imaging features (required)
    outFile:     <string> File name of clustering outputs
    numClusters: <int>    Number of sub-groups to find
    
    Options: (See reference for more details)
    
    lambda1:     <float>   The value of lambda1, for b (non-negative, default 10)
    lambda2:     <float>   The value of lambda2, for A (non-negative, default 100)
    transform:   <string>  Transformation to be used. (affine/duo/trans, default affine)
    eps:         <float>   Stopping critarion tolerance (default 0.001)
    max_iter:    <int>     Maximum iteration allowed (default 1000)
    numRun:      <int>     Number of Runs of optimization with different initialization (default 50)
    modelFile:   <string>  File name to save trained model (default not saving)
    r:           <float>   Weight of covariates distance (default auto)
    rs:          <float>   Weight of set distance (default 10)
    norm:        <int>     Data normalization. 0: no action; 1: 0-1 normalization; 2: z-score. (default 1)
    mode:        <int>     Save most reproducible (c=1)/ minimal energy (c=2) result among runs (default 2)
    verbose:     <bool>    Verbose output or not
    quite:       <bool>    Quiet for optimization objectives or not

    
    csv file data.csv should look like below (must contain fields IMG and Group):
        
        ID,        COVAR,COVAR, IMG,   IMG, ..., Group, Set
        ADNI_0001, 15.1, 0.454, 0.212, 0.13,....,0,     1
        ADNI_0002, 20.9, 0.121, 0.343, 1.32,..., 0,     2  
        ADNI_0003, 21.2, 0.141, 0.143, 0.21,..., 1,     2
        ...
        ...
    
    Reference:
    Dong, et al. "CHIMERA:Clustering heterogenous disease effects via distribution matching
    of imaging patterns" IEEE Transactions on Medical Imaging, 2016.

    Copyright (c) year University of Pennsylvania. All rights reserved.
    Contact: sbia-software@uphs.upenn.edu

    """

    config = {'verbose' : verbose,
              'lambda1' : lambda1,  #lambda1 for b
              'lambda2' : lambda2, #lambda2 for A
              'r'       : r,  #weight of covariate distance
              'rs'      : rs,    #weight of set distance
              'eps'     : eps, #precision tolerance
              'max_iter': max_iter,
              'numRun'  : numRun,   #number of different initializations
              'modelFile' : modelFile,
              'transform' : transform,
              'norm'      : norm,
              'mode'      : mode,   #save result based on reproducibility or energy function
              'quiet'     : quiet
              }
    
    sys.stdout.write("Parsing arguments...\n")
     
    # check input
    if not os.path.exists(dataFile):
        sys.stdout.write("File " + dataFile + " not found\n")
        sys.exit(1)
    if dataFile == outFile:
        sys.stdout.write("Input file and output file name should not be the same.\n")
        sys.exit(1)
    if numClusters <= 1:
        sys.stdout.write("Error: number of clusters should be larger than 1, got " + str(numClusters) +'\n')
        sys.exit(1)

    # check regularizations
    if (config['lambda1']<0) | (config['lambda2']<0) :
        sys.stdout.write("Error: lambda must be non-negative value.\n")
        sys.exit(1)

    # check transformation model
    if config['transform'] not in ["affine","duo","trans"] :
        sys.stdout.write("Error: transform should be chosen from affine/duo/trans.\n")
        sys.exit(1)
    
    # check norm and mode
    if config['norm'] not in [0,1,2]:
        sys.stdout.write("Error: normalization type should be chosen from 0/1/2.\n")
        sys.exit(1)
    if config['mode'] not in [1,2]:
        sys.stdout.write("Error:  should be chosen from 0/1/2.\n")
        sys.exit(1)
    
    # run optimzation
    sys.stdout.write("Starting CHIMERA clustering...\n")
    config.update({'K':numClusters})
    clustering_main(dataFile,outFile,config)
    
    # check output
    if not os.path.exists(outFile):
        sys.stdout.write("Error in generating clusters, " + outFile + " not generated\n")
        sys.exit(1)
    
    sys.stdout.write("Finished.\n")


def test(dataFile, outFile, modelFile):
    """
    CHIMERA--
    Clustering heterogenous disease effects via distribution matching of imaging patterns.
    
    *** Test out of samples using trained model ***
    *** Be careful to provide same data structure as used in training. ***

    Required:
    
    dataFile:    <string> File name of data in .csv format with header line, one subject per row 
                          Header line can have types of:
                            ID    : subject ID (optional)
                            Set   : subject dataset ID (optional)
                            COVAR : covariates (optional)
                            IMG   : imaging features (required)
    outFile:     <string> File name of clustering outputs
    modelFile:   <string> File name of trained model
    
    csv file data.csv should look like below (must contain fields IMG and Group):
        
        ID,        COVAR,COVAR, IMG,   IMG, ..., Set
        TEST_0001, 15.1, 0.454, 0.212, 0.13,..., 1
        TEST_0002, 20.9, 0.121, 0.343, 1.32,..., 2  
        TEST_0003, 21.2, 0.141, 0.143, 0.21,..., 2
        ...
        ...
    
    Reference:
    Dong, et al. "CHIMERA:Clustering heterogenous disease effects via distribution matching
    of imaging patterns" IEEE Transactions on Medical Imaging, 2016.

    Copyright (c) year University of Pennsylvania. All rights reserved.
    Contact: sbia-software@uphs.upenn.edu

    """

    # check input
    if not os.path.exists(dataFile):
        sys.stdout.write("File " + dataFile + " not found\n")
        sys.exit(1)
    if not os.path.exists(modelFile):
        sys.stdout.write("Model " + modelFile + " not found\n")
        sys.exit(1)
    if dataFile == outFile:
        sys.stdout.write("Input file and output file name should not be the same.\n")
        sys.exit(1)

    # run optimzation
    sys.stdout.write("Starting CHIMERA clustering...\n")
    clustering_test(dataFile,outFile,modelFile)
    
    # check output
    if not os.path.exists(outFile):
        sys.stdout.write("Error in generating clusters, " + outFile + " not generated\n")
        sys.exit(1)
    
    sys.stdout.write("Finished.\n")
