CHIMERA

  Section of Biomedical Image Analysis
  Department of Radiology
  University of Pennsylvania
  Richard Building
  3700 Hamilton Walk, 7th Floor
  Philadelphia, PA 19104

  Web:   https://www.cbica.upenn.edu/sbia/
  Email: sbia-software at uphs.upenn.edu

  Copyright (c) 2016 University of Pennsylvania. All rights reserved.
  See https://www.cbica.upenn.edu/sbia/software/license.html or COPYING file.

Author:
Aoyan Dong
cbica-software@uphs.upenn.edu

===============
1. INTRODUCTION
===============
This software performs clustering of heterogenous disease patterns within patient group. The clustering is based on imaging features, covariate features and dataset information.

===============
2. TESTING & INSTALLATION
===============

This software has been primarily implemented in Python for Linux operating systems.

----------------
 Dependencies
----------------
- Python 2.7.9
- Python library numpy 1.7.2
- Python library sklearn 0.15

----------------
 Installation
----------------
Run the following command:

   python setup.py install

This will automatically install the package and a standalone script "CHIMERA". 

-----------------
 Test
-----------------
We provided a test sample in test folder. Simply run the following command:

    ./test.py

This runs a test script which may take a few minutes. The test case contains a synthetic 20 dimensional data. Data file is named test_data.csv. The imaging features have a nonlinear correlation with covariate 1, and no correlation with covariate 2.

Test cases are used to check if standalone CHIMERA can run correctly, the expected adjusted rand index should be larger than 0.9. The sample output is also under test/ folder.

==========
3. USAGE
==========
Here is a brief introduction of running CHIMERA. For a complete list of parameters, see --help option.

To run this software you will need an input csv file, with the following mandatory fields:
(a) Subject group label (binary), header "Group". 1: patient; 0: normal control.
(b) At leat one imaging feature, header "IMG".

For a csv file data.csv that looks like below:
    
    ID,        COVAR,COVAR, IMG,   IMG, ..., Group, Set
    ADNI_0001, 15.1, 0.454, 0.212, 0.13,....,0,     1
    ADNI_0002, 20.9, 0.121, 0.343, 1.32,..., 0,     2
    ADNI_0003, 21.2, 0.141, 0.143, 0.21,..., 1,     2
    ...
    ...
    
If you install the package successfully, there will be two ways of running CHIMERA:

1. Running CHIMERA in command line (recommanded):

    CHIMERA -i data.csv -r output.txt -k 3 -o model.cpkl -m 20 -v

2. Running CHIMERA as a package, a simple example:

    import CHIMERA
    CHIMERA.run(dataFile, outFile, numClusters, verbose=True)


The software returns:
1. clustering labels in output.txt
2. transformation model in model.cpkl (cPickle binary mode)
 
** For the best performance, sample size should be large enough (100+) and parameters have to be cross validated.

===========
4. LICENSING
===========

  See https://www.cbica.upenn.edu/sbia/software/license.html or "licences/COPYING" file.
