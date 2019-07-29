## Extracting survival-relevant subnetworks from multi-scale omics data.


### Files description

-The file Library_MMAS.py is the library containing all the functions.

-The file first_run_MMAS.py contains the code with the upload of the data and the evaluation of the functions in Library_MMAS.py (with no correlation matrix).

-File Library_MMAS_corrMat.py is like Library_MMAS.py but modified to use the correlation matrix.
-File run_MMAS.py is adapted to use the just mentioned Library.



-The file Simulate.py contains the code that can be used to simulate new data.


The two file .csv are the matrix with simulated data (PPI of 50 nodes, 20 observations):

-gxSIM.csv is the matrix with gene expression data, survival data and censor information;

-netSIM.csv is the adjacency matrix of the scale-free simulated PPI network.
