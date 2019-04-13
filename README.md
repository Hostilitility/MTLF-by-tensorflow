# MTLF-by-tensorflow
MTLF algorithm for regression, written by Yu Ye

The algorithm was proposed by:
[1] Xu Y, Pan S J, Xiong H, et al. A unified framework for metric transfer learning[J].
    IEEE Transactions on Knowledge and Data Engineering, 2017, 29(6): 1158-1171.
    
Only the classification edition of MTLF on Matlab is given by the author.So we implement 
the algorithm on tensorflow, and replace SGD, the default optimization algorithm, with Adam.
