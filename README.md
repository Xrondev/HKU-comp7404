# HKU-comp7404

---
A rough implementation on the paper "_Breast Cancer Diagnosis Using Support Vector Machines Optimized by Whale Optimization and Dragonfly Algorithms_" https://ieeexplore.ieee.org/document/9805591

## Environment

The python code is implemented on Python 3.11 on Windows platform. 
you may need `matplotlib`, 'scikit-learn`, `numpy` to run the code.

## Files

```
-dataset/           the database files
-algo_util.py       base functions for woa and da
-config.py          set the global random state
-da_svm.py          DA-SVM implementation
-dataset.py         used for load the dataset and partition
-svm.py             basic Support Vector Mechine
-woa_svm.py         WOA-SVM implementation
```

## Dataset
- Wisconsin Diagnosis Breast Cancer (WDBC) databases
- Wisconsin Breast Cancer Database (WBCD)
Can be found on [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
) and [here](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)

  You can directly use the .data file from the dataset folder.
