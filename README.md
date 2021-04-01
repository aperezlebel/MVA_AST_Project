# MVA – Time series learning – Project
## Authors
**Hugo Laurençon, Alexandre Pérez**


## Install
To download the project and install its dependencies, please run the following command line commands in the directory of your choice.
```sh
git clone git@github.com:alexprz/MVA_AST_Project.git
python3 -m venv venv
source venv/bin/activate
pip install pip --upgrade
pip install -r requirements.txt
```
Note that our code has been checked for Python 3.9.2.

## Structure
The structure of our code is organized as follows.

`src/methods`:
* `DictionaryLearningMethod.py` Implement the dictionary learning based method studied in our project. It is based on the DictionaryLearning estimator of scikit-learn.

`src/datasets`: Implements some classes to load time series.
* `EquityDataset.py` The dataset the end-of-day stock price of 10 companies approximatly between 2000 and 2017.

`src/benchmark`: Implements the experiment that were run in our project.
* `SparsityBenchmark.py` Influence of the sparsity constraint on the quality of the results and on the compression rate (figure 1).
* `SizeBenchmark.py` Influence of the width and the stride on the quality of the results and on the compression rate (figure 2).
* `` Influence of the number of atoms on the quality of the results and on the compression rate (figure 3).
* `TrainSizeBenchmark.py`Influence of the size of the train set on the quality of the results (figure 4).


## Reproduce our experiments
The following commands permit to reproduce the figures of our report.

**Figure 1: Influence of the sparsity constraint**

a. RMSRE
```python
python main.py benchmark qts --splits 10 --dist rmsre --ds ety --w 14 --s 14 --iter 100
```
b. DTW
```python
python main.py benchmark qts --splits 10 --dist dtw --ds ety --w 14 --s 14 --iter 100
```


**Figure 2: Influence of the width and stride**
```python

```


**Figure 3: Influence of the number of atoms**
```python

```


**Figure 4: Influence of the size of the training set**

a. RMSRE
```python
python main.py benchmark qcr --splits 5 --dist rmsre --ds ety --w 14 --s 14 --iter 100
```

b. DTW
```python
python main.py benchmark qcr --splits 5 --dist dtw --ds ety --w 14 --s 14 --iter 100
```
