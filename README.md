## Improving Expressive Power of Spectral Graph Neural Networks with Eigenvalue Correction

This code contains a PyTorch implementation of "Improving Expressive Power of Spectral Graph Neural Networks with Eigenvalue Correction" (AAAI 2024)

## Environment Settings    
- pytorch 1.8.1
- numpy 1.18.1
- torch-geometric 1.6.3 
- tqdm 4.59.0
- scipy 1.6.2
- seaborn 0.11.1
- scikit-learn 0.24.1


## Histogram of eigenvalue distribution in Figure 1 in the paper

You can run the following Command:
```
python histogram.py
```

## The number of distinct eigenvalues in Table 1 in the paper

You can run the following Command:
```
python distinct_eigvalues.py
```

## Eigendecomposition

Run the following command to perform eigendecomposition on all ten datasets,
and will print the time required for each eigendecomposition.
```
python eigendecomposition.py
```
The eigenvalues and eigenvectors are stored in the data directory.

### Running the code

You can run the following script directly:
```sh
sh EC-Bern.sh
```
or run the following Command 
```sh
sh EC-GPR.sh
```
