# REU-2023
The files in this repo coincide with my paper "Dimension Reduction and the Fenchel Game". Here are some key notes:
- Please replace the file path in the laplacian_eigenmaps_nn_experiment with the path to your machine. 
- Feel free to test out various values of t nearest neighbors (hyperparameter) for the laplacian_eigenmaps_helix_experiment.
- The Averaged Laplacian Eigenmaps' output for t > 1 tends to shrink immensely, likely because the weight matrix's values
are severly diminished due to small denominators b in e^{-a^2/b^2}. These small denominators are the products of 
L2 normed distances between dense neighborhoods of points in 3d space (hence the denominators are on the order of 10^-3)
- To run the numpy_neuralnet.ipynb, please replace the "archive" file path to that on your machine linked to https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
