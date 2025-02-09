# Machine Learning & Adaptive Intelligence Assignment

# Part 1: Tracking Bees - Machine Learning
**Skills Gained**
- **Machine Learning Techniques**: Implemented supervised learning methods like **linear regression** for regression tasks.
- **Gaussian Basis Functions**: Applied **Gaussian functions** to model complex data patterns.
- **Model Optimization**: Used optimization techniques like **minimizing negative log likelihood** to fine-tune model parameters.
- **Data Visualization**: Visualized **flight path predictions** and compared them with actual data.
- **Python & Libraries**: Proficient in using **Numpy** for data manipulation and **Matplotlib** for visualization in machine learning tasks.


# Part 2: Machine Learning - Neural Networks, Dimensionality Reduction, Clustering & Denoising Autoencoders
**Skills Gained**
- **Neural Network Architectures**: Implemented and trained **Convolutional Neural Networks (CNNs)** and **Fully-Connected Neural Networks (FCNNs)** for image classification tasks.
- **Hyperparameter Tuning**: Optimized model performance using hyperparameter tuning techniques for learning rates, dropout rates, and the number of neurons.
- **Dimensionality Reduction**: Applied **PCA** to reduce data dimensions and retain the most important features for classification.
- **Clustering**: Used **KMeans clustering** to group similar data points and visualize class distributions.
- **Autoencoders**: Built and trained a **Denoising Autoencoder** to clean noisy images using convolutional layers and evaluated reconstruction error.
- **Machine Learning Libraries**: Proficient in **PyTorch** for building neural networks, **Scikit-Learn** for clustering, and **Matplotlib** for visualizations.


# Part 1: Tracking Bees - Machine Learning

## Overview
In this assignment, the task is to estimate the flight path of a bee based on detector observations. The model predicts the bee's location using **linear regression** and **Gaussian basis functions**, minimizing the total negative log likelihood.

## Libraries
- **Numpy**
- **Matplotlib**

## Dataset
The dataset `bee_flightpaths.npy` includes 30 flight paths with:
- `truepath`: The true flight path (100 points).
- `observations`: 17 observations, each containing the time, detector location, and the direction of the bee.

## Key Techniques
1. **Flight Path Reconstruction**: Use detector bearings to predict the bee's flight path.
2. **Negative Log Likelihood**: Implement a function to compute the error between predicted and observed locations.
3. **Linear Regression**: Predict the bee’s x and y coordinates over time using Gaussian basis functions.
4. **Optimization**: Minimize the negative log likelihood through hyperparameter optimization.

## Key Functions
- **negloglikelihood()**: Calculates negative log likelihood for a prediction.
- **getpred()**: Uses linear regression to predict bee positions.
- **totalnegloglikelihood()**: Computes the total negative log likelihood with regularization.

## Steps
1. **Plotting the True Path**: Visualize the true flight path of the bee and mark the starting point.
2. **Developing Likelihood Function**: Compute the error between predicted and observed directions.
3. **Prediction with Linear Regression**: Use Gaussian basis functions for predicting bee positions.
4. **Optimization**: Optimize model parameters with `scipy.optimize.minimize`.
5. **Visualizing the Predicted Path**: Plot the predicted path and compare it with the true path.

## Conclusion
This project demonstrates how to track the bee's flight path using machine learning techniques such as linear regression and Gaussian basis functions. The model parameters are optimized to minimize the error between predicted and true paths.


# Part 2: Machine Learning - Neural Networks, Dimensionality Reduction, Clustering & Denoising Autoencoders

## Overview
This part focuses on classification with neural networks, dimensionality reduction, clustering, and denoising autoencoders. Models are trained and evaluated on the **OrganAMNIST** and **CIFAR-10** datasets.

## Libraries
- **Numpy**
- **Scipy**
- **Matplotlib**
- **PyTorch**
- **Scikit-Learn**

## Task 1: Classification with Neural Networks
- Implemented **Convolutional Neural Networks (CNN)** and **Fully-Connected Neural Networks (FCNN)**.
- Applied **hyperparameter tuning** (learning rates, nodes, dropout) to improve performance.
- Best Model: **CNN** achieved 97.83% accuracy on **OrganAMNIST** dataset.

## Task 2: Dimensionality Reduction & Clustering
- Used **KMeans clustering** and **PCA** for dimensionality reduction.
- Visualized data in 3D to analyze class distributions and clustering performance.

## Task 3: Denoising Autoencoder (CIFAR-10)
- Added noise to CIFAR-10 images and trained a **Convolutional Autoencoder** to denoise them.
- Evaluated the model using **Mean Squared Error** loss between the noisy input and the original image.
- **Hyperparameter tuning** (learning rates, number of neurons) improved model performance.

### Key Findings:
1. **Learning Rate**: Lowering the learning rate improved performance up to a point (0.0001), after which performance worsened.
2. **Number of Neurons**: Increasing the number of neurons didn’t always improve results; a simpler model performed better.

## Summary
- **Skills**: Model development, hyperparameter tuning, clustering, dimensionality reduction, and autoencoder training.
- **Best Performance**: CNN (OrganAMNIST) and Convolutional Autoencoder (CIFAR-10).
