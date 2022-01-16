## Optimizer Development and Analysis for Activation Maximization

This is the project page for the project for testing and developing black box optimizers for searching for optimal images for neurons in artifical and biological neurons. 

### Rationale 

### Large-scale Benchmark of Gradient Free Optimizers
We first conducted a large scale benchmark of common gradient free optimizers in their ability to optimize stimuli for visually selective units. 

We used units from pre-trained CNNs as models of visual neuron. For these optimizers, each visual neuron or CNN unit form a different function over the image space, thus it's a different test function for the optimizer. 
We chose the classic and popular model [`AlexNet`](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) and a deeper and adversarially trained model [`ResNet50-robust`](https://github.com/MadryLab/robustness) as benchmark. ResNet50-robust was chosen due to its relative light weight and its higher similarity to the visual representations in the brain (high rank on the [Brain-Score](https://www.brain-score.org/)).

We tested 12 gradient-free optimizers as implemented / interfaced by [`nevergrad`](https://github.com/facebookresearch/nevergrad): NGOpt, DE, TwoPointsDE, ES, CMA, RescaledCMA, DiagonalCMA, SQPCMA, PSO, OnePlusOne, TBPSA, and RandomSearch. 

![performance]()

![runtime]()

We found that Covariance Matrix Adaptation Evolution Strategy (CMAES) and Diagonal CMAES are the top two algorithms in terms of the highest activation it achieved. We found the same result for units across models and layers that even with the default setting of hyperparameters. 

### Comparison with Previous Method Genetic Algorithm
！[](media/Figure_GA_CMA_cmp_vivo_silico-01.png)

### Benchmark Specific Type of CMAES Optimizer 


## Geometry of CMA-ES Optimizer
Given this unique success of CMA-ES type optimizer, we further analyzed the geometry of its evolution trajectory to gain insights of its working. 

### Sinusoidal Structure 
![](Figure TrajSinusoidal-01.png)
### Evolution Trajectory and Space Geometry 



## Re-design CMA-ES Optimizer

### Support or Contact
Contact binxu_wang@hms.harvard.edu if you have more questions about our work!
