---
layout: NotesPage
title: K-Means
permalink: /work_files/research/ml/kmeans
prevLink: /work_files/research/dl/ml.html
---

[Explanation](http://www.chioka.in/explain-to-myself-k-means-algorithm/)  



# K-Means

__K-Means:__{: style="color: red"}    
It is a method for cluster analysis. It aims to partition $$n$$ observations into $$k$$ clusters in which each observation belongs to the cluster with the nearest mean. It results in a partitioning of the data space into __Voronoi Cells__.  


__IDEA:__{: style="color: red"}  
* Minimizes the aggregate Intra-Cluster distance
* Equivalent to minimizing the Variance
* Thus, it finds k-clusters with __minimum aggregate Variance__.  


__Formal Description:__{: style="color: red"}    
Given a set of observations $$\left(\mathbf{x}_{1}, \mathbf{x} _{2}, \ldots, \mathbf{x}_{n}\right)$$, $$\mathbf{x}_ i \in \mathbb{R}^d$$, the algorithm aims to partition the $$n$$ observations into $$k$$ sets $$\mathbf{S}=\left\{S_{1}, S_{2}, \ldots, S_{k}\right\}$$ so as to minimize the __intra-cluster Sum-of-Squares__ (i.e. __variance__).  

The Objective:  
<p>$$\underset{\mathbf{S}}{\arg \min } \sum_{i=1}^{k} \sum_{\mathbf{x} \in S_{i}}\left\|\mathbf{x}-\boldsymbol{\mu}_{i}\right\|^{2}=\underset{\mathbf{S}}{\arg \min } \sum_{i=1}^{k}\left|S_{i}\right| \operatorname{Var} S_{i}$$</p>  
where $$\boldsymbol{\mu}_i$$ is the mean of points in $$S_i$$. 



__Algorithm:__{: style="color: red"}  
* Choose two random points, call them _"Centroids"_  
* Assign the closest $$N/2$$ points (Euclidean-wise) to each of the Centroids  
* Compute the mean of each _"group"/class_ of points  
* Re-Assign the centroids to the newly computed Means ↑
* REPEAT!

The "assignment" step is referred to as the "expectation step", while the "update step" is a maximization step, making this algorithm a variant of the generalized expectation-maximization algorithm.


__Complexity:__{: style="color: red"}    
The original formulation of the problem is __NP-Hard__; however, __EM__ algorithms (specifically, Coordinate-Descent) can be used as efficient heuristic algorithms that converge quickly to good local minima.  


__Convergence:__{: style="color: red"}    
Guaranteed to converge after a finite number of iterations  
* __Proof:__  
    The Algorithm Minimizes a __monotonically decreasing__, __Non-Negative__ _Energy function_ on a finite Domain:  
    By *__Monotone Convergence Theorem__* the objective Value Converges.


__Optimality:__{: style="color: red"}    
* __Locally optimal__: due to convergence property  
* __Non-Globally optimal:__  
    * The _objective function_ is *__non-convex__*  
    * Moreover, coordinate Descent doesn't converge to global minimum on non-convex functions.  


__Objective Function:__{: style="color: red"}    
<p>$$J(c, \mu)= \sum_{i=1}^{m} \| x^{(i)}-\mu_c \|^{2}$$</p>  
* __MLE__:  
    <p>$$\dfrac{\partial}{\partial \mu_c} J(c, \mu) = \dfrac{\sum_{i=1}^m x^{(i)}}{m}$$</p>