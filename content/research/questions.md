---
layout: NotesPage
title: Prep Questions (Learning)
permalink: /work_files/research/prep_qs
prevLink: /work_files/research.html
---

# CNNs
* __What is a CNN?__{: style="color: red"}  
* __What are the layers of a CNN?__{: style="color: red"}  
* __What are the four important ideas and their benefits that the convolution affords CNNs:__{: style="color: red"}  
* __What is the inspirational model for CNNs:__{: style="color: red"}  
* __Describe the connectivity pattern of the neurons in a layer of a CNN:__{: style="color: red"}  
* __Describe the process of a ConvNet:__{: style="color: red"}  
* __Convolution Operation:__{: style="color: red"}  
    * __Define:__{: style="color: blue"}  
    * __Formula (continuous):__{: style="color: blue"}  
    * __Formula (discrete):__{: style="color: blue"}  
    * __Define the following:__{: style="color: blue"}  
        * __Feature Map:__{: style="color: blue"}  
    * __Does the operation commute?__{: style="color: blue"}  
* __Cross Correlation:__{: style="color: red"}  
    * __Define:__{: style="color: blue"}  
    * __Formulae:__{: style="color: blue"}  
    * __What are the differences/similarities between convolution and cross-correlation:__{: style="color: blue"}  
* __Write down the Convolution operation and the cross-correlation over two axes and:__{: style="color: red"}  
    * __Convolution:__{: style="color: blue"}  
    * __Convolution (commutative):__{: style="color: blue"}  
    * __Cross-Correlation:__{: style="color: blue"}  
* __The Convolutional Layer:__{: style="color: red"}  
    * __What are the parameters and how do we choose them?__{: style="color: blue"}  
    * __Describe what happens in the forward pass:__{: style="color: blue"}  
    * __What is the output of the forward pass:__{: style="color: blue"}  
    * __How is the output configured?__{: style="color: blue"}  
* __Spatial Arrangements:__{: style="color: red"}  
    * __List the Three Hyperparameters that control the output volume:__{: style="color: blue"}  
    * __How to compute the spatial size of the output volume?__{: style="color: blue"}  
    * __How can you ensure that the input & output volume are the same?__{: style="color: blue"}  
    * __In the output volume, how do you compute the $$d$$-th depth slice:__{: style="color: blue"}  
* __Calculate the number of parameters for the following config:__{: style="color: red"}  
    > Given:  
        * __Input Volume__:  $$64\times64\times3$$  
        * __Filters__:  $$15 7\times7$$  
        * __Stride__:  $$2$$  
        * __Pad__:  $$3$$  
* __Definitions:__{: style="color: red"}  
    * __Receptive Field:__{: style="color: blue"}  
* __Suppose the input volume has size  $$[ 32 × 32 × 3 ]$$  and the receptive field (or the filter size) is  $$5 × 5$$ , then each neuron in the Conv Layer will have weights to a *\_\_Blank\_\_* region in the input volume, for a total of  *\_\_Blank\_\_* weights:__{: style="color: red"}  
* __How can we achieve the greatest reduction in the spatial dims of the network (for classification):__{: style="color: red"}  
* __Pooling Layer:__{: style="color: red"}  
    * __Define:__{: style="color: blue"}  
    * __List key ideas/properties and benefits:__{: style="color: blue"}  
    * __List the different types of Pooling:__{: style="color: blue"}  
        [Answer](http://localhost:8889/work_files/research/dl/nlp/cnnsNnlp#bodyContents12)  
    * __List variations of pooling and their definitions:__{: style="color: blue"}  
        <button>Show Questions</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        * __What is "Learned Pooling":__{: style="color: blue"}  
        * __What is "Dynamical Pooling":__{: style="color: blue"}  
        {: hidden=""}
    * __List the hyperparams of Pooling Layer:__{: style="color: blue"}  
    * __How to calculate the size of the output volume:__{: style="color: blue"}  
    * __How many parameters does the pooling layer have:__{: style="color: blue"}    
    * __What are other ways to perform downsampling:__{: style="color: blue"}  
* __Weight Priors:__{: style="color: red"}  
    * __Define "Prior Prob Distribution on the parameters":__{: style="color: blue"}  
    * __Define "Weight Prior" and its types/classes:__{: style="color: blue"}  
        <button>Show Questions</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        * __Weak Prior:__{: style="color: blue"}  
        * __Strong Prior:__{: style="color: blue"}  
        * __Infinitely Strong Prior:__{: style="color: blue"}  
        {: hidden=""}
    * __Describe the Conv Layer as a FC Layer using priors:__{: style="color: blue"}  
    * __What are the key insights of using this view:__{: style="color: blue"}  
        <button>Show Questions</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        * __When is the prior imposed by convolution INAPPROPRIATE:__{: style="color: blue"}  
        * __What happens when the priors imposed by convolution and pooling are not suitable for the task?__{: style="color: blue"}  
        * __What kind of other models should Convolutional models be compared to? Why?:__{: style="color: blue"}  
        {: hidden=""}
* __When do multi-channel convolutions commute?__{: style="color: red"}  
[Answer](/work_files/research/dl/archits/convnets#bodyContents61)
* __Why do we use several different kernels in a given conv-layer?__{: style="color: red"}  
* __Strided Convolutions__{: style="color: red"}    
    * __Define:__{: style="color: blue"}  
    * __What are they used for?__{: style="color: blue"}  
    * __What are they equivalent to?__{: style="color: blue"}  
    * __Formula:__{: style="color: blue"}  
* __Zero-Padding:__{: style="color: red"}  
    * __Definition/Usage:__{: style="color: blue"}  
    * __List the types of padding:__{: style="color: blue"}  
* __Locally Connected Layers/Unshared Convolutions:__{: style="color: red"}  
* __Bias Parameter:__{: style="color: red"}  
    * __How many bias terms are used per output channel in the tradional convolution:__{: style="color: blue"}  
* __Dilated Convolutions__{: style="color: red"}    
    * __Define:__{: style="color: blue"}  
    * __What are they used for?__{: style="color: blue"}  
* __Stacked Convolutions__{: style="color: red"}    
    * __Define:__{: style="color: blue"}  
    * __What are they used for?__{: style="color: blue"}  

* [Archits](http://localhost:8889/work_files/research/dl/arcts)


***

# Theory


# RNNs
* __What is an RNN?__{: style="color: red"}  
    * __Definition:__{: style="color: blue"}  
    * __What machine-type is the standard RNN:__{: style="color: blue"}  
* __What is the big idea behind RNNs?__{: style="color: red"}  
* __Dynamical Systems:__{: style="color: red"}  
    * __Standard Form:__{: style="color: blue"}  
    * __RNN as a Dynamical System:__{: style="color: blue"}  
* __Unfolding Computational Graphs__{: style="color: red"}  
    * __Definition:__{: style="color: blue"}  
    * __List the Advantages introduced by unfolding and the benefits:__{: style="color: blue"}  
    * __Graph and write the equations of Unfolding hidden recurrence:__{: style="color: blue"}    
* __Describe the State of the RNN, its usage, and extreme cases of the usage:__{: style="color: red"}  
* __RNN Architectures:__{: style="color: red"}  
    * __List the three standard architectures of RNNs:__{: style="color: blue"}  
        * __Graph:__{: style="color: blue"}  
        * __Architecture:__{: style="color: blue"}  
        * __Equations:__{: style="color: blue"}  
        * __Total Loss:__{: style="color: blue"}  
        * __Complexity:__{: style="color: blue"}  
        * __Properties:__{: style="color: blue"}  
* __Teacher Forcing:__{: style="color: red"}  
    * __Definition:__{: style="color: blue"}  
    * __Application:__{: style="color: blue"}  
    * __Disadvantages:__{: style="color: blue"}  
    * __Possible Solutions for Mitigation:__{: style="color: blue"}  


***

# Optimization
* __Define the *sigmoid* function and some of its properties:__{: style="color: red"}  
* __Backpropagation:__{: style="color: red"}  
    * __Definition:__{: style="color: blue"}  
    * __Derive Gradient Descent Update:__{: style="color: blue"}  
    * __Explain the difference kinds of gradient-descent optimization procedures:__{: style="color: blue"}  
    * __List the different optimizers and their properties:__{: style="color: blue"}  
* __Error-Measures:__{: style="color: red"}  
    * __Define what an error measure is:__{: style="color: blue"}  
    * __List the 5 most common error measures and where they are used:__{: style="color: blue"}  
    * __Specific Questions:__{: style="color: blue"}  
        <button>Show Questions</button>{: .showText value="show" onclick="showText_withParent_PopHide(event);"}
        * __Derive MSE carefully:__{: style="color: blue"}  
        * __Derive the Binary Cross-Entropy Loss function:__{: style="color: blue"}  
        * __Explain the difference between Cross-Entropy and MSE and which is better (for what task)?__{: style="color: blue"}  
        * __Describe the properties of the Hinge loss and why it is used?__{: style="color: blue"}  
        {: hidden=""}  
* __Show that the weight vector of a linear signal is orthogonal to the decision boundary?__{: style="color: red"}  
* __What does it mean for a function to be *well-behaved* from an optimization pov?__{: style="color: red"}  
* __Write $$\|\mathrm{Xw}-\mathrm{y}\|^{2}$$ as a summation__{: style="color: red"}  
* __Compute:__{: style="color: red"}  
    * __$$\dfrac{\partial}{\partial y}\vert{x-y}\vert=$$__{: style="color: blue"}  

***

# ML Theory
* __Explain intuitively why Deep Learning works?__{: style="color: red"}  
* __List the different types of Learning Tasks and their definitions:__{: style="color: red"}  
[answer](/concepts_#bodyContents64)  
* __Describe the relationship between supervised and unsupervised learning?__{: style="color: red"}  
[answer](/concepts_#bodyContents64)  
* __Describe the differences between Discriminative and Generative Models?__{: style="color: red"}  
* __Describe the curse of dimensionality and its effects on problem solving:__{: style="color: red"}  
* __Describe how to initialize a NN and any concerns w/ reasons:__{: style="color: red"}  
* __Describe the difference between Learning and Optimization in ML:__{: style="color: red"}  
* __List the 12 Standard Tasks in ML:__{: style="color: red"}  


***

# Statistical Learning Theory
* __Define Statistical Learning Theory:__{: style="color: red"}  
* __What assumptions are made by the theory?__{: style="color: red"}  
* __Give the Formal Definition of SLT:__{: style="color: red"}  
    <button>Show</button>{: .showText value="show"
     onclick="showText_withParent_PopHide(event);"}
    * __The Definitions:__{: style="color: blue"}  
    * __The Assumptions:__{: style="color: blue"}  
    * __The Inference Problem:__{: style="color: blue"}  
    * __The Expected Risk:__{: style="color: blue"}  
    * __The Target Function:__{: style="color: blue"}  
    * __The Empirical Risk:__{: style="color: blue"}  
    {: hidden=""}
* __Define Empirical Risk Minimization:__{: style="color: red"}  
* __What is the Complexity of ERM?__{: style="color: red"}  
    <button>Show</button>{: .showText value="show"
     onclick="showText_withParent_PopHide(event);"}
    * __How do you Cope with the Complexity?__{: style="color: blue"}  
    {: hidden=""}
* __Definitions:__{: style="color: red"}  
    * __Generalization:__{: style="color: blue"}  
    * __Generalization Error:__{: style="color: blue"}  
    * __Generalization Gap:__{: style="color: blue"}  
    * __Data-Generating Distribution:__{: style="color: blue"}  
* __Describe the difference between Learning and Optimization in ML:__{: style="color: red"}  
* __Why do we need the probabilistic framework?__{: style="color: red"}  
* __Why specifically do we need the assumption of a joint probability distribution $$P(x,y)$$ over the training data:__{: style="color: red"}  
    Note that the assumption of a joint probability distribution allows us to model uncertainty in predictions (e.g. from noise in data) because $${\displaystyle y}$$ is not a deterministic function of $${\displaystyle x}$$, but rather a random variable with conditional distribution $${\displaystyle P(y|x)}$$ for a fixed $${\displaystyle x}$$.  
* __Give the Formal Definition of SLT:__{: style="color: red"}  
* __What is the *Approximation-Generalization Tradeoff*:__{: style="color: red"}  
* __What are the factors determining how well an ML-algo will perform?__{: style="color: red"}  
* __Define the following and their usage/application & how they relate to each other:__{: style="color: red"}  
    * __Underfitting:__{: style="color: blue"}  
    * __Overfitting:__{: style="color: blue"}  
    * __Capacity:__{: style="color: blue"}  
        * Models with __Low-Capacity:__{: style="color: blue"}  
        * Models with __High-Capacity:__{: style="color: blue"}  
    * __VC-Dimension:__{: style="color: red"}  
        * __What does it measure?__{: style="color: blue"}  
    * __Graph the relation between Error, and Capacity in the ctxt of (Underfitting, Overfitting, Training Error, Generalization Err, and Generalization Gap):__{: style="color: blue"}  
* __What is the most important result in SLT that show that learning is feasible?__{: style="color: red"}  




***

# Bias-Variance Decomposition Theory
* __What is the Bias-Variance Decomposition Theory:__{: style="color: red"}  
* __What are the Assumptions made by the theory?__{: style="color: red"}  
* __What is the Bias-Variance Decomposition:__{: style="color: red"}  
* __Define:__{: style="color: red"}  
    * __Bias:__{: style="color: blue"}  
    * __Variance:__{: style="color: blue"}  
    * __Irreducible Error:__{: style="color: blue"}  
* __What does each of the following measure? Describe it in Words? Give their AKA in statistics?__{: style="color: red"}  
    * __Bias:__{: style="color: blue"}  
    * __Variance:__{: style="color: blue"}  
    * __Irreducible Error:__{: style="color: blue"}  
* __Give the Formal Definition of the Decomposition (Formula):__{: style="color: red"}  
    * __What is the Expectation over?__{: style="color: blue"}  
* __Define the *Bias-Variance Tradeoff*:__{: style="color: red"}  
    * __Effects of Bias:__{: style="color: blue"}  
    * __Effects of Variance:__{: style="color: blue"}  
    * __Draw the Graph of the Tradeoff:__{: style="color: blue"}  
* __Derive the Bias-Variance Decomposition with explanations:__{: style="color: red"}  
* __What are the key Takeaways from the Tradeoff?__{: style="color: red"}  
* __What are the most common way to negotiate the Tradeoff?__{: style="color: red"}  
    * Cross-Validation
    * MSE of the Estimates


***

# Activation Functions
* __List the different activation functions used in ML?__{: style="color: red"}  
    * __Names, Definitions, Properties, Applications, pros/cons__{: style="color: blue"}  
* <button>Show Questions</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    * __Tanh VS sigmoid for activation?__{: style="color: blue"}  
    * __ReLU:__{: style="color: blue"}  
        * __What makes it superior/advantageous?__{: style="color: blue"}  
        * __What problems does it have?__{: style="color: blue"}  
            * __What solution do we have to mitigate the problem?__{: style="color: blue"}  
    * __Compute the derivatives of all activation functions:__{: style="color: blue"}  
    * __Graph all activation functions and their derivatives:__{: style="color: blue"}  
    {: hidden=""}

***

# Kernels
* __Define "Local Kernel" and give an analogy to describe it:__{: style="color: red"}  
* __Write the following kernels:__{: style="color: red"}  
    * __Polynomial Kernel of degree, up to, $$d$$:__{: style="color: blue"}  
    * __Gaussian Kernel:__{: style="color: blue"}  
    * __Sigmoid Kernel:__{: style="color: blue"}  
    * __Polynomial Kernel of degree, exactly, $$d$$:__{: style="color: blue"}  
    


***

# Math
* __What is a metric?__{: style="color: red"}  
[Metric](http://localhost:8889/concepts_#bodyContents31)

* __Describe Binary Relations and their Properties?__{: style="color: red"}  
[answer](/concepts_#bodyContents32)


* __Formulas:__{: style="color: red"}  
    * __Set theory:__{: style="color: blue"}  
        * __Number of subsets of a set of $$N$$ elements:__{: style="color: blue"}  
        * __Number of pairs $$(a,b)$$ of a set of N elements:__{: style="color: blue"}  
    * __Binomial Theorem:__{: style="color: blue"}  
    * __Binomial Coefficient:__{: style="color: blue"}  
    * __Expansion of $$x^n - y^n = $$__{: style="color: blue"}  
    * __Number of ways to partition $$N$$ data points into $$k$$ clusters:__{: style="color: blue"}  
    * __$$\log_x(y) =$$__{: style="color: blue"}  
    * __The length of a vector $$\mathbf{x}$$  along a direction (projection):__{: style="color: blue"}  
        1. Along a unit-length vector $$\hat{\mathbf{w}}$$: 
        2. Along an unnormalized vector $$\mathbf{w}$$: 
    * __$$\sum_{i=1}^{n} 2^{i}=$$__{: style="color: blue"}  

* __List 6 proof methods:__{: style="color: red"}  
[answer](/concepts_#bodyContents34)

* __Something__{: style="color: red"}  

***

# Statistics
* __ROC curve:__{: style="color: red"}  
    * __Definition:__{: style="color: blue"}  
    * __Purpose:__{: style="color: blue"}  
    * __How to identify a good classifier:__{: style="color: blue"}  
    * __How to identify a bad classifier:__{: style="color: blue"}  
    [answer](http://localhost:8889/concepts_#bodyContents41)
* __AUC - AUROC:__{: style="color: red"}  
    * __Definition:__{: style="color: blue"}  
    * __Range:__{: style="color: blue"}  
    [answer](http://localhost:8889/concepts_#bodyContents42)
* __Define Statistical Efficiency (of an estimator)?__{: style="color: red"}  
[answer](http://localhost:8889/concepts_#bodyContents43)
* __Whats the difference between *Errors* and *Residuals*:__{: style="color: red"}  
[answer](http://localhost:8889/concepts_#bodyContents44)
    * __Compute the statistical errors and residuals of the univariate, normal distribution defined as $$X_{1}, \ldots, X_{n} \sim N\left(\mu, \sigma^{2}\right)$$:__{: style="color: blue"}  
* __Clearly Define MLE and derive the final formula:__{: style="color: red"}  
    * __What is the intuition behind using MLE?__{: style="color: blue"}  
    * __What kind of problem is MLE and how to solve for it?__{: style="color: blue"}  
    * __Explain clearly why we maximize the natural log of the likelihood__{: style="color: blue"}  

* __What is a biased estimator?__{: style="color: red"}  
    * __Why would we prefer biased estimators in some cases?__{: style="color: blue"}  
* __What is the difference between "Probability" and "Likelihood":__{: style="color: red"}  
* __Estimators:__{: style="color: red"}  
    * __Define:__{: style="color: blue"}  
    * __Formula:__{: style="color: blue"}  
    * __Whats a good estimator?__{: style="color: blue"}  
    * __What are the Assumptions made regarding the estimated parameter:__{: style="color: blue"}  
* __What is Function Estimation:__{: style="color: red"}  
    * __Whats the relation between the Function Estimator $$\hat{f}$$ and Point Estimator:__{: style="color: blue"}  
* __topic:__{: style="color: red"}  


***

# Text-Classification \| Classical
* __List some Classification Methods:__{: style="color: red"}  
* __List some Applications of Txt Classification:__{: style="color: red"}  

***

# NLP
* __List some problems in NLP:__{: style="color: red"}  
* __List the Solved Problems in NLP:__{: style="color: red"}  
* __List the "within reach" problems in NLP:__{: style="color: red"}  
* __List the Open Problems in NLP:__{: style="color: red"}  
* __Why is NLP hard? List Issues:__{: style="color: red"}  
* __Define:__{: style="color: red"}  
    * __Morphemes:__{: style="color: blue"}  

***

# Language Modeling
* __What is a Language Model?__{: style="color: red"}  
* __List some Applications of LMs:__{: style="color: red"}  
* __Traditional LMs:__{: style="color: red"}  
    * __How are they setup?__{: style="color: blue"}  
    * __What do they depend on?__{: style="color: blue"}  
    * __What is the Goal of the LM task? (in the ctxt of the problem setup)__{: style="color: blue"}  
    * __What assumptions are made by the problem setup? Why?__{: style="color: blue"}  
    * __What are the MLE Estimates for probabilities of the following:__{: style="color: blue"}  
        * __Bi-Grams:__{: style="color: blue"}  
            <p>$$p(w_2\vert w_1) = $$</p>  
        * __Tri-Grams:__{: style="color: blue"}  
            <p>$$p(w_3\vert w_1, w_2) = $$</p>  
    * __What are the issues w/ Traditional Approaches?__{: style="color: red"}  
* __What+How can we setup some NLP tasks as LM tasks:__{: style="color: red"}  
* __How does the LM task relate to Reasoning/AGI:__{: style="color: red"}  
* __Evaluating LM models:__{: style="color: red"}  
    * __List the Loss Functions (+formula) used to evaluate LM models? Motivate each:__{: style="color: blue"}  
    * __Which application of LM modeling does each loss work best for?__{: style="color: blue"}  
    <button>Show Questions</button>{: .showText value="show" onclick="showText_withParent_PopHide(event);"}
    * __Why Cross-Entropy:__{: style="color: blue"}  
    * __Which setting it used for?__{: style="color: blue"}  
    * __Why Perplexity:__{: style="color: blue"}  
    * __Which setting used for?__{: style="color: blue"}  
    * __If no surprise, what is the perplexity?__{: style="color: blue"}  
    * __How does having a good LM relate to Information Theory?__{: style="color: blue"}  
    {: hidden=""}
* __LM DATA:__{: style="color: red"}  
    * __How does the fact that LM is a time-series prediction problem affect the way we need to train/test:__{: style="color: blue"}  
    * __How should we choose a subset of articles for testing:__{: style="color: blue"}  
* __List three approaches to Parametrizing LMs:__{: style="color: red"}  
    <button>Show Questions</button>{: .showText value="show" onclick="showText_withParent_PopHide(event);"}
    * __Describe "Count-Based N-gram Models":__{: style="color: blue"}  
    * __What distributions do they capture?:__{: style="color: blue"}  
    * __Describe "Neural N-gram Models":__{: style="color: blue"}  
    * __What do they replace the captured distribution with?__{: style="color: blue"}  
    * __What are they better at capturing:__{: style="color: blue"}  
    * __Describe "RNNs":__{: style="color: blue"}  
    * __What do they replace/capture?__{: style="color: blue"}  
    * __How do they capture it?__{: style="color: blue"}  
    * __What are they best at capturing:__{: style="color: blue"}  
    {: hidden=""}
* __What's the main issue in LM modeling?__{: style="color: red"}  
    <button>Show Questions</button>{: .showText value="show" onclick="showText_withParent_PopHide(event);"}
    * __How do N-gram models capture/approximate the history?:__{: style="color: blue"}  
    * __How do RNNs models capture/approximate the history?:__{: style="color: blue"}  
    {: hidden=""}
    * __The Bias-Variance Tradeoff of the following:__{: style="color: blue"}  
        * __N-Gram Models:__{: style="color: blue"}  
        * __RNNs:__{: style="color: blue"}  
        * __An Estimate s.t. it predicts the probability of a sentence by how many times it has seen it before:__{: style="color: blue"}  
            * __What happens in the limit of infinite data?__{: style="color: blue"}  
* __What are the advantages of sub-word level LMs:__{: style="color: red"}  
* __What are the disadvantages of sub-word level LMs:__{: style="color: red"}  
* __What is a "Conditional LM"?__{: style="color: red"}  
* __Write the decomposition of the probability for the Conditional LM:__{: style="color: red"}  
* __Describe the Computational Bottleneck for Language Models:__{: style="color: red"}  
* __Describe/List some solutions to the Bottleneck:__{: style="color: red"}  
* __Complexity Comparison of the different solutions:__{: style="color: red"}  
    <button>Show Questions</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](/main_files/qs/1.png){: width="100%" hidden=""}   

***

# Regularization
