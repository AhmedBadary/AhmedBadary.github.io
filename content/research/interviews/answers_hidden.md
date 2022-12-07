---
layout: NotesPage
title: Answers to Prep Questions (Learning)
permalink: /work_files/research/answers_hidden
prevLink: /work_files/research.html
---

# Statistics
<button>Statistics</button>{: .showText value="show" onclick="showText_withParent_PopHide(event);"}
1. __ROC curve:__{: style="color: red"}  
    1. __Definition:__{: style="color: blue"}  
        A __receiver operating characteristic curve__, or __ROC curve__, is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.  
    1. __Purpose:__{: style="color: blue"}  
        A way to quantify how good a **binary classifier** separates two classes.  
    1. __How do you create the plot?__{: style="color: blue"}  
        The ROC curve is created by plotting the __true positive rate (TPR)__ against the __false positive rate (FPR)__ at various threshold settings.  
        \- $$\mathrm{TPR}=\frac{\mathrm{TP}}{\mathrm{P}}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}$$  
        \- $$\mathrm{FPR}=\frac{\mathrm{FP}}{\mathrm{N}}=\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}$$  
    1. __How to identify a good classifier:__{: style="color: blue"}  
        A Good classifier has a ROC curve that is near the top-left diagonal (hugging it).  
    1. __How to identify a bad classifier:__{: style="color: blue"}  
        A Bad Classifier has a ROC curve that is close to the diagonal line.  
    1. __What is its application in tuning the model?__{: style="color: blue"}  
        It allows you to set the **classification threshold**:  
        1. You can minimize False-positive rate or maximize the True-Positive Rate  
1. __AUC - AUROC:__{: style="color: red"}  
    1. __Definition:__{: style="color: blue"}  
        When using normalized units, the area under the curve (often referred to as simply the AUC) is equal to the probability that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one (assuming 'positive' ranks higher than 'negative').  
    1. __Range:__{: style="color: blue"}  
        Range $$ = 0.5 - 1.0$$, from poor to perfect, with an uninformative classifier yielding $$0.5$$    
    1. __What does it measure:__{: style="color: blue"}  
        It is a measure of aggregated classification performance.  
    1. __Usage in ML:__{: style="color: blue"}  
        For model comparison.  
1. __Define Statistical Efficiency (of an estimator)?__{: style="color: red"}  
    Essentially, a more efficient estimator, experiment, or test needs fewer observations than a less efficient one to achieve a given performance.  
    Efficiencies are often defined using the _variance_ or _mean square error_ as the measure of desirability.  
    An efficient estimator is also the minimum variance unbiased estimator (MVUE).  

    1. An Efficient Estimator has lower variance than an inefficient one  
    1. The use of an inefficient estimator gives results equivalent to those obtainable from a subset of data; and is therefor, wasteful of data  
1. __Whats the difference between *Errors* and *Residuals*:__{: style="color: red"}  
    The __Error__ of an observed value is the deviation of the observed value from the (unobservable) **_true_** value of a quantity of interest.  

    The __Residual__ of an observed value is the difference between the observed value and the *__estimated__* value of the quantity of interest.  
  
    1. __Compute the statistical errors and residuals of the univariate, normal distribution defined as $$X_{1}, \ldots, X_{n} \sim N\left(\mu, \sigma^{2}\right)$$:__{: style="color: blue"}  
        1. __Statistical Errors__:  
            <p>$$e_{i}=X_{i}-\mu$$</p>  
        1. __Residuals__:  
            <p>$$r_{i}=X_{i}-\overline {X}$$</p>  
        1. [**Example in Univariate Distributions**](https://en.wikipedia.org/wiki/Errors_and_residuals#In_univariate_distributions){: value="show" onclick="iframePopA(event)"}
        <a href="https://en.wikipedia.org/wiki/Errors_and_residuals#In_univariate_distributions"></a>
            <div markdown="1"> </div>    
1. __What is a biased estimator?__{: style="color: red"}  
    We define the __Bias__ of an estimator as:  
    <p>$$ \operatorname{bias}\left(\hat{\boldsymbol{\theta}}_{m}\right)=\mathbb{E}\left(\hat{\boldsymbol{\theta}}_{m}\right)-\boldsymbol{\theta} $$</p>  
    A __Biased Estimator__ is an estimator $$\hat{\boldsymbol{\theta}}_ {m}$$ such that:  
    <p>$$ \operatorname{bias}\left(\hat{\boldsymbol{\theta}}_ {m}\right) \geq 0$$</p>  
    1. __Why would we prefer biased estimators in some cases?__{: style="color: blue"}  
        Mainly, due to the *__Bias-Variance Decomposition__*. The __MSE__ takes into account both the _bias_ and the _variance_ and sometimes the biased estimator might have a lower variance than the unbiased one, which results in a total _decrease_ in the MSE.  
1. __What is the difference between "Probability" and "Likelihood":__{: style="color: red"}  
    __Probabilities__ are the areas under a fixed distribution  
    $$pr($$data$$|$$distribution$$)$$  
    i.e. probability of some _data_ (left hand side) given a distribution (described by the right hand side)  
    __Likelihoods__ are the y-axis values for fixed data points with distributions that can be moved..  
    $$L($$distribution$$|$$observation/data$$)$$  
    It is the likelihood of the parameter $$\theta$$ for the data $$\mathcal{D}$$.  
    > Likelihood is, basically, a specific probability that can only be calculated after the fact (of observing some outcomes). It is not normalized to $$1$$ (it is __not__ a probability). It is just a way to quantify how likely a set of observation is to occur given some distribution with some parameters; then you can manipulate the parameters to make the realization of the data more _"likely"_ (it is precisely meant for that purpose of estimating the parameters); it is a _function_ of the __parameters__.  
    Probability, on the other hand, is absolute for all possible outcomes. It is a function of the __Data__.  
1. __Estimators:__{: style="color: red"}  
    1. __Define:__{: style="color: blue"}  
        A __Point Estimator__ or __statistic__ is any function of the data.  
    1. __Formula:__{: style="color: blue"}  
        <p>$$\hat{\boldsymbol{\theta}}_{m}=g\left(\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(m)}\right)$$</p>  
    1. __Whats a good estimator?__{: style="color: blue"}  
        A good estimator is a function whose output is close to the true underlying $$ \theta $$ that generated the training data.  
    1. __What are the Assumptions made regarding the estimated parameter:__{: style="color: blue"}  
        We assume that the true $$\boldsymbol{\theta}$$ is fixed, and that $$\hat{\boldsymbol{\theta}}$$ is a function of the data, which is drawn from a random process, making $$\hat{\boldsymbol{\theta}}$$ a __random variable__.  
1. __What is Function Estimation:__{: style="color: red"}  
    __Function Estimation/Approximation__ refers to estimation of the relationship between _input_ and _target data_.  
    I.E. We are trying to predict a variable $$y$$ given an input vector $$x$$, and we assume that there is a function $$f(x)$$ that describes the approximate relationship between $$y$$ and $$x$$.  
    If we assume that: $$y = f(x) + \epsilon$$, where $$\epsilon$$ is the part of $$y$$ that is not predictable from $$x$$; then we are interested in approximating $$f$$ with a model or estimate $$ \hat{f} $$.  
    1. __Whats the relation between the Function Estimator $$\hat{f}$$ and Point Estimator:__{: style="color: blue"}  
        Function estimation is really just the same as estimating a parameter $$\boldsymbol{\theta}$$; the function estimator $$ \hat{f} $$ is simply a point estimator in function space.  
1. __Define "marginal likelihood" (wrt naive bayes):__{: style="color: red"}  
    Marginal likelihood is, the probability that the word ‘FREE’ is used in any message (not given any other condition?).   
{: hidden=""}
