---
layout: NotesPage
title: Regularization
permalink: /work_files/research/dl/theory/dl_book_regularization
prevLink: /work_files/research/dl/theory.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Regularization Basics and Definitions](#content1)
  {: .TOC1}
  * [Parameter Norm Penalties](#content2)
  {: .TOC2}
  * [Advanced Regularization Techniques](#content3)
  {: .TOC3}
</div>

***
***

[Regularization in FFN](/work_files/research/dl/nlp/dl_book_pt1#bodyContents133)  
[Regularization Concept](/concepts_#bodyContents616)  
[Regularization Ch.7 Summary](https://medium.com/inveterate-learner/deep-learning-book-chapter-7-regularization-for-deep-learning-937ff261875c)  
[How Regularization Reduces Variance from bias-var-decomp](http://cs229.stanford.edu/notes-spring2019/addendum_bias_variance.pdf)  
[Probabilistic Interpretation of Regularization (MAP)](http://bjlkeng.github.io/posts/probabilistic-interpretation-of-regularization)  
[The Math of Regularization](https://www.wikiwand.com/en/Regularization_(mathematics))  


## Regularization Basics and Definitions
{: #content1}

1. **Regularization:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}    
    __Regularization__ can be, loosely, defined as: any modification we make to a learning algorithm that is intended to _reduce_ its _generalization error_ but not its _training error_.  

    Formally, it is a set of techniques that impose certain restrictions on the hypothesis space (by adding information) in order to solve an __ill-posed__ problem or to prevent __overfitting__.[^1]  


2. **Theoretical Justification for Regularization:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}    
    A theoretical justification for regularization is that it attempts to impose Occam's razor on the solution.  
    From a Bayesian point of view, many regularization techniques correspond to imposing certain prior distributions on model parameters.
    <br>

3. **Regularization in Deep Learning:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    In the context of DL, most regularization strategies are based on __regularizing estimators__, which usually works by _trading increased bias for reduced variance_.  

    An effective regularizer is one that makes a profitable trade, reducing variance significantly while not overly increasing the bias.
    <br>

4. **Regularization and Data Domains in DL - A Practical Motivation:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    Most applications of DL are to domains where the true data-generating process is almost certainly outside the model family (hypothesis space). Deep learning algorithms are typically applied to extremely complicated domains such as images, audio sequences and text, for which the true generation process essentially involves simulating the entire universe.  

    Thus, controlling the complexity of the mdoel is not a simple matter of finding the model of the right size, with the right number of parameters; instead, the best fitting model (wrt. generalization error) is a large model that has been regularized appropriately.  


[^1]: Where we (Hadamard) define __Well-Posed Problems__ as having the properties (1) A Solution Exists (2) It is Unique (3) It's behavior changes continuously with the initial conditions.  

***

## Parameter Norm Penalties
{: #content2}

1. **Parameter Norms:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  
    Many regularization approaches are based on limiting the capacity of models by adding a parameter norm penalty $$\Omega(\boldsymbol{\theta})$$ to the objective function $$J$$. We denote the regularized objective function by $$\tilde{J}$$:  
    <p>$$\tilde{J}(\boldsymbol{\theta} ; \boldsymbol{X}, \boldsymbol{y})=J(\boldsymbol{\theta} ; \boldsymbol{X}, \boldsymbol{y})+\alpha \Omega(\boldsymbol{\theta}) \tag{7.1}$$</p>  
    where $$\alpha \in[0, \infty)$$ is a HP that weights the relative contribution of the norml penalty term, $$\Omega$$, relative to the standard objective function $$J$$.  
    * __Effects of $$\alpha$$__:  
        * $$\alpha = 0$$ results in NO regularization
        * Larger values of $$\alpha$$ correspond to MORE regularization

    The __effect of minimizing the regularized objective function__ is that it will *__decrease__*, both, _the original objective $$J$$_ on the training data and some _measure of the size of the parameters $$\boldsymbol{\theta}$$_.  

    Different choices for the parameter norm $$\Omega$$ can result in different solutions being preferred.  


2. **Parameter Penalties and the Bias parameter:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  
    In NN, we usually penalize __only the weights__ of the affine transformation at each layer and we leave the __biases unregularized__.  
    Biases typically require less data than the weights to fit accurately. The reason is that _each weight specifies how TWO variables interact_ so fitting the weights well, requires observing both variable sin a variety of conditions. However, _each bias controls only a single variable_, thus, we dont induce too much _variance_ by leaving the biases unregularized. If anything, regularizing the bias can introduce a significant amount of _underfitting_.  


3. **Note on the $$\alpha$$ parameter for different hidden layers:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}  
    In the context of neural networks, it is sometimes desirable to use a separate penalty with a different $$\alpha$$ coefficient for each layer of the network. Because it can be expensive to search for the correct value of multiple hyperparameters, it is still reasonable to use the same weight decay at all layers just to reduce the size of search space.  


4. **$$L^2$$ Parameter Regularization (Weight Decay):**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}  
    It is a regularization strategy that _drives the weights closer to the origin_[^2] by adding a regularization term:  
    <p>$$\Omega(\mathbf{\theta}) = \frac{1}{2}\|\boldsymbol{w}\|_ {2}^{2}$$</p>  
    to the objective function.  
    
    In statistics, $$L^2$$ regularization is also known as __Ridge Regression__ or __Tikhonov Regularization__.  

    __Analyzing Weight Decay:__{: style="color: red"}  
    <button>Show Analysis</button>{: .showText value="show" onclick="showText_withParent_PopHide(event);"}
    * __What happens in a Single Step__:  
        We can gain some insight into the behavior of weight decay regularization by studying the gradient of the regularized objective function.  
        Take the models objective function:  
        <p>$$\tilde{J}(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})=\frac{\alpha}{2} \boldsymbol{w}^{\top} \boldsymbol{w}+J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y}) \tag{7.2}$$</p>  
        with the corresponding _parameter gradient_:  
        <p>$$\nabla_{\boldsymbol{w}} \tilde{J}(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})=\alpha \boldsymbol{w}+\nabla_{\boldsymbol{w}} J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y}) \tag{7.3}$$</p>  
        The gradient descent update:  
        <p>$$\boldsymbol{w} \leftarrow \boldsymbol{w}-\epsilon\left(\alpha \boldsymbol{w}+\nabla_{\boldsymbol{w}} J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})\right) \tag{7.4}$$</p>  
        Equivalently:  
        <p>$$\boldsymbol{w} \leftarrow(1-\epsilon \alpha) \boldsymbol{w}-\epsilon \nabla_{\boldsymbol{w}} J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y}) \tag{7.5}$$</p>    

        Observe that the addition of the weight decay term has modified the learning rule to __multiplicatively shrink the weight vector by  a constant factor on each step__, just before performing the usual gradient update.  

    * __What happens over the Entire course of training__:  
        We simplify the analysis by making a quadratic (2nd-order Taylor) approximation to the objective function in the neighborhood of the optimal wight-parameter of the unregularized objective $$\mathbf{w}^{\ast} = \arg \min_{\boldsymbol{w}} J(\boldsymbol{w})$$.[^3]  
        The approximation $$\hat{J}$$:  
        <p>$$\hat{J}(\boldsymbol{\theta})=J\left(\boldsymbol{w}^{\ast}\right)+\frac{1}{2}\left(\boldsymbol{w}-\boldsymbol{w}^{\ast}\right)^{\top} \boldsymbol{H}(J(\boldsymbol{w}^{\ast}))\left(\boldsymbol{w}-\boldsymbol{w}^{\ast}\right)  \tag{7.6}$$</p>  
        where $$\boldsymbol{H}$$ is the Hessian matrix of $$J$$ with respect to $$\mathbf{w}$$ evaluated at $$\mathbf{w}^{\ast}$$.  

        __Notice:__  
        * There is no first-order term in this quadratic approximation, because $$\boldsymbol{w}^{\ast}$$  is defined to be a minimum, where the gradient vanishes.  
        * Because $$\boldsymbol{w}^{\ast}$$ is the location of a minimum of $$J$$, we can conclude that $$\boldsymbol{H}$$ is __positive semidefinite__.  

        The __gradient__ of $$\hat{J} + \Omega(\mathbf{\theta})$$:  
        <p>$$\nabla_{\boldsymbol{w}} \hat{J}(\boldsymbol{w})=\boldsymbol{H}(J(\boldsymbol{w}^{\ast}))\left(\tilde{\boldsymbol{w}}-\boldsymbol{w}^{\ast}\right) + \alpha \tilde{\boldsymbol{w}} \tag{7.7}$$</p>  
        And the __minimum__ is achieved at $$\nabla_{\boldsymbol{w}} \hat{J}(\boldsymbol{w}) = 0$$:  
        <p>$$\tilde{\boldsymbol{w}}=(\boldsymbol{H}+\alpha \boldsymbol{I})^{-1} \boldsymbol{H} \boldsymbol{w}^{\ast} \tag{7.10}$$</p>  

        __Effects:__  
        * As $$\alpha$$ approaches $$0$$: the regularized solution $$\tilde{\boldsymbol{w}}$$ approaches $$\boldsymbol{w}^{\ast}$$.  
        * As $$\alpha$$ grows: we apply __spectral decomposition__ to the __real and symmetric__ $$\boldsymbol{H} = \boldsymbol{Q} \boldsymbol{\Lambda} \boldsymbol{Q}^{\top}$$:  
            <p>$$\begin{aligned} \tilde{\boldsymbol{w}} &=\left(\boldsymbol{Q} \mathbf{\Lambda} \boldsymbol{Q}^{\top}+\alpha \boldsymbol{I}\right)^{-1} \boldsymbol{Q} \boldsymbol{\Lambda} \boldsymbol{Q}^{\top} \boldsymbol{w}^{\ast} \\ &=\left[\boldsymbol{Q}(\boldsymbol{\Lambda}+\alpha \boldsymbol{I}) \boldsymbol{Q}^{\top}\right]^{-1} \boldsymbol{Q} \boldsymbol{\Lambda} \boldsymbol{Q}^{\top} \boldsymbol{w}^{\ast} \\ &=\boldsymbol{Q}(\boldsymbol{\Lambda}+\alpha \boldsymbol{I})^{-1} \boldsymbol{\Lambda} \boldsymbol{Q}^{\top} \boldsymbol{w}^{\ast} \end{aligned} \tag{7.13}$$</p>  

        Thus, we see that the effect of weight decay is to rescale $$\boldsymbol{w}^{\ast}$$ along the axes defined by the eigenvector of $$\boldsymbol{H}$$ . Specifically, the component of $$\boldsymbol{w}^{\ast}$$ that is aligned with the $$i$$-th eigenvector of $$\boldsymbol{H}$$  is rescaled by a factor of $$\frac{\lambda_{i}}{\lambda_{i}+\alpha}$$.  

        ![img](/main_files/dl_book/regularization/1.png){: width="100%"}   

        __Summary:__  

        | __Condition__|__Effect of Regularization__ |   
        | $$\lambda_{i}>>\alpha$$ | Not much |  
        | $$\lambda_{i}<<\alpha$$ | The weight value almost shrunk to $$0$$ |  

    * __Applying $$L^2$$ regularization to *Linear Regression* :__  
        * <button>Application to Linear Regression</button>{: .showText value="show" onclick="showTextPopHide(event);"}
            ![img](/main_files/dl_book/regularization/2.png){: width="100%" hidden=""}   
    {: hidden=""}  
    <br>

    __$$L^2$$ Regularization Derivation:__{: style="color: red"}  
    $$L^2$$ regularization is equivalent to __MAP Bayesian inference with a Gaussian prior on the weights__.  

    __The MAP Estimate:__  
    <button>Show MAP Estimate Derivation</button>{: .showText value="show" onclick="showText_withParent_PopHide(event);"}
    <p hidden="">$$\begin{aligned} \hat{\theta}_ {\mathrm{MAP}} &=\arg \max_{\theta} P(\theta | y) \\ &=\arg \max_{\theta} \frac{P(y | \theta) P(\theta)}{P(y)} \\ &=\arg \max_{\theta} P(y | \theta) P(\theta) \\ &=\arg \max_{\theta} \log (P(y | \theta) P(\theta)) \\ &=\arg \max_{\theta} \log P(y | \theta)+\log P(\theta) \end{aligned}$$</p>  
    
    We place a __Gaussian Prior__ on the weights, with __zero mean__ and __equal variance $$\tau^2$$__:  
    <p>$$\begin{aligned} \hat{\theta}_ {\mathrm{MAP}} &=\arg \max_{\theta} \log P(y | \theta)+\log P(\theta) \\ &=\arg \max _{\boldsymbol{w}}\left[\log \prod_{i=1}^{n} \dfrac{1}{\sigma \sqrt{2 \pi}} e^{-\dfrac{\left(y_{i}-\boldsymbol{w}^T\boldsymbol{x}_i\right)^{2}}{2 \sigma^{2}}}+\log \prod_{j=0}^{p} \dfrac{1}{\tau \sqrt{2 \pi}} e^{-\dfrac{w_{j}^{2}}{2 \tau^{2}}} \right] \\ &=\arg \max _{\boldsymbol{w}} \left[-\sum_{i=1}^{n} \dfrac{\left(y_{i}-\boldsymbol{w}^T\boldsymbol{x}_i\right)^{2}}{2 \sigma^{2}}-\sum_{j=0}^{p} \dfrac{w_{j}^{2}}{2 \tau^{2}}\right] \\ &=\arg \min_{\boldsymbol{w}} \dfrac{1}{2 \sigma^{2}}\left[\sum_{i=1}^{n}\left(y_{i}-\boldsymbol{w}^T\boldsymbol{x}_i\right)^{2}+\dfrac{\sigma^{2}}{\tau^{2}} \sum_{j=0}^{p} w_{j}^{2}\right] \\ &=\arg \min_{\boldsymbol{w}} \left[\sum_{i=1}^{n}\left(y_{i}-\boldsymbol{w}^T\boldsymbol{x}_i\right)^{2}+\lambda \sum_{j=0}^{p} w_{j}^{2}\right] \\ &= \arg \min_{\boldsymbol{w}} \left[ \|XW - \boldsymbol{y}\|^2 + \lambda {\|\boldsymbol{w}\|_ 2}^2\right]\end{aligned}$$</p>  
    <button>Different Notation</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](/main_files/dl_book/regularization/4.png){: width="100%" hidden=""}   
    <br>

    __Properties:__{: style="color: red"}  
    * Notice that L2-regularization has a rotational invariance. This actually makes it more sensitive to irrelevant features.  [Ref](https://www.cs.ubc.ca/~schmidtm/Courses/540-W18/L6.pdf)  
    * Adding L2-regularization to a convex function gives a strongly-convex function. So L2-regularization can make gradient descent converge much faster.  (^ same ref)      
    <br>

5. **$$L^1$$ Regularization:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents25}  
    $$L^1$$ Regularization is another way to regulate the model by _penalizing the size of its parameters_; the technique adds a regularization term:  
    <p>$$\Omega(\boldsymbol{\theta})=\|\boldsymbol{w}\|_{1}=\sum_{i}\left|w_{i}\right| \tag{7.18}$$</p>  
    which is a sum of absolute values of the individual parameters.  

    The regularized objective function is given by:  
    <p>$$\tilde{J}(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})=\alpha\|\boldsymbol{w}\|_ {1}+J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y}) \tag{7.19}$$</p>  
    with the corresponding (sub) gradient:  
    <p>$$\nabla_{\boldsymbol{w}} \tilde{J}(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})=\alpha \operatorname{sign}(\boldsymbol{w})+\nabla_{\boldsymbol{w}} J(\boldsymbol{X}, \boldsymbol{y} ; \boldsymbol{w}) \tag{7.20}$$</p>  

    Notice that the regularization contribution to the gradient, __no longer scales linearly with each $$w_i$$__; instead it is a __constant factor with a sign = $$\text{sign}(w_i)$$__.  

    \[Analysis\]  

    __Sparsity of the $$L^1$$ regularization:__  
    In comparison to $$L^2$$, $$L^1$$ regularization results in a solution that is more __sparse__.  
    The _sparsity property_ has been used extensively as a __feature selection__ mechanism.  
    * __LASSO__: The Least Absolute Shrinkage and Selection Operator integrates an $$L^1$$ penalty with a _linear model_ and a _least-squares cost function_.  
        The $$L^1$$ penalty causes a subset of the weights to become __zero__, suggesting that the corresponding features may safely be discarded.  

    __$$L^1$$ Regularization Derivation:__{: style="color: red"}  
    $$L^1$$ regularization is equivalent to (the log-prior term in) __MAP Bayesian inference with an isotropic Laplace distribution prior on the weights__:  
    <p>$$\log p(\boldsymbol{w})=\sum_{i} \log \operatorname{Laplace}\left(w_{i} ; 0, \frac{1}{\alpha}\right)=-\alpha\|\boldsymbol{w}\|_ {1}+n \log \alpha-n \log 2 \tag{7.24}$$</p>  
    note that we can ignore the terms $$\log \alpha-\log 2$$ because they do not depend on $$\boldsymbol{w}$$.      
    <button>Derivation</button>{: .showText value="show" onclick="showText_withParent_PopHide(event);"}
    <p hidden="">$$\begin{aligned} \hat{\theta}_ {\mathrm{MAP}} &=\arg \max_{\theta} \log P(y | \theta)+\log P(\theta) \\  &=\arg \max _{\boldsymbol{w}}\left[\log \prod_{i=1}^{n} \dfrac{1}{\sigma \sqrt{2 \pi}} e^{-\dfrac{\left(y_{i}-\boldsymbol{w}^T\boldsymbol{x}_i\right)^{2}}{2 \sigma^{2}}}+\log \prod_{j=0}^{p} \dfrac{1}{2 b} e^{-\dfrac{\left|\theta_{j}\right|}{2 b}} \right] \\    &=\arg \max _{\boldsymbol{w}} \left[-\sum_{i=1}^{n} \dfrac{\left(y_{i}-\boldsymbol{w}^T\boldsymbol{x}_i\right)^{2}}{2 \sigma^{2}}-\sum_{j=0}^{p} \dfrac{\left|w_{j}\right|}{2 b}\right] \\    &=\arg \min_{\boldsymbol{w}} \dfrac{1}{2 \sigma^{2}}\left[\sum_{i=1}^{n}\left(y_{i}-\boldsymbol{w}^T\boldsymbol{x}_i\right)^{2}+\dfrac{\sigma^{2}}{b} \sum_{j=0}^{p}\left|w_{j}\right|\right] \\    &=\arg \min_{\boldsymbol{w}} \left[\sum_{i=1}^{n}\left(y_{i}-\boldsymbol{w}^T\boldsymbol{x}_i\right)^{2}+\lambda \sum_{j=0}^{p}\left|w_{j}\right|\right] \\    &= \arg \min_{\boldsymbol{w}} \left[ \|XW - \boldsymbol{y}\|^2 + \lambda \|\boldsymbol{w}\|_ 1\right]\end{aligned}$$</p>
    <br>


6. **$$L^1$$ VS $$L^2$$ Regularization:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents26}  
    
    
    * __Feature Correlation and Sparsity__:  
        * __Identical features__:   
            * $$L^1$$ regularization spreads weight arbitrarily (all weights same sign) 
            * $$L^2$$ regularization spreads weight evenly 
        * __Linearly related features__:   
            * $$L^1$$ regularization chooses variable with larger scale, $$0$$ weight to others  
            * $$L^2$$ prefers variables with larger scale — spreads weight proportional to scale  
        > [Reference](https://www.youtube.com/watch?v=KIoz_aa1ed4&list=PLnZuxOufsXnvftwTB1HL6mel1V32w0ThI&index=7)  

    
    __Interpreting Sparsity with an Example:__{: style="color: red"}  
    Let's imagine we are estimating two coefficients in a regression. In $$L^2$$ regularization, the solution $$\boldsymbol{w} =(0,1)$$ has the same weight as $$\boldsymbol{w}=(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}})$$  so they are both treated equally. In $$L^1$$ regularization, the same two solutions favor the sparse one:  
    <p>$$\|(1,0)\|_{1}=1<\left\|\left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)\right\|_{1}=\sqrt{2}$$</p>  
    So $$L^2$$ regularization doesn't have any specific built in mechanisms to favor zeroed out coefficients, while $$L^1$$ regularization actually favors these sparser solutions.  
    > [Extensive Discussions on Sparsity (Quora)](https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization-How-does-it-solve-the-problem-of-overfitting-Which-regularizer-to-use-and-when)  

    <br>


__Notes:__{: style="color: red"}  
* __Elastic Net Regularization:__  
    <p>$$\Omega = \lambda\left(\alpha\|w\|_{1}+(1-\alpha)\|w\|_{2}^{2}\right), \alpha \in[0,1]$$</p>  
    * Combines both $$L^1$$ and $$L^2$$  
    * Used to __produce sparse solutions__, but to avoid the problem of $$L^1$$ solutions being sometimes __Non-Unique__  
        * The problem mainly arises with __correlated features__  



[^2]: More generally, we could regularize the parameters to be near any specific point in space and, surprisingly, still get a regularization effect, but better results will be obtained for a value closer to the true one, with zero being a default value that makes sense when we do not know if the correct value should be positive or negative.  

[^3]: The approximation is perfect if the objective function is truly quadratic, as in the case of __linear regression w/ MSE__.  

***

## Advanced Regularization Techniques
{: #content3}

1. **Regularization and Under-Constrained Problems:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents31}  
    In some cases, regularization is necessary for machine learning problems to be properly define.  

    Many linear models (e.g. Linear Regression, PCA) depend on __inverting $$\boldsymbol{X}^T\boldsymbol{X}$$__. This is not possible if $$\boldsymbol{X}^T\boldsymbol{X}$$ is singular. In this case, many forms of regularization correspond to solving inverting $$\boldsymbol{X}^{\top} \boldsymbol{X}+\alpha \boldsymbol{I}$$ instead. This regularized matrix is __guaranteed to be invertible__.  
    * $$\boldsymbol{X}^T\boldsymbol{X}$$ can be singular if:  
        * The data-generating function truly has no variance in some direction.  
        * No Variance is _observed_ in some direction because there are fewer examples (rows of $$\boldsymbol{X}$$) than input features (columns).  

    Models with no closed-form solution can, also, be _underdetermined_:  
    Take __logistic regression on a linearly separable dataset__, if a weight vector $$\boldsymbol{w}$$ is able to achieve perfect classification, then so does $$2\boldsymbol{w}$$ but with even __higher likelihood__. Thus, an iterative optimization procedure (sgd) will continually increase the magnitude of $$\boldsymbol{w}$$ and, in theory, will __never halt__.  
    We can use regularization to guarantee the convergence of iterative methods applied to underdetermined problems: e.g. __weight decay__ will cause gradient descent to _quit increasing the magnitude of the weights when the **slope of the likelihood is equal to the weight decay coefficient**_.  

    __Linear Algebra Perspective:__  
    Given that the __Moore-Penrose pseudoinverse__ $$\boldsymbol{X}^{+}$$ of a matrix $$\boldsymbol{X}$$ can solve underdetermined linear equations:  
    <p>$$\boldsymbol{X}^{+}=\lim_{\alpha \searrow 0}\left(\boldsymbol{X}^{\top} \boldsymbol{X}+\alpha \boldsymbol{I}\right)^{-1} \boldsymbol{X}^{\top} \tag{7.29}$$</p>  
    we can now recognize the equation as __performing linear regression with weight-decay__.  
    Specifically, $$7.29$$ is the limit of eq $$7.17$$ as the _regularization coefficient shrinks to zero_.  
    We can thus interpret the pseudoinverse as __stabilizing underdetermined problems using regularization__.  


2. **Dataset Augmentation:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents32}  
    Having more data is the most desirable thing to improving a machine learning model’s performance. In many cases, it is relatively easy to artificially generate data.  
    * __Applications__: for certain problems like __classification__ this approach is readily usable. E.g. for a classification task, we require the model to be _invariant to certain types of transformations_, of which we can generate data by applying them on our current dataset.  
        The most successful application of data-augmentation has been in __object recognition__.  
    * __Non-Applicable__: this approach is not applicable to many problems, especially those that require us to learn the true data-distribution first E.g. Density Estimation.  

    __Noise Injection as Data-Augmentation:__{: style="color: red"}  
    Injecting noise in the _input_ to a NN _(Siestma and Dow, 1991)_ can also be seen as a form of data augmentation.  
    * __Motivation:__  
        * For many classification and (some) regression tasks: the task should be possible to solve even if small random noise is added to the input [(Local Constancy)](/work_files/research/dl/theory/dl_book_pt1#bodyContents32)  
        * Moreover, NNs prove not to be very robust to noise.  

    __Injecting Noise in the Hidden Units:__  
    It can be seen as doing data-augmentation at *__multiple levels of abstraction__*. This approach can be highly effective provided that the magnitude of the noise is carefully tuned _(Poole et al. 2014)_.  
    > __Dropout__ can be seen as a process of constructing new inputs by _multiplying_ by noise.  


3. **Noise Robustness:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents33}  
    We can apply __Noise Injection__ to different components of the model as a way to regularize the model:  
    __Injecting Noise in the Input Layer:__{: style="color: red"}  
    * __Motivation__:  
        We have motivated the injection of noise, to the inputs, as a dataset augmentation strategy.        
    * __Interpretation__:  
        For some models, the addition of noise with infinitesimal variance at the input of the model is equivalent to __imposing a penalty on the norm of the weights__ _(Bishop, 1995a,b)_.  

    __Injecting Noise in the Hidden Layers:__{: style="color: red"}  
    * __Interpretation__:  
        It can be seen as doing __data-augmentation__ at *__multiple levels of abstraction__*.  
    * __Applications__:  
        The most successful application of this type of noise injection is __Dropout__.  
        It can be seen as a process of constructing new inputs by _multiplying_ by noise.  

    __Injecting Noise in the Weight Matrices:__{: style="color: red"}  
    * __Interpretation__:  
        1. It can be interpreted as a stochastic implementation of Bayesian inference over the weights.  
            * __The Bayesian View__:  
                The Bayesian treatment of learning would consider the model weights to be _uncertain and representable via a probability distribution that reflects this uncertainty_. Adding noise to the weights is a practical, stochastic way to reflect this uncertainty.  
        2. It can, also, be interpreted as equivalent a more traditional form of regularization, _encouraging stability of the function to be learned_.  
            * <button>Analysis</button>{: .showText value="show" onclick="showTextPopHide(event);"}
                ![img](/main_files/dl_book/regularization/3.png){: width="100%" hidden=""}   
    * __Applications__:  
        This technique has been used primarily in the context of __recurrent neural networks__ _(Jim et al., 1996; Graves, 2011)_.  

    __Injecting Noise in the Output Layer:__{: style="color: red"}  
    * __Motivation__:  
        * Most datasets have some number of mistakes in the $$y$$ labels. It can be harmful to maximize $$\log p(y | \boldsymbol{x})$$ when $$y$$ is a mistake. One way to prevent this is to explicitly model the noise on the labels.  
        One can assume that for some small constant $$\epsilon$$, the training set label $$y$$ is correct with probability $$1-\epsilon$$.  
            This assumption is easy to incorporate into the cost function analytically, rather than by explicitly drawing noise samples (e.g. __label smoothing__).  
        * MLE with a softmax classifier and hard targets may never converge - the softmax can never predict a probability of exactly $$0$$ or $$1$$, so it will continue to learn larger and larger weights, making more extreme predictions forever.{: #bodyContents33mle}  
    * __Interpretation__:  
        For some models, the addition of noise with infinitesimal variance at the input of the 
    * __Applications__:  
        __Label Smoothing__ regularizes a model based on a softmax with $$k$$ output values by replacing the hard $$0$$ and $$1$$ classification targets with targets of $$\dfrac{\epsilon}{k-1}$$ and $$1-\epsilon$$, respectively.   
        * [__Applied to MLE problem:__](#bodyContents33mle) Label smoothing, compared to weight-decay, has the advantage of preventing the pursuit of hard probabilities without discouraging correct classification.  
        * Application in modern NN: _(Szegedy et al. 2015)_ 


4. **Semi-Supervised Learning:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents34}  
    __Semi-Supervised Learning__ is a class of ML tasks and techniques that makes use of both unlabeled examples from $$P(\mathbf{x})$$ and labeled examples from $$P(\mathbf{x}, \mathbf{y})$$ to estimate $$P(\mathbf{y} | \mathbf{x})$$ or predict $$\mathbf{y}$$ from $$\mathbf{x}$$.  

    In the context of Deep Learning, Semi-Supervised Learning usually refers to _learning a representation $$\boldsymbol{h}=f(\boldsymbol{x})$$_; the goal being to learn a representation such that __examples from the same class have similar representations__.   
    Usually, __Unsupervised Learning__ provides us clues (e.g. clustering) that influence the representation of the data.  
    > __PCA__, as a preprocessing step before applying a classifier, is a long-standing variant of this approach.  

    __Approach:__  
    Instead of separating the supervised and unsupervised criteria, we can instead have a generative model of $$P(\mathbf{x})$$ (or $$P(\mathbf{x}, \mathbf{y})$$) which shares parameters with a discriminative model $$P(\mathbf{y} \vert \mathbf{x})$$.  
    The idea is to share the unsupervised/generative criterion with the supervised criterion to _express a prior belief that the structure of $$P(\mathbf{x})$$ (or $$P(\mathbf{x}, \mathbf{y})$$) is connected to the structure of $$P(\mathbf{y} \vert \mathbf{x})$$_, which is captured by the _shared parameters_.  
    By controlling how much of the generative criterion is included in the total criterion, one can find a better trade-off than with a purely generative or a purely discriminative training criterion _(Lasserre et al., 2006; Larochelle and Bengio, 2008)_.  

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents35}  

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents36}  

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents37}  

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents38}  

