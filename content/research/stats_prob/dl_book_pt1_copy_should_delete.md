---
layout: NotesPage
title: Deep Learning <br > A Complete Overview - PT 1
permalink: /work_files/research/dl/nlp/dl_book_pt1_copy
prevLink: /work_files/research/dl/nlp.html
---


<div markdown="1" class = "TOC">
# Table of Contents

  * [Machine Learning Basics](#content1)
  {: .TOC1}
  * [The Mathematics of Neural Networks](#content2)
  {: .TOC2}
  * [Challenges in Deep Learning](#content3)
  {: .TOC3}
<!--   * [FOURTH](#content4)
  {: .TOC4}
  * [FIFTH](#content5)
  {: .TOC5}
  * [SIXTH](#content6)
  {: .TOC6} -->
</div>

***
***

## Machine Learning Basics
{: #content1}

1. **Introduction and Definitions:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}
    * __Two Approaches to Statistics__:  
        * Frequentest Estimators
        * Bayesian Inference  
    * __The Design Matrix__:  
        A common way for describing a dataset where it is a matrix containing a different example in each row. Each column of the matrix corresponds to a different feature.  
            

2. **Learning Algorithms:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}
    * __Learning__:  
        A computer program is said to learn from *__experience__* $$E$$ with respect to some class of *__tasks__* $$T$$ and *__performance measure__* $$P$$, if its performance at tasks in $$T$$, as measured by $$P$$, improves with experience $$E$$
    * __The Task $$T$$__: 
        * *__Classification__*:  
            A task where the computer program is asked to specify which of $$k$$ categories some input belongs to.  
            To solve this task, the learning algorithm is usually asked to produce a function $$f:\mathbb{R}^n \rightarrow {1, . . . , k}$$.  
            When $$y=f(x)$$, the model assigns an input described by vector $$x$$ to a category identified by numeric code $$y$$.  
            > e.g. Object Recognition
        * *__Classification with Missing Inputs__*:  
            Classification becomes more challenging if the computer program is not guaranteed that every measurement in its input vector will always be provided.  
            To solve this task, rather than providing a single classification function (as in the normal classification case), the learning algorithm must learn a set of functions, each corresponding to classifying $$x$$ with a different subset of its inputs missing.  

            One way to efficiently define such a large set of functions is to learn a probability distribution over all the relevant variables, then solve the classification task by marginalizing out the missing variables.  
            With $$n$$ input variables, we can now obtain all $$2^n$$ different classification functions needed for each possible set of missing inputs, but the computer program needs to learn only a single function describing the joint probability distribution.  
            > e.g. Medical Diagnosis (where some tests weren't conducted for any reason)  
        * *__Regression__*:  
            A computer is asked to predict a numerical value given some input.  
            To solve this task, the learning algorithm is asked to output a function $$f:\mathbb{R}^n \rightarrow R$$  
            > e.g. Object Localization
        * *__Transcription__*:  
            In this type of task, the machine learning system is asked to observe a relatively unstructured representation of some kind of data and transcribe the information into discrete textual form.  
            > e.g. OCR
        * *__Machine Translation__*:  
            In a machine translation task, the input already consists of a sequence of symbols in some language, and the computer program must convert this into a sequence of symbols in another language.  
            > e.g. Google Translate  
        * *__Structured Output__*:  
            Structured output tasks involve any task where the output is a vector (or other data structure containing multiple values) with important relationships between the different elements.  
            This is a broad category and subsumes the transcription and translation tasks described above, as well as many other tasks.  
            These tasks are called structured output tasks because the program must output several values that are all tightly interrelated. For example, the words produced by an image captioning program must form a valid sentence.  
            > e.g. Syntax Parsing, Image Segmentation  
        * *__Anomaly Detection__*:  
            In this type of task, the computer program sifts through a set of events or objects and ﬂags some of them as being unusual or atypical.  
            > e.g. Insider Trading Detection
        * *__Synthesis and Sampling__*:  
            In this type of task, the machine learning algorithm is asked to generate new examples that are similar to those in the training data.  
            This is a kind of structured output task, but with the added qualification that there is no single correct output for each input, and we explicitly desire a large amount of variation in the output, in order for the output to seem more natural and realistic.  
            > e.g. Image Synthesis, Speech Synthesis
        * *__Imputation__*:  
            In this type of task, the machine learning algorithm is given a new example $$x \in \mathbb{R}^n$$, but with some entries $$x_i$$ of $$x$$ missing. The algorithm must provide a prediction of the values of the missing entries.  
        * *__Denoising__*:  
            In this type of task, the machine learning algorithm is given as input a corrupted example $$\tilde{x} \in \mathbb{R}^n$$ obtained by an unknown corruption process from a clean example $$x \in \mathbb{R}^n$$. The learner must predict the clean example $$x$$ from its corrupted version $$\tilde{x}$$, or more generally predict the conditional probability distribution $$p(x |\tilde{x})$$.  
            > e.g. Signal Reconstruction, Image Artifact Removal  
        * *__Density (Probability Mass Function) Estimation__*:  
            In the density estimation problem, the machine learning algorithm is asked to learn a function $$p_\text{model}: \mathbb{R}^n \rightarrow R$$, where $$p_\text{model}(x)$$ can be interpreted as a probability density function (if $$x$$ is continuous) or a probability mass function (if $$x$$ is discrete) on the space that the examples were drawn from.  
            To do such a task well, the algorithm needs to learn the structure of the data it has seen. It must know where _examples cluster tightly_ and where they are _unlikely to occur_.  
            Most of the tasks described above require the learning algorithm to at least implicitly capture the structure of the probability distribution (i.e. it can be computed but we don't have an equation for it). Density estimation enables us to explicitly capture that distribution.  
            In principle,we can then perform computations on that distribution to solve the other tasks as well.  
            For example, if we have performed density estimation to obtain a probability distribution p(x), we can use that distribution to solve the missing value imputation task. Equivalently, if a value $$x_i$$ is missing, and all the other values, denoted $$x_{−i}$$, are given, then we know the distribution over it is given by $$p(x_i| x_{−i})$$.  
                In practice, density estimation does not always enable us to solve all these related tasks, because in many cases the required operations on p(x) are computationally intractable.  
            > e.g. Language Modeling  
        * *__A Lot More__*:  
            Their are many more tasks that could be defined for and solved by Machine Learning. However, this is a list of the most common problems, which have a well-known set of methods for handling them.  

    * __The Performance Measure $$P$$__:  
        A quantitative measure of the performance of a machine learning algorithm.  
        We often use __accuracy__ or __error rate__.

        
3. **Estimators:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    A __Point Estimator__ or __statistic__ is any function of the data:  
    <p>$$\hat{\boldsymbol{\theta}}_{m}=g\left(\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(m)}\right)$$</p>  
    such that a good estimator is a function whose output is close to the true underlying $$ \theta $$ that generated the training data.  
    > We assume that the true $$\boldsymbol{\theta}$$ is fixed, and that $$\hat{\boldsymbol{\theta}}$$ is a function of the data, which is drawn from a random process, making $$\hat{\boldsymbol{\theta}}$$ a __random variable__.  


    __Function Estimation/Approximation__ refers to estimation of the relationship between _input_ and _target data_.  
    I.E. We are trying to predict a variable $$y$$ given an input vector $$x$$, and we assume that there is a function $$f(x)$$ that describes the approximate relationship between $$y$$ and $$x$$.  
    If we assume that: $$y = f(x) + \epsilon$$, where $$\epsilon$$ is the part of $$y$$ that is not predictable from $$x$$; then we are interested in approximating $$f$$ with a model or estimate $$ \hat{f} $$.  
    > Function estimation is really just the same as estimating a parameter $$\boldsymbol{\theta}$$; the function estimator $$ \hat{f} $$ is simply a point estimator in function space.


4. **Properties of Estimators - Bias and Variance:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    The __Bias__ of an estimator is:  
    <p>$$ \operatorname{bias}\left(\hat{\boldsymbol{\theta}}_{m}\right)=\mathbb{E}\left(\hat{\boldsymbol{\theta}}_{m}\right)-\boldsymbol{\theta} $$</p>  
    where the expectation is over the data (seen as samples from a random variable) and $$ \theta $$ is the true underlying value of $$ \theta $$ used to define the data-generating distribution.  
    * __Unbiased Estimators:__ An estimator $$\hat{\boldsymbol{\theta}}_{m}$$ is said to be __unbiased__ if $$\operatorname{bias}\left(\hat{\boldsymbol{\theta}}_{m}\right)=\mathbf{0}$$, which implies that $$ \mathbb{E}\left(\hat{\boldsymbol{\theta}}_{m}\right)=\boldsymbol{\theta} $$.  
    * __Asymptotically Unbiased Estimators:__ An estimator is said to be __asymptotically unbiased__ if $$ \lim _{m \rightarrow \infty} \operatorname{bias}\left(\hat{\boldsymbol{\theta}}_{m}\right)=\mathbf{0},$$ which implies that $$\lim _{m \rightarrow \infty} \mathbb{E}\left(\hat{\boldsymbol{\theta}}_{m}\right)=\boldsymbol{\theta} $$  

    The __Variance__ of an estimator is a way to measure how much we expect the estimator to vary as a function of the data sample, defined, simply, as the variance over the training set random variable $$\hat{\theta}$$:  
    <p>$$ \operatorname{Var}(\hat{\theta}) $$</p>  

    The __Standard Error__ $$\operatorname{SE}(\hat{\theta})$$, of an estimator, is the square root of the variance.  
    * E.g. __The Standard Error of the Mean:__  
        $$\operatorname{SE}\left(\hat{\mu}_{m}\right)=\sqrt{\operatorname{Var}\left[\frac{1}{m} \sum_{i=1}^{m} x^{(i)}\right]}=\frac{\sigma}{\sqrt{m}}$$  
        Where $$\sigma^2$$ is the true variance of the samples $$x^i$$.  

    > Unfortunately, neither the square root of the sample variance nor the square root of the unbiased estimator of the variance provide an unbiased estimate of the standard deviation.


5. **Generalization Error from Standard Error (of the mean):**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    We often estimate the __generalization error__ by computing the __sample mean__ of the error on the test set.  
    Taking advantage of the central limit theorem, which tells us that the mean will be approximately distributed with a normal distribution, we can use the standard error to compute the probability that the true expectation falls in any chosen interval.  
    For example, the 95 percent confidence interval centered on the mean $$\hat{\mu}_ {m}$$ is:  
    <p>$$\left(\hat{\mu}_{m}-1.96 \mathrm{SE}\left(\hat{\mu}_{m}\right), \hat{\mu}_{m}+1.96 \mathrm{SE}\left(\hat{\mu}_{m}\right)\right)$$</p>  
    under the normal distribution with mean $$\hat{\mu}_{m}$$ and variance $$\mathrm{SE}\left(\hat{\mu}_{m}\right)^{2}$$.  
    We say that algorithm $$\boldsymbol{A}$$ is __better than__ algorithm $$\boldsymbol{B}$$ if the _upper bound_ of the $$95$$ percent confidence interval for the error of algorithm $$\boldsymbol{A}$$ is __less than__ the _lower bound_ of the $$95$$ percent confidence interval for the error of algorithm $$\boldsymbol{B}$$.  

6. **The Bias Variance Trade-off:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    Bias and variance measure two different sources of error in an estimator:  
    * __Bias__: measures the expected deviation from the true value of the function or parameter  
    * __Variance__: provides a measure of the deviation from the expected estimator value that any particular sampling of the data is likely to cause  

    __Evaluating Models - Trading off Bias and Variance:__  
    * The most common way to negotiate this trade-off is to use __cross-validation__
    * Alternatively, we can also compare the __mean squared error (MSE)__ of the estimates:  
        <p>$$\begin{aligned} \mathrm{MSE} &=\mathbb{E}\left[\left(\hat{\theta}_{m}-\theta\right)^{2}\right] \\ &=\operatorname{Bias}\left(\hat{\theta}_{m}\right)^{2}+\operatorname{Var}\left(\hat{\theta}_{m}\right) \end{aligned}$$</p>  
        The __MSE__ measures the overall expected deviation — in a squared error sense — between the estimator and the true value of the parameter $$\theta$$.  

    <button>Capacity and Bias/Variance</button>{: .showText value="show"
     onclick="showTextPopHide(event);"}
    ![img](/main_files/dl_book/1.png){: hidden=""}  


7. **Properties of Estimators - Consistency:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  
    __Consistency__ is a property implying that as the number of data points $$m$$ in our dataset increases, our point estimates converge to the true value of the corresponding parameters. Formally:  
    <p>$$\mathrm{plim}_{m \rightarrow \infty} \hat{\theta}_{m}=\theta$$</p>  
    Where:    
    $${\text { The symbol plim indicates convergence in probability, meaning that for any } \epsilon>0,}$$ 
    <p>$${P\left(\vert\hat{\theta}_{m}-\theta \vert>\epsilon\right) \rightarrow 0 \text { as } m \rightarrow \infty}$$</p>  
    > Sometimes referred to as __Weak Consistency__  

    __Strong Consistency__ applies to *__almost sure convergence__* of $$\hat{\theta}$$ to $$\theta$$.  

    __Consistency and Asymptotic Bias:__  
    * Consistency ensures that the bias induced by the estimator diminishes as the number of data examples grows.  
    * However, asymptotic unbiasedness does __not__ imply consistency


<!-- 8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}
 -->
***

## The Mathematics of Neural Networks
{: #content2}

0. **Derivative:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents20}
:   The derivative of a function is the amount that the value of a function changes when the input changes by an $$\epsilon$$ amount:  
:   $$f'(a)=\lim_{h\to 0}{\frac {f(a+h)-f(a)}{h}}. \\
    \text{i.e. } f(x + \epsilon)\approx f(x)+\epsilon f'(x)
    $$
:   __The Chain Rule__ is a way to compute the derivative of _composite functions_.  
    If $$y = f(x)$$ and $$z = g(y)$$:  
:   $$\dfrac{\partial z}{\partial x} = \dfrac{\partial z}{\partial y} \dfrac{\partial y}{\partial x}$$    

1. **Gradient: (Vector in, Scalar out)**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}
:   Gradients generalize derivatives to __*scalar functions*__ of several variables
:  ![Gradient](/main_files/math/calc/1.png){: width="80%"}
:   __Property:__ the gradient of a function $$\nabla f(x)$$ points in the direction of __steepest ascent__ from $$x$$. 

2. **The Jacobian: (Vector in, Vector out)**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}
:   The __Jacobian__ of $$f: \mathbb{R}^n \rightarrow \mathbb{R}^m $$ is a matrix of _first-order partial derivatives_ of a __*vector-valued function*__:  
:  ![Jacobian](/main_files/math/calc/2.png){: width="80%"}  
:   __The Chain Rule:__  
    Let $$f : \mathbb{R}^N \rightarrow \mathbb{R}^M$$ and $$g : \mathbb{R}^M \rightarrow \mathbb{R}^ K$$; and let  $$x \in \mathbb{R}^N, y \in \mathbb{R}^M$$, and $$z \in \mathbb{R}^K$$ with $$y = f(x)$$ and $$z = g(y)$$:  
:  $$\dfrac{\partial z}{\partial x} = \dfrac{\partial z}{\partial y} \dfrac{\partial y}{\partial x}$$    
:   where, $$\dfrac{\partial z}{\partial y} \in \mathbb{R}^{K \times M}$$ matrix, $$\dfrac{\partial y}{\partial x} \in \mathbb{R}^{M \times N}$$ matrix, and $$\dfrac{\partial z}{\partial x} \in \mathbb{R}^{K \times N}$$  matrix;  
    the multiplication of $$\dfrac{\partial z}{\partial y}$$  and $$\dfrac{\partial y}{\partial x}$$ is a matrix multiplication.  

2. **The Generalized Jacobian: (Tensor in, Tensor out)**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}
:   __A Tensor__ is a D-dimensional grid of number.  
:   Suppose that $$f: \mathbb{R}^{N_1 \times \cdots \times N_{D_x}} \rightarrow \mathbb{R}^{M_1 \times \cdots \times M_{D_y}} $$.  
    If $$y = f(x)$$ then the derivative $$\dfrac{\partial y}{\partial x}$$ is a __generalized Jacobian__ - an object with shape:  
:   $$(M_1 \times \cdots \times M_{D_y}) \times (N_1 \times \cdots \times N_{D_x})$$ 
:   >  we can think of the generalized Jacobian as generalization of a matrix, where each “row” has the same shape as $$y$$  and each “column” has the same shape as $$x$$.  
:    Just like the standard Jacobian, the generalized Jacobian tells us the relative rates of change between all elements of $$x$$  and all elements of $$y$$:  
:   $$(\dfrac{\partial y}{\partial x})_{i,j} = \dfrac{\partial y_i}{\partial x_j} \in \mathbb{R}$$
:   Just as the derivative, the generalized Jacobian gives us the relative change in $$y$$ given a small change in $$x$$:  
:   $$f(x + \delta x)\approx f(x)+ f'(x) \delta x = y + \dfrac{\partial y}{\partial x}\delta x$$  
:   where now, $$\delta x$$ is a tensor in $$\mathbb{R}{N_1 \cdots N_{d_x}}$$ and $$\dfrac{\partial y}{\partial x}$$ is a generalized matrix in $$\mathbb{R}^{(M_1 \times \cdots \times M_{D_y}) \times (N_1 \times \cdots \times N_{D_x})} $$.  
    The product $$\dfrac{\partial y_i}{\partial x_j} \delta x$$ is, therefore, a __*generalized matrix-vector multiply*__, which results in a tensor in $$\mathbb{R}^{M_1 \times \cdots \times M_{D_y}}$$.  
:   The __generalized matrix-vector multiply__ follows the same algebraic rules as a traditional matrix-vector multiply:  
:   ![matrix-vector mult](/main_files/math/calc/4.png){: width="80%"}  
:   ![matrix-vector mult-2](/main_files/math/calc/5.png){: width="100%"}  
:   __The Chain Rule:__  
:   ![chain rule](/main_files/math/calc/6.png){: width="100%"}  


3. **The Hessian:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}
:   The __Hessian__ Matrix of a _scalar function_ $$f: \mathbb{R}^d \rightarrow \mathbb{R} $$ is a matric of _second-order partial derivatives_:
:  ![Hessian](/main_files/math/calc/3.png){: width="80%"}
:   __Properties:__ 
        * The Hessian matrix is __*symmetric*__ - since we usually work with smooth/differentiable functions - due to _Clairauts Theorem_.  
        > __Clairauts Theorem:__ if the partial derivatives are continuous, the order of differentiation can be interchanged  
        * The Hessian is used in some optimization algorithms such as Newton’s method  
        * It is expensive to calculate but can drastically reduce the number of iterations needed to converge to a local minimum by providing information about the curvature of $$f$$


4. **Matrix Calculus:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}
:   __Important Identities:__  
:   $${\frac  {\partial {\mathbf  {a}}^{\top }{\mathbf  {x}}}{\partial {\mathbf  {x}}}}={\frac  {\partial {\mathbf  {x}}^{\top }{\mathbf  {a}}}{\partial {\mathbf  {x}}}}= a \\ 
    {\frac  {\partial {\mathbf  {x}}^{\top }{\mathbf  {A}}{\mathbf  {x}}}{\partial {\mathbf  {x}}}}=  ({\mathbf  {A}}+{\mathbf  {A}}^{\top }){\mathbf  {x}} \\ 
    {\frac  {\partial {\mathbf  {x}}^{\top }{\mathbf  {A}}{\mathbf  {x}}}{\partial {\mathbf  {x}}}}=  2{\mathbf  {A}}{\mathbf  {x}} \:\:\:\:\: \text{[Symmetric } A\text{]}$$
:   [Identities](https://en.wikipedia.org/wiki/Matrix_calculus)
:   __The Product Rule:__  
:   $${\displaystyle {\begin{aligned}\nabla (\mathbf {A} \cdot \mathbf {B} )&=(\mathbf {A} \cdot \nabla )\mathbf {B} +(\mathbf {B} \cdot \nabla )\mathbf {A} +\mathbf {A} \times (\nabla \times \mathbf {B} )+\mathbf {B} \times (\nabla \times \mathbf {A} )\\&=\mathbf {J} _{\mathbf {A} }^{\mathrm {T} }\mathbf {B} +\mathbf {J}_{\mathbf {B} }^{\mathrm {T} }\mathbf {A} \\&=\nabla \mathbf {A} \cdot \mathbf {B} +\nabla \mathbf {B} \cdot \mathbf {A} \ \end{aligned}}}\\ 
    \implies \\ 
    \nabla (fg) = (f')^T g + (g')^T f$$ 
:   Thus, we set our function $$h(x) = \langle f(x), g(x) \rangle = f(x)^T g(x)$$; then,  
:   $$\nabla h(x) = f'(x)^T g(x) + g'(x)^T f(x).$$


5. **Notes:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents25}
    * __Derivative Trick__:   
        Treat Least Square Matrix Forms as a simple square, but put the chain rule 2nd calculation to the left and transpose it (since it wasn't a square it was a transpose):  
        $$\nable_w \frac{1}{N}\|\mathrm{Xw}-\mathrm{y}\|^{2} = \frac{2}{N}\mathrm{X}^T(\mathrm{Xw}-\mathrm{y})$$  


6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents26}

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents27}

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents28}

***

## Challenges in Machine Learning
{: #content3}

1. **The Curse of Dimensionality:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents31}  
    It is a phenomena where many machine learning problems become exceedingly difficult when the number of dimensions in the data is high.  

    * The number of possible distinct configurations of a set of variables increases exponentially as the number of variables increases:  
        <button>Capacity and Bias/Variance</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        ![img](/main_files/dl_book/2.png){: hidden=""}  
        * __Statistical Challenge:__ the number of possible configurations of $$x$$ is much larger than the number of training examples  

2. **Local Constancy and Smoothness Regularization:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents32}  
    Prior believes about the particular data-set/learning-problem can be incorporated as:  
    * Beliefs about the __distribution__ of __parameters__ 
    * Beliefs about the __properties__ of the estimating __function__  
        > expressed implicitly by choosing algorithms that are biased toward choosing some class of functions over another, even though these biases may not be expressed (or even be possible to express) in terms of a probability distribution representing our degree of belief in various functions.  

    __The Local Constancy (Smoothness) Prior:__ states that the function we learn should not change very much within a small region.  
    Mathematically, traditional ML methods are designed to encourage the learning process to learn a function $$f^\ast$$ that satisfies the condition:  
    $$\:\:\:\:\:\:\:$$ $$\:\:\:\:\:\:\:$$ $$f^{*}(\boldsymbol{x}) \approx f^{*}(\boldsymbol{x}+\epsilon)$$  
    for most configurations $$x$$ and small change $$\epsilon$$.  
    * <button>Example: K-Means</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        ![img](/main_files/dl_book/7.png){: hidden=""}  


    A [__Local Kernel__](/concepts_#bodyContents60) can be thought of as a similarity function that performs template matching, by measuring how closely a test example $$x$$ resembles each training example $$x^{(i)}$$.  
    Much of the modern motivation for Deep Learning is derived from studying the limitations of local template matching and how deep models are able to succeed in cases where local template matching fails _(Bengio et al., 2006b)_.  

    <button>Example: Decision Trees</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    _Decision trees also suffer from the limitations of exclusively smoothness-based learning, because they break the input space into as many regions as there are leaves and use a separate parameter (or sometimes many parameters for extensions of decision trees) in each region. If the target function requires a tree with at least $$n$$ leaves to be represented accurately, then at least $$n$$ training examples are required to fit the tree. A multiple of $$n$$ is needed to achieve some level of statistical confidence in the predicted output._{: hidden=""}  

    In general, to distinguish $$\mathcal{O}(k)$$ regions in input space, all these methods require $$\mathcal{O}(k)$$ examples. Typically there are $$\mathcal{O}(k)$$ parameters, with $$\mathcal{O}(1)$$ parameters associated with each of the $$\mathcal{O}(k)$$ regions.  

    
    __Key Takeaways:__   
    * *__Is there a way to represent a complex function that has many more regions to be distinguished than the number of training examples?__*  
        Clearly, assuming only smoothness of the underlying function will not allow a learner to do that.  
        The smoothness assumption and the associated nonparametric learning algorithms work extremely well as long as there are enough examples for the learning algorithm to observe high points on most peaks and low points on most valleys of the true underlying function to be learned.  
    * *__Is it possible to represent a complicated function efficiently? and if it is complicated, Is it possible for the estimated function to generalize well to new inputs?__*  
        Yes.  
        The key insight is that a very large number of regions, such as $$\mathcal{O}(2^k)$$, can be deﬁned with $$\mathcal{O}(k)$$ examples, so long as we introduce some dependencies between the regions through additional assumptions about the underlying data-generating distribution.
        In this way, we can actually generalize non-locally _(Bengio and Monperrus, 2005; Bengio et al., 2006c)_.  
    * *__Deep Learning VS Machine Learning:__*  
        The core idea in deep learning is that we assume that the data was generated by the composition of factors, or features, potentially at multiple levels in a hierarchy.  
        These apparently mild assumptions allow an exponential gain in the relationship between the number of examples and the number of regions that can be distinguished.  
        The exponential advantages conferred by the use of deep distributed representations counter the exponential challenges posed by the curse of dimensionality.  
        > __Further Reading:__ (on the exponential gain) Sections: 6.4.1, 15.4 and 15.5.  
    

3. **Manifold Learning:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents33}  
    A __Manifold__ - a connected region - is a set of points associated with a neighborhood around each point. From any given point, the manifold locally appears to be a Euclidean space.  
    ![img](/main_files/dl_book/8.png){: width="100%"}  

    __Manifolds in ML:__  
    In ML, the term is used loosely to designate a connected set of points that can be approximated well by considering only a small number of degrees of freedom, or dimensions, embedded in a higher-dimensional space. Each dimension corresponds to a local direction of variation.  
    In the context of machine learning, we allow the dimensionality of the manifold to vary from one point to another. This often happens when a manifold intersects itself. For example, a figure eight is a manifold that has a single dimension in most places but two dimensions at the intersection at the center.  
    __Manifold Assumptions:__  
    <button>Discussion</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    _Many machine learning problems seem hopeless if we expect the machine learning algorithm to learn functions with interesting variations across all of $$\mathbb{R}^n$$. Manifold learning algorithms surmount this obstacle by assuming that most of $$\mathbb{R}^n$$ consists of invalid inputs, and that interesting inputs occur only a long a collection of manifolds containing a small subset of points, with interesting variations in the output of the learned function occurring only along directions that lie on the manifold, or with interesting variations happening only when we move from one manifold to another. Manifold learning was introduced in the case of continuous-valued data and in the unsupervised learning setting, although this probability concentration idea can be generalized to both discrete data and the supervised learning setting: the key assumption remains that probability mass is highly concentrated._{: hidden=""}  
    We assume that the *__data lies along a low-dimensional manifold__*:  
    * May not always be correct or useful  
    * In the context of AI tasks (e.g. processing images, sounds, or text): At least approximately correct.  
        To show that is true we need to argue two points:  
        * The probability distribution over images, text strings, and sounds that occur in real life is highly concentrated.  
            <button>Example/proof: The Manifold of Natural Images</button>{: .showText value="show" onclick="showTextPopHide(event);"}
            ![img](/main_files/dl_book/9.png){: hidden=""}   
            > Uniform noise essentially never resembles structured inputs from these domains.   
        * We must, also, establish that the examples we encounter are connected to each other by other examples, with each example surrounded by other highly similar examples that can be reached by applying transformations to traverse the manifold:  
            _Informally,_ we can imagine such neighborhoods and transformations:  
            In the case of images, we can think of many possible transformations that allow us to trace out a manifold in image space: we can gradually dim or brighten the lights, gradually move or rotate objects in the image, gradually alter the colors on the surfaces of objects, and so forth.  
            > Multiple manifolds are likely involved in most applications. For example,the manifold of human face images may not be connected to the manifold of cat face images.  
            > Rigorous Results: _(Cayton, 2005; Narayanan and Mitter,2010; Schölkopf et al., 1998; Roweis and Saul, 2000; Tenenbaum et al., 2000; Brand,2003; Belkin and Niyogi, 2003; Donoho and Grimes, 2003; Weinberger and Saul,2004)_  

    __Benefits:__  
    When the data lies on a low-dimensional manifold, it can be most natural for machine learning algorithms to represent the data in terms of coordinates on the manifold, rather than in terms of coordinates in $$\mathbb{R}^n$$.  
    E.g. In everyday life, we can think of roads as 1-D manifolds embedded in 3-D space. We give directions to specific addresses in terms of address numbers along these 1-D roads, not in terms of coordinates in 3-D space.  
    > Learning Manifold Structure: _figure 20.6_  


<!-- ## Activation Functions
{: #content4}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents41}

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents42}

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents43}

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents44}

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents45}

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents46}

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents47}

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents48}

*** -->