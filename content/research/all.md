---
layout: NotesPage
title: All
permalink: /all
prevLink: /work_files/research/dl/theory.html
---

<div markdown="1" class = "TOC">
# Table of Contents

</div>

***
***


## Data Drift/Shift
{: #content1}


1. **Types of SHIFT:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    * Dataset shift happens when the i.i.d. assumptions are not valid for out problem space  


1. **Types of SHIFT:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    * __Covariate Shift:__  

        ![img](https://cdn.mathpix.com/snip/images/q4r9dKK9TcR2xk-GzjpGu9ufWzttm4idv18GCoz86eM.original.fullsize.png){: width="40%"}  

    * __Prior Probability Shift__: $$P(x)$$  
        ![img](https://cdn.mathpix.com/snip/images/Qoxi81pa3lGfWuHOpMAKeth3IJZxA0UubqgBip2oWTU.original.fullsize.png){: width="40%"}  

    * __Covariate Shift__:  

        ![img](https://cdn.mathpix.com/snip/images/SnCgPXP8t5kPsPh2XMo4elCO87crgEgXzFdmCV3P7vs.original.fullsize.png){: width="40%"}  

    * __Internal CS__:  
        Researchers found that due to the variation in the distribution of activations from the output of a given hidden layer, which are used as the input to a subsequent layer, the network layers can suffer from covariate shift which can impede the training of deep neural networks.



        * Covariate shift is the change in the distribution of the covariates specifically, that is, the independent variables. This is normally due to changes in state of latent variables, which could be temporal (even changes to the stationarity of a temporal process), or spatial, or less obvious.  

        * IIt introduces BIAS to *__cross-validation__* 


    * The problem of dataset shift can stem from the way input features are utilized, the way training and test sets are selected, data sparsity, shifts in the data distribution due to non-stationary environments, and also from changes in the activation patterns within layers of deep neural networks.  

2. **General Data Distribution Shifts:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    * __Feature change__, such as when new features are added, older features are removed, or the set of all possible values of a feature changes:
        months to years  
    * __Label schema change__ is when the set of possible values for Y change. With label shift, P(Y) changes but P(X|Y) remains the same. With label schema change, both P(Y) and P(X|Y) change.  
        * *__CREDIT:__* * With regression tasks, label schema change could happen because of changes in the possible range of label values. Imagine you’re building a model to predict someone’s credit score. Originally, you used a credit score system that ranged from 300 to 850, but you switched to a new system that ranges from 250 to 900.  
    * 

3. **Causes of SHIFT:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    ![img](https://cdn.mathpix.com/snip/images/kOrj05WvKP5Za6xCtyQNlLv6kDvfErHCPWjVcCkWGt4.original.fullsize.png){: width="40%"}  
    * Dataset shift resulting from sample selection bias is especially relevant when dealing with imbalanced classification, because, in highly imbalanced domains, the minority class is particularly sensitive to singular classification errors, due to the typically low number of samples it presents.

    ![img](https://cdn.mathpix.com/snip/images/COPV9jUSmrT5WdsWaBhrUuzZKrdh2GmlL3uLwREiUqw.original.fullsize.png){: width="40%"}  


    * __IN CREDIT__{: style="color: red"}  
        * Non-stationary due to changing macro-economic state  
        * Adversarial Relationship / Fraud: people might try to game the system to get loans  

4. **Handling Data Distribution Shifts:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    1. DETECTION  
    2. HANDLING  

    * __DETECTION:__{: style="color: blue"}  
        ![img](https://cdn.mathpix.com/snip/images/Doczy4vS_FRRoH_miFbRcraLdzirPt8RHaWrNym5ReU.original.fullsize.png){: width="40%"}  
        ![img](https://cdn.mathpix.com/snip/images/38j2j-H44tajU8k5Khp1RVA7C42q1-Gb2Ga0LD2BDZ0.original.fullsize.png){: width="40%"}  
        * monitor your model’s accuracy-related metrics30 in production to see whether they have changed.
            * When ground truth labels are unavailable or too delayed to be useful, we can monitor other distributions of interest instead. The distributions of interest are the input distribution P(X), the label distribution P(Y), and the conditional distributions P(X|Y) and P(Y|X).  
            * In research, there have been efforts to understand and detect label shifts without labels from the target distribution. One such effort is Black Box Shift Estimation by Lipton et al., 2018.  
        * __Statistical Methods:__{: style="color: blue"}  
            *  a simple method many companies use to detect whether the two distributions are the same is to compare their statistics like mean, median, variance, quantiles, skewness, kurtosis, etc. (bad)   
                * If those metrics differ significantly, the inference distribution might have shifted from the training distribution. However, if those metrics are similar, there’s no guarantee that there’s no shift.  
            * 




## Encoding
{: #content2}

1. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  

2. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  

3. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}  

4. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}  


## Feature Importance
{: #content3}

1. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents31}  

2. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents32}  

3. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents33}  

4. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents34}  


## Feature Selection
{: #content4}

1. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents41}  

2. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents42}  

3. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents43}  

4. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents44}  


## Data Preprocessing and Normalization
{: #content5}

1. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents5 #bodyContents51}  

2. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents5 #bodyContents52}  

    __Data Regularization:__{: style="color: red"}  
    {: #lst-p}
    * The __Design Matrix__ contains sample points in each *__row__* 
    * __Feature Scaling/Mean Normalization (of data)__:  
        * Define the mean $$\mu_j$$ of each feature of the datapoints $$x^{(i)}$$:  
        <p>$$\mu_{j}=\frac{1}{m} \sum_{i=1}^{m} x_{j}^{(i)}$$</p>  
        * Replace each $$x_j^{(i)}$$ with $$x_j - \mu_j$$  
    * __Centering__:  subtracting $$\mu$$ from each row of $$X$$ 
    * __Sphering__:  applying the transform $$X' = X \Sigma^{-1/2}$$  
    * __Whitening__:  Centering + Sphering (also known as *__Decorrelating feature space__*)  

    __Why Normalize the Data/Signal?__  
    ![img](https://cdn.mathpix.com/snip/images/8aNuJetgTgCtv4pvqaI0dr96pDyUmfuX_d1aLK1lmaw.original.fullsize.png){: width="40%"}  

3. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents5 #bodyContents53}  

4. **CODE:**{: style="color: SteelBlue"}{: .bodyContents5 #bodyContents54}  
    * [Preprocessing for deep learning: from covariance matrix to image whitening (Blog)](https://hadrienj.github.io/posts/Preprocessing-for-deep-learning/)  
        * [NOTEBOOK](https://github.com/hadrienj/Preprocessing-for-deep-learning)  



## Validation & Evaluation - ROC, AUC, Reject Inference + Off-policy Evaluation
{: #content6}

1. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents61}  

2. **ROC:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents62}  
    * A way to quantify how good a **binary classifier** separates two classes
    * True-Positive-Rate / False-Positive-Rate
    * Good classifier has a ROC curve that is near the top-left diagonal (hugging it)
    * A Bad Classifier has a ROC curve that is close to the diagonal line
    * It allows you to set the **classification threshold**  
        * You can minimize False-positive rate or maximize the True-Positive Rate  

    __Notes:__{: style="color: red"}  
    {: #lst-p}
    * ROC curve is monotone increasing from 0 to 1 and is invariant to any monotone transformation of test results.
    * ROC curves (& AUC) are useful even if the __predicted probabilities__ are not *__"properly calibrated"__*  
    * ROC curves are not affected by monotonically increasing functions
    * [Scale and Threshold Invariance (Blog)](https://builtin.com/data-science/roc-curves-auc)  
    * Accuracy is neither a threshold-invariant metric nor a scale-invariant metric.  
    * When to use __PRECISION__: when data is *__imbalanced__* E.G. when the number of *__negative__* examples is __larger__ than *__positive__*.  
        Precision does not include __TN (True Negatives)__ so NOT AFFECTED.  
        In PRACTICE, e.g. studying *__RARE Disease__*.  
    <br>


    * ROC Curve only cares about the *__ordering__* of the scores, not the values.  
    * __Probability Calibration__ and ROC: The calibration doesn't change the order of the scores, it just scales them to make a better match, and the ROC score only cares about the ordering of the scores.  

    * [ROC and Credit Score Example (Blog)](https://kiwidamien.github.io/what-is-a-roc-curve-a-visualization-with-credit-scores.html)  

    * __AUC__: The AUC is also the probability that a randomly selected positive example has a higher score than a randomly selected negative example.

    <button>AUC Reliability (Equal AUC - different models)</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](https://cdn.mathpix.com/snip/images/GAyvZvN61xzDjklTeVepqrYYuWrXXfPnEHkNwM80p6k.original.fullsize.png){: width="100%" hidden=""}  


3. **AUC:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents63}  
    AUC provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example.  

    * Range $$ = 0.5 - 1.0$$, from poor to perfect  

    * __Pros:__{: style="color: blue"}  
        * AUC is *__scale-invariant__*: It measures how well predictions are ranked, rather than their absolute values.
        * AUC is *__classification-threshold-invariant__*: It measures the quality of the model's predictions irrespective of what classification threshold is chosen.

        > These properties make AUC pretty valuable for evaluating binary classifiers as it provides us with a way to compare them without caring about the classification threshold.  
        
    * __Cons__{: style="color: red"}   
        However, both these reasons come with caveats, which may limit the usefulness of AUC in certain use cases:

        * Scale invariance is not always desirable. For example, sometimes we really do need well calibrated probability outputs, and AUC won’t tell us about that.

        * Classification-threshold invariance is not always desirable. In cases where there are wide disparities in the cost of false negatives vs. false positives, it may be critical to minimize one type of classification error. For example, when doing email spam detection, you likely want to prioritize minimizing false positives (even if that results in a significant increase of false negatives). AUC isn't a useful metric for this type of optimization.  


    __Notes:__{: style="color: red"}  
    {: #lst-p}
    * __Partial AUC__ can be used when only a portion of the entire ROC curve needs to be considered.  
    <br>


4. **Reject Inference and Off-policy Evaluation:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents64}  
    * __Reject inference__ is a method for performing off-policy evaluation, which is a way to estimate the performance of a policy (a decision-making strategy) based on data generated by a different policy. In reject inference, the idea is to use importance sampling to weight the data in such a way that the samples generated by the behavior policy (the one that generated the data) are down-weighted, while the samples generated by the target policy (the one we want to evaluate) are up-weighted. This allows us to focus on the samples that are most relevant to the policy we are trying to evaluate, which can improve the accuracy of our estimates.

    * __Off-policy evaluation__ is useful in situations where it is not possible or practical to directly evaluate the performance of a policy. For example, in a real-world setting, it may not be possible to directly evaluate a new policy because it could be risky or expensive to implement. In such cases, off-policy evaluation can help us estimate the performance of the policy using data generated by a different, perhaps safer or more easily implemented, policy.


5. **Validation:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents65}  

    1. Validate a model with given constraints (see above).
    
        Spoke to the recruiter who was super nice and transparent. Scheduled a technical screening afterwards. The question was to validate a model only knowing the true values and predicted values. The interviewer wanted to incorporate the business value of the model. I found this to be interesting and odd as how can the business value validate any model. As we walked through the problem, the interviewer did not care about traditional statistical error measures and techniques in model validation. The interviewer wanted to incorporate the business cases (i.e. __total loss in revenue and gains__{: style="color: goldenrod"}) to validate the model. To me, it felt more business intelligence rather than traditional statistics/machine learning model validation. I am uncertain if data scientists at Affirm are just BI with Python skills.


6. **Evaluation:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents66}  
    * __Precision vs Recall Tradeoff:__{: style="color: blue"}  
        * __Recall__ is more important where Overlooked Cases (False Negatives) are more costly than False Alarms (False Positive). The focus in these problems is finding the positive cases.

        * __Precision__ is more important where False Alarms (False Positives) are more costly than Overlooked Cases (False Negatives). The focus in these problems is in weeding out the negative cases.





## Regularization
{: #content7}

1. **Regularization:**{: style="color: SteelBlue"}{: .bodyContents7 #bodyContents71}  

2. **Norms:**{: style="color: SteelBlue"}{: .bodyContents7 #bodyContents72}  

3. **AUC:**{: style="color: SteelBlue"}{: .bodyContents7 #bodyContents73}  

4. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents7 #bodyContents74}  



## Interpretability
{: #content8}

1. **Regularization:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents81}  

2. **Norms:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents82}  

3. **AUC:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents83}  

4. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents84}  




## Decision Trees, Random Forests, XGB, and Gradient Boosting
{: #content9}

1. **Regularization:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents91}  

2. **Norms:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents92}  

3. **AUC:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents93}  

4. **Boosting:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents94}  

5. **Boosting:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents95}  
    * __Boosting__: create different hypothesis $$h_i$$s sequentially + make each new hypothesis __decorrelated__ with previous hypothesis.  
        * Assumes that this will be combined/ensembled  
        * Ensures that each new model/hypothesis will give a different/independent output  



## [Uncertainty and Probabilistic Calibration](/work_files/calibration)  
{: #content10}

* [CALIBRATION (website)](/work_files/calibration)  

1. **Uncertainty:**{: style="color: SteelBlue"}{: .bodyContents10 #bodyContents101}  
    * __Aleatoric vs Epistemic:__{: style="color: red"}  
        Aleatoric uncertainty and epistemic uncertainty are two types of uncertainty that can arise in statistical modeling and machine learning. Aleatoric uncertainty is a type of uncertainty that arises from randomness or inherent noise in the data. It is inherent to the system being studied and cannot be reduced through additional data or better modeling. On the other hand, epistemic uncertainty is a type of uncertainty that arises from incomplete or imperfect knowledge about the system being studied. It can be reduced through additional data or better modeling.  

2. **Model Uncertainty, Softmax, and Dropout:**{: style="color: SteelBlue"}{: .bodyContents10 #bodyContents102}   
    __Interpreting Softmax Output Probabilities:__{: style="color: red"}  
    Softmax outputs only measure [__Aleatoric Uncertainty__](https://en.wikipedia.org/wiki/Uncertainty_quantification#Aleatoric_and_epistemic_uncertainty).  
    In the same way that in regression, a NN with two outputs, one representing mean and one variance, that parameterise a Gaussian, can capture aleatoric uncertainty, even though the model is deterministic.  
    Bayesian NNs (dropout included), aim to capture epistemic (aka model) uncertainty.  

    __Dropout for Measuring Model (epistemic) Uncertainty:__{: style="color: red"}  
    Dropout can give us principled uncertainty estimates.  
    Principled in the sense that the uncertainty estimates basically approximate those of our [Gaussian process](/work_files/research/dl/archits/nns#bodyContents13).  

    __Theoretical Motivation:__ dropout neural networks are identical to <span>variational inference in Gaussian processes</span>{: style="color: purple"}.  
    __Interpretations of Dropout:__  
    {: #lst-p}
    * Dropout is just a diagonal noise matrix with the diagonal elements set to either 0 or 1.  
    * [What My Deep Model Doesn't Know (Blog! - Yarin Gal)](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html)  
    <br>


3. **Calibration in Deep Networks:**{: style="color: SteelBlue"}{: .bodyContents10 #bodyContents103}  
    * [READ THIS (Blog!)](http://alondaks.com/2017/12/31/the-importance-of-calibrating-your-deep-model/)  

4. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents10 #bodyContents104}  




## Extra: Bandit, bootstrapping, and prediction interval estimation, Linear Models in Credit
{: #content11}

1. **Bandit:**{: style="color: SteelBlue"}{: .bodyContents11 #bodyContents111}  

2. **bootstrapping:**{: style="color: SteelBlue"}{: .bodyContents11 #bodyContents112}  

3. **prediction interval estimation:**{: style="color: SteelBlue"}{: .bodyContents11 #bodyContents113}  

4. **Linear Models in Credit Analysis:**{: style="color: SteelBlue"}{: .bodyContents11 #bodyContents114}  
    ![img](https://cdn.mathpix.com/snip/images/fa_yKNL9BXfeHhGkOvnXhNmgSYR8TF0B-hCfZn-0whc.original.fullsize.png){: width="40%"}  s


5. **Errors vs Residuals:**{: style="color: SteelBlue"}{: .bodyContents11 #bodyContents115}  
    The __Error__ of an observed value is the deviation of the observed value from the (unobservable) **_true_** value of a quantity of interest.  

    The __Residual__ of an observed value is the difference between the observed value and the *__estimated__* value of the quantity of interest.  




## Notes from Affirm Blog
{: #content12}

1. **Bandit:**{: style="color: SteelBlue"}{: .bodyContents12 #bodyContents121}  

2. **bootstrapping:**{: style="color: SteelBlue"}{: .bodyContents12 #bodyContents122}  

3. **prediction interval estimation:**{: style="color: SteelBlue"}{: .bodyContents12 #bodyContents123}  

4. **Linear Models in Credit Analysis:**{: style="color: SteelBlue"}{: .bodyContents12 #bodyContents124}  
    ![img](https://cdn.mathpix.com/snip/images/fa_yKNL9BXfeHhGkOvnXhNmgSYR8TF0B-hCfZn-0whc.original.fullsize.png){: width="40%"}  s




***
***


![img](https://cdn.mathpix.com/snip/images/TNzZfbJuHJsESt3Ds0LpsEBVdRsi56VBP8RK7r54Vc0.original.fullsize.png){: width="80%"}  


## Information Theory
{: #content1}

1. **Information Theory:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    __Information theory__ is a branch of applied mathematics that revolves around quantifying how much information is present in a signal.  
    In the context of machine learning, we can also apply information theory to continuous variables where some of these message length interpretations do not apply, instead, we mostly use a few key ideas from information theory to characterize probability distributions or to quantify similarity between probability distributions.  
    <br>

2. **Motivation and Intuition:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    The basic intuition behind information theory is that learning that an unlikely event has occurred is more informative than learning that a likely event has occurred. A message saying “the sun rose this morning” is so uninformative as to be unnecessary to send, but a message saying “there was a solar eclipse this morning” is very informative.  
    Thus, information theory quantifies information in a way that formalizes this intuition:  
    {: #lst-p}
    * Likely events should have low information content - in the extreme case, guaranteed events have no information at all  
    * Less likely events should have higher information content
    * Independent events should have additive information. For example, finding out that a tossed coin has come up as heads twice should convey twice as much information as finding out that a tossed coin has come up as heads once.  
    <br>

33. **Measuring Information:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents133}  
    In Shannons Theory, to __transmit $$1$$ bit of information__ means to __divide the recipients *Uncertainty* by a factor of $$2$$__.  

    Thus, the __amount of information__ transmitted is the __logarithm__ (base $$2$$) of the __uncertainty reduction factor__.  

    The __uncertainty reduction factor__ is just the __inverse of the probability__ of the event being communicated.  

    Thus, the __amount of information__ in an event $$\mathbf{x} = x$$, called the *__Self-Information__*  is:  
    <p>$$I(x) = \log (1/p(x)) = -\log(p(x))$$</p>  

    __Shannons Entropy:__  
    It is the __expected amount of information__ of an uncertain/stochastic source. It acts as a measure of the amount of *__uncertainty__* of the events.  
    Equivalently, the amount of information that you get from one sample drawn from a given probability distribution $$p$$.  
    <br>

3. **Self-Information:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    The __Self-Information__ or __surprisal__ is a synonym for the surprise when a random variable is sampled.  
    The __Self-Information__ of an event $$\mathrm{x} = x$$:  
    <p>$$I(x) = - \log P(x)$$</p>  
    Self-information deals only with a single outcome.  

    <button>Graph $$\log_2{1/x}$$</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](https://cdn.mathpix.com/snip/images/NQ7sPTwgM1cNWpDRoclXezOYWAE8lttajOy5ofO3UQ4.original.fullsize.png){: width="28%" hidden=""}  
    <br>

4. **Shannon Entropy:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    To quantify the amount of uncertainty in an entire probability distribution, we use __Shannon Entropy__.  
    __Shannon Entropy__ is defined as the average amount of information produced by a stochastic source of data.  
    <p>$$H(x) = {\displaystyle \operatorname {E}_{x \sim P} [I(x)]} = - {\displaystyle \operatorname {E}_{x \sim P} [\log P(X)] = -\sum_{i=1}^{n} p\left(x_{i}\right) \log p\left(x_{i}\right)}$$</p>  
    __Differential Entropy__ is Shannons entropy of a __continuous__ random variable $$x$$.  
    ![img](/main_files/math/prob/11.png){: width="60%"}  
    <br>

5. **Distributions and Entropy:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    Distributions that are nearly deterministic (where the outcome is nearly certain) have low entropy; distributions that are closer to uniform have high entropy.
    <br>

6. **Relative Entropy \| KL-Divergence:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    The __Kullback–Leibler divergence__ (__Relative Entropy__) is a measure of how one probability distribution diverges from a second, expected probability distribution.  
    __Mathematically:__  
    <p>$${\displaystyle D_{\text{KL}}(P\parallel Q)=\operatorname{E}_{x \sim P} \left[\log \dfrac{P(x)}{Q(x)}\right]=\operatorname{E}_{x \sim P} \left[\log P(x) - \log Q(x)\right]}$$</p>  
    * __Discrete__:  
    <p>$${\displaystyle D_{\text{KL}}(P\parallel Q)=\sum_{i}P(i)\log \left({\frac {P(i)}{Q(i)}}\right)}$$  </p>  
    * __Continuous__:  
    <p>$${\displaystyle D_{\text{KL}}(P\parallel Q)=\int_{-\infty }^{\infty }p(x)\log \left({\frac {p(x)}{q(x)}}\right)\,dx,}$$ </p>  

    __Interpretation:__  
    {: #lst-p}
    * __Discrete variables__:  
        it is the extra amount of information needed to send a message containing symbols drawn from probability distribution $$P$$, when we use a code that was designed to minimize the length of messages drawn from probability distribution $$Q$$.  
    * __Continuous variables__:  

    __Properties:__  
    {: #lst-p}
    * Non-Negativity:  
            $${\displaystyle D_{\mathrm {KL} }(P\|Q) \geq 0}$$  
    * $${\displaystyle D_{\mathrm {KL} }(P\|Q) = 0 \iff}$$ $$P$$ and $$Q$$ are:
        * *__Discrete Variables__*:  
                the same distribution 
        * *__Continuous Variables__*:  
                equal "almost everywhere"  
    * Additivity of _Independent Distributions_:  
            $${\displaystyle D_{\text{KL}}(P\parallel Q)=D_{\text{KL}}(P_{1}\parallel Q_{1})+D_{\text{KL}}(P_{2}\parallel Q_{2}).}$$  
    * $${\displaystyle D_{\mathrm {KL} }(P\|Q) \neq D_{\mathrm {KL} }(Q\|P)}$$  
        > This asymmetry means that there are important consequences to the choice of the ordering  
    * Convexity in the pair of PMFs $$(p, q)$$ (i.e. $${\displaystyle (p_{1},q_{1})}$$ and  $${\displaystyle (p_{2},q_{2})}$$ are two pairs of PMFs):  
            $${\displaystyle D_{\text{KL}}(\lambda p_{1}+(1-\lambda )p_{2}\parallel \lambda q_{1}+(1-\lambda )q_{2})\leq \lambda D_{\text{KL}}(p_{1}\parallel q_{1})+(1-\lambda )D_{\text{KL}}(p_{2}\parallel q_{2}){\text{ for }}0\leq \lambda \leq 1.}$$  

    __KL-Div as a Distance:__  
    Because the KL divergence is non-negative and measures the difference between two distributions, it is often conceptualized as measuring some sort of distance between these distributions.  
    However, it is __not__ a true distance measure because it is __*not symmetric*__.  
    > KL-div is, however, a *__Quasi-Metric__*, since it satisfies all the properties of a distance-metric except symmetry  

    __Applications__  
    Characterizing:  
    {: #lst-p}
    * Relative (Shannon) entropy in information systems
    * Randomness in continuous time-series 
    * It is a measure of __Information Gain__; used when comparing statistical models of inference  

    __Example Application and Direction of Minimization__  
    Suppose we have a distribution $$p(x)$$ and we wish to _approximate_ it with another distribution $$q(x)$$.  
    We have a choice of _minimizing_ either:  
    1. $${\displaystyle D_{\text{KL}}(p\|q)} \implies q^\ast = \operatorname {arg\,min}_q {\displaystyle D_{\text{KL}}(p\|q)}$$  
        Produces an approximation that usually places high probability anywhere that the true distribution places high probability.  
    2. $${\displaystyle D_{\text{KL}}(q\|p)} \implies q^\ast \operatorname {arg\,min}_q {\displaystyle D_{\text{KL}}(q\|p)}$$  
        Produces an approximation that rarely places high probability anywhere that the true distribution places low probability.  
        > which are different due to the _asymmetry_ of the KL-divergence  

    <button>Choice of KL-div Direction</button>{: .showText value="show"  
     onclick="showTextPopHide(event);"}
    ![img](/main_files/math/infothry/1.png){: width="100%" hidden=""}
    <br>

7. **Cross Entropy:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  
    The __Cross Entropy__ between two probability distributions $${\displaystyle p}$$ and $${\displaystyle q}$$ over the same underlying set of events measures the average number of bits needed to identify an event drawn from the set, if a coding scheme is used that is optimized for an "unnatural" probability distribution $${\displaystyle q}$$, rather than the "true" distribution $${\displaystyle p}$$.  
    <p>$$H(p,q) = \operatorname{E}_{p}[-\log q]= H(p) + D_{\mathrm{KL}}(p\|q) =-\sum_{x }p(x)\,\log q(x)$$</p>  
  
    It is similar to __KL-Div__ but with an additional quantity - the entropy of $$p$$.  
  
    Minimizing the cross-entropy with respect to $$Q$$ is equivalent to minimizing the KL divergence, because $$Q$$ does not participate in the omitted term.  
  
    We treat $$0 \log (0)$$ as $$\lim_{x \to 0} x \log (x) = 0$$.  
    <br>

8. **Mutual Information:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}  
    The __Mutual Information (MI)__ of two random variables is a measure of the mutual dependence between the two variables.  
    More specifically, it quantifies the "amount of information" (in bits) obtained about one random variable through observing the other random variable.  
  
    It can be seen as a way of measuring the reduction in uncertainty (information content) of measuring a part of the system after observing the outcome of another parts of the system; given two R.Vs, knowing the value of one of the R.Vs in the system gives a corresponding reduction in (the uncertainty (information content) of) measuring the other one.  

    __As KL-Divergence:__  
    Let $$(X, Y)$$ be a pair of random variables with values over the space $$\mathcal{X} \times \mathcal{Y}$$ . If their joint distribution is $$P_{(X, Y)}$$ and the marginal distributions are $$P_{X}$$ and $$P_{Y},$$ the mutual information is defined as:  
    <p>$$I(X ; Y)=D_{\mathrm{KL}}\left(P_{(X, Y)} \| P_{X} \otimes P_{Y}\right)$$</p>  

    __In terms of PMFs for discrete distributions:__  
    The mutual information of two jointly discrete random variables $$X$$ and $$Y$$ is calculated as a double sum:  
    <p>$$\mathrm{I}(X ; Y)=\sum_{y \in \mathcal{Y}} \sum_{x \in \mathcal{X}} p_{(X, Y)}(x, y) \log \left(\frac{p_{(X, Y)}(x, y)}{p_{X}(x) p_{Y}(y)}\right)$$</p>  
    where $${\displaystyle p_{(X,Y)}}$$ is the joint probability mass function of $${\displaystyle X}$$ X and $${\displaystyle Y}$$, and $${\displaystyle p_{X}}$$ and $${\displaystyle p_{Y}}$$ are the marginal probability mass functions of $${\displaystyle X}$$ and $${\displaystyle Y}$$ respectively.  

    __In terms of PDFs for continuous distributions:__  
    In the case of jointly continuous random variables, the double sum is replaced by a double integral:  
    <p>$$\mathrm{I}(X ; Y)=\int_{\mathcal{Y}} \int_{\mathcal{X}} p_{(X, Y)}(x, y) \log \left(\frac{p_{(X, Y)}(x, y)}{p_{X}(x) p_{Y}(y)}\right) d x d y$$</p>  
    where $$p_{(X, Y)}$$ is now the joint probability density function of $$X$$ and $$Y$$ and $$p_{X}$$ and $$p_{Y}$$ are the marginal probability density functions of $$X$$ and $$Y$$ respectively.  


    __Intuitive Definitions:__{: style="color: red"}  
    {: #lst-p}
    * Measures the information that $$X$$ and $$Y$$ share:  
        It measures how much knowing one of these variables reduces uncertainty about the other.  
        * __$$X, Y$$ Independent__  $$\implies I(X; Y) = 0$$: their MI is zero  
        * __$$X$$ deterministic function of $$Y$$ and vice versa__ $$\implies I(X; Y) = H(X) = H(Y)$$ their MI is equal to entropy of each variable  
    * It's a Measure of the inherent dependence expressed in the joint distribution of  $$X$$ and  $$Y$$ relative to the joint distribution of $$X$$ and $$Y$$ under the assumption of independence.  
        i.e. The price for encoding $${\displaystyle (X,Y)}$$ as a pair of independent random variables, when in reality they are not.  

    __Properties:__{: style="color: red"}  
    {: #lst-p}
    * The KL-divergence shows that $$I(X; Y)$$ is equal to zero precisely when <span>the joint distribution conicides with the product of the marginals i.e. when </span>{: style="color: goldenrod"} __$$X$$ and $$Y$$ are *independent*__{: style="color: goldenrod"}.  
    * The MI is __non-negative__: $$I(X; Y) \geq 0$$  
        * It is a measure of the price for encoding $${\displaystyle (X,Y)}$$ as a pair of independent random variables, when in reality they are not.  
    * It is __symmetric__: $$I(X; Y) = I(Y; X)$$  
    * __Related to conditional and joint entropies:__  
        <p>$${\displaystyle {\begin{aligned}\operatorname {I} (X;Y)&{}\equiv \mathrm {H} (X)-\mathrm {H} (X|Y)\\&{}\equiv \mathrm {H} (Y)-\mathrm {H} (Y|X)\\&{}\equiv \mathrm {H} (X)+\mathrm {H} (Y)-\mathrm {H} (X,Y)\\&{}\equiv \mathrm {H} (X,Y)-\mathrm {H} (X|Y)-\mathrm {H} (Y|X)\end{aligned}}}$$</p>  
        where $$\mathrm{H}(X)$$ and $$\mathrm{H}(Y)$$ are the marginal entropies, $$\mathrm{H}(X | Y)$$ and $$\mathrm{H}(Y | X)$$ are the conditional entopries, and $$\mathrm{H}(X, Y)$$ is the joint entropy of $$X$$ and $$Y$$.  
        * Note the _analogy to the **union, difference, and intersection of two sets**_:  
            ![img](https://cdn.mathpix.com/snip/images/aT2_JfK4TlRP9b5JawVqQigLD7dzxOrFjDIapoSF-F4.original.fullsize.png){: width="35%" .center-image}  
    * __Related to KL-div of conditional distribution:__  
        <p>$$\mathrm{I}(X ; Y)=\mathbb{E}_{Y}\left[D_{\mathrm{KL}}\left(p_{X | Y} \| p_{X}\right)\right]$$</p>  
    * [MI (tutorial #1)](https://www.youtube.com/watch?v=U9h1xkNELvY)  
    * [MI (tutorial #2)](https://www.youtube.com/watch?v=d7AUaut6hso)  


    __Applications:__{: style="color: red"}  
    <button>Show Lists</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    * In search engine technology, mutual information between phrases and contexts is used as a feature for k-means clustering to discover semantic clusters (concepts)  
    * Discriminative training procedures for hidden Markov models have been proposed based on the maximum mutual information (MMI) criterion.
    * Mutual information has been used as a criterion for feature selection and feature transformations in machine learning. It can be used to characterize both the relevance and redundancy of variables, such as the minimum redundancy feature selection.
    * Mutual information is used in determining the similarity of two different clusterings of a dataset. As such, it provides some advantages over the traditional Rand index.
    * Mutual information of words is often used as a significance function for the computation of collocations in corpus linguistics.  
    * Detection of phase synchronization in time series analysis
    * The mutual information is used to learn the structure of Bayesian networks/dynamic Bayesian networks, which is thought to explain the causal relationship between random variables  
    * Popular cost function in decision tree learning.
    * In the infomax method for neural-net and other machine learning, including the infomax-based Independent component analysis algorithm
    {: hidden=""}



    __Independence assumptions and low-rank matrix approximation (alternative definition):__{: style="color: red"}  
    <button>show analysis</button>{: .showText value="show"  
     onclick="showTextPopHide(event);"}
     ![img](https://cdn.mathpix.com/snip/images/jzmGBSoIKS4x2IrykIaQR3P21Y2z8RS_VmZNKkOfjZQ.original.fullsize.png){: width="100%" hidden=""}  


    __As a Metric (relation to Jaccard distance):__{: style="color: red"}  
    <button>show analysis</button>{: .showText value="show"  
     onclick="showTextPopHide(event);"}
    ![img](https://cdn.mathpix.com/snip/images/dOB7qr575sMswJ1MPEbHmFvlqvJp8ncf9ulYlR4aKDY.original.fullsize.png){: width="100%" hidden=""}
    <br>


9. **Pointwise Mutual Information (PMI):**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents19}  
    The PMI of a pair of outcomes $$x$$ and $$y$$ belonging to discrete random variables $$X$$ and $$Y$$ quantifies the discrepancy between the probability of their coincidence given their joint distribution and their individual distributions, assuming independence. Mathematically:  
    <p>$$\operatorname{pmi}(x ; y) \equiv \log \frac{p(x, y)}{p(x) p(y)}=\log \frac{p(x | y)}{p(x)}=\log \frac{p(y | x)}{p(y)}$$</p>  
    In contrast to mutual information (MI) which builds upon PMI, it refers to single events, whereas MI refers to the average of all possible events.  
    The mutual information (MI) of the random variables $$X$$ and $$Y$$ is the expected value of the PMI (over all possible outcomes).  

### More
* __Conditional Entropy__: $$H(X \mid Y)=H(X)-I(X, Y)$$  
* __Independence__: $$I(X, Y)=0$$
* __Independence Relations__:  $$H(X \mid Y)=H(X)$$



***
***



# DATA PROCESSING

* [Data Wrangling Techniques (Blog!)](https://theprofessionalspoint.blogspot.com/2019/03/data-wrangling-techniques-steps.html)  

* [Non-Negative Matrix Factorization NMF Tutorial](http://mlexplained.com/2017/12/28/a-practical-introduction-to-nmf-nonnegative-matrix-factorization/)  
* [How to Use t-SNE Effectively (distill blog!)](https://distill.pub/2016/misread-tsne/)  

* [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction [better than t-sne?] (Library Code!)](https://umap-learn.readthedocs.io/en/latest/)  


## Dimensionality Reduction
{: style="font-size: 1.60em"}
{: #content1}


### **Dimensionality Reduction**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents1 #bodyContents11}  
__Dimensionality Reduction__ is the process of reducing the number of random variables under consideration by obtaining a set of principal variables. It can be divided into __feature selection__{: style="color: goldenrod"} and __feature extraction__{: style="color: goldenrod"}.  
<br>

**Dimensionality Reduction Methods:**
{: #lst-p}
* PCA
* Heatmaps
* t-SNE
* Multi-Dimensional Scaling (MDS)

### **t-SNE \| T-distributed Stochastic Neighbor Embeddings**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents1 #bodyContents12}  

<button>Paper</button>{: .showText value="show"
     onclick="showText_withParent_PopHide(event);"}
<iframe hidden="" src="https://docs.google.com/viewer?url=http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf&amp;embedded=true" frameborder="0" height="840" width="646" title="Layer Normalization" scrolling="auto"></iframe>

[Understanding t-SNE Part 1: SNE algorithm and its drawbacks](https://medium.com/@layog/i-dont-understand-t-sne-part-1-50f507acd4f9)  
[Understanding t-SNE Part 2: t-SNE improvements over SNE](https://medium.com/@layog/i-do-not-understand-t-sne-part-2-b2f997d177e3)  
[t-SNE (statwiki)](https://wiki.math.uwaterloo.ca/statwiki/index.php?title=visualizing_Data_using_t-SNE)  
[t-SNE tutorial (video)](https://www.youtube.com/watch?v=W-9L6v_rFIE)  
[series (deleteme)](https://www.youtube.com/watch?v=FQmCzpKWD48&list=PLupD_xFct8mHqCkuaXmeXhe0ajNDu0mhZ)  



__SNE - Stochastic Neighbor Embeddings:__{: style="color: red"}  
__SNE__ is a method that aims to _match_ __distributions of distances__ between points in high and low dimensional space via __conditional probabilities__.  
It Assumes distances in both high and low dimensional space are __Gaussian-distributed__.  
* [**Algorithm**](https://www.youtube.com/embed/ohQXphVSEQM?start=130){: value="show" onclick="iframePopA(event)"}
<a href="https://www.youtube.com/embed/ohQXphVSEQM?start=130"></a>
    <div markdown="1"> </div>    
    ![img](/main_files/dl/concepts/data_proc/2.png){: width="65%"}  
<br> 


__t-SNE:__{: style="color: red"}  
__t-SNE__ is a machine learning algorithm for visualization developed by Laurens van der Maaten and Geoffrey Hinton.  
It is a *__nonlinear__* *__dimensionality reduction__* technique well-suited for _embedding high-dimensional data for visualization in a low-dimensional space_ of _two or three dimensions_.  
Specifically, it models each high-dimensional object by a two- or three-dimensional point in such a way that <span>similar objects are modeled by nearby points</span>{: style="color: goldenrod"} and <span>dissimilar objects are modeled by distant points</span>{: style="color: goldenrod"}  __with high probability__.  
> It tends to *preserve __local structure__*, while at the same time, *preserving the __global structure__* as much as possible.  

<br>

__Stages:__{: style="color: red"}  
{: #lst-p}
1. It Constructs a probability distribution over pairs of high-dimensional objects in such a way that similar objects have a high probability of being picked while dissimilar points have an extremely small probability of being picked.  
2. It Defines a similar probability distribution over the points in the low-dimensional map, and it minimizes the __Kullback–Leibler divergence__ between the two distributions with respect to the locations of the points in the map.  
<br>

__Key Ideas:__{: style="color: red"}  
It solves two big problems that __SNE__ faces:  
{: #lst-p}
1. __The Crowding Problem:__  
    The "crowding problem" that are addressed in the paper is defined as: "the area of the two-dimensional map that is available to accommodate moderately distant datapoints will not be nearly large enough compared with the area available to accommodate nearby datepoints". This happens when the datapoints are distributed in a region on a high-dimensional manifold around i, and we try to model the pairwise distances from i to the datapoints in a two-dimensional map. For example, it is possible to have 11 datapoints that are mutually equidistant in a ten-dimensional manifold but it is not possible to model this faithfully in a two-dimensional map. Therefore, if the small distances can be modeled accurately in a map, most of the moderately distant datapoints will be too far away in the two-dimensional map. In SNE, this will result in very small attractive force from datapoint i to these too-distant map points. The very large number of such forces collapses together the points in the center of the map and prevents gaps from forming between the natural clusters. This phenomena, crowding problem, is not specific to SNE and can be observed in other local techniques such as Sammon mapping as well.  
    * __Solution - Student t-distribution for $$q$$__:  
        Student t-distribution is used to compute the similarities between data points in the low dimensional space $$q$$.  
2. __Optimization Difficulty of KL-div:__  
    The KL Divergence is used over the conditional probability to calculate the error in the low-dimensional representation. So, the algorithm will be trying to minimize this loss and will calculate its gradient:  
    <p>$$\frac{\delta C}{\delta y_{i}}=2 \sum_{j}\left(p_{j | i}-q_{j | i}+p_{i | j}-q_{i | j}\right)\left(y_{i}-y_{j}\right)$$</p>  
    This gradient involves all the probabilities for point $$i$$ and $$j$$. But, these probabilities were composed of the exponentials. The problem is that: We have all these exponentials in our gradient, which can explode (or display other unusual behavior) very quickly and hence the algorithm will take a long time to converge.  
    * __Solution - Symmetric SNE__:  
        The Cost Function is a __symmetrized__ version of that in SNE. i.e. $$p_{i\vert j} = p_{j\vert i}$$ and $$q_{i\vert j} = q_{j\vert i}$$.  
<br>

__Application:__{: style="color: red"}  
It is often used to visualize high-level representations learned by an __artificial neural network__.  
<br>

__Motivation:__{: style="color: red"}  
There are a lot of problems with traditional dimensionality reduction techniques that employ _feature projection_; e.g. __PCA__. These techniques attempt to *__preserve the global structure__*, and in that process they *__lose the local structure__*. Mainly, projecting the data on one axis or another, may (most likely) not preserve the _neighborhood structure_ of the data; e.g. the clusters in the data:  
![img](/main_files/dl/concepts/data_proc/1.png){: width="70%"}  
t-SNE finds a way to project data into a low dimensional space (1-d, in this case) such that the clustering ("local structure") in the high dimensional space is preserved.  
<br>


__t-SNE Clusters:__{: style="color: red"}  
While t-SNE plots often seem to display clusters, the visual clusters can be influenced strongly by the chosen parameterization and therefore a good understanding of the parameters for t-SNE is necessary. Such "clusters" can be shown to even appear in non-clustered data, and thus may be false findings.  
It has been demonstrated that t-SNE is often able to _recover well-separated clusters_, and with special parameter choices, [approximates a simple form of __spectral clustering__](https://arxiv.org/abs/1706.02582).  
<br>

__Properties:__{: style="color: red"}  
{: #lst-p}
* It preserves the _neighborhood structure_ of the data  
* Does NOT preserve _distances_ nor _density_  
* Only to some extent preserves _nearest-neighbors_?  
    [discussion](https://stats.stackexchange.com/questions/263539/clustering-on-the-output-of-t-sne/264647#264647)  
* It learns a __non-parametric mapping__, which means that it does NOT learn an _explicit function_ that maps data from the input space to the map  
<br>

__Algorithm:__{: style="color: red"}  
<button>Algorithm Details (wikipedia)</button>{: .showText value="show"
     onclick="showText_withParent_PopHide(event);"}
<iframe hidden="" src="https://www.wikiwand.com/en/T-distributed_stochastic_neighbor_embedding#/Details" frameborder="0" height="840" width="646" title="Layer Normalization"></iframe>

<br>

__Issues/Weaknesses/Drawbacks:__{: style="color: red"}  
{: #lst-p}
1. The paper only focuses on the date visualization using t-SNE, that is, embedding high-dimensional date into a two- or three-dimensional space. However, this behavior of t-SNE presented in the paper cannot readily be extrapolated to $$d>3$$ dimensions due to the heavy tails of the Student t-distribution.  
2. It might be less successful when applied to data sets with a high intrinsic dimensionality. This is a result of the *__local linearity assumption__ on the manifold* that t-SNE makes by employing Euclidean distance to present the similarity between the datapoints. 
3. The cost function is __not convex__. This leads to the problem that several optimization parameters (hyperparameters) need to be chosen (and tuned) and the constructed solutions depending on these parameters may be different each time t-SNE is run from an initial random configuration of the map points.  
4. It cannot work __"online"__. Since it learns a non-parametric mapping, which means that it does not learn an explicit function that maps data from the input space to the map. Therefore, it is not possible to embed test points in an existing map. You have to re-run t-SNE on the full dataset.  
    A potential approach to deal with this would be to train a multivariate regressor to predict the map location from the input data.  
    Alternatively, you could also [make such a regressor minimize the t-SNE loss directly (parametric t-SNE)](https://lvdmaaten.github.io/publications/papers/AISTATS_2009.pdf).  

<br>

__t-SNE Optimization:__{: style="color: red"}  
{: #lst-p}
* [Accelerating t-SNE using Tree-Based Algorithms](https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf)  
* [Barnes-Hut-SNE Optimization](https://arxiv.org/pdf/1301.3342.pdf)  

<br>

__Discussion and Information:__{: style="color: red"}  
{: #lst-p}
* __What is perplexity?__  
    Perplexity is a measure for information that is defined as 2 to the power of the Shannon entropy. The perplexity of a fair die with k sides is equal to k. In t-SNE, the perplexity may be viewed as a knob that sets the number of effective nearest neighbors. It is comparable with the number of nearest neighbors k that is employed in many manifold learners.  
* __Choosing the perplexity hp:__   
    The performance of t-SNE is fairly robust under different settings of the perplexity. The most appropriate value depends on the density of your data. Loosely speaking, one could say that a larger / denser dataset requires a larger perplexity. Typical values for the perplexity range between $$5$$ and $$50$$.  
* __Every time I run t-SNE, I get a (slightly) different result?__  
    In contrast to, e.g., PCA, t-SNE has a non-convex objective function. The objective function is minimized using a gradient descent optimization that is initiated randomly. As a result, it is possible that different runs give you different solutions. Notice that it is perfectly fine to run t-SNE a number of times (with the same data and parameters), and to select the visualization with the lowest value of the objective function as your final visualization.  
* __Assessing the "Quality of Embeddings/visualizations":__  
    Preferably, just look at them! Notice that t-SNE does not retain distances but probabilities, so measuring some error between the Euclidean distances in high-D and low-D is useless. However, if you use the same data and perplexity, you can compare the Kullback-Leibler divergences that t-SNE reports. It is perfectly fine to run t-SNE ten times, and select the solution with the lowest KL divergence.  
        



<!-- __Advantages:__{: style="color: red"}  
{: #lst-p}
1. Reduces time and 

### **Feature Selection**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents1 #bodyContents12}   -->
<br>

<!--  
### **Asynchronous**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents1 #bodyContents13}  
<br>
### **Asynchronous**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents1 #bodyContents14}  
<br>
### **Asynchronous**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents1 #bodyContents15}  
<br>
### **Asynchronous**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents1 #bodyContents16}  
<br>
### **Asynchronous**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents1 #bodyContents17}  
<br>
### **Asynchronous**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents1 #bodyContents18}  
 -->

***
***

## Feature Selection
{: style="font-size: 1.60em"}
{: #content2}


### **Feature Selection**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents2 #bodyContents21}  
__Feature Selection__ is the process of selecting a subset of relevant features (variables, predictors) for use in model construction.  

__Applications:__{: style="color: red"}  
{: #lst-p}
* Simplification of models to make them easier to interpret by researchers/users  
* Shorter training time  
* A way to handle _curse of dimensionality_  
* Reduction of Variance $$\rightarrow$$ Reduce Overfitting $$\rightarrow$$ Enhanced Generalization  

__Strategies/Approaches:__{: style="color: red"}  
{: #lst-p}
* __Wrapper Strategy__:  
    Wrapper methods use a predictive model to score feature subsets. Each new subset is used to train a model, which is tested on a hold-out set. Counting the number of mistakes made on that hold-out set (the error rate of the model) gives the score for that subset. As wrapper methods train a new model for each subset, they are very computationally intensive, but usually provide the best performing feature set for that particular type of model.  
    __e.g.__ __Search Guided by Accuracy__{: style="color: goldenrod"}, __Stepwise Selection__{: style="color: goldenrod"}   
* __Filter Strategy__:  
    Filter methods use a _proxy measure_ instead of the error rate _to score a feature subset_. This measure is chosen to be fast to compute, while still capturing the usefulness of the feature set.  
    Filter methods produce a feature set which is _not tuned to a specific model_, usually giving lower prediction performance than a wrapper, but are more general and more useful for exposing the relationships between features.  
    __e.g.__ __Information Gain__{: style="color: goldenrod"}, __pointwise-mutual/mutual information__{: style="color: goldenrod"}, __Pearson Correlation__{: style="color: goldenrod"}    
* __Embedded Strategy:__  
    Embedded methods are a catch-all group of techniques which perform feature selection as part of the model construction process.  
    __e.g.__ __LASSO__{: style="color: goldenrod"}  


<br>

### **Correlation Feature Selection**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents2 #bodyContents22}  
The __Correlation Feature Selection (CFS)__ measure evaluates subsets of features on the basis of the following hypothesis:  
"__Good feature subsets contain features highly correlated with the classification, yet uncorrelated to each other__{: style="color: goldenrod"}".  

The following equation gives the __merit of a feature subset__ $$S$$ consisting of $$k$$ features:  
<p>$${\displaystyle \mathrm {Merit} _{S_{k}}={\frac {k{\overline {r_{cf}}}}{\sqrt {k+k(k-1){\overline {r_{ff}}}}}}.}$$</p>  
where, $${\displaystyle {\overline {r_{cf}}}}$$ is the average value of all feature-classification correlations, and $${\displaystyle {\overline {r_{ff}}}}$$ is the average value of all feature-feature correlations.  

The __CFS criterion__ is defined as follows:  
<p>$$\mathrm {CFS} =\max _{S_{k}}\left[{\frac {r_{cf_{1}}+r_{cf_{2}}+\cdots +r_{cf_{k}}}{\sqrt {k+2(r_{f_{1}f_{2}}+\cdots +r_{f_{i}f_{j}}+\cdots +r_{f_{k}f_{1}})}}}\right]$$</p>  

<br>

### **Feature Selection Embedded in Learning Algorithms**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents2 #bodyContents23}  
* $$l_{1}$$-regularization techniques, such as sparse regression, LASSO, and $${\displaystyle l_{1}}$$-SVM
* Regularized trees, e.g. regularized random forest implemented in the RRF package
* Decision tree
* Memetic algorithm
* Random multinomial logit (RMNL)
* Auto-encoding networks with a bottleneck-layer
* Submodular feature selection

<br>

### **Information Theory Based Feature Selection Mechanisms**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents2 #bodyContents24}  
There are different Feature Selection mechanisms around that __utilize mutual information for scoring the different features__.  
They all usually use the same algorithm:  
1. Calculate the mutual information as score for between all features ($${\displaystyle f_{i}\in F}$$) and the target class ($$c$$)
1. Select the feature with the largest score (e.g. $${\displaystyle argmax_{f_{i}\in F}(I(f_{i},c))}$$) and add it to the set of selected features ($$S$$)
1. Calculate the score which might be derived form the mutual information
1. Select the feature with the largest score and add it to the set of select features (e.g. $${\displaystyle {\arg \max }_{f_{i}\in F}(I_{derived}(f_{i},c))}$$)
5. Repeat 3. and 4. until a certain number of features is selected (e.g. $${\displaystyle \vert S\vert =l}$$)  


<!-- <br> ### **Asynchronous**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents2 #bodyContents25}  
### **Asynchronous**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents2 #bodyContents26}   -->

***
***

## Feature Extraction
{: style="font-size: 1.60em"}
{: #content3}

### **Feature Extraction**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents3 #bodyContents31}  
__Feature Extraction__ starts from an initial set of measured data and builds derived values (features) intended to be informative and non-redundant, facilitating the subsequent learning and generalization steps, and in some cases leading to better human interpretations.  

In __dimensionality reduction__, feature extraction is also called __Feature Projection__, which is a method that transforms the data in the high-dimensional space to a space of fewer dimensions. The data transformation may be linear, as in principal component analysis (PCA), but many nonlinear dimensionality reduction techniques also exist.  

__Methods/Algorithms:__{: style="color: red"}  
{: #lst-p}
* Independent component analysis  
* Isomap  
* Kernel PCA  
* Latent semantic analysis  
* Partial least squares  
* Principal component analysis  
* Autoencoder  
* Linear Discriminant Analysis (LDA)  
* Non-negative matrix factorization (NMF)


<br>



### **Data Imputation**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents3 #bodyContents32}  


__Resources:__  
{: #lst-p}
* [Imputation Solutions (Product)](https://www.interpretable.ai/products/optimpute/)  
* [Robust Data Pipeline Design (Product/Case)](https://www.interpretable.ai/solutions/data-pipeline/)  


<!-- ### ****{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents3 #bodyContents32}  
### ****{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents3 #bodyContents33}   -->

* [How to Make Your Machine Learning Models Robust to Outliers (Blog!)](https://heartbeat.fritz.ai/how-to-make-your-machine-learning-models-robust-to-outliers-44d404067d07)  



[Outliers](https://en.wikipedia.org/wiki/Outlier#Working_with_outliers)  
[Replacing Outliers](https://en.wikipedia.org/wiki/Robust_statistics#Replacing_outliers_and_missing_values)  
[Data Transformation - Outliers - Standardization](https://en.wikipedia.org/wiki/Data_transformation_(statistics))  
[PreProcessing in DL - Data Normalization](https://hadrienj.github.io/posts/Preprocessing-for-deep-learning/)  
[Imputation and Feature Scaling](https://towardsdatascience.com/the-complete-beginners-guide-to-data-cleaning-and-preprocessing-2070b7d4c6d)  
[Missing Data - Imputation](https://en.wikipedia.org/wiki/Missing_data#Techniques_of_dealing_with_missing_data)  
[Dim-Red - Random Projections](https://en.wikipedia.org/wiki/Random_projection)  
[F-Selection - Relief](https://en.wikipedia.org/wiki/Relief_(feature_selection))  
[Box-Cox Transf - outliers](https://www.statisticshowto.datasciencecentral.com/box-cox-transformation/)  
[ANCOVA](https://en.wikipedia.org/wiki/Analysis_of_covariance)  
[Feature Selection Methods](https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/)  



***
***

# PRactiical Concepts

* [A Cookbook for Machine Learning: Vol 1 (Blog!)](https://www.inference.vc/design-patterns/)  
    * [Reddit Blog](https://www.reddit.com/r/MachineLearning/comments/7dd45h/d_a_cookbook_for_machine_learning_a_list_of_ml/)  
* [Deep Learning Cookbook (book)](http://noracook.io/Books/MachineLearning/deeplearningcookbook.pdf)  


<!-- ## FIRST
{: #content1} --> 

1. **Data Snooping:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  

    __The Principle:__  
    If a data set has __*affected*__ any step in the __learning process__, its __ability to *assess the outcome*__ has been compromised.  

    __Analysis:__  
    {: #lst-p}
    * Making decision by __examining the dataset__ makes *__you__* a part of the learning algorithm.  
        However, you didn't consider your contribution to the learning algorithm when making e.g. VC-Analysis for Generalization.  
    * Thus, you are __vulnerable__ to designing the model (or choices of learning) according to the *__idiosyncrasies__* of the __dataset__.  
    * The real problem is that you are not _"charging" for the decision you made by examining the dataset_.    

    __What's allowed?__  
    {: #lst-p}
    * You are allowed (even encouraged) to look at all other information related to the __target function__ and __input space__.  
        e.g. number/range/dimension/scale/etc. of the inputs, correlations, properties (monotonicity), etc.  
    * EXCEPT, for the __*specific* realization of the training dataset__.  


    __Manifestations of Data Snooping with Examples (one/manifestation):__{: style="color: red"}  
    {: #lst-p}
    * __Changing the Parameters of the model (Tricky)__:  
        * __Complexity__:  
            Decreasing the order of the fitting polynomial by observing geometric properties of the __training set__.  
    * __Using statistics of the Entire Dataset (Tricky)__:  
        * __Normalization__:  
            Normalizing the data with the mean and variance of the __entire dataset (training+testing)__.  
            * E.g. In Financial Forecasting; the average affects the outcome by exposing the trend.  
    * __Reuse of a Dataset__:  
        If you keep Trying one model after the other *on the* __same data set__, you will eventually 'succeed'.  
        _"If you torture the data long enough, it will confess"_.  
        This bad because the final model you selected, is the __*union* of all previous models__: since some of those models were *__rejected__* by __you__ (a *__learning algorithm__*).  
        * __Fixed (deterministic) training set for Model Selection__:  
            Selecting a model by trying many models on the __same *fixed (deterministic)* Training dataset__.  
    * __Bias via Snooping__:  
        By looking at the data in the future when you are not allowed to have the data (it wouldn't have been possible); you are creating __sampling bias__ caused by _"snooping"_.  
        * E.g. Testing a __Trading__ algorithm using the *__currently__* __traded companies__ (in S&P500).  
            You shouldn't have been able to know which companies are being *__currently__* traded (future).  



    __Remedies/Solutions to Data Snooping:__{: style="color: red"}  
    {: #lst-p}
    1. __Avoid__ Data Snooping:  
        A strict discipline (very hard).  
    2. __Account for__ Data Snooping:  
        By quantifying "How much __data contamination__".  

    <br>


2. **Mismatched Data:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  

3. **Mismatched Classes:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  

4. **Sampling Bias:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    __Sampling Bias__ occurs when: $$\exists$$ Region with zero-probability $$P=0$$ in training, but with positive-probability $$P>0$$ in testing.  

    __The Principle:__  
    If the data is sampled in a biased way, learning will produce a similarly biased outcome.  

    __Example: 1948 Presidential Elections__  
    {: #lst-p}
    * Newspaper conducted a *__Telephone__* poll between: __Jackson__ and __Truman__  
    * __Jackson__ won the poll __decisively__.  
    * The result was NOT __unlucky__:  
        No matter how many times the poll was re-conducted, and no matter how many times the sample sized is increased; the outcome will be fixed.  
    * The reason is the *__Telephone__*:  
        (1) Telephones were __expensive__ and only __rich people__ had Telephones.  
        (2) Rich people favored __Jackson__.  
        Thus, the result was __well reflective__ of the (mini) population being sampled.  

    __How to sample:__{: style="color: red"}  
    Sample in a way that <span>matches the __distributions__ of __train__ and __test__</span>{: style="color: purple"} samples.  

    The solution __Fails__ (doesn't work) if:  
    $$\exists$$ Region with zero-probability $$P=0$$ in training, but with positive-probability $$P>0$$ in testing.  
    > This is when sampling bias exists.  


    __Notes:__{: style="color: red"}  
    {: #lst-p}
    * Medical sources sometimes refer to sampling bias as __ascertainment bias__.  
    * Sampling bias could be viewed as a subtype of __selection bias__.  
    <br>


5. **Model Uncertainty:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  

    __Interpreting Softmax Output Probabilities:__{: style="color: red"}  
    Softmax outputs only measure [__Aleatoric Uncertainty__](https://en.wikipedia.org/wiki/Uncertainty_quantification#Aleatoric_and_epistemic_uncertainty).  
    In the same way that in regression, a NN with two outputs, one representing mean and one variance, that parameterise a Gaussian, can capture aleatoric uncertainty, even though the model is deterministic.  
    Bayesian NNs (dropout included), aim to capture epistemic (aka model) uncertainty.  

    __Dropout for Measuring Model (epistemic) Uncertainty:__{: style="color: red"}  
    Dropout can give us principled uncertainty estimates.  
    Principled in the sense that the uncertainty estimates basically approximate those of our [Gaussian process](/work_files/research/dl/archits/nns#bodyContents13).  

    __Theoretical Motivation:__ dropout neural networks are identical to <span>variational inference in Gaussian processes</span>{: style="color: purple"}.  
    __Interpretations of Dropout:__  
    {: #lst-p}
    * Dropout is just a diagonal noise matrix with the diagonal elements set to either 0 or 1.  
    * [What My Deep Model Doesn't Know (Blog! - Yarin Gal)](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html)  
    <br>

6. **Probability Calibration:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    Modern NN are __miscalibrated__: not well-calibrated. They tend to be very confident. We cannot interpret the softmax probabilities as reflecting the true probability distribution or as a measure of confidence.  

    __Miscalibration:__ is the discrepancy between model confidence and model accuracy.  
    You assume that if a model gives $$80\%$$ confidence for 100 images, then $$80$$ of them will be accurate and the other $$20$$ will be inaccurate.  
    <button>Miscalibration in Modern Neural Networks</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](https://cdn.mathpix.com/snip/images/boMaW8Wx2tXfUYTJpd-rhcVGWnrtpC4_2AGbXPxtocc.original.fullsize.png){: width="100%" hidden=""}  

    __Model Confidence:__ probability of correctness.  
    __Calibrated Confidence (softmax scores) $$\hat{p}$$:__ $$\hat{p}$$ represents a true probability.  

    <button>Bias of Different Classical ML Models</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](https://cdn.mathpix.com/snip/images/m91-I3AcQ52sbAjr2gzeBlv_SlmZSh5Hb_knOLkTOMk.original.fullsize.png){: width="100%" hidden=""}  

    <button>Summary On Practical Use of Model Scores (sklearn)</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](https://cdn.mathpix.com/snip/images/f7hsQi4QKi0wejzS4YNwKVf_AaYVjOjqZFdt5UcSvDc.original.fullsize.png){: width="100%" hidden=""}  


    __Probability Calibration:__{: style="color: red"}  
    Predicted scores (model outputs) of many classifiers do not represent _"true" probabilities_.  
    They only respect the _mathematical definition_ (conditions) of what a probability function is:  
    1. Each "probability" is between 0 and 1  
    2. When you sum the probabilities of an observation being in any particular class, they sum to 1.  

    * __Calibration Curves__: A calibration curve plots the predicted probabilities against the actual rate of occurance.  
        I.E. It plots the *__predicted__* probabilities against the *__actual__* probabilities.  
        <button>Example: Rain Prediction with Naive Bayes Model</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        ![img](https://cdn.mathpix.com/snip/images/SgB0b51NGZ0_McTgF0d25jWXx43A-cA8MSSco6jvzZA.original.fullsize.png){: width="100%" hidden=""}  



    * __Approach__:  
        Calibrating a classifier consists of fitting a regressor (called a calibrator) that maps the output of the classifier (as given by ```decision_function``` or ```predict_proba``` - sklearn) to a calibrated probability in $$[0, 1]$$.  
        Denoting the output of the classifier for a given sample by $$f_i$$, the calibrator tries to predict $$p\left(y_i=1 \mid f_i\right)$$.  



    * [__Methods__](https://scikit-learn.org/stable/modules/calibration.html):  
        * __Platt Scaling__: Platt scaling basically fits a logistic regression on the original model's.  
            The closer the calibration curve is to a sigmoid, the more effective the scaling will be in correcting the model.  
            <button>Model Definition</button>{: .showText value="show" onclick="showTextPopHide(event);"}
            ![img](https://cdn.mathpix.com/snip/images/F_DDq98LBIJPNFg6AjRQI4OpJJ_ozb4ZM1NdrOaxfOk.original.fullsize.png){: width="100%" hidden=""}  

            * __Assumptions__:  
                The sigmoid method assumes the calibration curve can be corrected by applying a sigmoid function to the raw predictions.  
                This assumption has been empirically justified in the case of __Support Vector Machines__ with __common kernel functions__ on various benchmark datasets but does not necessarily hold in general.  

            * __Limitations__:  
                * The logistic model works best if the __calibration error__ is *__symmetrical__*, meaning the classifier output for each binary class is *__normally distributed__* with the *__same variance__*.  
                    This can be a problem for highly imbalanced classification problems, where outputs do not have equal variance.  


        * __Isotonic Method__:  The ‘isotonic’ method fits a non-parametric isotonic regressor, which outputs a step-wise non-decreasing function.  
            <button>Objective/Loss</button>{: .showText value="show" onclick="showTextPopHide(event);"}
            ![img](https://cdn.mathpix.com/snip/images/XEjg4c6wis3M51_xsrTRzg00BRdtFK8_4CNOd7IcZ-I.original.fullsize.png){: width="100%" hidden=""}  

            This method is more general when compared to ‘sigmoid’ as the only restriction is that the mapping function is monotonically increasing. It is thus more powerful as it can correct any monotonic distortion of the un-calibrated model. However, it is more prone to overfitting, especially on small datasets.  
        
        * __Comparison:__{: style="color: blue"}  
            * Platt Scaling is most effective when the un-calibrated model is under-confident and has similar calibration errors for both high and low outputs.  
            * Isotonic Method is more powerful than Platt Scaling:  Overall, ‘isotonic’ will perform as well as or better than ‘sigmoid’ when there is enough data (greater than ~ 1000 samples) to avoid overfitting.  


        

    * [Limitations of recalibration:](https://kiwidamien.github.io/are-you-sure-thats-a-probability.html#Limitations-of-recalibration)  
        Different calibration methods have different weaknesses depending on the shape of the _calibration curve_.  
        E.g. _Platt Scaling_ works better the more the _calibration curve_ resembles a *__sigmoid__*.  

        <button>Example of Platt Scaling Failure</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        ![img](https://cdn.mathpix.com/snip/images/I8sRhwL5JnmbjJ39hRcuIc5jomlUD4O2rrC1wMA6H6M.original.fullsize.png){: width="100%" hidden=""}  


    * [__Multi-Class Support:__](https://scikit-learn.org/stable/modules/calibration.html#multiclass-support){: style="color: blue"}  


    __Note:__ The samples that are used to fit the calibrator should not be the same samples used to fit the classifier, as this would introduce bias. This is because performance of the classifier on its training data would be better than for novel data. Using the classifier output of training data to fit the calibrator would thus result in a biased calibrator that maps to probabilities closer to 0 and 1 than it should.  





    * [On Calibration of Modern Neural Networks](https://arxiv.org/pdf/1706.04599.pdf)    
        Paper that defines the problem and gives multiple effective solution for calibrating Neural Networks. 
    * [Calibration of Convolutional Neural Networks (Thesis!)](file:///Users/ahmadbadary/Downloads/Kängsepp_ComputerScience_2018.pdf)  
    * For calibrating output probabilities in Deep Nets; Temperature scaling outperforms Platt scaling. [paper](https://arxiv.org/pdf/1706.04599.pdf)  
    * [Plot and Explanation](https://scikit-learn.org/stable/modules/calibration.html)  
    * [Blog on How to do it](http://alondaks.com/2017/12/31/the-importance-of-calibrating-your-deep-model/)  
    * [Interpreting outputs of a logistic classifier (Blog)](https://kiwidamien.github.io/are-you-sure-thats-a-probability.html)    
    <br>

7. **Debugging Strategies for Deep ML Models:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  
    <button>Strategies</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    1. __Visualize the model in action:__  
        Directly observing qualitative results of a model (e.g. located objects, generated speech) can help avoid __evaluation bugs__ or __mis-leading evaluation results__. It can also help guide the expected quantitative performance of the model.  
    2. __Visualize the worst mistakes:__  
        By viewing the training set examples that are the hardest to model correctly by using a confidence measure (e.g. softmax probabilities), one can often discover problems with the way the data have been __preprocessed__ or __labeled__.  
    3. __Reason about Software using Training and Test *Error*:__  
        It is hard to determine whether the underlying software is correctly implemented.  
        We can use the training/test errors to help guide us:  
        * If training error is low but test error is high, then:  
            * it is likely that that the training procedure works correctly,and the model is overfitting for fundamental algorithmic reasons.  
            * or that the test error is measured incorrectly because of a problem with saving the model after training then reloading it for test set evaluation, or because the test data was prepared differently from the training data.  
        * If both training and test errors are high, then:  
            it is difficult to determine whether there is a software defect or whether the model is underfitting due to fundamental algorithmic reasons.  
            This scenario requires further tests, described next.  
    3. __Fit a *Tiny Dataset:*__  
        If you have high error on the training set, determine whether it is due to genuine underfitting or due to a software defect.  
        Usually even small models can be guaranteed to be able fit a suﬃciently small dataset. For example, a classification dataset with only one example can be fit just by setting the biase sof the output layer correctly.  
        This test can be extended to a small dataset with few examples.  
    4. __Monitor histograms of *Activations* and *Gradients:*__  
        It is often useful to visualize statistics of neural network activations and gradients, collected over a large amount of training iterations (maybe one epoch).  
        The __preactivation value__ of __hidden units__ can tell us if the units <span>__saturate__</span>{: style="color: purple"}, or how often they do.  
        For example, for rectifiers,how often are they off? Are there units that are always off?  
        For tanh units,the average of the absolute value of the preactivations tells us how saturated the unit is.  
        In a deep network where the propagated gradients quickly grow or quickly vanish, optimization may be hampered.  
        Finally, it is useful to compare the magnitude of parameter gradients to the magnitude of the parameters themselves. As suggested by Bottou (2015), we would like the magnitude of parameter updates over a minibatch to represent something like 1 percent of the magnitude of the parameter, not 50 percent or 0.001 percent (which would make the parametersmove too slowly). It may be that some groups of parameters are moving at a good pace while others are stalled. When the data is sparse (like in natural language) some parameters may be very rarely updated, and this should be kept in mind when monitoring their evolution.  
    5. Finally, many deep learning algorithms provide some sort of guarantee about the results produced at each step.  
        For example, in part III, we will see some approximate inference algorithms that work by using algebraic solutions to optimization problems.  
        Typically these can be debugged by testing each of their guarantees.Some guarantees that some optimization algorithms offer include that the objective function will never increase after one step of the algorithm, that the gradient with respect to some subset of variables will be zero after each step of the algorithm,and that the gradient with respect to all variables will be zero at convergence.Usually due to rounding error, these conditions will not hold exactly in a digital computer, so the debugging test should include some tolerance parameter. 
    {: hidden=""}
    <br>

8. **The Machine Learning Algorithm Recipe:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}  
    Nearly all deep learning algorithms can be described as particular instances of a fairly simple recipe in both Supervised and Unsupervised settings:  
    * A combination of:  
        * A specification of a dataset
        * A cost function
        * An optimization procedure
        * A model
    * __Ex: Linear Regression__  
        <button>Example</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        * A specification of a dataset:  
            The Dataset consists of $$X$$ and $$y$$.  
        * A cost function:  
            $$J(\boldsymbol{w}, b)=-\mathbb{E}_{\mathbf{x}, \mathbf{y} \sim \hat{p}_{\text {data }}} \log p_{\text {model }}(y | \boldsymbol{x})$$  
        * An optimization procedure:  
            in most cases, the optimization algorithm is defined by solving for where the gradient of the cost is zero using the normal equation.  
        * A model:  
            The Model Specification is:  
            $$p_{\text {model}}(y \vert \boldsymbol{x})=\mathcal{N}\left(y ; \boldsymbol{x}^{\top} \boldsymbol{w}+b, 1\right)$$  
        {: hidden=""}
    * __Ex: PCA__  
        <button>Example</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        * A specification of a dataset:  
            $$X$$  
        * A cost function:  
            $$J(\boldsymbol{w})=\mathbb{E}_{\mathbf{x} \sim \hat{p}_{\text {data }}}\|\boldsymbol{x}-r(\boldsymbol{x} ; \boldsymbol{w})\|_ {2}^{2}$$  
        * An optimization procedure:  
            Constrained Convex optimization or Gradient Descent.  
        * A model:  
            Defined to have $$\boldsymbol{w}$$ with __norm__ $$1$$ and __reconstruction function__ $$r(\boldsymbol{x})=\boldsymbol{w}^{\top} \boldsymbol{x} \boldsymbol{w}$$.  
        {: hidden=""}


    * __Specification of a Dataset__:  
        Could be __labeled (supervised)__ or __unlabeled (unsupervised)__.  
    * __Cost Function__:  
        The cost function typically includes at least one term that causes the learning process to perform __statistical estimation__. The most common cost function is the negative log-likelihood, so that minimizing the cost function causes maximum likelihood estimation.  
    * __Optimization Procedure__:  
        Could be __closed-form__ or __iterative__ or __special-case__.  
        If the cost function does not allow for __closed-form__ solution (e.g. if the model is specified as __non-linear__), then we usually need __iterative__ optimization algorithms e.g. __gradient descent__.  
        If the cost can't be computed for __computational problems__ then we can approximate it with an iterative numerical optimization as long as we have some way to <span>approximating its __gradients__</span>{: style="color: purple"}.   
    * __Model__:  
        Could be __linear__ or __non-linear__.  

    If a machine learning algorithm seems especially unique or hand designed, it can usually be understood as using a __special-case optimizer__.  
    Some models, such as __decision trees__ and __k-means__, require *__special-case optimizers__* because their __cost functions__ have *__flat regions__* that make them inappropriate for minimization by gradient-based optimizers.  

    Recognizing that most machine learning algorithms can be described using this recipe helps to see the different algorithms as part of a taxonomy of methods for doing related tasks that work for similar reasons, rather than as a long list of algorithms that each have separate justifications.  

    <br>



__Recall__ is more important where Overlooked Cases (False Negatives) are more costly than False Alarms (False Positive). The focus in these problems is finding the positive cases.

__Precision__ is more important where False Alarms (False Positives) are more costly than Overlooked Cases (False Negatives). The focus in these problems is in weeding out the negative cases.

* [Interview practice with P and R (Blog)](https://kiwidamien.github.io/interview-practice-with-precision-and-recall.html)  



__ROC Curve and AUC:__

Note:  
* ROC Curve only cares about the *__ordering__* of the scores, not the values.  
    * __Probability Calibration__ and ROC: The calibration doesn't change the order of the scores, it just scales them to make a better match, and the ROC score only cares about the ordering of the scores.  

* [ROC and Credit Score Example (Blog)](https://kiwidamien.github.io/what-is-a-roc-curve-a-visualization-with-credit-scores.html)  

* __AUC__: The AUC is also the probability that a randomly selected positive example has a higher score than a randomly selected negative example.

<button>AUC Reliability (Equal AUC - different models)</button>{: .showText value="show" onclick="showTextPopHide(event);"}
![img](https://cdn.mathpix.com/snip/images/GAyvZvN61xzDjklTeVepqrYYuWrXXfPnEHkNwM80p6k.original.fullsize.png){: width="100%" hidden=""}  


<button>ROC Diagonal</button>{: .showText value="show" onclick="showTextPopHide(event);"}
![img](https://cdn.mathpix.com/snip/images/GPRX_7Ca-eJdTm04-HQO1Mc8E0cLi1FbFqjnp1z04Yk.original.fullsize.png){: width="100%" hidden=""}  

<button>Comparing Thresholds on an ROC</button>{: .showText value="show" onclick="showTextPopHide(event);"}
![img](https://cdn.mathpix.com/snip/images/QPD_iV87ZGNW8Mpk-apO69uGKfiy400nrHgO8Cuc-cc.original.fullsize.png){: width="100%" hidden=""}  

<button>AUC to Compare two Classifiers</button>{: .showText value="show" onclick="showTextPopHide(event);"}
![img](https://cdn.mathpix.com/snip/images/L3X5BVmBkVbzzZmHlQRM4r69hOzfo3EFLWEvm6VeZVM.original.fullsize.png){: width="100%" hidden=""}  

<button>PR Curve</button>{: .showText value="show" onclick="showTextPopHide(event);"}
![img](https://cdn.mathpix.com/snip/images/fMMLgJHugy3Ebj7OG7Lq4X3HX5ZTGL6gzFi4IaP8hT4.original.fullsize.png){: width="100%" hidden=""}  

<button>When to use Precision instead of FPR</button>{: .showText value="show" onclick="showTextPopHide(event);"}
![img](https://cdn.mathpix.com/snip/images/lZkRZCj8kSX4xK6JcEcnm8wVVEe5856uFO5mNs0VRY8.original.fullsize.png){: width="100%" hidden=""}  


<button>Why use Precision</button>{: .showText value="show" onclick="showTextPopHide(event);"}
![img](https://cdn.mathpix.com/snip/images/IQ7O2yvLTJqG70NIzPxMdZySzEXQI39YgIcz5D4YXrQ.original.fullsize.png){: width="100%" hidden=""}  


<button>Example when Precision is favorable to FPR</button>{: .showText value="show" onclick="showTextPopHide(event);"}
![img](https://cdn.mathpix.com/snip/images/8ZcbFolG4VmYWakNzewSVx4oThTkxP1jS4pGZJ3oUbc.original.fullsize.png){: width="100%" hidden=""}  


* [ROC in Radiology (Paper)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2698108/)  
    Includes discussion for *__Partial AUC__* when only a portion of the entire ROC curve needs to be considered.  


***

## SECOND
{: #content2}

<!-- 
1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}
2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}
3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}
4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}
 -->




***
***


# Calibration

Modern NN are __miscalibrated__: not well-calibrated. They tend to be very confident. We cannot interpret the softmax probabilities as reflecting the true probability distribution or as a measure of confidence.  

__Miscalibration:__ is the discrepancy between model confidence and model accuracy.  
You assume that if a model gives $$80\%$$ confidence for 100 images, then $$80$$ of them will be accurate and the other $$20$$ will be inaccurate.  
<button>Miscalibration in Modern Neural Networks</button>{: .showText value="show" onclick="showTextPopHide(event);"}
![img](https://cdn.mathpix.com/snip/images/boMaW8Wx2tXfUYTJpd-rhcVGWnrtpC4_2AGbXPxtocc.original.fullsize.png){: width="40%"}  

__Model Confidence:__ probability of correctness.  
__Calibrated Confidence (softmax scores) $$\hat{p}$$:__ $$\hat{p}$$ represents a true probability.  

<button>Bias of Different Classical ML Models</button>{: .showText value="show" onclick="showTextPopHide(event);"}
![img](https://cdn.mathpix.com/snip/images/m91-I3AcQ52sbAjr2gzeBlv_SlmZSh5Hb_knOLkTOMk.original.fullsize.png){: width="40%"}  

<button>Summary On Practical Use of Model Scores (sklearn)</button>{: .showText value="show" onclick="showTextPopHide(event);"}
![img](https://cdn.mathpix.com/snip/images/f7hsQi4QKi0wejzS4YNwKVf_AaYVjOjqZFdt5UcSvDc.original.fullsize.png){: width="40%"}  


__Probability Calibration:__{: style="color: red"}  
Predicted scores (model outputs) of many classifiers do not represent _"true" probabilities_.  
They only respect the _mathematical definition_ (conditions) of what a probability function is:  
1. Each "probability" is between 0 and 1  
2. When you sum the probabilities of an observation being in any particular class, they sum to 1.  

* __Calibration Curves__: A calibration curve plots the predicted probabilities against the actual rate of occurance.  
    I.E. It plots the *__predicted__* probabilities against the *__actual__* probabilities.  
    <button>Example: Rain Prediction with Naive Bayes Model</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](https://cdn.mathpix.com/snip/images/SgB0b51NGZ0_McTgF0d25jWXx43A-cA8MSSco6jvzZA.original.fullsize.png){: width="40%"}  



* __Approach__:  
    Calibrating a classifier consists of fitting a regressor (called a calibrator) that maps the output of the classifier (as given by ```decision_function``` or ```predict_proba``` - sklearn) to a calibrated probability in $$[0, 1]$$.  
    Denoting the output of the classifier for a given sample by $$f_i$$, the calibrator tries to predict $$p\left(y_i=1 \mid f_i\right)$$.  



* [__Methods__](https://scikit-learn.org/stable/modules/calibration.html):  
    * __Platt Scaling__: Platt scaling basically fits a logistic regression on the original model's.  
        The closer the calibration curve is to a sigmoid, the more effective the scaling will be in correcting the model.  
        <button>Model Definition</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        ![img](https://cdn.mathpix.com/snip/images/F_DDq98LBIJPNFg6AjRQI4OpJJ_ozb4ZM1NdrOaxfOk.original.fullsize.png){: width="40%"}  

        * __Assumptions__:  
            The sigmoid method assumes the calibration curve can be corrected by applying a sigmoid function to the raw predictions.  
            This assumption has been empirically justified in the case of __Support Vector Machines__ with __common kernel functions__ on various benchmark datasets but does not necessarily hold in general.  

        * __Limitations__:  
            * The logistic model works best if the __calibration error__ is *__symmetrical__*, meaning the classifier output for each binary class is *__normally distributed__* with the *__same variance__*.  
                This can be a problem for highly imbalanced classification problems, where outputs do not have equal variance.  


    * __Isotonic Method__:  The ‘isotonic’ method fits a non-parametric isotonic regressor, which outputs a step-wise non-decreasing function.  
        <button>Objective/Loss</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        ![img](https://cdn.mathpix.com/snip/images/XEjg4c6wis3M51_xsrTRzg00BRdtFK8_4CNOd7IcZ-I.original.fullsize.png){: width="40%"}  

        This method is more general when compared to ‘sigmoid’ as the only restriction is that the mapping function is monotonically increasing. It is thus more powerful as it can correct any monotonic distortion of the un-calibrated model. However, it is more prone to overfitting, especially on small datasets.  
    
    * __Comparison:__{: style="color: blue"}  
        * Platt Scaling is most effective when the un-calibrated model is under-confident and has similar calibration errors for both high and low outputs.  
        * Isotonic Method is more powerful than Platt Scaling:  Overall, ‘isotonic’ will perform as well as or better than ‘sigmoid’ when there is enough data (greater than ~ 1000 samples) to avoid overfitting.  


    

* [Limitations of recalibration:](https://kiwidamien.github.io/are-you-sure-thats-a-probability.html#Limitations-of-recalibration)  
    Different calibration methods have different weaknesses depending on the shape of the _calibration curve_.  
    E.g. _Platt Scaling_ works better the more the _calibration curve_ resembles a *__sigmoid__*.  

    <button>Example of Platt Scaling Failure</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](https://cdn.mathpix.com/snip/images/I8sRhwL5JnmbjJ39hRcuIc5jomlUD4O2rrC1wMA6H6M.original.fullsize.png){: width="40%"}  


* [__Multi-Class Support:__](https://scikit-learn.org/stable/modules/calibration.html#multiclass-support){: style="color: blue"}  


__Note:__ The samples that are used to fit the calibrator should not be the same samples used to fit the classifier, as this would introduce bias. This is because performance of the classifier on its training data would be better than for novel data. Using the classifier output of training data to fit the calibrator would thus result in a biased calibrator that maps to probabilities closer to 0 and 1 than it should.  





* [On Calibration of Modern Neural Networks](https://arxiv.org/pdf/1706.04599.pdf)    
    Paper that defines the problem and gives multiple effective solution for calibrating Neural Networks. 
* [Calibration of Convolutional Neural Networks (Thesis!)](file:///Users/ahmadbadary/Downloads/Kängsepp_ComputerScience_2018.pdf)  
* For calibrating output probabilities in Deep Nets; Temperature scaling outperforms Platt scaling. [paper](https://arxiv.org/pdf/1706.04599.pdf)  
* [Plot and Explanation](https://scikit-learn.org/stable/modules/calibration.html)  
* [Blog on How to do it](http://alondaks.com/2017/12/31/the-importance-of-calibrating-your-deep-model/)  
* [Interpreting outputs of a logistic classifier (Blog)](https://kiwidamien.github.io/are-you-sure-thats-a-probability.html)    
<br>



***
***



# Answers Hidden

# Data Processing and Analysis
<button>Data Processing and Analysis</button>{: .showText value="show" onclick="showText_withParent_PopHide(event);"}
1. __What are 3 data preprocessing techniques to handle outliers?__{: style="color: red"}  
    1. Winsorizing/Winsorization (cap at threshold).
    2. Transform to reduce skew (using Box-Cox or similar).
    3. Remove outliers if you're certain they are anomalies or measurement errors.
1. __Describe the strategies to dimensionality reduction?__{: style="color: red"}  
    1. Feature Selection  
    2. Feature Projection/Extraction  
1. __What are 3 ways of reducing dimensionality?__{: style="color: red"}  
    1. Removing Collinear Features
    2. Performing PCA, ICA, etc. 
    3. Feature Engineering
    4. AutoEncoder
    5. Non-negative matrix factorization (NMF)
    6. LDA
    7. MSD
1. __List methods for Feature Selection__{: style="color: red"}  
    1. Variance Threshold: normalize first (variance depends on scale)
    1. Correlation Threshold: remove the one with larger mean absolute correlation with other features.  
    1. Genetic Algorithms
    1. Stepwise Search: bad performance, regularization much better, it's a greedy algorithm (can't account for future effects of each change)    
    1. LASSO, Elastic-Net  
1. __List methods for Feature Extraction__{: style="color: red"}  
    1. PCA, ICA, CCA
    1. AutoEncoders
    1. LDA: LDA is a supervised linear transformation technique since the dependent variable (or the class label) is considered in the model. It Extracts the k new independent variables that __maximize the separation between the classes of the dependent variable__.  
        1. Linear discriminant analysis is used to find a linear combination of features that characterizes or separates two or more classes (or levels) of a categorical variable.  
        1. Unlike PCA, LDA extracts the k new independent variables that __maximize the separation between the classes of the dependent variable__. LDA is a supervised linear transformation technique since the dependent variable (or the class label) is considered in the model.  
    1. Latent Semantic Analysis
    1. Isomap
1. __How to detect correlation of "categorical variables"?__{: style="color: red"}  
    1. Chi-Squared test: it is a statistical test applied to the groups of categorical features to evaluate the likelihood of correlation or association between them using their frequency distribution.  
1. __Feature Importance__{: style="color: red"}  
    1. Use linear regression and select variables based on $$p$$ values
    1. Use Random Forest, Xgboost and plot variable importance chart
    1. Lasso
    1. Measure information gain for the available set of features and select top $$n$$ features accordingly.
    1. Use Forward Selection, Backward Selection, Stepwise Selection
    1. Remove the correlated variables prior to selecting important variables
    1. In linear models, feature importance can be calculated by the scale of the coefficients  
    1. In tree-based methods (such as random forest), important features are likely to appear closer to the root of the tree. We can get a feature's importance for random forest by computing the averaging depth at which it appears across all trees in the forest   
1. __Capturing the correlation between continuous and categorical variable? If yes, how?__{: style="color: red"}  
    Yes, we can use ANCOVA (analysis of covariance) technique to capture association between continuous and categorical variables.  
    [ANCOVA Explained](https://www.youtube.com/watch?v=a61mkzQRf6c&t=2s)  
1. __What cross validation technique would you use on time series data set?__{: style="color: red"}  
    [Forward chaining strategy](https://en.wikipedia.org/wiki/Forward_chaining) with k folds.  
1. __How to deal with missing features? (Imputation?)__{: style="color: red"}  
    1. Assign a unique category to missing values, who knows the missing values might decipher some trend.  
    2. Remove them blatantly
    3. we can sensibly check their distribution with the target variable, and if found any pattern we’ll keep those missing values and assign them a new category while removing others.  
1. __Do you suggest that treating a categorical variable as continuous variable would result in a better predictive model?__{: style="color: red"}  
    For better predictions, categorical variable can be considered as a continuous variable only when the variable is ordinal in nature.  
1. __What are collinearity and multicollinearity?__{: style="color: red"}  
    1. __Collinearity__ occurs when two predictor variables (e.g., $$x_1$$ and $$x_2$$) in a multiple regression have some correlation.  
    1. __Multicollinearity__ occurs when more than two predictor variables (e.g., $$x_1, x_2, \text{ and } x_3$$) are inter-correlated.  
1. __What is data normalization and why do we need it?__{: style="color: red"}  
    ![img](https://cdn.mathpix.com/snip/images/8aNuJetgTgCtv4pvqaI0dr96pDyUmfuX_d1aLK1lmaw.original.fullsize.png){: width="80%"}  
{: hidden=""}

***


# ML/Statistical Models
<button>ML/Statistical Models</button>{: .showText value="show" onclick="showText_withParent_PopHide(event);"}
1. __What are parametric models?__{: style="color: red"}  
    Parametric models are those with a finite number of parameters. To predict new data, you only need to know the parameters of the model. Examples include linear regression, logistic regression, and linear SVMs.
1. __What is a classifier?__{: style="color: red"}  
    A function that maps... 
{: hidden=""}

***



***
***


# Effect Size

list two methods for calculating the effect size in applied machine learning and when they might be useful

There are several different methods for calculating the effect size in applied machine learning, and which one is most appropriate to use depends on the specific context and type of data being analyzed. Some common methods for calculating effect size include:

Cohen's d: This method is commonly used when comparing the means of two groups, such as in a controlled experiment where one group receives a treatment and the other serves as a control. It is calculated by taking the difference between the two group means and dividing by the pooled standard deviation.

Pearson's r: This method is commonly used when analyzing the relationship between two continuous variables, such as the relationship between a person's height and weight. It is calculated by taking the covariance of the two variables and dividing by the product of their standard deviations.

These methods might be useful in applied machine learning when you are trying to evaluate the performance of a model or the impact of a specific feature on the outcome of a model. For example, if you are comparing the performance of two different machine learning algorithms on a classification task, you might use Cohen's d to compare the mean accuracy of the two algorithms and determine which one is more effective. Alternatively, if you are trying to understand the relationship between a particular feature and the outcome of a regression model, you might use Pearson's r to determine the strength of that relationship.



***
***


# Affirm Interview QA

how would you implement a machine learning data processing pipeline?

To implement a machine learning data processing pipeline, you can follow these steps:

Collect the data that you want to process. This data should be relevant to the problem that you want to solve with machine learning.

Clean and preprocess the data to prepare it for machine learning. This may include tasks such as removing missing or invalid data points, transforming the data into a suitable format, and scaling the data so that all features have the same range.

Split the preprocessed data into training, validation, and test sets. The training set is used to train the machine learning model, the validation set is used to evaluate the model during training, and the test set is used to evaluate the final model.

Choose a machine learning model that is suitable for the problem you want to solve. You may need to experiment with different models and hyperparameters to find the best one.

Train the machine learning model on the training set. This involves using the training data to adjust the model's parameters so that it can make accurate predictions on new data.

Evaluate the trained model on the validation set. This will give you an idea of how well the model is performing and whether it is overfitting or underfitting the data.

Fine-tune the model by adjusting its hyperparameters, training it on different subsets of the training data, or using regularization techniques to prevent overfitting.

Once you are satisfied with the model's performance on the validation set, evaluate it on the test set to get an estimate of its performance on unseen data.

Use the trained and fine-tuned model to make predictions on new data. This may involve using the model to make predictions on the entire dataset, or using it to make predictions on individual data points as they are received.

Overall, implementing a machine learning data processing pipeline requires a combination of data preparation, model selection and training, and model evaluation and fine-tuning. By following these steps, you can build a pipeline that can process data efficiently and accurately for a variety of machine learning tasks.




AJ
how would you process data to be fed to gradient boosting models?

To process data for use with gradient boosting models, you can follow these steps:

Collect the data that you want to process. This data should be relevant to the problem that you want to solve with gradient boosting.

Clean and preprocess the data to prepare it for use with gradient boosting. This may include tasks such as removing missing or invalid data points, transforming the data into a suitable format, and scaling the data so that all features have the same range.

Split the preprocessed data into training, validation, and test sets. The training set is used to train the gradient boosting model, the validation set is used to evaluate the model during training, and the test set is used to evaluate the final model.

Choose a gradient boosting model that is suitable for the problem you want to solve. You may need to experiment with different models and hyperparameters to find the best one.

Train the gradient boosting model on the training set. This involves using the training data to adjust the model's parameters so that it can make accurate predictions on new data.

Evaluate the trained model on the validation set. This will give you an idea of how well the model is performing and whether it is overfitting or underfitting the data.

Fine-tune the model by adjusting its hyperparameters, training it on different subsets of the training data, or using regularization techniques to prevent overfitting.

Once you are satisfied with the model's performance on the validation set, evaluate it on the test set to get an estimate of its performance on unseen data.

Use the trained and fine-tuned model to make predictions on new data. This may involve using the model to make predictions on the entire dataset, or using it to make predictions on individual data points as they are received.

Overall, processing data for use with gradient boosting models involves many of the same steps as processing data for other machine learning models. The key difference is that you need to carefully choose a gradient boosting model that is suitable for the problem you want to solve, and you may need to experiment with different hyperparameters to achieve the best performance.




AJ
How should data be formatted to be fed to xgboost?

To be fed to XGBoost, data should be formatted as a matrix with rows representing the samples and columns representing the features. Each sample should be labeled with the correct class, if the problem is a supervised learning task. The matrix should be stored in a NumPy array or a Pandas DataFrame, which are the two most common formats for storing data in Python.

Additionally, XGBoost requires that the data be preprocessed in a specific way. For example, all categorical variables should be encoded as numeric values, and all continuous variables should be scaled so that they have the same range. You can use the DMatrix class in XGBoost to convert the data into the appropriate format for training and testing.

Overall, the key to formatting data for use with XGBoost is to ensure that it is organized as a matrix with rows representing samples and columns representing features, and that it is preprocessed in a way that is suitable for gradient boosting. By following these guidelines, you can prepare your data for use with XGBoost and other gradient boosting models.




AJ
How we can measure the importance of features in regression? and then some special cases.


2 / 2

There are several ways to measure the importance of features in regression. Some common methods include:

Coefficient of determination (R²): This is a measure of how well the model fits the data. It ranges from 0 to 1, with a higher value indicating a better fit.

Coefficient of variation (CV): This is a measure of how much the model's predictions vary from the true values. It is calculated as the standard deviation of the residuals (the difference between the predicted values and the true values) divided by the mean of the true values.

F-test: This is a statistical test that measures how well the model fits the data compared to a simpler model. The F-statistic is calculated as the ratio of the variance of the residuals in the full model to the variance of the residuals in the simpler model.

Feature importance: Some machine learning algorithms, such as random forests and gradient boosting, can calculate the importance of each feature in the model. This is typically measured as the decrease in model performance when the feature is removed or perturbed.

In some special cases, such as when the data has multicollinearity (meaning that the features are highly correlated with each other), it may be difficult to accurately measure the importance of individual features. In these cases, it may be necessary to use regularization techniques or other methods to improve the model's ability to accurately measure feature importance.



***
***


# Regularization

## Parameter Norm Penalties
{: #content2}

1. **Parameter Norms:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  
    Many regularization approaches are based on limiting the capacity of models by adding a parameter norm penalty $$\Omega(\boldsymbol{\theta})$$ to the objective function $$J$$. We denote the regularized objective function by $$\tilde{J}$$:  
    <p>$$\tilde{J}(\boldsymbol{\theta} ; \boldsymbol{X}, \boldsymbol{y})=J(\boldsymbol{\theta} ; \boldsymbol{X}, \boldsymbol{y})+\alpha \Omega(\boldsymbol{\theta}) \tag{7.1}$$</p>  
    where $$\alpha \in[0, \infty)$$ is a HP that weights the relative contribution of the norm penalty term, $$\Omega$$, relative to the standard objective function $$J$$.  
    * __Effects of $$\alpha$$__:  
        * $$\alpha = 0$$ results in NO regularization
        * Larger values of $$\alpha$$ correspond to MORE regularization

    The __effect of minimizing the regularized objective function__ is that it will *__decrease__*, both, _the original objective $$J$$_ on the training data and some _measure of the size of the parameters $$\boldsymbol{\theta}$$_.  

    Different choices for the parameter norm $$\Omega$$ can result in different solutions being preferred.  
    <br>


2. **Parameter Penalties and the Bias parameter:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  
    In NN, we usually penalize __only the weights__ of the affine transformation at each layer and we leave the __biases unregularized__.  
    Biases typically require less data than the weights to fit accurately. The reason is that _each weight specifies how TWO variables interact_ so fitting the weights well, requires observing both variables in a variety of conditions. However, _each bias controls only a single variable_, thus, we don't induce too much _variance_ by leaving the biases unregularized. If anything, regularizing the bias can introduce a significant amount of _underfitting_.  
    <br>


3. **Note on the $$\alpha$$ parameter for different hidden layers:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}  
    In the context of neural networks, it is sometimes desirable to use a separate penalty with a different $$\alpha$$ coefficient for each layer of the network. Because it can be expensive to search for the correct value of multiple hyperparameters, it is still reasonable to use the same weight decay at all layers just to reduce the size of search space.  
    <br>


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

        Thus, we see that the effect of weight decay is to rescale $$\boldsymbol{w}^{\ast}$$ along the axes defined by the eigenvector of $$\boldsymbol{H}$$. Specifically, the component of $$\boldsymbol{w}^{\ast}$$ that is aligned with the $$i$$-th eigenvector of $$\boldsymbol{H}$$  is rescaled by a factor of $$\frac{\lambda_{i}}{\lambda_{i}+\alpha}$$.  

        ![img](/main_files/dl_book/regularization/1.png){: width="100%"}   

        __Summary:__  

        | __Condition__|__Effect of Regularization__ |   
        | $$\lambda_{i}>>\alpha$$ | Not much |  
        | $$\lambda_{i}<<\alpha$$ | The weight value almost shrunk to $$0$$ |  

    * __Applying $$L^2$$ regularization to *Linear Regression* :__  
        * <button>Application to Linear Regression</button>{: .showText value="show" onclick="showTextPopHide(event);"}
            ![img](/main_files/dl_book/regularization/2.png){: width="100%" hidden=""}   
    * [Weight Decay Analysis (blog)](https://himarora.github.io/regularization/understanding-the-mathematics-of-weight-decay/)  
    {: hidden=""}  
    <br>

    __$$L^2$$ Regularization Derivation:__{: style="color: red"}  
    $$L^2$$ regularization is equivalent to __MAP Bayesian inference with a Gaussian prior on the weights__.  

    __The MAP Estimate:__  
    <button>Show MAP Estimate Derivation</button>{: .showText value="show" onclick="showText_withParent_PopHide(event);"}
    <p hidden="">$$\begin{aligned} \hat{\theta}_ {\mathrm{MAP}} &=\arg \max_{\theta} P(\theta \vert y) \\ &=\arg \max_{\theta} \frac{P(y \vert \theta) P(\theta)}{P(y)} \\ &=\arg \max_{\theta} P(y \vert \theta) P(\theta) \\ &=\arg \max_{\theta} \log (P(y \vert \theta) P(\theta)) \\ &=\arg \max_{\theta} \log P(y \vert \theta)+\log P(\theta) \end{aligned}$$</p>  
    
    We place a __Gaussian Prior__ on the weights, with __zero mean__ and __equal variance $$\tau^2$$__:  
    <p>$$\begin{aligned} \hat{\theta}_ {\mathrm{MAP}} &=\arg \max_{\theta} \log P(y \vert \theta)+\log P(\theta) \\ &=\arg \max _{\boldsymbol{w}}\left[\log \prod_{i=1}^{n} \dfrac{1}{\sigma \sqrt{2 \pi}} e^{-\dfrac{\left(y_{i}-\boldsymbol{w}^{\top}\boldsymbol{x}_i\right)^{2}}{2 \sigma^{2}}}+\log \prod_{j=0}^{p} \dfrac{1}{\tau \sqrt{2 \pi}} e^{-\dfrac{w_{j}^{2}}{2 \tau^{2}}} \right] \\ &=\arg \max _{\boldsymbol{w}} \left[-\sum_{i=1}^{n} \dfrac{\left(y_{i}-\boldsymbol{w}^{\top}\boldsymbol{x}_i\right)^{2}}{2 \sigma^{2}}-\sum_{j=0}^{p} \dfrac{w_{j}^{2}}{2 \tau^{2}}\right] \\ &=\arg \min_{\boldsymbol{w}} \dfrac{1}{2 \sigma^{2}}\left[\sum_{i=1}^{n}\left(y_{i}-\boldsymbol{w}^{\top}\boldsymbol{x}_i\right)^{2}+\dfrac{\sigma^{2}}{\tau^{2}} \sum_{j=0}^{p} w_{j}^{2}\right] \\ &=\arg \min_{\boldsymbol{w}} \left[\sum_{i=1}^{n}\left(y_{i}-\boldsymbol{w}^{\top}\boldsymbol{x}_i\right)^{2}+\lambda \sum_{j=0}^{p} w_{j}^{2}\right] \\ &= \arg \min_{\boldsymbol{w}} \left[ \|XW - \boldsymbol{y}\|^2 + \lambda {\|\boldsymbol{w}\|_ 2}^2\right]\end{aligned}$$</p>  
    <button>Different Notation</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](/main_files/dl_book/regularization/4.png){: width="100%" hidden=""}   
    <br>

    __Properties:__{: style="color: red"}  
    {: #lst-p}
    * Notice that L2-regularization has a rotational invariance. This actually makes it more sensitive to irrelevant features.  [\[Ref\]](https://www.cs.ubc.ca/~schmidtm/Courses/540-W18/L6.pdf)  
        > [Paper](https://icml.cc/Conferences/2004/proceedings/papers/354.pdf)  
    * Adding L2-regularization to a convex function gives a strongly-convex function. So L2-regularization can make gradient descent converge much faster.  (^ same ref)      

    __Notes:__{: style="color: red"}  
    {: #lst-p}
    * [L2-reg and Adversarial Examples (New Angle)](https://thomas-tanay.github.io/post--L2-regularization/)  
    * [Weight Decay Analysis (blog)](https://himarora.github.io/regularization/understanding-the-mathematics-of-weight-decay/)  
    * __Interpreting the Penalty term $$\lambda$$:__  
        In the Bayesian Interpretation of Weight Decay; by assuming a __Gaussian Distribution__ on the errors:  
        the Penalty term $$\lambda$$ can be interpreted as the (squared) <span>ratio of two __variances__</span>{: style="color: purple"} of the __data__ and the __weights__ respectively: $$\lambda = \frac{\sigma_{D}^{2}}{\sigma_{W}^{2}}$$.  
        * [Learning the __VARIANCES__ of the distributions/Priors (Hinton!)](https://www.youtube.com/watch?v=_WZAD2uhvUM&list=PLiPvV5TNogxKKwvKb1RKwkq2hm7ZvpHz0&index=60)  
    * [A visual explanation for regularization of linear models (Blog)](https://explained.ai/regularization/index.html)  
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
    <p hidden="">$$\begin{aligned} \hat{\theta}_ {\mathrm{MAP}} &=\arg \max_{\theta} \log P(y \vert \theta)+\log P(\theta) \\  &=\arg \max _{\boldsymbol{w}}\left[\log \prod_{i=1}^{n} \dfrac{1}{\sigma \sqrt{2 \pi}} e^{-\dfrac{\left(y_{i}-\boldsymbol{w}^{\top}\boldsymbol{x}_i\right)^{2}}{2 \sigma^{2}}}+\log \prod_{j=0}^{p} \dfrac{1}{2 b} e^{-\dfrac{\left|\theta_{j}\right|}{2 b}} \right] \\    &=\arg \max _{\boldsymbol{w}} \left[-\sum_{i=1}^{n} \dfrac{\left(y_{i}-\boldsymbol{w}^{\top}\boldsymbol{x}_i\right)^{2}}{2 \sigma^{2}}-\sum_{j=0}^{p} \dfrac{\left|w_{j}\right|}{2 b}\right] \\    &=\arg \min_{\boldsymbol{w}} \dfrac{1}{2 \sigma^{2}}\left[\sum_{i=1}^{n}\left(y_{i}-\boldsymbol{w}^{\top}\boldsymbol{x}_i\right)^{2}+\dfrac{\sigma^{2}}{b} \sum_{j=0}^{p}\left|w_{j}\right|\right] \\    &=\arg \min_{\boldsymbol{w}} \left[\sum_{i=1}^{n}\left(y_{i}-\boldsymbol{w}^{\top}\boldsymbol{x}_i\right)^{2}+\lambda \sum_{j=0}^{p}\left|w_{j}\right|\right] \\    &= \arg \min_{\boldsymbol{w}} \left[ \|XW - \boldsymbol{y}\|^2 + \lambda \|\boldsymbol{w}\|_ 1\right]\end{aligned}$$</p>

    __Properties:__{: style="color: red"}  
    {: #lst-p}
    * $$L^1$$ regularization can occasionally produce non-unique solutions. A simple example is provided in the figure when the space of possible solutions lies on a 45 degree line. 
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
    * Elastic net regularization tends to have a grouping effect, where correlated input features are assigned equal weights.  



[^2]: More generally, we could regularize the parameters to be near any specific point in space and, surprisingly, still get a regularization effect, but better results will be obtained for a value closer to the true one, with zero being a default value that makes sense when we do not know if the correct value should be positive or negative.  

[^3]: The approximation is perfect if the objective function is truly quadratic, as in the case of __linear regression w/ MSE__.  
