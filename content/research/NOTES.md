---
layout: NotesPage
title: Notes
permalink: /work_files/notes
prevLink: /work_files/research/dl/theory.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [FIRST](#content1)
  {: .TOC1}
  * [SECOND](#content2)
  {: .TOC2}
<!--     * [Feature Importance](#content3)
  {: .TOC3}
  * [FOURTH](#content4)
  {: .TOC4}
  * [FIFTH](#content5)
  {: .TOC5}
  * [SIXTH](#content6)
  {: .TOC6} -->
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
        * credit bank has only data from customers whose loan has been approved. This set of customers will be generally a biased sample of the whole population or the set of potential customers.  
        <button>Bounding selection sample by unlabaled data</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        ![img](https://cdn.mathpix.com/snip/images/h2GhVO85H70x3AnmvbgreGBSIV_YHdYoOwEzpvZqexw.original.fullsize.png){: width="50%" hidden=""}  
        <button>Propensity Scores and Sample Selection Bias in Credit</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        ![img](https://cdn.mathpix.com/snip/images/r2l867YNrqKQ8UU0KQsFpmTdRAa3dz58CCLMWQxaokA.original.fullsize.png){: width="60%" hidden=""}  
        ![img](https://cdn.mathpix.com/snip/images/EaqGdt8_rvnPS8xBCN9fkMFP87MvxEKRUcIrBECe1ZA.original.fullsize.png){: width="60%"}  
        NEW WORK:  
        ![img](https://cdn.mathpix.com/snip/images/lH6tljfOukqecy8Qjf25QLGCc9ISHPT9Fh-1fhf92rY.original.fullsize.png){: width="30%"}  

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

            * ![img](https://cdn.mathpix.com/snip/images/CfH9SccuC_rsfylUX1wcmilIuu1VJfeavad4ydHZv7I.original.fullsize.png){: width="50%"}  
                * Because two-sample tests often work better on low-dimensional data than on high-dimensional data, it’s highly recommended that you reduce the dimensionality of your data before performing a two-sample test on them  


    * __HAndling__{: style="color: red"}  
        ![img](https://cdn.mathpix.com/snip/images/MgGLgMYqBmLEu8QwjmF7cnKx71zt1fheSsB4OyJIH88.original.fullsize.png){: width="40%"}  

        ![img](https://cdn.mathpix.com/snip/images/Hi_luNIO0SgICUxX76D7QzV47lkgARIkV9HBI3tAD04.original.fullsize.png){: width="40%"}  

        ![img](https://cdn.mathpix.com/snip/images/R16l_8va3w_As48hLZwqaljdt750PokepcDU-SMrbjc.original.fullsize.png){: width="40%"}  

        * CHip  

        ![img](https://cdn.mathpix.com/snip/images/6wgY91SaB8iucfzSSWVD3Nd4-PPyE5tH6r2miYcqe2s.original.fullsize.png){: width="40%"}  

        ![img](https://cdn.mathpix.com/snip/images/6FaS3KTCfYIba5S6LGk4mTH24U129Un1mqZX2bJWoHM.original.fullsize.png){: width="50%"}  


    MONITORING:  
    ![img](https://cdn.mathpix.com/snip/images/DbAOKG_hm5eh-G5NPGsKy3LMb4wDBIRcDDU1izxama8.original.fullsize.png){: width="50%"}  
    Monitoring Raw Inputs


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