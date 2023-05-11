---
layout: NotesPage
title: Practical Concepts in Machine Learning
permalink: /work_files/research/dl/practical/practical_concepts
prevLink: /work_files/research/dl/practical.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [FIRST](#content1)
  {: .TOC1}
<!--   * [SECOND](#content2)
  {: .TOC2}
  * [THIRD](#content3)
  {: .TOC3} -->
</div>

***
***


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


12. **Limited Training Data:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents112}  
    * [LinkedIn Post on Dealing with Limited Training Data in Supervised Learning (Blog)](https://www.linkedin.com/posts/sebastianraschka_machinelearning-deeplearning-ai-activity-7024742743079886849-NUGk?utm_source=share&utm_medium=member_desktop)  


***

## SECOND
{: #content2}

<!-- 
1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}
2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}
3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}
4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}
 -->
