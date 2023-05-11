---
layout: NotesPage
title: Probability Calibration
permalink: /work_files/calibration
prevLink: /work_files/research/dl/practical.html
---

[ens](/work_files/research/ml/ens_lern)

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
