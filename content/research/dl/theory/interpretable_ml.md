---
layout: NotesPage
title: Interpretable Machine Learning (Models)
permalink: /work_files/research/dl/theory/interpretable_ml
prevLink: /work_files/research/dl/theory.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [FIRST](#content1)
  {: .TOC1}
  * [SECOND](#content2)
  {: .TOC2}
<!--     * [THIRD](#content3)
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

* [__The Ultimate Guide on Interpretability:__ Interpretable Machine Learning (book!)](https://christophm.github.io/interpretable-ml-book/)  

* [The Building Blocks of Interpretability (distill)](https://distill.pub/2018/building-blocks/)  
* [Explaining a Black-box Using Deep Variational Information Bottleneck Approach](https://blog.ml.cmu.edu/2019/05/17/explaining-a-black-box-using-deep-variational-information-bottleneck-approach/)  
* [Paper Dissected: Understanding Black Box Predictions via Influence Functions Explained (blog)](http://mlexplained.com/2018/06/01/paper-dissected-understanding-black-box-predictions-via-influence-functions/)  
* [Interpretability via attentional and memory-based interfaces, using TensorFlow (blog!)](https://www.oreilly.com/ideas/interpretability-via-attentional-and-memory-based-interfaces-using-tensorflow)  
* [Explainable Artificial Intelligence – Model Interpretation Strategies (blog)](https://www.kdnuggets.com/2018/12/explainable-ai-model-interpretation-strategies.html/2)  
* [Black-box vs. white-box models - Interpretability Techniques (blog)](https://towardsdatascience.com/machine-learning-interpretability-techniques-662c723454f3)  
* [Ideas on Machine Learning Interpretability (H2O Vid)](https://www.youtube.com/watch?v=Ds1eRF7wpCU)  
* [Interpretable AI Solutions](https://www.interpretable.ai/products/)  


* __Interpretable Models__:  
    <button>Table: Algorithms, Properties, and Tasks</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](https://cdn.mathpix.com/snip/images/y57lxIe5xFF-Gmkct2LwnQEvVYKqbM53HTSunJjKrho.original.fullsize.png){: width="100%" hidden=""}   

    * __Linearity__: Typically we have a linear model if the association between features and target is modeled linearly.
    * __Monotonicity__: A monotonic model ensures that the relationship between a feature and the target outcome is always in one consistent direction (increase or decrease) over the feature (in its entirety of its range of values).
    * __Interactions:__ You can always add interaction features, non-linearity to a model with manual feature engineering. Some models create it automatically also.


    * __Complexity of Learned Functions (increasing)__:  
        * Linear, Monotonic
        * Non-linear, Monotonic
        * Non-linear, Non-monotonic  


    * [__Approaches to interpretability of ML models based on agnosticism__](https://arxiv.org/pdf/2208.13405.pdf):  
        ![img](https://cdn.mathpix.com/snip/images/Ggit47XWdqbrdsbmaiqUKpSvtU2ikrj5H-pdL13KEoI.original.fullsize.png){: width="80%"}  