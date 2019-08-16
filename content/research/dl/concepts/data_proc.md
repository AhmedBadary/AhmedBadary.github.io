---
layout: NotesPage
title: Data Processing
permalink: /work_files/dl/concepts/data_proc
prevLink: /work_files/research/dl/concepts
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Dimensionality Reduction](#content1)
  {: .TOC1}
  * [Feature Selection](#content2)
  {: .TOC2}
  * [Feature Extraction](#content3)
  {: .TOC3}
  * [Feature Importance](#content4)
  {: .TOC4}
  * [Imputation](#content5)
  {: .TOC5}
  * [Normalization](#content6)
  {: .TOC6}
  * [Outliers Handling](#content7)
  {: .TOC7}
</div>

***
***

## Dimensionality Reduction
{: style="font-size: 1.60em"}
{: #content1}


### **Dimensionality Reduction**{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents1 #bodyContents11}  
__Dimensionality Reduction__ is the process of reducing the number of random variables under consideration by obtaining a set of principal variables. It can be divided into __feature selection__{: style="color: goldenrod"} and __feature extraction__{: style="color: goldenrod"}.  
<br>

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

<!-- ### ****{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents3 #bodyContents32}  
### ****{: style="color: SteelBlue; font-size: 1.15em"}{: .bodyContents3 #bodyContents33}   -->

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