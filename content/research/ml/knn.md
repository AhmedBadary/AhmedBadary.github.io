---
layout: NotesPage
title: KNN <br> K-Nearest Neighbor
permalink: /work_files/research/ml/knn
prevLink: /work_files/research/ml.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [FIRST](#content1)
  {: .TOC1}
  * [SECOND](#content2)
  {: .TOC2}
<!--   * [THIRD](#content3)
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

## FIRST
{: #content1}

1. **KNN:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}    
    __KNN__ is a _non-parametric_ method used for classification and regression.  
    <br>

2. **Structure:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    In both cases, the input consists of the $$k$$ closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression:  
    {: #lst-p}
    * In __k-NN classification__, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its $$k$$ nearest neighbors ($$k$$ is a positive integer, typically small). If $$k = 1$$, then the object is simply assigned to the class of that single nearest neighbor.  
    * In __k-NN regression__, the output is the property value for the object. This value is the average of the values of $$k$$ nearest neighbors.
    <br>

3. **Formal Description - Statistical Setting:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    Suppose we have pairs $${\displaystyle (X_{1},Y_{1}),(X_{2},Y_{2}),\dots ,(X_{n},Y_{n})}$$ taking values in $${\displaystyle \mathbb {R} ^{d}\times \{1,2\}}$$, where $$Y$$ is the class label of $$X$$, so that $${\displaystyle X|Y=r\sim P_{r}}$$ for $${\displaystyle r=1,2}$$ (and probability distributions $${\displaystyle P_{r}}$$. Given some norm $${\displaystyle \|\cdot \|}$$ on $${\displaystyle \mathbb {R} ^{d}}$$ and a point $${\displaystyle x\in \mathbb {R} ^{d}}$$, let $${\displaystyle (X_{(1)},Y_{(1)}),\dots ,(X_{(n)},Y_{(n)})}$$ be a reordering of the training data such that $${\displaystyle \|X_{(1)}-x\|\leq \dots \leq \|X_{(n)}-x\|}$$.  
    <br>

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  

    __Properties:__  
    * k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until classification.  
    * It is sensitive to the local structure of the data.  


5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    :   __Complexity__:  
        :   * _Training_: $$\:\:\:\:\mathcal{O}(1)$$   
            * _Predict_: $$\:\:\:\:\mathcal{O}(N)$$ 


6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}  

***

## SECOND
{: #content2}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}  

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}  

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents25}  

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents26}  

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents27}  

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents28}  

***