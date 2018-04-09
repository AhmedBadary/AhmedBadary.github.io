---
layout: NotesPage
title: Concepts
permalink: /concepts_
prevLink: /
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [ConvNets](#content1)
  {: .TOC1}
  * [RNNs](#content2)
  {: .TOC2}
  * [Math](#content3)
  {: .TOC3}
  * [Statistics](#content4)
  {: .TOC4}
  * [Optimization](#content5)
  {: .TOC5}
  * [Machine Learning](#content6)
  {: .TOC6}
  * [Computer Vision](#content7)
  {: .TOC7}
  * [NLP](#content8)
  {: .TOC8}
  * [Physics](#content9)
  {: .TOC9}
  * [Game Theory](#content10)
  {: .TOC10}
  * [Misc.](#content11)
  {: .TOC11}
</div>

***
***

## ConvNets
{: #content1}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    :   

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    :   

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    :   

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}  
    :   

***

## RNNs
{: #content2}

1. **Gradient Clipping Intuition:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  
    :   ![img](/main_files/main/concepts/1.png){: width="100%"}  
    :   * The image above is that of the __Error Surface__ of a _single hidden unit RNN_ 
        * The observation here is that there exists __High Curvature Walls__ 
            This Curvature Wall will move the gradient to a very different/far, probably less useful area. 
            Thus, if we clip the gradients we will avoid the walls and will remain in the more useful area that we were exploring already. 
    :   Draw a line between the original point on the Error graph and the End (optimized) point then evaluate the Error on points on that line and look at the changes -> this shows changes in the curvature.


2. **PeepHole Connection:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  
    :   is an addition on the equations of the __LSTM__ as follows: 
    :   $$ \Gamma_o = \sigma(W_o[a^{(t-1)}, x^{(t)}] + b_o) \\
        \implies 
        \sigma(W_o[a^{(t-1)}, x^{(t)}, c^{(t-1)}] + b_o)$$
    :   Thus, we add the term $$c^{(t-1)}$$ to the output gate.

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}  
    :   

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents25}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents26}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents27}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents28}  
    :   

***

## Maths
{: #content3}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents31}  
    :   

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents32}  
    :   

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents33}  
    :   

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents34}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents35}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents36}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents37}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents38}  
    :   

***

## Statistics
{: #content4}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents41}  
    :   

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents42}  
    :   

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents43}  
    :   

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents44}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents45}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents46}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents47}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents48}  
    :   

***

## Optimization
{: #content5}

1. **Sigmoid:**{: style="color: SteelBlue"}{: .bodyContents5 #bodyContents51}  
    :   $$\sigma(-x) = 1 - \sigma(x)$$

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents5 #bodyContents52}  
    :   

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents5 #bodyContents53}  
    :   

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents5 #bodyContents54}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents5 #bodyContents55}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents5 #bodyContents56}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents5 #bodyContents57}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents5 #bodyContents58}  
    :   

***

## Machine Learning
{: #content6}

1. **Why NNs are not enough:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents61}  
    :   The gist of it is this: neural nets do *pattern recognition*, which achieves *local generalization* (which works great for supervised perception). But many simple problems require some (small) amount of abstract modeling, which modern neural nets can't learn

2. **The Big Formulations:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents62}  
    :   * __Sequence Labeling__: 
            * *__Problems__*:  
                * Speech Recognition 
                * OCR
                * Semantic Segmentation
            * *__Approaches__*:  
                * CTC - Bi-directional LSTM
                * Listen Attend and Spell (LAS)
                * HMMs 
                * CRFs
                
            

3. **What is ML?:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents63}  
    :   Improve on __TASK T__ with respect to __PERFORMANCE METRIC P__ based on __EXPERIENCE E__.  
    :   __T:__ Categorize email messages as spam or legitimate 
        __P:__ Percentage of email messages correctly classified 
        __E:__ Database of emails, some with 
        human-given labels 

        __T:__ Recognizing hand-written words 
        __P:__ Percentage of words correctly classified 
        __E:__ Database of human-labeled images of 
        handwritten words 

        __T:__ playing checkers 
        __P:__ percentage of games won against an arbitrary opponent 
        __E:__ Playing practice games against itself 


        __T:__ Driving on four-lane highways using vision sensors 
        __P:__ Average distance traveled before a human-judged error 
        __E:__ A seq of images and steering commands recorded while observing a human driver  

4. **Graphical Models:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents64}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents65}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents66}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents67}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents68  
    :   

***

## Computer Vision
{: #content7}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents7 #bodyContents71}  
    :   

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents7 #bodyContents72}  
    :   

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents7 #bodyContents73}  
    :   

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents7 #bodyContents74}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents7 #bodyContents75}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents7 #bodyContents76}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents7 #bodyContents77}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents7 #bodyContents78}  
    :   

***

## NLP
{: #content8}

1. **Towards Better Language Modelling (Lec.9 highlight, 38m):**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents81}  
    :   To improve a _Language Model_:  
        1. __Better Inputs__: 
            Word -> Subword -> Char  
            ![img](/main_files/main/concepts/2.png){: width="100%"}  
            _Subword Language Modeling , Mikolov et al. 2012_  
            _Character-Aware Neural Language Model , Kim et al. 2015_.  
        2. __Better Regularization/Preprocessing__:  
            Similar to computer vision, we can do both Regularization and Preprocessing on the data to increase its relevance to the true distribution.  
            Preprocessing acts as a *__data augmentation__* technique. This allows us to achieve a __Smoother__ distribution, since we are removing more common words and re-enforcing rarer words.  
            _Zoneout, Kruger et al. 2016_  
            _Data Noising as Smoothing, Xie et al. 2016_       
            * *__Regularization__*:  
                * Use Dropout (Zaremba, et al. 2014). 
                * Use Stochastic FeedForward depth (Huang et al. 2016)
                * Use Norm Stabilization (Memisevic 2015)
                ...  
            * *__Preprocessing__*:  
                 * Randomly replacing words in a sentence with other words  
                 * Use bigram statistics to generate _Kneser-Ney_ inspired replacement (Xie et al. 2016). 
                 * Replace a word with __fixed__ drop rate
                 * Replace a word with __adaptive__ drop rate, by how rare two words appear together (i.e. "Humpty Dumpty"), and replace by a unigram draw over vocab
                 * Replace a word with __adaptive__ drop rate, and draw word from a __proposal distribution__ (i.e. "New York") 
        3. __Better Model__ (+ all above)

2. **Language Models:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents82}  
    :   * __The ML-Estimate of $$p(w_i \vert w_{i-1})$$__ $$ = \dfrac{c(w_{i-1}\: w_i)}{\sum_{w_i} c(w_{i-1}\: w_i)}$$
            

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents83}  
    :   

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents84}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents85}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents86}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents87}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents88}  
    :   

***

## Physics
{: #content9}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents91}  
    :   

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents92}  
    :   

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents93}  
    :   

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents94}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents95}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents96}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents97}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents98}  
    :   

***

## Game Theory
{: #content10}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents10 #bodyContents101}  
    :   

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents10 #bodyContents102}  
    :   

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents10 #bodyContents103}  
    :   

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents10 #bodyContents104}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents10 #bodyContents105}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents10 #bodyContents106}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents10 #bodyContents107}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents10 #bodyContents108}  
    :   

***

## Misc.
{: #content11}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents11 #bodyContents111}  
    :   

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents11 #bodyContents112}  
    :   

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents11 #bodyContents113}  
    :   

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents11 #bodyContents114}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents11 #bodyContents115}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents11 #bodyContents116}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents11 #bodyContents117}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents11 #bodyContents118}  
    :   