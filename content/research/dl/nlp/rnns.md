---
layout: NotesPage
title: Recurrent Neural Networks (RNNs) <br /> Language Modeling
permalink: /work_files/research/dl/nlp/rnns
prevLink: /work_files/research/dl/nlp.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Introduction to and History of Language Models](#content1)
  {: .TOC1}
  * [Recurrent Neural Networks](#content2)
  {: .TOC2}
  * [RNN Language Models](#content3)
  {: .TOC3}
  * [Training RNNs](#content4)
  {: .TOC4}
  * [RNNs in Sequence Modeling](#content4)
  {: .TOC4}
  * [Bidirectional and Deep RNNs](#content4)
  {: .TOC4}
</div>

***
***

## Introduction to and History of Language Models 
{: #content1}

1. **Language Models:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    :   A __Language Model__ is a statistical model that computes a _probability distribution_ over sequences of words.  

2. **Applications:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    :   * __Machine Translation (MT)__:   
            * Word Ordering:  
                p("the cat is small") > p("small the cat is")  
            * Word Choice:  
                p("walking home after school") > p("walking house after school")
        * __Speech Recognition__:     
            * Word Disambiguation:  
                p("The listeners _recognize speech_") > p("The listeners _wreck a nice beach_")  
        * __Information Retrieval__: 
            * Used in _query likelihood model_
            
3. **Traditional Language Models:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    :   * Probability is usually conditioned on window of $$n$$ previous words
        * An incorrect but necessary Markov assumption  
    :   $$P(w_1, ldots, w_m) = \prod_{i=1}^m P(W_i \| w_1, \ldots, w_{i-1}) \approx \prod_{i=1}^m P(w_i \| w_{i-(n-1)}, ldots, w_{i-1})$$  
        * To estimate probabilities, compute for  
            * Unigrams:  
                $$P(w_2 \| w_1) = \dfrac{\text{count}(w_1, w_2)}{\text{count}((w_1)}$$  
            * Bigrams:  
                $$P(w_3 \| w_1, w_2) = \dfrac{\text{count}(w_1, w_2, w_3)}{\text{count}((w_1, w_2)}$$  

4. **Issues with the Traditional Approaches:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    :   * To improve performance we need to:  
            * Keep higher n-gram counts
            * Use Smoothing
            * Use Backoff (trying n-gram, (n-1)-gram, (n-2)-grams, ect.)  
    :   However, 
        * There are __A LOT__ of n-grams
            * $$\implies$$ Gigantic RAM requirements

***

## Recurrent Neural Networks
{: #content2}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  
    :   

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  
    :   

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

## THIRD
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