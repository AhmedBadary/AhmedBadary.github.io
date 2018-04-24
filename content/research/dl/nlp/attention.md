---
layout: NotesPage
title: Attention Mechanism for DNNs 
permalink: /work_files/research/dl/nlp/attention
prevLink: /work_files/research/dl/nlp.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Introduction](#content1)
  {: .TOC1}
  * [Specialized Attention Varieties](#content2)
  {: .TOC2}
</div>

***
***

## Introduction
{: #content1}

1. **Motivation:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    :   In __Vanilla Seq2Seq models__, the only representation of the input is the _fixed-dimensional vector representation $$(y)$$_, that we need to carry through the entire decoding process.   
        This presents a __bottleneck__ in condensing all of the information of the _entire input sequence_ into just one _fixed-length_ vector representation.  

2. **Attention:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    :   Attention is a mechanism that allows DNNs to focus on (view) certain local or global features of the input sequence as a whole or in part.     
    :   Attention involves focus on _certain parts_ of the input, while having a _low-resolution_ view of the rest of the input -- similar to human attention in vision/audio.  
    :   An __Attention Unit__ considers all sub regions and contexts as its input and it outputs the weighted arithmetic mean of these regions.  
        > The __arithmetic mean__ is the inner product of actual values and their probabilities.  
    :   $$m_i = \tanh (x_iW_{x_i} + CW_C)$$ 
    :   These __probabilities__ are calculated using the *__context__*.  
        The __Context__ $$C$$ represents everything the RNN has outputted until now.  
    :   The difference between using the _hyperbolic tanh_ and a _dot product_ is the __granularity__ of the output regions of interest - __tanh__ is more fine-grained with less choppy and smoother sub-regions chosen.  
    :   The probabilities are interpreted as corresponding to the relevance of the sub-region $$x_i$$ given context C

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    :   

4. **Types of Attention:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    :   * __Soft Attention__: we consider different parts of _different subregions_   
            * Soft Attention is __deterministic__ 
        * __Hard Attention__: we consider only _one subregion_  
            * Hard Attention is a __stochastic__ process 
                

5. **Strategy:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    :   * Encode each word in the sentence into a vector (representation)
        * When decoding, perform a linear combination of these vectors, weighted by _attention weights_ 
        * Use this combination in picking the next word (subregion)

6. **Calculating Attention:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    :   * Use __query__ vector (decoder state) and __key__ vectors (all encoder states)
        * For each query-key pair, calculate weight 
        * Normalize to add to one using softmax 
        * Combine together value vectors (usually encoder states, like key vectors) by taking the weighted sum
        * Use this in any part of the model

7. **Attention Score Functions:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  
    :   $$q$$ is the query, $$k$$ is the key:  
        * __Multi-Layer Perceptron__ _(Bahdanau et al. 2015)_:  
            * Flexible, often very good with large data   
        <p>$$a(q,k) = w_2^T \tanh (W_1[q;k])$$</p>   
    :   * __Bilinear__ _(luong et al. 2015)_:  
            * Not used widely in Seq2Seq models
            * Results are inconsistent
    :   $$a(q,k) = q^TWk$$ 
    :   * __Dot Product__ _(luong et al. 2015)_:  
            * No parameters
            * Requires the sizes to be the same
    :   $$a(q,k) = q^Tk$$  
    :   * __Scaled Dot Product__ _(Vaswani et al. 2017)_:  
            * Solves the scale problem of the dot-product: the scale of the dot product increases as dimensions get larger
    :   $$a(q,k) = \dfrac{q^Tk}{\sqrt{\vert k \vert}}$$ 
            
                
8. **What to Attend to?**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}  
    :   * __Input Sentence__:  
            * A previous word for translation - [Neural Machine Translation](/)  
            * Copying Mechanism - [Gu et al. 2016](/)  
            * Lexicon bias [Arthur et al. 2016](/)  
        * __Previously Generated Things__:  
            * In *__language modeling__*: attend to the previous words - [Merity et al. 2016](/)   
                > Attend to the previous words that you generated and decide whether to use them again (copy)  
            * In __*translation*__: attend to either input or previous output - [Vaswani et al. 2017](/)

9. **Modalities:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents19}
    :   * __Images__ (Xu et al. 2015)    
        * __Speech__ (Chan et al. 2015)  
        * __Hierarchical Structures__ (Yang et al. 2016):  
            * Encode with attention over each sentence then attention over each sentence in the document  
        * __Multiple Sources__:    
            * Attend to multiple sentences in different languages to be translated to one target language (Zoph et al. 2015)  
            * Attend to a sentence and an image (Huang et al. 2016)  
                
10. **Intra-Attention/Self-Attention:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents110}      
    :   Each element in the sentence attends to other elements -- context sensitive encodings.  
    :   It behaves similar to a __Bi-LSTM__ in that it tries to encode information about the context (words around the current input) into the representation of the word.  
        It differs however:  
        1. Intra-Attention is much more direct, as it takes the context directly without being influenced by many steps inside the RNN 
        2. It is much faster as it is only a dot/matrix product  

11. **Improvement to Attention:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents111}  
    :   * __The Coverage Problem:__{: style="color: red"}  Neural models tend to drop or repeat content when tested on data not very similar to the training set  
        * __Solution:__{: style="color: red"} Model how many times words have been covered  
            * __Impose a penalty__ if attention is not $$\approx 1$$ for each word (Cohn et al. 2015)   
                It forces the system to translate each word at least once.  
            * __Add embeddings indicating coverage__ (Mi.. et al. 2016)  
            * Incorporating Markov Properties (Cohn et al. 2015)  
                * Intuition: attention from last time tends to be correlated with attention this time
                * Strategy: Add information about the last attention when making the next decision
            * __Bidirectional Training__ (Cohn et al. 2015): 
                * Intuition: Our attention should be roughly similar in forward and backward directions
                * Method: Train so that we get a bonus based on the trace of the matrix product for training in both directions  
                    $$\mathrm{Tr} (A_{X \rightarrow Y}A^T_{Y \rightarrow X})$$  
            * __Supervised Training__ (Mi et al. 2016):   
                * Sometimes we can get "gold standard" alignments a-priori:  
                    * Manual alignments
                    * Pre-trained with strong alignment model
                * Train the model to match these strong alignments (bias the model)

12. **Attention is not Alignment _(Koehn and Knowles 2017)_:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents121}  
    :   * Attention is often blurred
        * Attention is often off by one:  
            Since the DNN has already seen parts of the information required to generate previous outputs, it might not need all of the information from the word that is actually matched with its current output.  
    :   Thus, even if _Supervised training_ is used to increase alignment accuracy, the overall error rate of the task might not actually decrease. 

***

## Specialized Attention Varieties
{: #content2}

1. **Hard Attention _(Xu et al. 2015)_:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  
    :   * Instead of a _soft interpolation_, make a __Zero-One decision__ about where to attend (Xu et al. 2015)
            * Harder to train - requires reinforcement learning methods
        * It helps interpretability (Lei et al. 2016)   

2. **Monotonic Attention _(Yu et al. 2016)_:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  
    :   * In some cases, we might know the output will be the same order as the input:  
            * Speech Recognition
            * Incremental Translation
            * Morphological Inflection - sometimes
            * Summarization - sometimes
        * Hard decisions about whether to read more

3. **Convolutional Attention _(Allamanis et al. 2016)_:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}  
    :   * __Intuition__: we might want to be able to attend to "the word after 'Mr.'"  

4. **Multi-headed Attention:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}  
    :   * __Idea__: multiple _attention heads_ focus on different parts of the sentence
        * Different heads for "copy" vs regular (Allamanis et al. 2016)   
        * Multiple independently learned heads (Vaswani et al. 2017)

5. **Tips:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents25}  
    :   * Don't use attention with very long sequences - especially those you want to summarize and process efficiently 
        * __Fertility__: we impose the following heuristic "It is bad to pay attention to the same subregion many times" 
