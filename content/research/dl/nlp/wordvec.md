---
layout: NotesPage
title: Word Vector Representations <br /> word2vec
permalink: /work_files/research/dl/nlp/wordvec
prevLink: /work_files/research/dl/nlp.html
---


<div markdown="1" class = "TOC">
# Table of Contents

  * [Word Meaning](#content1)
  {: .TOC1}
  * [Word Embeddings](#content2)
  {: .TOC2}
  * [Word2Vec](#content3)
  {: .TOC3}
  * [FOURTH](#content4)
  {: .TOC4}
</div>

***
***

## Word Meaning
{: #content1}

1. **Representing the Meaning of a Word:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    :   Commonest linguistic way of thinking of meaning:  
        Signifier $$\iff$$ Signified (idea or thing) = denotation
    
2. **How do we have usable meaning in a computer:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    :   Commonly:  Use a taxonomy like WordNet that has hypernyms (is-a) relationships and synonym sets
    
3. **Problems with this discrete representation:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    :   * __Great as a resource but missing nuances__:  
            * Synonyms:  
                adept, expert, good, practiced, proficient, skillful
        * __Missing New Words__
        * __Subjective__  
        * __Requires human labor to create and adapt__  
        * __Hard to compute accurate word similarity__:  
            * _One-Hot Encoding_: in vector space terms, this is a vector with one 1 (at the position of the word) and a lot of zeroes (elsewhere).  
                * It is a __localist__ representation   
                * There is __no__ natural __notion of similarity__ in a set of one-hot vectors   
    
4. **Distributed Representations of Words:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    :   A method where vectors encode the similarity between the words.  
    :   The meaning is represented with real-valued numbers and is "_smeared_" across the vector.  
    :   Contrast with __one-hot encoding__.
    
5. **Distributional Similarity:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    :   is an idea/hypothesis that one can describe the meaning of words by the context in which they appear in.   
    :   Contrast with __Denotational Meaning__ of words.
    
6. **The Big Idea:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    :   We will build a dense vector for each word type, chosen so that it is good at predicting other words appearing in its context.  
    
7. **Learning Neural Network Word Embeddings:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  
    :   We define a model that aims to predict between a center word $$w_t$$ and context words in terms of word vectors.  
    :   $$p(\text{context} \| w_t) = \ldots$$ 
    :   __The Loss Function__:  
    :   $$J = 1 - p(w_{-t} \| w_t)$$  
    :   We look at many positions $$t$$ in a big language corpus
    :   We keep adjusting the vector representations of words to minimize this loss

    
8. **Relevant Papers:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}  
    :   * Learning representations by back-propagating errors (Rumelhart et al., 1986) 
        * A neural probabilistic language model (Bengio et al., 2003) 
        * NLP (almost) from Scratch (Collobert & Weston, 2008) 
        * A recent, even simpler and faster model: word2vec (Mikolov et al. 2013) à intro now
    
***

## Word Embeddings
{: #content2}

1. **Main Ideas:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  
    :   * Words are represented as vectors of real numbers
        * Words with similar vectors are _semantically_ similar 
        * Sometimes vectors are low-dimensional compared to the vocabulary size  
    
2. **The Clusterings:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  
    :   * __Relationships (attributes) Captured__:    
            * Synonyms: car, auto
            * Antonyms: agree, disagree
            * Values-on-a-scale: hot, warm, cold
            * Hyponym-Hypernym: "Truck" is a type of "car", "dog" is a type of "pet"
            * Co-Hyponyms: "cat"&"dog" is a type of "pet"
            * Context: (Drink, Eat), (Talk, Listen)
    
3. **Word Embeddings Theory:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}  
    :   Distributional Similarity Hypothesis

4. **History and Terminology:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  
    :   Word Embeddings \\
        = \\
        Distributional Semantic Model \\
        = \\
        Distributed Representation \\
        = \\
        Semantic Vector Space \\
        = \\
        Vector Space Model 

5. **Applications:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents25}  
    :   * Word Similarity
        * Word Grouping
        * Features in Text-Classification
        * Document Clustering
        * NLP:  
            * POS-Tagging
            * Semantic Analysis
            * Syntactic Parsing

6. **Approaches:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents26}  
    :   * __Count__: word count/context co-occurrences   
            * *__Distributional Semantics__*:    
                1. Summarize the occurrence statistics for each word in a large document set:   
                    ![img](/main_files/dl/nlp/1/1.png){: width="40%"}  
                2. Apply some dimensionality reduction transformation (SVD) to the counts to obtain dense real-valued vectors:   
                    ![img](/main_files/dl/nlp/1/2.png){: width="40%"}  
                3. Compute similarity between words as vector similarity:  
                    ![img](/main_files/dl/nlp/1/3.png){: width="40%"}  
        * __Predict__: word based on context  
            * __word2vec__:  
                1. In one setup, the goal is to predict a word given its context.  
                    ![img](/main_files/dl/nlp/1/4.png){: width="80%"}   
                2. Update word representations for each context in the data set  
                3. Similar words would be predicted by similar contexts

7. **Parameters:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents27}
    :   * Underlying Document Set   
        * Context Size
        * Context Type

8. **Software:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents28}  
    :   ![img](/main_files/dl/nlp/1/5.png){: width="80%"}
    
***

## Word2Vec
{: #content3}

1. **Main Idea:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents31}  
    :   Predict between every word and its context words.  
    
2. **Algorithms:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents32}  
    :   1. __Skip-grams (SG)__:  
            Predict context words given target (position independent)
        2. __Continuous Bag of Words (CBOW)__:  
            Predict target word from bag-of-words context
    
3. **Training Methods:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents33}  
    :   * __Basic__:    
            1. Naive Softmax  
        * __(Moderately) Efficient__:  
            1. Hierarchical Softmax
            2. Negative Sampling   
    
4. **Skip-Gram Prediction Method:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents34}  
    :   Skip-Gram Models aim to predict the _distribution (probability)_ of context words from a center word.  
        > CBOW does the opposite, and aims to predict a center word from the surrounding context in terms of word vectors.  
    :   * __The Algorithm__:    
            1. We generate our one hot input vector $$x \in \mathbf{R}^\|V\|$$ of the center word.  
            2. We get our embedded word vector for the center word $$v_c = V_x \in \mathbf{R}^n$$  
            3. Generate a score vector $$z = \mathcal{U}_{v_c}$$ 
            4. Turn the score vector into probabilities, $$\hat{y} = \text{softmax}(z)$$ 
                > Note that $$\hat{y}_{c−m}, \ldots, \hat{y}_{c−1}, \hat{y}_{c+1}, \ldots, \hat{y}_{c+m}$$ are the probabilities of observing each context word.  
            5. We desire our probability vector generated to match the true probabilities, which is  
                $$ s y^{(c−m)} , ldots, y^{(c−1)} , y^{(c+1)} , ldots, y^{(c+m)}$$,  
                 the one hot vectors of the actual output.  
            6. 
    
5. **Word2Vec Details:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents35}  
    :   * For each word $$t = 1 \ldots T$$, predict surrounding words in a window of “radius” $$m$$ of every word.  
        * For $$p(w_{t+j} \| w_t)$$ the simplest first formulation is:  
        :   $$\\p(o \| c) = \dfrac{e^{u_o^Tv_c}}{\sum_{w=1}^V e^{u_w^Tv_c}}\\$$  
        where, ...

    
6. **The Objective:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents36}  
    :   Maximize the probability of any context word given the current center word.  
    :   
    
7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents37}  
    :   
    
8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents38}  
    :   
    
***