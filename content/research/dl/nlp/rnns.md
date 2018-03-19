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
    :   $$P(w_1, \ldots, w_m) = \prod_{i=1}^m P(W_i \| w_1, \ldots, w_{i-1}) \approx \prod_{i=1}^m P(w_i \| w_{i-(n-1)}, \ldots, w_{i-1})$$  
    :   * To estimate probabilities, compute for  
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

1. **Recurrent Neural Networks:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyCoxqntents21}  
    :   An __RNN__ is a class of artificial neural network where connections between units form a directed cycle, allowing it to exhibit dynamic temporal behavior.
    :   The standard RNN is a nonlinear dynamical system that maps sequences to sequences.  

2. **The Structure of an RNN:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  
    :   The RNN is parameterized with three weight matrices and three bias vectors:  
    :   $$ \theta = [W_{hv}, W_{hh}, W_{oh}, b_h, b_o, h_0] $$
    :   These parameter completely describe the RNN.  

3. **The Algorithm:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}  
    :   Given an _input sequence_ $$\hat{x} = [x_1, \ldots, x_T]$$, the RNN computes a sequence of hidden states $$h_1^T$$ and a sequence of outputs $$y_1^T$$ in the following way:  
        __for__ $$t$$ __in__ $$[1, ..., T]$$ __do__  
            $$\ \ \ \ \ \ \ \ \ \ $$ $$u_t \leftarrow W_{hv}x_t + W_{hh}h_{t-1} + b_h$$  
            $$\ \ \ \ \ \ \ \ \ \ $$ $$h_t \leftarrow g_h(u_t)$$  
            $$\ \ \ \ \ \ \ \ \ \ $$ $$o_t \leftarrow W_{oh}h_{t} + b_o$$  
            $$\ \ \ \ \ \ \ \ \ \ $$ $$y_t \leftarrow g_y(o_t)$$   

4. **The Loss:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}  
    :   The loss of an RNN is commonly a sum of per-time losses:  
    :   $$L(y, z) = \sum_{t=1}^TL(y_t, z_t)$$
    :   * __Language Modelling__: 
            We use the *__Cross Entropy__* Loss function but predicting _words_ instead of classes
    :   $$ J^{(t)}(\theta) = - \sum_{j=1}^{\vert V \vert} y_{t, j} \log \hat{y_{t, j}}$$
    :   $$\implies$$
    :   $$L(y,z) = J = -\dfrac{1}{T} \sum_{t=1}^{T} \sum_{j=1}^{\vert V \vert} y_{t, j} \log \hat{y_{t, j}}$$
    :   To __Evaluate__ the model, we use *__Preplexity__* : 
    :   $$ 2^J$$
    :   > Lower Preplexity is _better_

5. **Analyzing the Gradient:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents25}  
    :   Assuming the following formulation of an __RNN__:  
    :   $$h_t = Wf(h_{t-1}) + W^{(hx)}x_{[t]} \\
        \hat{y_t} = W^{(S)}f(h_t)$$
    :   * The __Total Error__ is the sum of each error at each time step $$t$$:  
    :   $$ \dfrac{\partial E}{\partial W} = \sum_{t=1}^{T} \dfrac{\partial E_t}{\partial W}$$
    :   * The __local Error__ at a time step $$t$$:  
    :   $$\dfrac{\partial E_t}{\partial W} = \sum_{k=1}^{t} \dfrac{\partial E_t}{\partial y_t} \dfrac{\partial y_t}{\partial h_t} \dfrac{\partial h_t}{\partial h_k} \dfrac{\partial h_k}{\partial W}$$
    :   * To compute the _local derivative_ we need to compute:  
    :   $$\dfrac{\partial h_t}{\partial h_k}$$
    :   
    :   $$\begin{align}
        \dfrac{\partial h_t}{\partial h_k} &= \prod_{j=k+1}^t \dfrac{\partial h_j}{\partial h_{j-1}} \\
        &= \prod_{j=k+1}^t J_{j, j-1}
        \end{align}$$
    :   $$\:\:\:\:\:\:\:\:$$ where each $$J_{j, j-1}$$ is the __jacobina matrix__ of the partial derivatives of each respective  
        $$\:\:\:\:\:\:\:\:$$ hidden layer.

9. **The Vanishing Gradient Problem:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents29}  
    :   * __Analyzing the Norms of the Jacobians__ of each partial:  
    :   $$\| \dfrac{\partial h_j}{\partial h_{j-1}} \| \leq \| W^T \| \cdot \| \text{ diag}[f'(h_{j-1})] \| \leq \beta_W \beta_h$$
    :   $$\:\:\:\:\:\:\:$$ where we defined the $$\beta$$s as _upper bounds_ of the _norms_.        
    :   * __The Gradient is the product of these Jacobian Matrices__ (each associated with a step in the forward computation):  
    :   $$ \| \dfrac{\partial h_t}{\partial h_k} \| = \| \prod_{j=k+1}^t \dfrac{\partial h_j}{\partial h_{j-1}} \| \leq (\beta_W \beta_h)^{t-k}$$
    :   * *__Conclusion__*:  
            Now, as the exponential $$(t-k) \rightarrow \infty$$:  
            * __If $$(\beta_W \beta_h) < 1$$__:   
                $$(\beta_W \beta_h)^{t-k} \rightarrow 0$$.  
                known as __Vanishing Gradient__. 
            * __If $$(\beta_W \beta_h) > 1$$__:  
                $$(\beta_W \beta_h)^{t-k} \rightarrow \infty$$.  
                known as __Exploding Gradient__. 
    :   As the bound can become __very small__ or __very large__ quickly, the _locality assumption of gradient descent_ breaks down.

                    
            

                


6. **BPTT:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents26}  
    :   __for__ $$t$$ __from__ $$T$$ __to__ $$1$$ __do__  
            $$\ \ \ \ \ \ \ \ \ \ $$ $$do_t \leftarrow g_y'(o_t) · dL(y_t ; z_t)/dy_t$$  
            $$\ \ \ \ \ \ \ \ \ \ $$ $$db_o \leftarrow db_o + do_t$$  
            $$\ \ \ \ \ \ \ \ \ \ $$ $$dW_{oh} \leftarrow dW_{oh} + do_th_t^T$$  
            $$\ \ \ \ \ \ \ \ \ \ $$ $$dh_t \leftarrow dh_t + W_{oh}^T do_t$$  
            $$\ \ \ \ \ \ \ \ \ \ $$ $$dy_t \leftarrow g_h'(y_t) · dh_t$$  
            $$\ \ \ \ \ \ \ \ \ \ $$ $$dW_{hv} \leftarrow dW_{hv} + dy_tx_t^T$$  
            $$\ \ \ \ \ \ \ \ \ \ $$ $$db_h \leftarrow db_h + dy_t$$  
            $$\ \ \ \ \ \ \ \ \ \ $$ $$dW_{hh} \leftarrow dW_{hh} + dy_th_{t-1}^T$$  
            $$\ \ \ \ \ \ \ \ \ \ $$ $$dh_{t-1} \leftarrow W_{hh}^T dy_t$$  
        __Return__ $$\:\:\:\: d\theta = [dW_{hv}, dW_{hh}, dW_{oh}, db_h, db_o, dh_0]$$


7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents27}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents28}  
    :   

***

## RNNs Extra!
{: #content3}


5. **Vanishing/Exploding Gradients:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents35}  
    :   * __Exploding Gradients__:  
            * Truncated BPTT 
            * Clip gradients at threshold 
            * RMSprop to adjust learning rate 
        * __Vanishing Gradient__:   
            * Harder to detect 
            * Weight initialization 
            * ReLu activation functions 
            * RMSprop 
            * LSTM, GRUs 

1. **Applications:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents31}  
    :   * __NER__  
        * __Entity Level Sentiment in context__  
        * __Opinionated Expressions__

2. **Bidirectional RNNs:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents32}  
    :   * __Motivation__:  
            For _classification_ we need to incorporate information from words both preceding and following the word being processed
    :   ![img](/main_files/dl/nlp/rnn/1.png){: width="100%"}  
    :   $$\:\:\:\:$$ Here $$h = [\overrightarrow{h};\overleftarrow{h}]$$ represents (summarizes) the _past_ and the _future_ around a single token.
    :   * __Deep Bidirectional RNNs__:  
    :   ![img](/main_files/dl/nlp/rnn/2.png){: width="100%"}  
    :   $$\:\:\:\:\:$$ Each memory layer passes an _intermediate sequential representation_ to the next.

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents33}  
    :   

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents34}  
    :   

5. **Vanishing/Exploding Gradients:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents35}  
    :   * __Exploding Gradients__:  
            * Truncated BPTT 
            * Clip gradients at threshold 
            * RMSprop to adjust learning rate 
        * __Vanishing Gradient__:   
            * Harder to detect 
            * Weight initialization 
            * ReLu activation functions 
            * RMSprop 
            * LSTM, GRUs 