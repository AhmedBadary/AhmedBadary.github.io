---
layout: NotesPage
title: Gated Units <br /> RNN Architectures
permalink: /work_files/research/dl/nlp/gated_units
prevLink: /work_files/research/dl/nlp.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [GRUs](#content2)
  {: .TOC2}
  * [LSTMs](#content3)
  {: .TOC3}
</div>

***
***

## GRUs
{: #content2}

1. **Gated Recurrent Units:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21} 
    :   __Gated Recurrent Units (GRUs)__ are a class of modified (_**Gated**_) RNNs that allow them to combat the _vanishing gradient problem_ by allowing them to capture more information/long range connections about the past (_memory_) and decide how strong each signal is.  

2. **Main Idea:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22} 
    :   Unlike _standard RNNs_ which compute the hidden layer at the next time step directly first, __GRUs__ computes two additional layers (__gates__):  
        > Each with different weights
    :   * *__Update Gate__*:  
    :   $$z_t = \sigma(W^{(z)}x_t + U^{(z)}h_{t-1})$$  
    :   * *__Reset Gate__*:  
    :   $$r_t = \sigma(W^{(r)}x_t + U^{(r)}h_{t-1})$$  
    :   The __Update Gate__ and __Reset Gate__ computed, allow us to more directly influence/manipulate what information do we care about (and want to store/keep) and what content we can ignore.  
        We can view the actions of these gates from their respecting equations as:  
    :   * *__New Memory Content__*:  
            at each hidden layer at a given time step, we compute some new memory content,  
            if the reset gate $$ = ~0$$, then this ignores previous memory, and only stores the new word information.  
    :   $$ \tilde{h}_t = \tanh(Wx_t + r_t \odot Uh_{t-1})$$
    :   * *__Final Memory__*:  
            the actual memory at a time step $$t$$, combines the _Current_ and _Previous time steps_,  
            if the _update gate_ $$ = ~0$$, then this, again, ignores the _newly computed memory content_, and keeps the old memory it possessed.  
    :   $$h_t = z_t \odot h_{t-1} + (1-z_t) \odot \tilde{h}_t$$  

***

## Long Short-Term Memory
{: #content3}

1. **LSTM:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents31} 
    :   The __Long Short-Term Memory__ (LSTM) Network is a special case of the Recurrent Neural Network (RNN) that uses special gated units (a.k.a LSTM units) as building blocks for the layers of the RNN.  

2. **Architecture:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents32} 
    :   The LSTM, usually, has four gates:  
    :   * __Input Gate__: 
            The input gate determines how much does the _current input vector (current cell)_ matters      
    :   $$i_t = \sigma(W^{(i)}x_t + U^{(i)}h_{t-1})$$ 
    :   * __Forget Gate__: 
            Determines how much of the _past memory_, that we have kept, is still needed   
    :   $$i_t = \sigma(W^{(i)}x_t + U^{(i)}h_{t-1})$$ 
    :   * __Output Gate__: 
            Determines how much of the _current cell_ matters for our _current prediction (i.e. passed to the sigmoid)_
    :   $$i_t = \sigma(W^{(i)}x_t + U^{(i)}h_{t-1})$$  
    :   * __Memory Cell__: 
            The memory cell is the cell that contains the _short-term memory_ collected from each input
    :   $$\begin{align}
            \tilde{c}_t & = \tanh(W^{(c)}x_t + U^{(c)}h_{t-1}) & \text{New Memory} \\
            c_t & = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t & \text{Final Memory}
        \end{align}$$
    :   The __Final Hidden State__ is calculated as follows:  
    :   $$h_t = o_t \odot \sigma(c_t)$$
     

3. **Properties:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents33} 
    :   * __Syntactic Invariance__:  
            When one projects down the vectors from the _last time step hidden layer_ (with PCA), one can observe the spatial localization of _syntacticly-similar sentences_  
            ![img](/main_files/dl/nlp/9/5.png){: width="100%"}  
