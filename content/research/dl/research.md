---
layout: NotesPage
title: Deep Learning <br /> Research Papers
permalink: /work_files/research/dl/nlp/research
prevLink: /work_files/research/dl/nlp.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Sequence to Sequence Learning with Neural Network](#content1)
  {: .TOC1}
  * [Towards End-to-End Speech Recognition with Recurrent Neural Networks](#content2)
  {: .TOC2}
  * [Attention-Based Models for Speech Recognition](#content3)
  {: .TOC3}
  * [4](#content4)
  {: .TOC4}
  * [5](#content5)
  {: .TOC5}
  * [6](#content6)
  {: .TOC6}
  * [7](#content7)
  {: .TOC7}
  * [8](#content8)
  {: .TOC8}
</div>

***
***

## Sequence to Sequence Learning with Neural Network
{: #content1}

1. **Introduction:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    This paper presents a general end-to-end approach to sequence learning that makes minimal assumptions (Domain-Independent) on the sequence structure.  
    It introduces __Seq2Seq__. 

2. **Structure:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}    
    * __Input__: sequence of input vectors  
    * __Output__: sequence of output labels
                
3. **Strategy:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    The idea is to use one LSTM to read the input sequence, one time step at a time, to obtain large fixed dimensional vector representation, and then to use another LSTM to extract the output sequence from that vector.  
    The second LSTM is essentially a recurrent neural network language model except that it is __conditioned__ on the __input sequence__.

4. **Solves:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    * Despite their flexibility and power, DNNs can only be applied to problems whose inputs and targets can be sensibly encoded with vectors of fixed dimensionality. It is a significant limitation, since many important problems are best expressed with sequences whose lengths are not known a-priori.  
        The RNN can easily map sequences to sequences whenever the alignment between the inputs the outputs is known ahead of time. However, it is not clear how to apply an RNN to problems whose input and the output sequences have different lengths with complicated and non-monotonic relationship.  


5. **Key Insights:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    * Uses LSTMs to capture the information present in a sequence of inputs into one vector of features that can then be used to decode a sequence of output features  
    * Uses two different LSTM, for the encoder and the decoder respectively  
    * Reverses the words in the source sentence to make use of short-term dependencies (in translation) that led to better training and convergence 

6. **Preparing Data (Pre-Processing):**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    :   
                    

7. **Architecture:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  
    :   * __Encoder__:  
            * *__LSTM:__* 
                * 4 Layers:    
                    * 1000 Dimensions per layer
                    * 1000-dimensional word embeddings
        * __Decoder__:  
            * *__LSTM:__* 
                * 4 Layers:    
                    * 1000 Dimensions per layer
                    * 1000-dimensional word embeddings
        * An __Output__ layer made of a standard __softmax function__  
            > over 80,000 words  
        * __Objective Function__:  
            <p>$$\dfrac{1}{\vert \mathbb{S} \vert} \sum_{(T,S) \in \mathbb{S}} \log p(T \vert S)
            $$</p>  
            where $$\mathbb{S}$$ is the training set.  
                
8. **Algorithm:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}  
:   * Train a large deep LSTM 
    * Train by maximizing the log probability of a correct translation $$T$$  given the source sentence $$S$$  
    * Produce translations by finding the most likely translation according to the LSTM:   
        <p>$$\hat{T} = \mathrm{arg } \max_{T} p(T \vert S)$$</p>
    * Search for the most likely translation using a simple left-to-right beam search decoder which maintains a small number B of partial hypotheses  
        > A __partial hypothesis__ is a prefix of some translation  
    * At each time-step we extend each partial hypothesis in the beam with every possible word in the vocabulary  
        > This greatly increases the number of the hypotheses so we discard all but the $$B$$  most likely hypotheses according to the model’s log probability  
    * As soon as the “<EOS>” symbol is appended to a hypothesis, it is removed from the beam and is added to the set of complete hypotheses  
    *

9. **Training:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents19}  
    :   * SGD
        * Momentum 
        * Half the learning rate every half epoch after the 5th epoch
        * Gradient Clipping  
            > enforce a hard constraint on the norm of the gradient
        * Sorting input

10. **Parameters:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents110}  
    :   * __Initialization__ of all the LSTM params with __uniform distribution__ $$\in [-0.08, 0.08]$$  
        * __Learning Rate__: $$0.7$$ 
        * __Batches__: $$28$$ sequences
        * __Clipping__: 
    :   $$g = 5g/\|g\|_2 \text{ if } \|g\|_2 > 5 \text{ else } g$$ 
                  

11. **Issues/The Bottleneck:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents111}  
    :   * The decoder is __approximate__  
        * The system puts too much pressure on the last encoded vector to capture all the (long-term) dependencies

12. **Results:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents112}  
    :   

13. **Discussion:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents113}  
    :   * Sequence to sequence learning is a framework that attempts to address the problem of learning variable-length input and output sequences. It uses an encoder RNN to map the sequential variable-length input into a fixed-length vector. A decoder RNN then uses this vector to produce the variable-length output sequence, one token at a time. During training, the model feeds the groundtruth labels as inputs to the decoder. During inference, the model performs a beam search to generate suitable candidates for next step predictions.

14. **Further Development:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents114}  
    :   

***

## Towards End-to-End Speech Recognition with Recurrent Neural Networks
{: #content2}

1. **Introduction:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  
    :   This paper presents an ASR system that directly transcribes audio data with text, __without__ requiring an _intermediate phonetic representation_.

2. **Structure:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}    
    :   * __Input__: 
        * __Output__:  
                

3. **Strategy:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}  
    :   The goal of this paper is a system where as much of the speech pipeline as possible is replaced by a single recurrent neural network (RNN) architecture.  
        The language model, however, will be lacking due to the limitation of the audio data to learn a strong LM. 

4. **Solves:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}  
    :   * First attempts used __RNNs__ or standard __LSTMs__. These models lacked the complexity that was needed to capture all the models required for ASR. 

5. **Key Insights:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents25}  
    :   * The model uses Bidirectional LSTMs to capture the nuances of the problem.  
        * The system uses a new __objective function__ that trains the network to directly optimize the __WER__.  

6. **Preparing the Data (Pre-Processing):**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents26}  
    :   The paper uses __spectrograms__ as a minimal preprocessing scheme.  

7. **Architecture:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents27}  
    :   The system is composed of:  
        * A __Bi-LSTM__  
        * A __CTC output layer__  
        * A __combined objective function__:  
            The new objective function at allows an RNN to be trained to optimize the expected value of an arbitrary loss function defined over output transcriptions (such as __WER__).  
            Given input sequence $$x$$, the distribution $$P(y\vert x)$$ over transcriptions sequences $$y$$ defined by CTC, and a real-valued transcription loss function $$\mathcal{L}(x, y)$$, the expected transcription loss $$\mathcal{L}(x)$$ is defined:  
            <p>$$\begin{align}
                \mathcal{L}(x) &= \sum_y P(y \vert x)\mathcal{L}(x,y) \\ 
                &= \sum_y \sum_{a \in \mathcal{B}^{-1}(y)} P(a \vert x)\mathcal{L}(x,y) \\
                &= \sum_a P(a \vert x)\mathcal{L}(x,\mathcal{B}(a))
                \end{align}$$</p>  
        <button>Show Derivation</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![Approximation and Differentiation](/main_files/dl/nlp/speech_research/3.png){: hidden="" width="80%"}


8. **Algorithm:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents28}  
    :   

9. **Issues/The Bottleneck:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents29}  
    :   

10. **Results:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents210}  
    :   * __WSJC__ (
    WER): 
            * Standard: $$27.3\%$$  
            * w/Lexicon of allowed words: $$21.9\%$$ 
            * Trigram LM: $$8.2\%$$ 
            * w/Baseline system: $$6.7\%$$

***

## Attention-Based Models for Speech Recognition
{: #content3}

1. **Introduction:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents31}  
    :   This paper introduces and extends the attention mechanism with features needed for ASR. It adds location-awareness to the attention mechanism to add robustness against different lengths of utterances.  

2. **Motivation:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents32}    
    :   Learning to recognize speech can be viewed as learning to generate a sequence (transcription) given another sequence (speech).  
        From this perspective it is similar to machine translation and handwriting synthesis tasks, for which attention-based methods have been found suitable. 
    :   __How ASR differs:__  
        Compared to _Machine Translation_, speech recognition differs by requesting much longer input sequences which introduces a challenge of distinguishing similar speech fragments in a single utterance.  
        > thousands of frames instead of dozens of words   
    :   It is different from _Handwriting Synthesis_, since the input sequence is much noisier and does not have a clear structure.  

2. **Structure:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents32}    
    :   * __Input__: $$x=(x_1, \ldots, x_{L'})$$ is a sequence of feature vectors  
            * Each feature vector is extracted from a small overlapping window of audio frames
        * __Output__: $$y$$ a sequence of __phonemes__   

3. **Strategy:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents33}  
    :   The goal of this paper is a system, that uses attention-mechanism with location awareness, whose performance is comparable to that of the conventional approaches.   
    :   * For each generated phoneme, an attention mechanism selects or weighs the signals produced by a trained feature extraction mechanism at potentially all of the time steps in the input sequence (speech frames).  
        * The weighted feature vector then helps to condition the generation of the next element of the output sequence.  
        * Since the utterances in this dataset are rather short (mostly under 5 seconds), we measure the ability of the considered models in recognizing much longer utterances which were created by artificially concatenating the existing utterances.

4. **Solves:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents34}  
    :   * __Problem__:  
            The [attention-based model proposed for NMT](https://arxiv.org/abs/1409.0473) demonstrates vulnerability to the issue of similar speech fragments with __longer, concatenated utterances__.  
            The paper argues that  this model adapted to track the absolute location in the input sequence of the content it is recognizing, a strategy feasible for short utterances from the original test set but inherently unscalable.  
        * __Solution__:  
            The attention-mechanism is modified to take into account the location of the focus from the previous step and the features of the input sequence by adding as inputs to the attention mechanism auxiliary *__Convolutional Features__* which are extracted by convolving the attention weights from the previous step with trainable filters.  

5. **Key Insights:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents35}  
    :   * Introduces attention-mechanism to ASR
        * The attention-mechanism is modified to take into account:  
            * location of the focus from the previous step  
            * features of the input sequence
        * Proposes a generic method of adding location awareness to the attention mechanism
        * Introduce a modification of the attention mechanism to avoid concentrating the attention on a single frame  

7. **Attention-based Recurrent Sequence Generator (ARSG):**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents37}  
    :   is a recurrent neural network that stochastically generates an output sequence $$(y_1, \ldots, y_T)$$ from an input $$x$$.  
    In practice, $$x$$ is often processed by an __encoder__ which outputs a sequential input representation $$h = (h_1, \ldots, h_L)$$ more suitable for the attention mechanism to work with.  
    :   The __Encoder__: a deep bidirectional recurrent network.  
        It forms a sequential representation h of length $$L = L'$$.  
    :   __Structure:__{: style="color: red"}    
        * *__Input__*: $$x = (x_1, \ldots, x_{L'})$$ is a sequence of feature vectors   
            > Each feature vector is extracted from a small overlapping window of audio frames.  
        * *__Output__*: $$y$$ is a sequence of phonemes
    :   __Strategy:__{: style="color: red"}    
        At the $$i$$-th step an ARSG generates an output $$y_i$$ by focusing on the relevant elements of $$h$$:  
    :   $$\begin{align}
        \alpha_i &= \text{Attend}(s_{i-1}, \alpha _{i-1}), h) & (1) \\
        g_i &= \sum_{j=1}^L \alpha_{i,j} h_j & (2) //
        y_i &\sim \text{Generate}(s_{i-1}, g_i) & (3)  
        \end{align}$$
    :   where $$s_{i−1}$$ is the $$(i − 1)$$-th state of the recurrent neural network to which we refer as the __generator__, $$\alpha_i \in \mathbb{R}^L$$ is a vector of the _attention weights_, also often called the __alignment__; and $$g_i$$ is the __glimpse__.  
        The step is completed by computing a *__new generator state__*:  
    :   $$s_i = \text{Recurrency}(s_{i-1}, g_i, y_i)$$  
    :   where the _Recurrency_ is an RNN.  
    :   ![img](/main_files/dl/nlp/speech_research/4.png){: width="100%"}  

12. **Attention-mechanism Types and Speech Recognition:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents312}  
    :   __Types of Attention:__{: style="color: red"}      
        * (Generic) Hybrid Attention: $$\alpha_i = \text{Attend}(s_{i-1}, \alpha_{i-1}, h)$$  
        * Content-based Attention: $$\alpha_i = \text{Attend}(s_{i-1}, h)$$   
            In this case, Attend is often implemented by scoring each element in h separately and normalizing the scores:  
            $$e_{i,j} = \text{Score}(s_{i-1}, h_j) \\$$ 
              $$\alpha_{i,j} = \dfrac{\text{exp} (e_{i,j}) }{\sum_{j=1}^L \text{exp}(e_{i,j})}$$  
            * __Limitations__:  
                The main limitation of such scheme is that identical or very similar elements of $$h$$ are scored equally regardless of their position in the sequence.  
                Often this issue is partially alleviated by an encoder such as e.g. a BiRNN or a deep convolutional network that encode contextual information into every element of h . However, capacity of h elements is always limited, and thus disambiguation by context is only possible to a limited extent.  
        * Location-based Attention: $$\alpha_i = \text{Attend}(s_{i-1}, \alpha_{i-1})$$   
            a location-based attention mechanism computes the alignment from the generator state and the previous alignment only.  
            * __Limitations__:  
                the model would have to predict the distance between consequent phonemes using $$s_{i−1}$$ only, which we expect to be hard due to large variance of this quantity.  
    :   Thus, we conclude that the __*Hybrid Attention*__ mechanism is a suitable candidate.  
        Ideally, we need an attention model that uses the previous alignment $$\alpha_{i-1}$$ to select a short list of elements from $$h$$, from which the content-based attention, will select the relevant ones without confusion.  

6. **Preparing the Data (Pre-Processing):**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents36}  
    :   The paper uses __spectrograms__ as a minimal preprocessing scheme.  

7. **Architecture:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents37}  
    :   Start with the __ARSG__-based model:  
        * __Encoder__: is a __Bi-RNN__  
        <p>$$e_{i,j} = w^T \tanh (Ws_{i-1} + Vh_j + b)$$</p>
        * __Attention__: Content-Based Attention extended for _location awareness_  
            <p>$$e_{i,j} = w^T \tanh (Ws_{i-1} + Vh_j + Uf_{i,j} + b)$$</p>
    :   __Extending the Attention Mechanism:__  
        Content-Based Attention extended for _location awareness_ by making it take into account the alignment produced at the previous step.  
        * First, we extract $$k$$ vectors $$f_{i,j} \in \mathbb{R}^k$$ for every position $$j$$ of the previous alignment $$\alpha_{i−1}$$ by convolving it with a matrix $$F \in \mathbb{R}^{k\times r}$$:  
            <p>$$f_i = F * \alpha_{i-1}$$</p>
        * These additional vectors $$f_{i,j} are then used by the scoring mechanism $$e_{i,j}$$:  
            <p>$$e_{i,j} = w^T \tanh (Ws_{i-1} + Vh_j + Uf_{i,j} + b)$$</p>  

                
            

8. **Algorithm:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents38}  
    :   

9. **Issues/The Bottleneck:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents39}  
    :   

10. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents310}  
    :   

***

## A Neural Transducer
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

## FIFTH
{: #content5}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents5 #bodyContents51}  
    :   

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

## Sixth
{: #content6}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents61}  
    :   

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents62}  
    :   

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents63}  
    :   

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents64}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents65}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents66}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents67}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents6 #bodyContents68  
    :   

## Seven
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

## Eight
{: #content8}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents81}  
    :   

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents82}  
    :   

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

## Nine
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

## Ten
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
