---
layout: NotesPage
title: ASR <br /> Research Papers
permalink: /work_files/research/dl/nlp/speech_research
prevLink: /work_files/research/dl/nlp.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Deep Speech](#content1)
  {: .TOC1}
  * [Towards End-to-End Speech Recognition with Recurrent Neural Networks](#content2)
  {: .TOC2}
  * [A Neural Transducer](#content3)
  {: .TOC3}
  * [FOURTH](#content4)
  {: .TOC4}
  * [FIFTH](#content5)
  {: .TOC5}
  * [SIXTH](#content6)
  {: .TOC6}
</div>

***
***

## Deep Speech 
{: #content2}

1. **Introduction:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    :   This paper takes a first attempt at an End-to-End system for ASR.  

2. **Structure:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}    
    :   * __Input__: vector of speech spectrograms  
            * An *__utterance__* $$x^{(i)}$$: is a time-series of length $$T^{(i)}$$ composed of time-slices where each is a vector of audio (spectrogram) features $$x_{t,p}^{(i)}, t=1,...,T^{(i)}$$, where $$p$$ denotes the power of the p'th frequency bin in the audio frame at time $$t$$.  
        * __Output__: English text transcript $$y$$  
    :   * __Goal__:  
            The goal of the RNN is to convert an input sequence $$x$$ into a sequence of character probabilities for the transcription $$y$$, with $$\tilde{y}_t = P(c_t\vert x)$$, where $$c_t \in \{\text{a, b, c, } \ldots \text{,  z, space,  apostrophe, blank}\}$$.
                
3. **Strategy:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    :   The goal is to replace the multi-part model with a single RNN network that captures as much of the information needed to do transcription in a single system.  

4. **Solves:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    :   * Previous models only used DNNs as a single component in a complex pipeline.  
            NNs are trained to classify __individual frames of acoustic data__, and then, their output distributions are reformulated as emission probabilities for a HMM.  
            In this case, the objective function used to train the networks is therefore substantially different from the true performance measure (sequence-level transcription accuracy.  
            This leads to problems where one system might have an improved accuracy rate but the overall transcription accuracy can still decrease.  
        *  An additional problem is that the frame-level training targets must be inferred from the alignments determined by the HMM. This leads to an awkward iterative procedure, where network retraining is alternated with HMM re-alignments to generate more accurate targets.  

5. **Key Insights:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    :   * As an __End-to-End__ model, this system avoids the problems of __multi-part__ systems that lead to inconsistent training criteria and difficulty of integration.   
            The network is trained directly on the text transcripts: no phonetic representation (and hence no pronunciation dictionary or state tying) is used.  
        * Using __CTC__ objective, the system is able to better approximate and solve the alignment problem avoiding HMM realignment training.  
            Since CTC integrates out over all possible input-output alignments, no forced alignment is required to provide training targets.  
        * The Dataset is augmented with newly synthesized data and modified to include all the variations and effects that face ASR problems.    
            This greatly increases the system performance on particularly noisy/affected speech.  

6. **Preparing Data (Pre-Processing):**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    :   The paper uses __spectrograms__ as a minimal preprocessing scheme.  
    :   ![img](/main_files/dl/nlp/speech_research/2.png){: width="60%"}    

7. **Architecture:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  
    :   The system is composed of:  
        ![img](/main_files/dl/nlp/speech_research/1.png){: width="80%"}    
        * An __RNN__:    
            * 5 layers of __hidden units__:  
                * 3 Layer of __Feed-forward Nets__:  
                    * For the __input layer__, the output depends on the spectrogram frame $$x_t$$ along with a context of $$C$$ frames on each side.  
                        > $$C \in \{5, 7, 9\}$$  
                    * The non-recurrent layers operate on independent data for each time step:  
                        $$h_t^{(l)} = g(W^{(l)} h_{(t)}^{(l-1)} + b^{(l)}),$$  
                        where $$g(z) = \min \{\max \{0, z\}, 20\}$$ is the *clipped RELU*.    
                * 2 layers of __Recurrent Nets__:  
                    * 1 layer of a __Bi-LSTM__:  
                        * Includes two sets of hidden units: 
                            A set with forward recurrence $$h^{(f)}$$  
                            A set with backward recurrence $$h^{(b)}$$:  
                            $$h_t^{(f)} = g(W^{(4)}h_t^{(3)} + W_r^{(b)} h_{t-1}^{(b)} + b ^{(4)}) \\ 
                            h_t^{(b)} = g(W^{(4)}h_t^{(3)} + W_r^{(b)} h_{t+1}^{(b)} + b ^{(4)})$$  
                            > Note that $$h^{(f)}$$ must be computed sequentially from $$t = 1$$ to $$t = T^{(i)}$$ for the i’th utterance, while the units $$h^{(b)}$$ must be computed sequentially in reverse from $$t = T^{(i)}$$ to $$t = 1$$.  
                    * 1 layer of __Feed-forward Nets__:   
                        * The fifth (non-recurrent) layer takes both the forward and backward units as inputs:  
                            $$h_t^{(5)} = g(W ^{(5)}h_t ^{(4)} + b ^{(5)}),$$  
                            where $$h_t^{(4)} = h_t^{(f)} + h_t^{(b)}$$ 
            * An __Output__ layer made of a standard __softmax function__ that yields the predicted character probabilities for each time-slice $$t$$ and character $$k$$ in the alphabet:   
                $$\displaystyle{h _{(t,k)} ^{(6)} = \hat{y} _{(t,k)} = P(c_t = k \vert x) = \dfrac{\exp (W_k ^{(6)} h_t ^{(5)} + b_k ^{(6)})}{\sum_j \exp (W_j ^{(6)}h_t ^{(5)} + b_j ^{(6)})}},$$  
                where $$W_k ^{(6)}$$ and $$b_k ^{(6)}$$ denote the k'th column of the weight matrix and k'th bias.  
        * A *__CTC__* __Loss Function__ $$\mathcal{L}(\hat{y}, y)$$  
        * An *__N-gram Language Model__* 
        * A __combined Objective Function__:  
    :   $$Q(c) = \log (P(x \vert x)) + \alpha \log (P_{\text{LM}}(c) + \beta \text{word_count}(c))$$ 

8. **Algorithm:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}  
    :   * Given the output $$P(c \vert x)$$ of the RNN: perform a __search__ to find the sequence of characters $$c_1, c_2, ...$$ that is most probable according to both:  
            1. The RNN Output
            2. The Language Model  
        * We maximize the combined objective:  
            $$Q(c) = \log (P(x \vert x)) + \alpha \log (P_{\text{LM}}(c) + \beta \text{word_count}(c))$$  
            where the term $$P_{\text{lm}} denotes the probability of the sequence $$c$$ according to the N-gram model.  
        * The objective is maximized using a highly optimized __beam search__ algorithm  
            > beam size: 1000-8000

9. **Training:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents19}  
    :   * The gradient of the CTC Loss $$\nabla_{\hat{y}} \mathcal{L}(\hat{y}, y)$$ with respect to the net outputs given the ground-truth character sequence $$y$$ is computed
    :   * Nesterov’s Accelerated gradient
        * Nesterov Momentum
        * Annealing the learning rate by a constant factor
        * Dropout  
        * Striding -- shortening the recurrent layers by taking strides of size $$2$$.  
            The unrolled RNN will have __half__ as many steps.  
            > similar to a convolutional network with a step-size of 2 in the first layer.  

10. **Parameters:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents110}  
    :   * __Momentum__: $$0.99$$ 
        * __Dropout__: $$5-10 \%$$ (FFN only)   
        * __Trade-Off Params__: use cross-validation for $$\alpha, \beta$$  

11. **Issues/The Bottleneck:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents111}  
    :   

12. **Results:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents112}  
    :   * __SwitchboardHub5’00__  (
    WER): 
            * Standard: $$16.0\%$$  
            * w/Lexicon of allowed words: $$21.9\%$$ 
            * Trigram LM: $$8.2\%$$ 
            * w/Baseline system: $$6.7\%$$

13. **Discussion:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents113}  
    :   * __Why avoid LSTMs__:  
            One disadvantage of LSTM cells is that they require computing and storing multiple gating neuron responses at each step.  
            Since the forward and backward recurrences are sequential, this small additional cost can become a computational bottleneck.  
    :   * __Why a homogeneous model__:  
             By using a homogeneous model we have made the computation of the recurrent activations as efficient as possible: computing the ReLu outputs involves only a few highly optimized BLAS operations on the GPU and a single point-wise nonlinearity.

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
    :   

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

## A Neural Transducer
{: #content3}

1. **Introduction:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents31}  
    :   This paper presents an ASR system that directly transcribes audio data with text, __without__ requiring an _intermediate phonetic representation_.

2. **Structure:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents32}    
    :   * __Input__:  
        * __Output__:  
                

3. **Strategy:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents33}  
    :   The goal of this paper is a system where as much of the speech pipeline as possible is replaced by a single recurrent neural network (RNN) architecture.  
        The language model, however, will be lacking due to the limitation of the audio data to learn a strong LM. 

4. **Solves:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents34}  
    :   * Previous models only used DNNs as a single component in a complex pipeline.  
            NNs are trained to classify __individual frames of acoustic data__, and then, their output distributions are reformulated as emission probabilities for a HMM.  
            In this case, the objective function used to train the networks is therefore substantially different from the true performance measure (sequence-level transcription accuracy.  
            This leads to problems where one system might have an improved accuracy rate but the overall transcription accuracy can still decrease.  
        *  An additional problem is that the frame-level training targets must be inferred from the alignments determined by the HMM. This leads to an awkward iterative procedure, where network retraining is alternated with HMM re-alignments to generate more accurate targets. 

5. **Key Insights:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents35}  
    :   * As an __End-to-End__ model, this system avoids the problems of __multi-part__ systems that lead to inconsistent training criteria and difficulty of integration.   
            The network is trained directly on the text transcripts -- no phonetic representation (and hence no pronunciation dictionary or state tying) is used.  
        * Using __CTC__ objective, the system is able to better approximate and solve the alignment problem avoiding HMM realignment training.  
            Since CTC integrates out over all possible input-output alignments, no forced alignment is required to provide training targets.  
        * The system uses a new __objective function__ that trains the network to directly optimize the __WER__.  

6. **Preparing the Data (Pre-Processing):**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents36}  
    :   The paper uses __spectrograms__ as a minimal preprocessing scheme.  

7. **Architecture:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents37}  
    :   

8. **Algorithm:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents38}  
    :   

9. **Issues/The Bottleneck:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents39}  
    :   

10. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents310}  
    :   

***

## A Neural Transducer
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

## FOURTH
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

