---
layout: NotesPage
title: ASR <br /> Automatic Speech Recognition
permalink: /work_files/research/dl/nlp/speech
prevLink: /work_files/research/dl/nlp.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Introduction](#content1)
  {: .TOC1}
  * [Connectionist Temporal Classification](#content2)
  {: .TOC2}
  * [LAS - Seq2Seq with Attention](#content3)
  {: .TOC3}
  * [Online Seq2Seq Models](#content4)
  {: .TOC4}
</div>

***
***

## Introduction
{: #content1}

1. **Classical Approach:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    :   Classically, _Speech Recognition_ was developed as a big machine incorporating different models from different fields.  
        The models were _statistical_ and they started from _text sequences_ to _audio features_.  
        Typically, a _generative language model_ is trained on the sentences for the intended language, then, to make the features, _pronunciation models_, _acoustic models_, and _speech processing models_ had to be developed. Those required a lot of feature engineering and a lot of human intervention and expertise and were very fragile.
    :   ![img](/main_files/dl/nlp/12/1.png){: width="100%"}  
    :   __Recognition__ was done through __*Inference*__: Given audio features $$\mathbf{X}=x_1x_2...x_t$$ infer the most likely tedxt sequence $$\mathbf{Y}^\ast=y_1y_2...y_k$$ that caused the audio features.
    :   $$\displaystyle{\mathbf{Y}^\ast =\mathrm{arg\,min}_{\mathbf{Y}} p(\mathbf{X} \vert \mathbf{Y}) p(\mathbf{Y})}$$

2. **The Neural Network Age:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    :   Researchers realized that each of the (independent) components/models that make up the ASR can be improved if it were replaced by a _Neural Network Based Model_.  
    :   ![img](/main_files/dl/nlp/12/2.png){: width="100%"}  

3. **The Problem with the component-based System:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    :   * Each component/model is trained _independently_, with a different _objective_  
        * Errors in one component may not behave well with errors in another component

4. **Solution to the Component-Based System:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    :   We aim to train models that encompass all of these components together, i.e. __End-to-End Model__:  
        * __Connectionist Temporal Classification (CTC)__
        * __Sequence-to-Sequence Listen Attend and Spell (LAS)__
                    
5. **End-to-End Speech Recognition:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    :   We treat __End-to-End Speech Recognition__ as a _modeling task_.
    :   Given __Audio__ $$\mathbf{X}=x_1x_2...x_t$$ (audio/processed spectogram) and corresponding output text $$\mathbf{Y}=y_1y_2...y_k$$  (transcript), we want to learn a *__Probabilistic Model__* $$p(\
    mathbf{Y} \vert \mathbf{X})$$ 

***

## Connectionist Temporal Classification
{: #content2}

1. **Motivation:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  
    :   * RNNs require a _target output_ at each time step 
        * Thus, to train an RNN, we need to __segment__ the training output (i.e. tell the network which label should be output at which time-step) 
        * This problem usually arises when the timing of the input is variable/inconsistent (e.g. people speaking at different rates/speeds)

2. **Connectionist Temporal Classification (CTC):**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  
    :   __CTC__ is a type of _neural network output_ and _associated scoring function_, for training recurrent neural networks (RNNs) such as LSTM networks to tackle sequence problems where the _timing is variable_.  
    :   Due to time variability, we don't know the __alignment__ of the __input__ with the __output__.  
        Thus, CTC considers __all possible alignments__.  
        Then, it gets a __closed formula__ for the __probability__ of __all these possible alignments__ and __maximizes__ it.

1. **Structure:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  
    * __Input__:  
        A sequence of _observations_
    * __Output__:  
        A sequence of _labels_

3. **Algorithm :**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}  
   ![img](/main_files/dl/nlp/12/3.png){: width="80%"}  
    1. Extract the (*__LOG MEL__*) _Spectogram_ from the input  
        > Use raw audio iff there are multiple microphones
    2. Feed the _Spectogram_ into a _(bi-directional) RNN_
    3. At each frame, we apply a _softmax_ over the entire vocabulary that we are interested in (plus a _blank token_), producing a prediction _log probability_ (called the __score__) for a _different token class_ at that time step.   
        * Repeated Tokens are duplicated
        * Any original transcript is mapped to by all the possible paths in the duplicated space
        * The __Score (log probability)__ of any path is the sum of the scores of individual categories at the different time steps
        * The probability of any transcript is the sum of probabilities of all paths that correspond to that transcript
        * __Dynamic Programming__ allopws is to compute the log probability $$p(\mathbf{Y} \vert \mathbf{X})$$ and its gradient exactly.  
    ![img](/main_files/dl/nlp/12/4.png){: width="80%"}  

5. **Analysis:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents25}  
    :   The _ASR_ model consists of an __RNN__ plus a __CTC__ layer.    
        Jointly, the model learns the __pronunciation__ and __acoustic__ model _together_.  
        However, a __language model__ is __not__ learned, because the RNN-CTC model makes __strong conditional independence__ assumptions (similar to __HMMs__).  
        Thus, the RNN-CTC model is capable of mapping _speech acoustics_ to _English characters_ but it makes many _spelling_ and _grammatical_ mistakes.  
        Thus, the bottleneck in the model is the assumption that the _network outputs_ at _different times_ are __conditionally independent__, given the _internal state_ of the network. 

4. **Improvements:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}  
    :   * Add a _language model_ to CTC during training time for _rescoring_.
           This allows the model to correct spelling and grammar.
        * Use _word targets_ of a certain vocabulary instead of characters 

7. **Applications:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents27}  
    :   * on-line Handwriting Recognition
        * Recognizing phonemes in speech audio  
        * ASR

***

## LAS - Seq2Seq with Attention
{: #content3}

1. **Motivation:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents31}  
    :   The __CTC__ model can only make predictions based on the data; once it has made a prediction for a given frame, it __cannot re-adjust__ the prediction.  
    :   Moreover, the _strong independence assumptions_ that the CTC model makes doesn't allow it to learn a _language model_.   

2. **Listen, Attend and Spell (LAS):**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents32}  
    :   __LAS__ is a neural network that learns to transcribe speech utterances to characters.  
        In particular, it learns all the components of a speech recognizer jointly.
    :   ![img](/main_files/dl/nlp/12/5.png){: width="80%"}  
    :   The model is a __seq2seq__ model; it learns a _conditional probability_ of the next _label/character_ given the _input_ and _previous predictions_ $$p(y_{i+1} \vert y_{1..i}, x)$$.  
    :   The approach that __LAS__ takes is similar to that of __NMT__.     
        Where, in translation, the input would be the _source sentence_ but in __ASR__, the input is _the audio sequence_.  
    :   __Attention__ is needed because in speech recognition tasks, the length of the input sequence is very large; for a 10 seconds sample, there will be ~10000 frames to go through.      

3. **Structure:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents33}  
    :   The model has two components:  
        * __A listener__: a _pyramidal RNN **encoder**_ that accepts _filter bank spectra_ as inputs
        * __A Speller__: an _attention_-based _RNN **decoder** _ that emits _characters_ as outputs 
    :   * __Input__:  
            
        * __Output__:  
            

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents34}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents35}  
    :   

6. **Limitations:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents36}  
    :   * Not an online model - input must all be received before transcripts can be produced
        * Attention is a computational bottleneck since every output token pays attention to every input time step
        * Length of input has a big impact on accuracy


***

## Online Seq2Seq Models
{: #content4}

1. **Motivation:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents41}  
    :   * __Overcome limitations of seq2seq__:  
            * No need to wait for the entire input sequence to arrive
            * Avoids the computational bottleneck of Attention over the entire sequence
        * __Produce outputs as inputs arrive__:  
            * Solves this problem: When has enough information arrived that the model is confident enough to output symbols 

2. **A Neural Transducer:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents42}  
    :    Neural Transducer is a more general class of seq2seq learning models. It avoids the problems of offline seq2seq models by operating on local chunks of data instead of the whole input at once. It is able to make predictions _conditioned on partially observed data and partially made predictions_.    

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

## Eight
{: #content8}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents81}  
    :   

2. **Speech Problems and Considerations:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents82}  
    :   * __ASR__:  
            * Spontaneous vs Read speech
            * Large vs Small Vocabulary
            * Noisy vs Clear input
            * Low vs High Resources 
            * Near-field vs Far-field input
            * Accent-independence 
            * Speaker-Adaptive vs Stand-Alone (speaker-independent) 
            * The cocktail party problem 
        * __TTS__:  
            * Low Resource
            * Realistic prosody
        * __Speaker Identification__
        * __Speech Enhancement__
        * __Speech Separation__       

3. **Acoustic Representation:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents83}  
    :   __What is speech?__{: style="color: red"}  
        * Waves of changing air pressure - Longitudinal Waves (consisting of compressions and rarefactions)
        * Realized through excitation from the vocal cords
        * Modulated by the vocal tract and the articulators (tongue, teeth, lips) 
        * Vowels are produced with an open vocal tract (stationary)
            > parametrized by position of tongue
        * Consonants are constrictions of vocal tract
        * They get __converted__ to _Voltage_ with a microphone
        * They are __sampled__ (and quantized) with an _Analogue to Digital Converter_ 
    :   __Speech as waves:__{: style="color: red"}  
        * Human hearing range is: $$~50 HZ-20 kHZ$$
        * Human speech range is: $$~85 HZ-8 kHZ$$
        * Telephone speech sampling is $$8 kHz$$ and a bandwidth range of $$300 Hz-4 kHz$$ 
        * 1 bit per sample is intelligible
        * Contemporary Speech Processing mostly around 16 khz 16 bits/sample  
            > A lot of data to handle
    :   __Speech as vectors (digits):__{: style="color: red"}  
        * We seek a *__low-dimensional__* representation to ease the computation  
        * The low-dimensional representation needs to be __invariant to__:  
            * Speaker
            * Background noise
            * Rate of Speaking
            * etc.
        * We apply __Fourier Analysis__ to see the energy in different frequency bands, which allows analysis and processing
            * Specifically, we apply _windowed short-term_ *__Fast Fourier Transform (FFT)__*  
                > e.g. FFT on overlapping $$25ms$$ windows (400 samples) taken every $$10ms$$  
        * FFT is still too high-dimensional  
            * We __Downsample__ by local weighted averages on _mel scale_ non-linear spacing, an d take a log:  
                $$ m = 1127 \ln(1+\dfrac{f}{700})$$  
            * This results in *__log-mel features__*, $$40+$$ dimensional features per frame    
                > Default for NN speech modelling  
    :   __Speech dimensionality for different models:__{: style="color: red"}  
        * __Gaussian Mixture Models (GMMs)__: 13 *__MFCCs__*  
            * *__MFCCs - Mel Frequency Cepstral Coefficients__*: are the discrete cosine transformation (DCT) of the mel filterbank energies \| Whitened and low-dimensional.  
                They are similar to _Principle Components_ of log spectra.  
            __GMMs__ used local differences (deltas) and second-order differences (delta-deltas) to capture the dynamics of the speech $$(13 \times 3 \text{ dim})$$
        * __FC-DNN__: 26 stacked frames of *__PLP__*  
            * *__PLP - Perceptual Linear Prediction__*: a common alternative representation using _Linear Discriminant Analysis (LDA)_  
                > Class aware __PCA__    
        * __LSTM/RNN/CNN__: 8 stacked frames of *__PLP__*  
    :   __Speech as Communication:__{: style="color: red"}      
        * Speech Consists of sentences (in ASR we usually talk about "utterances")  
        * Sentences are composed of words 
        * Minimal unit is a "phoneme" Minimal unit that distinguishes one word from another.
            * Set of 40-60 distinct sounds.
            * Vary per language 
            * Universal representations: 
                * *__IPA__* : international phonetic alphabet
                * *__X-SAMPA__* : (ASCII) 
        * *__Homophones__* : distinct words with the same pronunciation. (e.g. "there" vs "their") 
        * *__Prosody__* : How something is said can convey meaning. (e.g. "Yeah!" vs "Yeah?")  

4. **(Approximate) History of ASR:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents84}  
    * 1960s Dynamic Time Warping 
    * 1970s Hidden Markov Models 
    * Multi-layer perdptron 1986 
    * Speech recognition with neural networks 1987-1995 
    * Superseded by GMMs 1995-2009 
    * Neural network features 2002— 
    * Deep networks 2006— (Hinton, 2002) 
    * Deep networks for speech recognition:
        * Good results on TIMIT (Mohamed et al., 2009) 
        * Results on large vocabulary systems 2010 (Dahl et al., 2011) * Google launches DNN ASR product 2011
        * Dominant paradigm for ASR 2012 (Hinton et al., 2012) 
    * Recurrent networks for speech recognition 1990, 2012 - New models (CTC attention, LAS, neural transducer) 

5. **Datasets:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents85}  
    * __TIMIT__: 
        * Hand-marked phone boundaries are given 
        * 630 speakers $$\times$$ 10 utterances 
    * __Wall Street Journal (WSJ)__ 1986 Read speech. WSJO 1991, 30k vocab 
    * __Broadcast News (BN)__ 1996 104 hours 
    * __Switchboard (SWB)__ 1992. 2000 hours spontaneous telephone speech -  500 speakers 
    * __Google voice search__ - anonymized live traffic 3M utterances 2000 hours hand-transcribed 4M vocabulary. Constantly refreshed, synthetic reverberation + additive noise 
    * __DeepSpeech__ 5000h read (Lombard) speech + SWB with additive noise. 
    * __YouTube__ 125,000 hours aligned captions (Soltau et al., 2016) 


6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents86}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents87}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents8 #bodyContents88}  
    :   

***

## The Methods and Models of Speech Recognition
{: #content9}

1. **Probabilistic Speech Recognition:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents91}    
    :   We can view the problem of __ASR__ as a _sequence labeling_ problem, and, so, use statistical models (such as HMMs) to model the conditional probabilities between the states/words by viewing speech signal as a piecewise stationary signal or a short-time stationary signal. 
    :   * __Representation__: we _represent_ the _speech signal_ as an *__observation sequence__* $$o = \{o_t\}$$  
        * __Goal__: find the most likely _word sequence_ $$\hat{w}$$   
        * __Set-Up__:  
            * The system has a set of discrete states
            * The transitions from state to state are markovian and are according to the transition probabilities  
                > __Markovian__: Memoryless  
            * The _Acoustic Observations_ when making a transition are conditioned on _the state alone_ $$P(o_t \vert c_t)$$
            * The _goal_ is to _recover the state sequence_ and, consequently, the _word sequence_  
                
2. **Fundamental Equation of Speech Recognition:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents92}  
    :   We set the __decoders output__ as the *__most likely sequence__* $$\hat{w}$$ from all the possible sequences, $$\mathcal{S}$$, for an observation sequence $$o$$:  
    :   $$\begin{align}
            \hat{w} & = \mathrm{arg } \max_{w \in \mathcal{S}} P(w \vert o) & (1) \\
            & = \mathrm{arg } \max_{w \in \mathcal{S}} P(o \vert w) P(w) & (2)
            \end{align}
        $$  
    :   The __Conditional Probability of a sequence of observations given a sequence of (predicted) word__ is a _product_ of an __Acoustic Model__ and a __Language Model__ scores:  
    :   $$P(o \vert w) = \sum_{d,c,p} P(o \vert c) P(c \vert p) P(p \vert w)$$ 
    :   where $$p$$ is the __phone sequence__ and $$c$$ is the __state sequence__.  

3. **Speech Recognition as Transduction:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents93}  
    :   The problem of speech recognition can be seen as a transduction problem - mapping different forms of energy to other forms (representations).  
        Basically, we are going from __Signal__ to __Language__.  
        ![img](/main_files/dl/nlp/12/6.png){: width="60%"}    

4. **Gaussian Mixture Models:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents94}  
    :   * Dominant paradigm for ASR from 1990 to 2010 
        * Model the probability distribution of the acoustic features for each state.  
            $$P(o_t \vert c_i) = \sum_j w_{ij} N(o_t; \mu_{ij}, \sigma_{ij})$$   
        * Often use diagonal covariance Gaussians to keep number of parameters under control. 
        * Train by the E-M (Expectation Maximization) algorithm (Dempster et al., 1977) alternating:  
            * __M__: forced alignment computing the maximum-likelihood state sequence for each utterance 
            * __E__: parameter $$(\mu , \sigma)$$ estimation  
        * Complex training procedures to incrementally fit increasing numbers of components per mixture:  
            * More components, better fit - 79 parameters component. 
        * Given an alignment mapping audio frames to states, this is parallelizable by state.   
        * Hard to share parameters/data across states.  
    :   __Forced Alignment:__  
        * Forced alignment uses a model to compute the maximum likelihood alignment between speech features and phonetic states. 
        * For each training utterance, construct the set of phonetic states for the ground truth transcription. 
        * Use Viterbi algorithm to find ML monotonic state sequence 
        * Under constraints such as at least one frame per state. 
        * Results in a phonetic label for each frame. 
        * Can give hard or soft segmentation.  
        ![img](/main_files/dl/nlp/12/7.png){: width="60%"}  
    * <button>Algorithm/Training</button>{: .showText value="show"
     onclick="showTextPopHide(event);"}
    ![formula](/main_files/dl/nlp/12/8.png){: width="70%" hidden=""}   
    * __Decoding:__   
        ![img](/main_files/dl/nlp/12/9.png){: width="20%"}  
        * Speech recognition Unfolds in much the same way.
        *  Now we have a graph instead of a straight-through path.
        *  Optional silences between words Alternative pronunciation paths.
        *  Typically use max probability, and work in the log domain.
        *  Hypothesis space is huge, so we only keep a "beam" of the best paths, and can lose what would end up being the true best path.   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents95}  
    :   * __Two Paradigms of Neural Networks for Speech__:  
            * Use neural networks to compute nonlinear feature representations:      
                * "Bottleneck" or "tandem" features (Hermansky et al., 2000)
                * Low-dimensional representation is modelled conventionally with GMMs.
                * Allows all the GMM machinery and tricks to be exploited. 
                * _Bottleneck features_ outperform _Posterior features_ (Grezl et al. 2017)
                * Generally, __DNN features + GMMs__ reach the same performance as hybrid __DNN-HMM__ systems but are much more _complex_
            * Use neural networks to estimate phonetic unit probabilities  

6. **Hybrid Networks:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents96}  
    :   * Train the network as a classifier with a softmax across the __phonetic units__  
        * Train with __cross-entropy__
        * Softmax:   
    :   $$y(i) = \dfrac{e^{\psi(i, \theta)}}{\sum_{j=1}^N e^{\psi(j, \theta)}}$$ 
    :   * We _converge to/learn_ the __posterior probability across phonetic states__:  
    :   $$P(c_i \vert o_t)$$   
    :   * We, then, model $$P(o \vert c)$$ with a __Neural-Net__ instead of a __GMM__:   
            > We can ignore $$P(o_t)$$ since it is the same for all decoding paths   
    :   $$\begin{align}
            P(o \vert c) & = \prod_t P(o_t \vert c_t) & (3) \\
            P(o_t \vert c_t) & = \dfrac{P(c_t \vert o_t) P(o_t)}{P(c_t)} & (4) \\
            & \propto \dfrac{P(c_t \vert o_t)}{P(c_t)} & (5) \\
            \end{align}
        $$  
    :   * The __log scaled posterior__  from the last term:  
    :   $$\log P(o_t \vert c_t) = \log P(c_t \vert o_t) - \alpha \log P(c_t)$$ 
    :   * Empirically, a *__prior smoothing__* on $$\alpha$$ $$(\alpha \approx 0.8)$$ works better 
    :   * __Input Features__:  
            * NN can handle high-dimensional, correlated, features
            * Use (26) stacked filterbank inputs (40-dim mel-spaced filterbanks)
    :   * __NN Architectures for ASR__:  
            * *__Fully-Connected DNN__*  
            * *__CNNs__*: 
                * Time delay neural networks: 
                    * Waibel et al. (1989) 
                    * Dilated convolutions (Peddinti et al., 2015)  
                        > Pooling in time results in a loss of information.  
                        > Pooling in frequency domain is more tolerable  
                * CNNs in time or frequency domain:
                    * Abdel-Hamid et al. (2014)
                    * Sainath et al. (2013) 
                * Wavenet (van den Oord et al., 2016) 
            * *__RNNs__* :  
                * RNN (Robinson and Fallside, 1991) 
                * LSTM Graves et al. (2013)
                * Deep LSTM-P Sak et al. (2014b)
                * CLDNN (Sainath et al , 2015a)
                * GRU. DeepSpeech 1/2 (Amodei et al., 2015)

                * Bidirectional (Schuster and Paliwal, 1997) helps, but introduces latency. 
                * Dependencies not long at speech frame rates (100Hz).
                * Frame stacking and down-sampling help. 


7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents97}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents98}  
    :   