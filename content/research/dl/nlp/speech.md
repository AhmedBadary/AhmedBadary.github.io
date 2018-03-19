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

2. **Connectionist Temporal Classification (CTC):**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  
    :   __CTC__ is a type of _neural network output_ and _associated scoring function_, for training recurrent neural networks (RNNs) such as LSTM networks to tackle sequence problems where the _timing is variable_.  

1. **Structure:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  
    * __Input__:  
        A sequence of _observations_
    * __Output__:  
        A sequence of _labels_

3. **Algorithm :**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}  
    1. Extract the _Spectogram_ from the input
    2. Feed the _Spectogram_ into a _(bi-directional) RNN_
    3. At each frame, we apply a _softmax_ over the entire vocabulary that we are interested in (plus a _blank token_), producing a prediction _log probability_ for a _different token class_ at that time step called the __score__   

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents25}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents26}

7. **Applications:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents27}  
    :   * on-line Handwriting Recognition
        * Recognizing phonemes in speech audio  
        * ASR
