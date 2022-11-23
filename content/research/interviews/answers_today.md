---
layout: NotesPage
title: Answers to Prep Questions (Learning)
permalink: /work_files/research/answers_today
prevLink: /work_files/research.html
---


1. __Write down the Convolution operation and the cross-correlation over two axes and:__{: style="color: red"}  
    1. __Convolution:__{: style="color: blue"}  
    1. __Convolution (commutative):__{: style="color: blue"}  
    1. __Cross-Correlation:__{: style="color: blue"}  
1. __Measuring Information - Definitions and Formulas:__{: style="color: red"}  
    1. __In Shannons Theory, how do we quantify *"transmitting 1 bit of information"*?__{: style="color: blue"}  
        To __transmit $$1$$ bit of information__ means to __divide the recipients *Uncertainty* by a factor of $$2$$__.  
    1. __What is *the amount of information transmitted*?__{: style="color: blue"}  
        The __amount of information__ transmitted is the __logarithm__ (base $$2$$) of the __uncertainty reduction factor__.   
    1. __What is the *uncertainty reduction factor*?__{: style="color: blue"}  
        It is the __inverse of the probability__ of the event being communicated.  
    1. __What is the *amount of information in an event $$x$$*?__{: style="color: blue"}  
        The __amount of information__ in an event $$\mathbf{x} = x$$, called the *__Self-Information__*  is:  
        <p>$$I(x) = \log (1/p(x)) = -\log(p(x))$$</p>  
1. __RMSprop__{: style="color: red"}  
    1. __Motivation:__{: style="color: blue"}  
        RMSprop and Adadelta have both been developed independently around the same time stemming from the need to __resolve Adagrad's radically diminishing learning rates__.  
    1. __Definitions/Algorithm:__{: style="color: blue"}  
        __RMSprop__ is an unpublished, adaptive learning rate method proposed by Geoff Hinton in [Lecture 6e of his Coursera Class](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).  
    1. __Intuition:__{: style="color: blue"}  
        RMSprop in fact is identical to the first update vector of Adadelta that we derived above:  
        <p>$$\begin{align}  \begin{split}  E[g^2]_t &= 0.9 E[g^2]_{t-1} + 0.1 g^2_t \\  \theta_{t+1} &= \theta_{t} - \dfrac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_{t}  \end{split}  \end{align}$$</p>  

        RMSprop as well __divides the learning rate by an exponentially decaying average of squared gradients__{: style="color: goldenrod"}.  
    1. __Parameter Settings:__{: style="color: blue"}  
        Hinton suggests $$\gamma$$ to be set to $$0.9$$, while a good default value for the learning rate $$\eta$$ is $$0.001$$.  
    1. __Properties:__{: style="color: blue"}  
        Works well with RNNs.  
