---
layout: NotesPage
title: Generative Adversarial Networks
permalink: /work_files/research/dl/gans
prevLink: /work_files/research/dl/cv.html
---

<div markdown="1" class = "TOC">
# Table of Contents
  * [Variational Auto-Encoders](#content1)
  {: .TOC1}
</div>

***
***

## Generative Adversarial Networks (GANs)
{: #content1}

0. **Auto-Regressive Models VS Variational Auto-Encoders VS GANs:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents40}  
    :   __Auto-Regressive Models__ defined a *__tractable__* (discrete) density function and, then, optimized the likelihood of training data:   
    :   $$p_\theta(x) = p(x_0) \prod_1^n p(x_i | x_{i<})$$  
    :   While __VAEs__ defined an *__intractable__* (continuous) density function with latent variable $$z$$:  
    :   $$p_\theta(x) = \int p_\theta(z) p_\theta(x|z) dz$$
    :   but cannot optimize directly; instead, derive and optimize a lower bound on likelihood instead.  
    :   On the other hand, __GANs__ rejects explicitly defining a probability density function, in favor of only being able to sample.     

1. **Generative Adversarial Networks:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    :   are a class of AI algorithms used in unsupervised machine learning, implemented by a system of two neural networks contesting with each other in a zero-sum game framework.

2. **Motivation:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    :   * __Problem__: we want to sample from complex, high-dimensional training distribution; there is no direct way of doing this.  
        * __Solution__: we sample from a simple distribution (e.g. random noise) and learn a transformation that maps to the training distribution, by using a __neural network__.  
    :   * __Generative VS Discriminative__: discriminative models had much more success because deep generative models suffered due to the difficulty of approximating many intractable probabilistic computations that arise in maximum likelihood estimation and related strategies, and due to difficulty of leveraging the benefits of piecewise linear units in the generative context.  
        GANs propose a new framework for generative model estimation that sidesteps these difficulties.      

3. **Structure:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    :   * __Goal__: estimating generative models that capture the training data distribution  
        * __Framework__: an adversarial process in which two models are simultaneously trained a generative model $$G$$ that captures the data distribution, and a discriminative model $$D$$ that estimates the probability that a sample came from the training data rather than $$G$$.  
        * __Training__:  
            * $$G$$ maximizes the probability of $$D$$ making a mistake       
