---
layout: NotesPage
title: Information Theory
permalink: /work_files/research/dl/theory/infothry
prevLink: /work_files/research/dl/theory.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Information Theory](#content1)
  {: .TOC1}
</div>

[A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8)  
[Deep Learning Information Theory (Cross-Entropy and MLE)](https://jhui.github.io/2017/01/05/Deep-learning-Information-theory/)  


## Information Theory
{: #content1}

1. **Information Theory:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    __Information theory__ is a branch of applied mathematics that revolves around quantifying how much information is present in a signal.    
    In the context of machine learning, we can also apply information theory to continuous variables where some of these message length interpretations do not apply, instead, we mostly use a few key ideas from information theory to characterize probability distributions or to quantify similarity between probability distributions.  
    <br>

2. **Motivation and Intuition:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    The basic intuition behind information theory is that learning that an unlikely event has occurred is more informative than learning that a likely event has occurred. A message saying “the sun rose this morning” is so uninformative as to be unnecessary to send, but a message saying “there was a solar eclipse this morning” is very informative.  
    Thus, information theory quantifies information in a way that formalizes this intuition:    
    * Likely events should have low information content - in the extreme case, guaranteed events have no information at all  
    * Less likely events should have higher information content
    * Independent events should have additive information. For example, finding out that a tossed coin has come up as heads twice should convey twice as much information as finding out that a tossed coin has come up as heads once.  
    <br>

33. **Measuring Information:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents133}  
    In Shannons Theory, to __transmit $$1$$ bit of information__ means to __divide the recipients *Uncertainty* by a factor of $$2$$__.  

    Thus, the __amount of information__ transmitted is the __logarithm__ (base $$2$$) of the __uncertainty reduction factor__.   

    The __uncertainty reduction factor__ is just the __inverse of the probability__ of the event being communicated.  

    Thus, the __amount of information__ in an event $$\mathbf{x} = x$$, called the *__Self-Information__*  is:  
    <p>$$I(x) = \log (1/p(x)) = -\log(p(x))$$</p>  

    __Shannons Entropy:__  
    It is the __expected amount of information__ of an uncertain/stochastic source. It acts as a measure of the amount of *__uncertainty__* of the events.  
    Equivalently, the amount of information that you get from one sample drawn from a given probability distribution $$p$$.  
    <br>

3. **Self-Information:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    The __Self-Information__ or __surprisal__ is a synonym for the surprise when a random variable is sampled.  
    The __Self-Information__ of an event $$\mathrm{x} = x$$:    
    <p>$$I(x) = - \log P(x)$$</p>  
    Self-information deals only with a single outcome.  

4. **Shannon Entropy:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    To quantify the amount of uncertainty in an entire probability distribution, we use __Shannon Entropy__.    
    __Shannon Entropy__ is defined as the average amount of information produced by a stochastic source of data.  
    <p>$$H(x) = {\displaystyle \operatorname {E}_{x \sim P} [I(x)]} = - {\displaystyle \operatorname {E}_{x \sim P} [\log P(X)] = -\sum_{i=1}^{n} p\left(x_{i}\right) \log p\left(x_{i}\right)}$$</p>  
    __Differential Entropy__ is Shannons entropy of a __continuous__ random variable $$x$$  

5. **Distributions and Entropy:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    Distributions that are nearly deterministic (where the outcome is nearly certain) have low entropy; distributions that are closer to uniform have high entropy.   

6. **Relative Entropy \| KL-Divergence:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    The __Kullback–Leibler divergence__ (__Relative Entropy__) is a measure of how one probability distribution diverges from a second, expected probability distribution.    
    __Mathematically:__    
    <p>$${\displaystyle D_{\text{KL}}(P\parallel Q)=\operatorname{E}_{x \sim P} \left[\log \dfrac{P(x)}{Q(x)}\right]=\operatorname{E}_{x \sim P} \left[\log P(x) - \log Q(x)\right]}$$</p>  
    * __Discrete__:    
    <p>$${\displaystyle D_{\text{KL}}(P\parallel Q)=\sum_{i}P(i)\log \left({\frac {P(i)}{Q(i)}}\right)}$$  </p>  
    * __Continuous__:    
    <p>$${\displaystyle D_{\text{KL}}(P\parallel Q)=\int_{-\infty }^{\infty }p(x)\log \left({\frac {p(x)}{q(x)}}\right)\,dx,}$$ </p>  

    __Interpretation:__    
    * __Discrete variables__:  
        it is the extra amount of information needed to send a message containing symbols drawn from probability distribution $$P$$, when we use a code that was designed to minimize the length of messages drawn from probability distribution $$Q$$.  
    * __Continuous variables__:  
            
    __Properties:__   
    * Non-Negativity:  
            $${\displaystyle D_{\mathrm {KL} }(P\|Q) \geq 0}$$  
    * $${\displaystyle D_{\mathrm {KL} }(P\|Q) = 0 \iff}$$ $$P$$ and $$Q$$ are:
        * *__Discrete Variables__*:  
                the same distribution 
        * *__Continuous Variables__*:  
                equal "almost everywhere"  
    * Additivity of _Independent Distributions_:  
            $${\displaystyle D_{\text{KL}}(P\parallel Q)=D_{\text{KL}}(P_{1}\parallel Q_{1})+D_{\text{KL}}(P_{2}\parallel Q_{2}).}$$  
    * $${\displaystyle D_{\mathrm {KL} }(P\|Q) \neq D_{\mathrm {KL} }(Q\|P)}$$  
        > This asymmetry means that there are important consequences to the choice of the ordering   
    * Convexity in the pair of PMFs $$(p, q)$$ (i.e. $${\displaystyle (p_{1},q_{1})}$$ and  $${\displaystyle (p_{2},q_{2})}$$ are two pairs of PMFs):  
            $${\displaystyle D_{\text{KL}}(\lambda p_{1}+(1-\lambda )p_{2}\parallel \lambda q_{1}+(1-\lambda )q_{2})\leq \lambda D_{\text{KL}}(p_{1}\parallel q_{1})+(1-\lambda )D_{\text{KL}}(p_{2}\parallel q_{2}){\text{ for }}0\leq \lambda \leq 1.}$$  

    __KL-Div as a Distance:__     
    Because the KL divergence is non-negative and measures the difference between two distributions, it is often conceptualized as measuring some sort of distance between these distributions.  
    However, it is __not__ a true distance measure because it is __*not symmetric*__.  
    > KL-div is, however, a *__Quasi-Metric__*, since it satisfies all the properties of a distance-metric except symmetry  

    __Applications__      
    Characterizing:  
    * Relative (Shannon) entropy in information systems
    * Randomness in continuous time-series 
    * Information gain when comparing statistical models of inference  
    ![img](/main_files/math/prob/11.png){: width="100%"}    
    
    __Example Application and Direction of Minimization__    
    Suppose we have a distribution $$p(x)$$ and we wish to _approximate_ it with another distribution $$q(x)$$.  
    We have a choice of _minimizing_ either:  
    1. $${\displaystyle D_{\text{KL}}(p\|q)} \implies q^\ast = \operatorname {arg\,min}_q {\displaystyle D_{\text{KL}}(p\|q)}$$  
        Produces an approximation that usually places high probability anywhere that the true distribution places high probability.  
    2. $${\displaystyle D_{\text{KL}}(q\|p)} \implies q^\ast \operatorname {arg\,min}_q {\displaystyle D_{\text{KL}}(q\|p)}$$  
        Produces an approximation that rarely places high probability anywhere that the true distribution places low probability.  
        > which are different due to the _asymmetry_ of the KL-divergence  

    <button>Choice of KL-div Direction</button>{: .showText value="show"  
     onclick="showTextPopHide(event);"}
    ![img](/main_files/math/infothry/1.png){: width="100%" hidden=""}



7. **Cross Entropy:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  
    The __Cross Entropy__ between two probability distributions $${\displaystyle p}$$ and $${\displaystyle q}$$ over the same underlying set of events measures the average number of bits needed to identify an event drawn from the set, if a coding scheme is used that is optimized for an "unnatural" probability distribution $${\displaystyle q}$$, rather than the "true" distribution $${\displaystyle p}$$.    
    <p>$$H(p,q) = \operatorname{E}_{p}[-\log q]= H(p) + D_{\mathrm{KL}}(p\|q) =-\sum_{x }p(x)\,\log q(x)$$</p>  
    
    It is similar to __KL-Div__ but with an additional quantity - the entropy of $$p$$.    
    
    Minimizing the cross-entropy with respect to $$Q$$ is equivalent to minimizing the KL divergence, because $$Q$$ does not participate in the omitted term.  
    
    We treat $$0 \log (0)$$ as $$\lim_{x \to 0} x \log (x) = 0$$.    


> __Further Info (Lecture):__ https://www.youtube.com/watch?v=XL07WEc2TRI