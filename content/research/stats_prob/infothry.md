---
layout: NotesPage
title: Information Theory
permalink: /work_files/research/dl/nlp/infothry
prevLink: /work_files/research/dl/nlp.html
---

## Information Theory
{: #content1}

1. **Information Theory:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    :   __Information theory__ is a branch of applied mathematics that revolves around quantifying how much information is present in a signal.  
    :    In the context of machine learning, we can also apply information theory to continuous variables where some of these message length interpretations do not apply, instead, we mostly use a few key ideas from information theory to characterize probability distributions or to quantify similarity between probability distributions.

2. **Motivation and Intuition:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    :   The basic intuition behind information theory is that learning that an unlikely event has occurred is more informative than learning that a likely event has occurred. A message saying “the sun rose this morning” is so uninformative as to be unnecessary to send, but a message saying “there was a solar eclipse this morning” is very informative.
    :   Thus, information theory quantifies information in a way that formalizes this intuition:  
        * Likely events should have low information content - in the extreme case, guaranteed events have no information at all  
        * Less likely events should have higher information content
        * Independent events should have additive information. For example, finding out that a tossed coin has come up as heads twice should convey twice as much information as finding out that a tossed coin has come up as heads once. 

3. **Self-Information:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    :   The __Self-Information__ or __surprisal__ is a synonym for the surprise when a random variable is sampled.
    :   The __Self-Information__ of an event $$\mathrm{x} = x$$:  
    :   $$I(x) = - \log P(x)$$
    :   Self-information deals only with a single outcome.

4. **Shannon Entropy:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    :   To quantify the amount of uncertainty in an entire probability distribution, we use __Shannon Entropy___.  
    :   __Shannon Entropy__ is defined as the average amount of information produced by a stochastic source of data.
    :   $$H(x) = {\displaystyle \operatorname {E}_{x \sim P} [I(x)]} = - {\displaystyle \operatorname {E}_{x \sim P} [\log P(X)]}$$
    :   __Differential Entropy__ is shannons entropy of a __continuous__ random variable $$x$$

5. **Distributions and Entropy:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    :   Distributions that are nearly deterministic (where the outcome is nearly certain)have low entropy; distributions that are closer to uniform have high entropy. 

6. **Relative Entropy \| K-L Divergence:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    :   The __Kullback–Leibler divergence__ (__Relative Entropy__) is a measure of how one probability distribution diverges from a second, expected probability distribution.  

    :   __Interpetation:__  
        * __Discrete variables__:  
            it is the extra amount of information needed to send a message containing symbols drawn from probability distribution $$P$$, when we use a code that was designed to minimize the length of messages drawn from probability distribution $$Q$$.  
        * __Continuous variables__:  
            
    :   __Properties:__ 
        * Non-Negativity:  
            $${\displaystyle D_{\mathrm {KL} }(P\|Q) \geq 0}$$  
        * $${\displaystyle D_{\mathrm {KL} }(P\|Q) = 0 \iff}$$ $$P$$ and $$Q$$ are:
            * *__Discrete Variables__*:  
                the same distribution 
            * *__Continuous Variables__*:  
                equal "almost everywhere"
        * $${\displaystyle D_{\mathrm {KL} }(P\|Q) \neq D_{\mathrm {KL} }(Q\|P)}$$  
            > This asymmetry means that there are important consequences to the choice of the ordering 
    :   __KL-Div as a Distance:__   
        Because the KL divergence is non-negative and measures the difference between two distributions, it is often conceptualized as measuring some sort of distance between these distributions.  
        However, it is __not__ a true distance measure because it is __*not symmetric*__.
    :   __Applications__    
        Characterizing:  
        * Relative (Shannon) entropy in information systems
        * Randomness in continuous time-series 
        * Information gain when comparing statistical models of inference  
    :   ![img](/main_files/math/prob/11.png){: width="100%"}

7. **Cross Entropy:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  
    :   The __Cross Entropy__ between two probability distributions $${\displaystyle p}$$ and $${\displaystyle q}$$ over the same underlying set of events measures the average number of bits needed to identify an event drawn from the set, if a coding scheme is used that is optimized for an "unnatural" probability distribution $${\displaystyle q}$$, rather than the "true" distribution $${\displaystyle p}$$.  
    :   $$H(p,q) = \operatorname{E}_{p}[-\log q]= H(p) + D_{\mathrm{KL}}(p\|q) =-\sum_{x }p(x)\,\log q(x)$$
    :   It is similar to __KL-Div__ but with an additional quantity - the entropy of $$p$$.  
    :   Minimizing the cross-entropy with respect to $$Q$$ is equivalent to minimizing the KL divergence, because $$Q$$ does not participate in the omitted term.
    :    We treat $$0 \log (0)$$ as $$\lim_{x \to 0} x \log (x) = 0$$.  