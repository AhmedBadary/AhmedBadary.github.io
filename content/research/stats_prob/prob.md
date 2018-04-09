---
layout: NotesPage
title: ASR <br /> Automatic Speech Recognition
permalink: /work_files/research/dl/nlp/pgm
prevLink: /work_files/research/dl/nlp.html
---



<div markdown="1" class = "TOC">
# Table of Contents

  * [FIRST](#content1)
  {: .TOC1}
  * [SECOND](#content2)
  {: .TOC2}
  * [THIRD](#content3)
  {: .TOC3}
</div>

***
***

## FIRST
{: #content1}

1. **Uncertainty in General Systems and the need for a Probabilistic Framework:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    :   1. __Inherent stochasticity in the system being modeled:__  
            Take Quantum Mechanics, most interpretations of quantum mechanics describe the dynamics of sub-atomic particles as being probabilistic.  
        2. __Incomplete observability__:  
            Deterministic systems can appear stochastic even when we cannot observe all the variables that drive the behavior of the system.  
            > i.e. Point-of-View determinism (Monty-Hall)  
        3. __Incomplete modeling__:  
            Building a system that makes strong assumptions about the problem and discards (observed) information result in uncertainty in the predictions.    
2. **Bayesian Probabilities and Frequentist Probabilities:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    :   __Frequentist Probabilities__ describe the predicted number of times that a __repeatable__ process will result in a given output in an absolute scale.  
        __Bayesian Probabilities__ describe the _degree of belief_ that a certain __non-repeatable__ event is going to result in a given output, in an absolute scale.      
    :   We assume that __Bayesian Probabilities__ behaves in exactly the same way as __Frequentist Probabilities__.  
        This assumption is derived from a set of _"common sense"_ arguments that end in the logical conclusion that both approaches to probabilities must behave the same way - [Truth and probability (Ramsey 1926)](https://socialsciences.mcmaster.ca/econ/ugcm/3ll3/ramseyfp/ramsess.pdf).

3. **Probability as an extension of Logic:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    :   "Probability can be seen as the extension of logic to deal with uncertainty. Logic provides a set of formal rules for determining what propositions are implied to be true or false given the assumption that some other set of propositions is true or false. Probability theory provides a set of formal rules for determining the likelihood of a proposition being true given the likelihood of other propositions." - deeplearningbook p.54

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}  
    :   

***

## SECOND
{: #content2}

1. **Random Variables:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  
    :   A __Random Variable__ is a variable that can take on different values randomly.  
        Precisely, it is a _function_ that maps outcomes to numerical quantities (labels), typically real numbers.  
    :   * __Types__:  
            * *__Discrete__*: is a variable that has a finite or countably infinite number of states  
            * *__Continuous__*: is a variable that is a real value  

2. **Probability Distributions:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  
    :   A __Probability Distribution__ is a function that describes the likelihood that a random variable (or a set of r.v.) will take on each of its possible states.  
        Probability Distributions are defined in terms of the __Sample Space__.  
    :   * __Classes__:  
            * *__Discrete Probability Distribution__*: is encoded by a discrete list of the probabilities of the outcomes, known as a __Probability Mass Function (PMF)__.  
            * *__Continuous Probability Distribution__*: is described by a __Probability Density Function (PDF)__.  
    :   * __Types__:  
            * *__Univariate Distributions__*: are those whose sample space is $$\mathrm{R}$$.  
            They give the probabilities of a single random variable taking on various alternative values 
            * *__Multivariate Distributions__* (also known as *__Joint Probability distributions__*):  are those whose sample space is a vector space.   
            They give the probabilities of a random vector taking on various combinations of values.   

3. **Probability Mass Function:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}  
    :   A __Probability Mass Function (PMF)__ is a function that gives the probability that a discrete random variable is exactly equal to some value.  
    :   __Mathematical Definition__:  
        Suppose that $$X: S \rightarrow A (A {\displaystyle \subseteq }  R)$$ is a discrete random variable defined on a sample space $$S$$. Then the probability mass function $$f_X: A \rightarrow [0, 1]$$ for $$X$$ is defined as   
    :   $$f_{X}(x)=\Pr(X=x)=\Pr(\{s\in S:X(s)=x\})$$  
    :   The total probability for all hypothetical outcomes $$x$$ is always conserved:  
    :   $$\sum _{x\in A}f_{X}(x)=1$$
    :   __Joint Probability Distribution__ is a PMF over many variables, denoted $$P(\mathrm{x} = x, \mathrm{y} = y)$$ or $$P(x, y)$$.  
    :   A __PMF__ must satisfy these properties:  
        * The domain of $$P$$ must be the set of all possible states of $$x$$.  
        * $$\forall x \in \mathrm{x}, \: 0 \leq P(x) \leq 1$$. Impossible events has probability $$0$$. Guaranteed events have probability $$1$$.  
        * $$\sum_{x \in \mathrm{x}} P(x) = 1$$, i.e. the PMF must be normalized

            

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents25}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents26}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents27}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents28}  
    :   


## Tips and Tricks
{: #content10}

* It is more practical to use a simple but uncertain rule rather than a complex but certain one, even if the true rule is deterministic and our modeling system has the ﬁdelity to accommodate a complex rule.  
    For example, the simple rule “Most birds ﬂy” is cheap to develop and is broadly useful, while a rule of the form, “Birds ﬂy, except for very young birds that have not yet learned to ﬂy, sick or injured birds that have lost the ability to ﬂy, ﬂightless species of birds including the cassowary, ostrich and kiwi. . .” is expensive to develop, maintain and communicate and, after all this effort, is still brittle and prone to failure.