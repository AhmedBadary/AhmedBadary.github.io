---
layout: NotesPage
title: Probability Theory <br /> Mathematics of Deep Learning
permalink: /work_files/research/dl/nlp/probability
prevLink: /work_files/research/dl/nlp.html
---



<div markdown="1" class = "TOC">
# Table of Contents

  * [Motivation](#content1)
  {: .TOC1}
  * [Basics](#content2)
  {: .TOC2}
<!--   * [THIRD](#content3)
  {: .TOC3} -->
  * [Discrete Distributions](#content9)
  {: .TOC9}
  * [Notes, Tips, and Tricks](#content10)
  {: .TOC10}
</div>

***
***

## Motivation
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


***

## Basics
{: #content2}

0. **Elements of Probability:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents20}  
    :   * __Sample Space $$\Omega$$__: The set of all the outcomes of a stochastic experiment; where each _outcome_ is a complete description of the state of the real world at the end of the experiment.  
        * __Event Space $${\mathcal {F}}$$__: A set of _events_; where each event $$A \in \mathcal{F}$$ is a subset of the sample space $$\Omega$$ - it is a collection of possible outcomes of an experiment.  
        * __Probability Measure $$\operatorname {P}$$__: A function $$\operatorname {P}: \mathcal{F} \rightarrow \mathbb{R}$$ that satisfies the following properties:  
            * $$\operatorname {P}(A) \geq 0, \: \forall A \in \mathcal{f}$$, 
            * $$\operatorname {P}(\Omega) = 1$$, 
            * $${\displaystyle \operatorname {P}(\bigcup_i A_i) = \sum_i \operatorname {P}(A_i) }$$, where $$A_1, A_2, ...$$ are [_disjoint_ events](#bodyContents102)
                

1. **Random Variables:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  
    :   A __Random Variable__ is a variable that can take on different values randomly.  
        Formally, a random variable $$X$$ is a _function_ that maps outcomes to numerical quantities (labels), typically real numbers:
    :   $${\displaystyle X\colon \Omega \to \mathbb{R}}$$ 
    :   * __Types__:  
            * *__Discrete__*: is a variable that has a finite or countably infinite number of states  
            * *__Continuous__*: is a variable that is a real value  

2. **Probability Distributions:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  
    :   A __Probability Distribution__ is a function that describes the likelihood that a random variable (or a set of r.v.) will take on each of its possible states.  
        Probability Distributions are defined in terms of the __Sample Space__.  
    :   * __Classes__:  
            * *__Discrete Probability Distribution:__* is encoded by a discrete list of the probabilities of the outcomes, known as a __Probability Mass Function (PMF)__.  
            * *__Continuous Probability Distribution:__* is described by a __Probability Density Function (PDF)__.  
    :   * __Types__:  
            * *__Univariate Distributions:__* are those whose sample space is $$\mathbb{R}$$.  
            They give the probabilities of a single random variable taking on various alternative values 
            * *__Multivariate Distributions__* (also known as *__Joint Probability distributions__*):  are those whose sample space is a vector space.   
            They give the probabilities of a random vector taking on various combinations of values.  
    :   A __Cumulative Distribution Function (CDF)__: is a general functional form to describe a probability distribution.  
        > Because a probability distribution P on the real line is determined by the probability of a scalar random variable X being in a half-open interval (−∞, x], the probability distribution is completely characterized by its cumulative distribution function (i.e. one can calculate the probability of any event in the event space):
    :   $${\displaystyle F(x)=\operatorname {P} [X\leq x]\qquad {\text{ for all }}x\in \mathbb {R} .}$$
 

3. **Probability Mass Function:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}  
    :   A __Probability Mass Function (PMF)__ is a function (probability distribution) that gives the probability that a discrete random variable is exactly equal to some value.  
    :   __Mathematical Definition__:  
        Suppose that $$X: S \rightarrow A, \:\:\: (A {\displaystyle \subseteq }  R)$$ is a discrete random variable defined on a sample space $$S$$. Then the probability mass function $$f_X: A \rightarrow [0, 1]$$ for $$X$$ is defined as   
    :   $$f_{X}(x)=\Pr(X=x)=\Pr(\{s\in S:X(s)=x\})$$  
    :   The total probability for all hypothetical outcomes $$x$$ is always conserved:  
    :   $$\sum _{x\in A}f_{X}(x)=1$$
    :   __Joint Probability Distribution__ is a PMF over many variables, denoted $$P(\mathrm{x} = x, \mathrm{y} = y)$$ or $$P(x, y)$$.  
    :   A __PMF__ must satisfy these properties:  
        * The domain of $$P$$ must be the set of all possible states of $$\mathrm{x}$$.  
        * $$\forall x \in \mathrm{x}, \: 0 \leq P(x) \leq 1$$. Impossible events has probability $$0$$. Guaranteed events have probability $$1$$.  
        * $$\sum_{x \in \mathrm{x}} P(x) = 1$$, i.e. the PMF must be normalized.
            
4. **Probability Density Function:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}  
    :   A __Probability Density Function (PDF)__ is a function (probability distribution) whose value at any given sample (or point) in the sample space can be interpreted as providing a relative likelihood that the value of the random variable would equal that sample.  
    :   The __PDF__ is defined as the _derivative_ of the __CDF__:  
    :   $$f_{X}(x) = \dfrac{dF_{X}(x)}{dx}$$ 
    :    A Probability Density Function $$p(x)$$ does not give the probability of a specific state directly; instead the probability of landing inside an infinitesimal region with volume $$\delta x$$ is given by $$p(x)\delta x$$.  
        We can integrate the density function to find the actual probability mass of a set of points. Specifically, the probability that $$x$$ lies in some set $$S$$ is given by the integral of $$p(x)$$ over that set.  
    > In the __Univariate__ example, the probability that $$x$$ lies in the interval $$[a, b]$$ is given by $$\int_{[a, b]} p(x)dx$$  
    :   A __PDF__ must satisfy these properties:  
        * The domain of $$P$$ must be the set of all possible states of $$x$$.  
        * $$\forall x \in \mathrm{x}, \: 0 \leq P(x) \leq 1$$. Impossible events has probability $$0$$. Guaranteed events have probability $$1$$.  
        * $$\int p(x)dx = 1$$, i.e. the integral of the PDF must be normalized.  

44. **Cumulative Distribution Function:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents244}  
    :   A __Cumulative Distribution Function (CDF)__ is a function (probability distribution) of a real-valued random variable $$X$$, or just distribution function of $$X$$, evaluated at $$x$$, is the probability that $$X$$ will take a value less than or equal to $$x$$.  
    :   $$F_{X}(x)=\operatorname {P} (X\leq x)$$ 
    :   The probability that $$X$$ lies in the semi-closed interval $$(a, b]$$, where $$a  <  b$$, is therefore
    :   $${\displaystyle \operatorname {P} (a<X\leq b)=F_{X}(b)-F_{X}(a).}$$
    :   *__Properties__*:  
        * $$0 \leq F(x) \leq 1$$, 
        * $$\lim_{x \rightarrow -\infty} F(x) = 0$$, 
        * $$\lim_{x \rightarrow \infty} F(x) = 1$$, 
        * $$x \leq y \implies F(x) \leq F(y)$$. 

5. **Marginal Probability:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents25}  
    :   The __Marginal Distribution__ of a subset of a collection of random variables is the probability distribution of the variables contained in the subset.  
    :   __Two-variable Case__:  
        Given two random variables $$X$$ and $$Y$$ whose joint distribution is known, the marginal distribution of $$X$$ is simply the probability distribution of $$X$$ averaging over information about $$Y$$.
    :   * __Discrete__:  
    :   $${\displaystyle \Pr(X=x)=\sum _{y}\Pr(X=x,Y=y)=\sum _{y}\Pr(X=x\mid Y=y)\Pr(Y=y)}$$  
    :   * __Continuous__:  
    :   $${\displaystyle p_{X}(x)=\int _{y}p_{X,Y}(x,y)\,\mathrm {d} y=\int _{y}p_{X\mid Y}(x\mid y)\,p_{Y}(y)\,\mathrm {d} y}$$
    :   * *__Marginal Probability as Expectation__*:  
    :   $${\displaystyle p_{X}(x)=\int _{y}p_{X\mid Y}(x\mid y)\,p_{Y}(y)\,\mathrm {d} y=\mathbb {E} _{Y}[p_{X\mid Y}(x\mid y)]}$$
    :   <button>Intuitive Explanation</button>{: .showText value="show"
     onclick="showTextPopHide(event);"}
    ![img](/main_files/math/prob/1.png){: width="100%" hidden=""}
    :   > __Marginalization:__ the process of forming the marginal distribution with respect to one variable by summing out the other variable
            

6. **Conditional Probability:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents26}  
    :   __Conditional Probability__ is a measure of the probability of an event given that another event has occurred.  
        Conditional Probability is only defined when $$P(x) > 0$$ - We cannot compute the conditional probability conditioned on an event that never happens.   
    :   __Definition__:    
    :   $$P(A|B)={\frac {P(A\cap B)}{P(B)}} = {\frac {P(A, B)}{P(B)}}$$

7. **The Chain Rule of Conditional Probability:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents27}  
    :   Any joint probability distribution over many random variables may be decomposed into conditional distributions over only one variable.  
        The chain rule permits the calculation of any member of the joint distribution of a set of random variables using only conditional probabilities:    
    :   $$\mathrm {P} \left(\bigcap _{k=1}^{n}A_{k}\right)=\prod _{k=1}^{n}\mathrm {P} \left(A_{k}\,{\Bigg |}\,\bigcap _{j=1}^{k-1}A_{j}\right)$$

8. **Independence and Conditional Independence:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents28}  
    :   Two random variables $$x$$ and $$y$$ are __independent__ if their probability distribution can be expressed as a product of two factors, one involving only $$x$$ and one involving only $$y$$:  
    :   $$\mathrm{P}(A \cap B) = \mathrm{P}(A)\mathrm{P}(B)$$
    :   Two random variables $$A$$ and $$B$$ are conditionally independent given a random variable $$Y$$ if the conditional probability distribution over $$A$$ and $$B$$ factorizes in this way for every value of $$Y$$:  
    :   $$\Pr(A\cap B\mid Y)=\Pr(A\mid Y)\Pr(B\mid Y)$$
    :   or equivalently,
    :   $$\Pr(A\mid B\cap Y)=\Pr(A\mid Y)$$
    :    > In other words, $$A$$ and $$B$$ are conditionally independent given $$Y$$ if and only if, given knowledge that $$Y$$ occurs, knowledge of whether $$A$$ occurs provides no information on the likelihood of $$B$$ occurring, and knowledge of whether $$B$$ occurs provides no information on the likelihood of $$A$$ occurring.  
    :   __Notation:__  
        * *__$$A$$ is Independent from $$B$$__*:  $$A{\perp}B$$
        * *__$$A$$ and $$B$$ are conditionally Independent given $$Y$$__*:  $$A{\perp}B \:\vert Y$$
                
9. **Expectation:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents29}  
    :   The __expectation__, or __expected value__, of some function $$f(x)$$ with respect to a probability distribution $$P(x)$$ is the _"theoretical"_ average, or mean value, that $$f$$ takes on when $$x$$ is drawn from $$P$$.  
        > The Expectation of a R.V. is a weighted average of the values $$x$$ that the R.V. can take -- $$\operatorname {E}[X] = \sum_{x \in X} x \cdot p(x)$$  
    :   * __Discrete case__:  
    :   $${\displaystyle \operatorname {E}_{x \sim P} [f(X)]=f(x_{1})p(x_{1})+f(x_{2})p(x_{2})+\cdots +f(x_{k})p(x_{k})} = \sum_x P(x)f(x)$$             
    :   * __Continuous case__:  
    :   $${\displaystyle \operatorname {E}_{x \sim P} [f(X)] = \int p(x)f(x)dx}$$ 
    :   __Linearity of Expectation:__ 
    :   $${\displaystyle {\begin{aligned}\operatorname {E} [X+Y]&=\operatorname {E} [X]+\operatorname {E} [Y],\\[6pt]\operatorname {E} [aX]&=a\operatorname {E} [X],\end{aligned}}}$$ 
    :   __Independence:__   
        If $$X$$ and $$Y$$ are independent $$\implies \operatorname {E} [X+Y] = \operatorname {E} [X] \operatorname {E} [X]$$ 

10. **Variance:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents210}  
    :   __Variance__ is the expectation of the squared deviation of a random variable from its mean.  
        It gives a measure of how much the values of a function of a random variable $$x$$ vary as we sample different values of $$x$$ from its probability distribution:  
    :   $$\operatorname {Var} (f(x))=\operatorname {E} \left[(f(x)-\mu )^{2}\right] = \sum_{x \in X} (x - \mu)^2 \cdot p(x)$$  
    :   __Variance expanded__:  
    :   $${\displaystyle {\begin{aligned}\operatorname {Var} (X)&=\operatorname {E} \left[(X-\operatorname {E} [X])^{2}\right]\\&=\operatorname {E} \left[X^{2}-2X\operatorname {E} [X]+\operatorname {E} [X]^{2}\right]\\&=\operatorname {E} \left[X^{2}\right]-2\operatorname {E} [X]\operatorname {E} [X]+\operatorname {E} [X]^{2}\\&=\operatorname {E} \left[X^{2}\right]-\operatorname {E} [X]^{2}\end{aligned}}}$$   
    :   __Variance as Covariance__: 
            Variance can be expressed as the covariance of a random variable with itself: 
    :   $$\operatorname {Var} (X)=\operatorname {Cov} (X,X)$$ 
    :   __Properties:__  
        * $$\operatorname {Var} [a] = 0, \forall a \in \mathbb{R}$$ (constant $$a$$)  
        * $$\operatorname {Var} [af(X)] = a^2 \operatorname {Var} [f(X)]$$ (constant $$a$$)
        * $$\operatorname {Var} [X + Y] = a^2 \operatorname {Var} [X] + \operatorname {Var} [Y] + 2 \operatorname {Cov} [X, Y]$$  

11. **Standard Deviation:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents211}  
    :   The __Standard Deviation__ is a measure that is used to quantify the amount of variation or dispersion of a set of data values.  
        It is defined as the square root of the variance:  
    :   $${\displaystyle {\begin{aligned}\sigma &={\sqrt {\operatorname {E} [(X-\mu )^{2}]}}\\&={\sqrt {\operatorname {E} [X^{2}]+\operatorname {E} [-2\mu X]+\operatorname {E} [\mu ^{2}]}}\\&={\sqrt {\operatorname {E} [X^{2}]-2\mu \operatorname {E} [X]+\mu ^{2}}}\\&={\sqrt {\operatorname {E} [X^{2}]-2\mu ^{2}+\mu ^{2}}}\\&={\sqrt {\operatorname {E} [X^{2}]-\mu ^{2}}}\\&={\sqrt {\operatorname {E} [X^{2}]-(\operatorname {E} [X])^{2}}}\end{aligned}}}$$ 
    :   __Properties:__  
        * 68% of the data-points lie within $$1 \cdot \sigma$$s from the mean
        * 95% of the data-points lie within $$2 \cdot \sigma$$s from the mean
        * 99% of the data-points lie within $$3 \cdot \sigma$$s from the mean


12. **Covariance:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents212}  
    :   __Covariance__ is a measure of the joint variability of two random variables.  
        It gives some sense of how much two values are linearly related to each other, as well as the scale of these variables:  
    :   $$\operatorname {cov} (X,Y)=\operatorname {E} { {\big[ }(X-\operatorname {E} [X])(Y-\operatorname {E} [Y]){ \big] } }$$ 
    :   __Covariance expanded:__  
    :   $${\displaystyle {\begin{aligned}\operatorname {cov} (X,Y)&=\operatorname {E} \left[\left(X-\operatorname {E} \left[X\right]\right)\left(Y-\operatorname {E} \left[Y\right]\right)\right]\\&=\operatorname {E} \left[XY-X\operatorname {E} \left[Y\right]-\operatorname {E} \left[X\right]Y+\operatorname {E} \left[X\right]\operatorname {E} \left[Y\right]\right]\\&=\operatorname {E} \left[XY\right]-\operatorname {E} \left[X\right]\operatorname {E} \left[Y\right]-\operatorname {E} \left[X\right]\operatorname {E} \left[Y\right]+\operatorname {E} \left[X\right]\operatorname {E} \left[Y\right]\\&=\operatorname {E} \left[XY\right]-\operatorname {E} \left[X\right]\operatorname {E} \left[Y\right].\end{aligned}}}$$ 
    :   > when $${\displaystyle \operatorname {E} [XY]\approx \operatorname {E} [X]\operatorname {E} [Y]} $$, this last equation is prone to catastrophic cancellation when computed with floating point arithmetic and thus should be avoided in computer programs when the data has not been centered before.  
    :   __Covariance of Random Vectors__:  
    :   $${\begin{aligned}\operatorname {cov} (\mathbf {X} ,\mathbf {Y} )&=\operatorname {E} \left[(\mathbf {X} -\operatorname {E} [\mathbf {X} ])(\mathbf {Y} -\operatorname {E} [\mathbf {Y} ])^{\mathrm {T} }\right]\\&=\operatorname {E} \left[\mathbf {X} \mathbf {Y} ^{\mathrm {T} }\right]-\operatorname {E} [\mathbf {X} ]\operatorname {E} [\mathbf {Y} ]^{\mathrm {T} },\end{aligned}}$$ 
    :   __The Covariance Matrix__ of a random vector $$x \in \mathbb{R}^n$$ is an $$n \times n$$ matrix, such that:    
    :   $$ \operatorname {cov} (X)_{i,j} = \operatorname {cov}(x_i, x_j) \\
        \operatorname {cov}(x_i, x_j) = \operatorname {Var} (x_i)$$ 
    :   __Interpretations__:  
        * High absolute values of the covariance mean that the values change very much and are both far from their respective means at the same time.
        * __The sign of the covariance__:   
            The sign of the covariance shows the tendency in the linear relationship between the variables:  
            * *__Positive__*:  
                the variables tend to show similar behavior
            * *__Negative__*:  
                the variables tend to show opposite behavior  
            * __Reason__:  
            If the greater values of one variable mainly correspond with the greater values of the other variable, and the same holds for the lesser values, (i.e., the variables tend to show similar behavior), the covariance is positive. In the opposite case, when the greater values of one variable mainly correspond to the lesser values of the other, (i.e., the variables tend to show opposite behavior), the covariance is negative. 
    :   __Covariance and Independence:__  
        * Independence $$\Rightarrow$$ Zero Covariance  
        * Zero Covariance $$\nRightarrow$$ Independence


13. **Mixtures of Distributions:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents213}  
    :   It is also common to define probability distributions by combining other simpler probability distributions. One common way of combining distributions is to construct a __mixture distribution__.  
    :   A __Mixture Distribution__ is the probability distribution of a random variable that is derived from a collection of other random variables as follows: first, a random variable is selected by chance from the collection according to given probabilities of selection, and then the value of the selected random variable is realized.  
        On each trial, the choice of which component distribution should generate the sample is determined by sampling a component identity from a multinoulli distribution:  
    :   $$P(x) = \sum_i P(x=i)P(x \vert c=i)$$  
    :   where $$P(c)$$ is the multinoulli distribution over component identities.  

14. **Bayes' Rule:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents214}  
    :   __Bayes' Rule__ describes the probability of an event, based on prior knowledge of conditions that might be related to the event.  
    :   $${\displaystyle P(A\mid B)={\frac {P(B\mid A)\,P(A)}{P(B)}}}$$
    :   where, 
    :   $$P(B) =\sum_A P(B \vert A) P(A)$$

***

## Discrete Distributions
{: #content9}

1. **Uniform Distribution:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents91}  
    :   

2. **Bernoulli Distribution:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents92}  
    :   A distribution over a single binary random variable.  
        It is controlled by a single parameter $$\phi \in [0, 1]$$, which fives the probability of the r.v. being equal to $$1$$.  
        > It models the probability of a single experiment with a boolean outcome (e.g. coin flip $$\rightarrow$$ {heads: 1, tails: 0})  
    :   __PMF:__  
    :   $${\displaystyle P(x)={\begin{cases}p&{\text{if }}p=1,\\q=1-p&{\text{if }}p=0.\end{cases}}}$$  
    :   __Properties:__  
        * $$P(X=1) = \phi$$
        * $$P(X=0) = 1 - \phi$$
        * $$P(X=x) = \phi^x (1 - \phi)^{1-x}$$
        * $$\operatorname {E}[X] = \phi$$
        * $$\operatorname {Var}(X) = \phi (1 - \phi)$$

3. **Binomial Distribution:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents93}  
    :   
    :   > $${\binom {n}{k}}={\frac {n!}{k!(n-k)!}}$$ is the number of possible ways of getting $$x$$ successes and $$n-x$$ failures

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

***

## Continuous Distributions
{: #content99}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents99 #bodyContents991}  
    :   

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents99 #bodyContents992}  
    :   

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents99 #bodyContents993}  
    :   

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents99 #bodyContents994}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents99 #bodyContents995}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents99 #bodyContents996}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents99 #bodyContents997}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents99 #bodyContents998}  
    :   
***

## Notes, Tips and Tricks
{: #content10}

* It is more practical to use a simple but uncertain rule rather than a complex but certain one, even if the true rule is deterministic and our modeling system has the fidelity to accommodate a complex rule.  
    For example, the simple rule “Most birds ﬂy” is cheap to develop and is broadly useful, while a rule of the form, “Birds ﬂy, except for very young birds that have not yet learned to ﬂy, sick or injured birds that have lost the ability to ﬂy, ﬂightless species of birds including the cassowary, ostrich and kiwi. . .” is expensive to develop, maintain and communicate and, after all this effort, is still brittle and prone to failure.

* __Disjoint Events (Mutually Exclusive):__{: .bodyContents10 #bodyContents102} are events that cannot occur together at the same time
    Mathematically:  
    * $$A_i \cap A_j = \varnothing$$ whenever $$i \neq j$$  
    * $$p(A_i, A_j) = 0$$,  

* __Describing a Probability Distribution__:  
    A description of a probability distribution is _exponential_ in the number of variables it models