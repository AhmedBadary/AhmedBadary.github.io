---
layout: NotesPage
title: PGMs <br /> Probabilistic Graphical Models
permalink: /work_files/research/dl/archits/pgm
prevLink: /work_files/research/dl/nlp.html
---



<div markdown="1" class = "TOC">
# Table of Contents

  * [Introduction](#content1)
  {: .TOC1}
  * [Bayesian Networks](#content2)
  {: .TOC2}
<!--   * [THIRD](#content3)
  {: .TOC3} -->
</div>

***
***

[An Introduction to Probabilistic Graphical Models: Conditional Independence and Factorization (M Jordan)](http://people.eecs.berkeley.edu/~jordan/prelims/chapter2.pdf)  
[An Intro to PGMs: The Elimination Algorithm (M Jordan)](http://people.eecs.berkeley.edu/~jordan/prelims/chapter3.pdf)  
[An Intro to PGMs: Probability Propagation and Factor Graphs](http://people.eecs.berkeley.edu/~jordan/prelims/chapter4.pdf)  
[An Intro to PGMs: The EM algorithm](http://people.eecs.berkeley.edu/~jordan/prelims/chapter11.pdf)  
[An Intro to PGMs: Hidden Markov Models](http://people.eecs.berkeley.edu/~jordan/prelims/chapter12.pdf)  




## Introduction
{: #content1}

0. **Motivation:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents10}  
    Machine learning algorithms often involve probability distributions over a very large number of random variables. Often, these probability distributions involve direct interactions between relatively few variables. Using a single function to describe the entire joint probability distribution can be very inefficient (both computationally and statistically).  

    > A description of a probability distribution is _exponential_ in the number of variables it models.  



1. **Graphical Model:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    A __graphical model__ or __probabilistic graphical model (PGM)__ or __structured probabilistic model__ is a probabilistic model for which a graph expresses the conditional dependence structure (factorization of a probability distribution) between random variables.  
    > Generally, this is one of the most common _statistical models_  

    __Graph Structure:__  
    A PGM uses a graph $$\mathcal{G}$$ in which each _node_ in the graph corresponds to a _random variable_, and an _edge_ connecting two r.vs means that the probability distribution is able to _represent interactions_ between those two r.v.s.  
    
    __Types:__  
    {: #lst-p}
    * __Directed__:  
        Directed models use graphs with directed edges, and they represent factorizations into conditional probability distributions.  
        They contain one factor for every random variable $$x_i$$ in the distribution, and that factor consists of the conditional distribution over $$x_i$$ given the parents of $$x_i$$.  
    * __UnDirected__:  
        Undirected models use graphs with undirected edges, and they represent factorizations into a set of functions; unlike in the directed case, these functions are usually not probability distributions of any kind.  
                

<!-- 2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    :   

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    :   

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    :   

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    :   

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}  
    :   --> 

***

## Bayesian Network
{: #content2}

1. **Bayesian Network:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  
    A __Bayesian network__, __Bayes network__, __belief network__, or __probabilistic directed acyclic graphical model__ is a probabilistic graphical model that represents a set of variables and their conditional dependencies via a directed acyclic graph (__DAG__).  
    > E.g. a Bayesian network could represent the probabilistic relationships between diseases and symptoms.  


    __Bayes Nets (big picture):__{: style="color: red"}  
    __Bayes Nets:__ a technique for describing complex joint distributions (models) using simple, local distributions (conditional probabilities).  

    In other words, they are a device for describing a complex distribution, over a large number of variables, that is built up of small pieces (_local interactions_); with the assumptions necessary to conclude that the product of those local interactions describe the whole domain.  

    __Formally,__ a Bayes Net consists of:  
    {: #lst-p}
    1. A __directed acyclic graph of nodes__, one per variable $$X$$ 
    2. A __conditional distribution for each node $$P(X\vert A_1\ldots A_n)$$__, where $$A_i$$ is the $$i$$th parent of $$X$$, stored as a *__conditional probability table__* or *__CPT__*.  
        Each CPT has $$n+2$$ columns: one for the values of each of the $$n$$ parent variables $$A_1 \ldots A_n$$, one for the values of $$X$$, and one for the conditional probability of $$X$$.  

    Each node in the graph represents a single random variable and each directed edge represents one of the conditional probability distributions we choose to store (i.e. an edge from node $$A$$ to node $$B$$ indicates that we store the probability table for $$P(B\vert A)$$).  
    <span>Each node is conditionally independent of all its ancestor nodes in the graph, given all of its parents.</span>{: style="color: goldenrod"} Thus, if we have a node representing variable $$X$$, we store $$P(X\vert A_1,A_2,...,A_N)$$, where $$A_1,\ldots,A_N$$ are the parents of $$X$$.   


    __The _local probability tables (of conditional distributions)_ and the _DAG_ together encode enough information to compute any probability distribution that we could have otherwise computed given the entire joint distribution.__{: style="color: goldenrod"}  


    __Motivation:__{: style="color: red"}  
    There are problems with using full join distribution tables as our probabilistic models:  
    {: #lst-p}
    * Unless there are only a few variables, the joint is WAY too big to represent explicitly
    * Hard to learn (estimate) anything empirically about more than a few variables at a time  


    __Examples of Bayes Nets:__{: style="color: red"}  
    {: #lst-p}
    * __Coin Flips__:  
        img1
    * __Traffic__:  
        img2
    * __Traffic II__:  
        img3
    * __Alarm Network__:  
        img4


    __Probabilities in BNs:__{: style="color: red"}  
    Bayes Nets *__implicitly__* encode joint distributions:  
    * Encoded as a _product of local conditional distributions_  
        <p>$$p(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} p(x_i \vert \text{parents}(X_i)) \tag{1.1}$$</p>  
    We are guaranteed that $$1.1$$ results in a proper joint distribution:  
    1. Chain Rule is valid for all distributions:  
        <p>$$p(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} p(x_i \vert x_1, \ldots, x_{i-1})$$</p>  
    2. Conditional Independences Assumption:  
        <p>$$p(x_1, x_2, \ldots, x_{i-1}) = \prod_{i=1}^{n} p(x_i \vert \text{parents}(X_i))$$</p>
    $$\implies$$  
    <p>$$p(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} p(x_i \vert \text{parents}(X_i)) \tag{1.1}$$</p>  

    From above, Not every BN can represent every joint distribution. The topology enforces certain conditional independencies that need to be met.  
    > e.g. Only distributions whose variables are _absolutely independent_ can be represented by a Bayes Net with no arcs  


    __Causality:__{: style="color: red"}  
    Although the structure of the BN might be in a way that encodes causality, it is not necessary to define the joint distribution graphically. The two definitions below are the same:  
    img5
    To summarize:  
    * When BNs reflect the true causal patterns:  
        * Often simpler (nodes have fewer parents)
        * Often easier to think about
        * Often easier to elicit from experts 
    * BNs need NOT be causal:  
        * Sometimes no causal net exists over the domain (especially if variables are missing)  
            * e.g. consider the variables $$\text{Traffic}$$ and $$\text{Drips}$$  
        * Results in arrows that reflect __correlation__, not __causation__  
    * The meaning of the arrows:  
        * The topology may happen to encode causal structure
        * But, the <span>topology really encodes conditional independence </span>{: style="color: goldenrod"}   
            <p>$$p(x_1, x_2, \ldots, x_{i-1}) = \prod_{i=1}^{n} p(x_i \vert \text{parents}(X_i))$$</p>


    
    <br>

    __Notes:__{: style="color: red"}  
    {: #lst-p}
    * The __acyclicity__ gives an order to the (order-less) chain-rule of conditional probabilities  
    * Think of the conditional distribution for each node as a _description of a noisy "causal" process_  



<!-- 2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  
    :   

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}  
    :   

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
 -->

***

<!-- ## THIRD
{: #content3}

1. **HMMs:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents31}  
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
 -->