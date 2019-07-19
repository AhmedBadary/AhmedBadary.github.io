---
layout: NotesPage
title: PGMs <br /> Probabilistic Graphical Models
permalink: /work_files/research/dl/nlp/pgm
prevLink: /work_files/research/dl/nlp.html
---



<div markdown="1" class = "TOC">
# Table of Contents

  * [Introduction](#content1)
  {: .TOC1}
  * [SECOND](#content2)
  {: .TOC2}
  * [THIRD](#content3)
  {: .TOC3}
</div>

***
***

## Introduction
{: #content1}

0. **Motivation:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents10}  
    :   Machine learning algorithms often involve probability distributions over a very large number of random variables. Often, these probability distributions involve direct interactions between relatively few variables. Using a single function to describe the entire joint probability distribution can be very inefficient (both computationally and statistically).  
        > A description of a probability distribution is _exponential_ in the number of variables it models.

1. **Graphical Model:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    :   A __graphical model__ or __probabilistic graphical model (PGM)__ or __structured probabilistic model__ is a probabilistic model for which a graph expresses the conditional dependence structure (factorization of a probability distribution) between random variables.  
        > Generally, this is one of the most common _statistical models_  
    :   __Graph Structure:__  
        A PGM uses a graph $$\mathcal{G}$$ in which each _node_ in the graph corresponds to a _random variable_, and an _edge_ connecting two r.vs means that the probability distribution is able to _represent interactions_ between those two r.vs.  
    :   __Types:__  
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
    :   A __Bayesian network__, __Bayes network__, __belief network__, or __probabilistic directed acyclic graphical model__ is a probabilistic graphical model that represents a set of variables and their conditional dependencies via a directed acyclic graph (__DAG__).  
        > E.g. a Bayesian network could represent the probabilistic relationships between diseases and symptoms. 

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