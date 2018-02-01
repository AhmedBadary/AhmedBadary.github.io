---
layout: NotesPage
title: Topology and Smooth Manifolds
permalink: /work_files/research/math/manifolds
prevLink: /work_files/research/dl/cv.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Introduction and Definitions](#content1)
  {: .TOC1}
  * [Manifolds](#content2)
  {: .TOC2}
</div>

***
***

## Introduction and Definitions
{: #content1}

1. **Topology:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    :   is a mathematical field concerned with the properties of space that are preserved under continuous deformations, such as stretching, crumpling and bending, but not tearing or gluing

2. **Topological Space:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    :   is defined as a set of points $$\mathbf{X}$$, along with a set of neighbourhoods (sub-sets) $$\mathbf{T}$$ for each point, satisfying the following set of axioms relating points and neighbourhoods:  
        * __$$\mathbf{T}$$ is the Open Sets__:     
            1. The __Empty Set__ $$\emptyset$$ is in $$\mathbf{T}$$
            2. $$\mathbf{X}$$ is in $$\mathbf{T}$$
            3. The __Intersection of a finite number of Sets__ in $$\mathbf{T}$$ is, also, in $$\mathbf{T}$$
            4. The __Union of an arbitrary number of Sets__ in $$\mathbf{T}$$ is, also, in $$\mathbf{T}$$  
        * __$$\mathbf{T}$$ is the Closed Sets__:     
            1. The __Empty Set__ $$\emptyset$$ is in $$\mathbf{T}$$
            2. $$\mathbf{X}$$ is in $$\mathbf{T}$$
            3. The __Intersection of an arbitrary number of Sets__ in $$\mathbf{T}$$ is, also, in $$\mathbf{T}$$
            4. The __Union of a finite number of Sets__ in $$\mathbf{T}$$ is, also, in $$\mathbf{T}$$

3. **Homeomorphism:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    :   Intuitively, a __Homeomorphism__ or __Topological Isomorphism__ or __bi-continuous Function__ is a continuous function between topological spaces that has a continuous inverse function.  
    :   Mathematically, a function $${\displaystyle f:X\to Y}$$ between two topological spaces $${\displaystyle (X,{\mathcal {T}}_{X})}$$ and $${\displaystyle (Y,{\mathcal {T}}_{Y})}$$ is called a __Homeomorphism__ if it has the following properties:  
        * $$f$$ is a bijection (one-to-one and onto)  
        * $$f$$ is continuous
        * the inverse function $${\displaystyle f^{-1}}$$ is continuous ($${\displaystyle f}$$ is an open mapping).  
    :   > i.e. There exists a __continuous map__ with a __continuous inverse__

4. **Maps and Spaces:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    :   | __Map__ | __Space__ | __Preserved Property__ |  
        | Linear Map | Vector Space | Linear Structure: $$f(aw+v) = af(w)+f(v)$$ |  
        | Group Homomorphism | Group | Group Structure: $$f(x \ast y) = f(x) \ast f(y)$$ |  
        | Continuous Map | Topological Space | Openness/Closeness: $$f^{-1}(\{\text{open}\}) \text{ is open}$$ |  
        | _Smooth Map_ | _Topological Space_ | 

5. **Smooth Maps:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    :   
        * __Continuous__: 
        * __Unique Limits__:       

6. **Hausdorff:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    :   

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  
    :   

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}  
    :   

***

## Manifolds
{: #content2}

1. **Manifold:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  
    :   is a topological space that locally resembles Euclidean space near each point  
        > i.e. around every point, there is a neighborhood that is topologically the same as the open unit ball in $$\mathbb{R}^n$$  
    :   

2. **Smooth Manifold:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  
    :   A topological space $$M$$ is called a __$$n$$-dimensional smooth manifold__ if:  
        * Is is __Hausdorff__
        * It is __Second-Countable__
        * It comes with a family $$\{(U_\alpha, \phi_\alpha)\}$$ with:  
            * __Open sets__ $$U_\alpha \subset_\text{open} M$$ 
            * __Homeomorphisms__ $$\phi_\alpha : U_\alpha \rightarrow \mathbb{R}^n$$   
    such that $${\displaystyle M = \bigcup_\alpha U_\alpha}$$  
    and given $${\displaystyle U_\alpha \cap U_\beta \neq \emptyset}$$ the map $$\phi_\beta \circ \phi_\alpha^{-1}$$ is smooth

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
