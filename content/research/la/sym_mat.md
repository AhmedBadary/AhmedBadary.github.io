---
layout: NotesPage
title: Classes of Matrices
permalink: /work_files/research/la/cls_mat
prevLink: /work_files/research/la
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Diagonal Matrices](#content1)
  {: .TOC1} 
  * [Symmetric Matrices](#content2)
  {: .TOC2}
  * [Skew-Symmetric Matrices](#content3)
  {: .TOC3}
  * [Covariance Matrices](#content4)
  {: .TOC4}
  * [Positive Semi-Definite Matrices](#content5)
  {: .TOC5}
  * [Positive Definite Matrices](#content6)
  {: .TOC6}
  * [Orthogonal Matrices](#content7)
  {: .TOC7}
  * [Dyads](#content8)
  {: .TOC8}
  * [Normalized Dyads](#content9)
  {: .TOC9}
</div>

***
***

## Diagonal Matrices
{: #content1}

1. **Definition.**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents11}
    :   **Diagonal matrices** are square matrices $$A$$ with $$A_{ij} = 0 \text{, when } i \ne j.$$

2. **Properties:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents12}
    * 

***

## Symmetric Matrices
{: #content2}

1. **Definition.**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents21} 
    :   **Symmetric matrices** are square matrices that satisfy $$A_{ij} = A_{ji}$$ for every pair $$(i,j).$$

2. **Properties:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents22} 

3. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents23} 

*** 

## Skew-Symmetric Matrices
{: #content3}

1. **Definition.**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents31}

2. **Properties:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents32} 

3. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents33} 

***

## Covariance Matrices
{: #content4}

1. **Definition.**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents41} 

2. **Properties:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents42} 

3. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents43} 

***

## Positive Semi-Definite Matrices
{: #content5}

1. **Definition**{: style="color: SteelBlue  "}{: .bodyContents5 #bodyContents51}

2. **Properties:**{: style="color: SteelBlue  "}{: .bodyContents5 #bodyContents52} 

3. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents5 #bodyContents53} 

***

## Positive Definite Matrices
{: #content6}

1. **Definition.**{: style="color: SteelBlue  "}{: .bodyContents6 #bodyContents61}

2. **Properties:**{: style="color: SteelBlue  "}{: .bodyContents6 #bodyContents62}

3. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents6 #bodyContents63}

***

## Orthogonal Matrices
{: #content7}

1. **Definition.**{: style="color: SteelBlue  "}{: .bodyContents7 #bodyContents71}
    :   Orthogonal (or, unitary) matrices are square matrices, such that the columns form an orthonormal basis.

2. **Properties:**{: style="color: SteelBlue  "}{: .bodyContents7 #bodyContents72}
    1. If $$U = [u_1, \ldots, u_n]$$ is an orthogonal matrix, then  
        $$u_i^Tu_j = \left\{ \begin{array}{ll} 1 & \mbox{if } i=j, \\  0 & \mbox{otherwise.} \end{array} \right. $$

    2. $$U^TU = I_n$$.

    3. $$UU^T = I_n$$.

    4. Orthogonal matrices correspond to rotations or reflections across a direction.
        > i.e. they preserve length and angles!  
        > Proof. Part 5 and 6.

    5. For all vectors $$\vec{x}$$,  
        $$ \|Ux\|_2^2 = (Ux)^T(Ux) = x^TU^TUx = x^Tx = \|x\|_2^2 .$$  
        > Known as, _the rotational invariance_ of the Euclidean norm.

    6. If $$x, y$$ are two vectors with unit norm, then the angle $$\theta$$ between them satisfies $$\cos \theta = x^Ty$$  
    while the angle $$\theta'$$ between the rotated vectors $$x' = Ux, y' = Uy$$ satisfies $$\cos \theta' = (x')^Ty'$$.  
    Since, $$(Ux)^T(Uy) = x^T U^TU y = x^Ty,$$ we obtain that the angles are the same.

    7. Geometrically, orthogonal matrices correspond to rotations (around a point) or reflections (around a line passing through the origin).


3. **Examples:**{: style="color: SteelBlue  "}{: .bodyContents7 #bodyContents73}
    * [**Permutation Matrices**](http://livebooklabs.com/keeppies/c5a5868ce26b8125/72793237f1b79da9){: value="show" onclick="iframePopA(event)"}
        <a href="http://livebooklabs.com/keeppies/c5a5868ce26b8125/72793237f1b79da9">` Visit the Book`</a>
        <div markdown="1"> </div>

***

## Dyads
{: #content8}

1. **Definition.**{: style="color: SteelBlue  "}{: .bodyContents8 #bodyContents81}
    :   A matrix $$A \in \mathbf{R}^{m \times n}$$ is a dyad if it is of the form $$A = uv^T$$ for some vectors $$u \in \mathbf{R}^m, v \in \mathbf{R}^n$$.  
        The dyad acts on an input vector $$x \in \mathbf{R}^n$$ as follows:  
    :    $$ Ax = (uv^T) x = (v^Tx) u.$$

2. **Properties:**{: style="color: SteelBlue  "}{: .bodyContents8 #bodyContents82}
    1. The output always points in the same direction $$u$$ in output space ($$\mathbf{R}^m$$), no matter what the input $$x$$ is.

    2. The output is always a simple scaled version of $$u$$.

    3. The amount of scaling depends on the vector $$v$$, via the linear function $$x \rightarrow v^Tx$$.

3. **Examples:**{: style="color: SteelBlue  "}{: .bodyContents8 #bodyContents83}
    * [**Single factor model of financial price data**](http://livebooklabs.com/keeppies/c5a5868ce26b8125/438c7a0d0fc50d09){: value="show" onclick="iframePopA(event)"}
        <a href="http://livebooklabs.com/keeppies/c5a5868ce26b8125/438c7a0d0fc50d09">` Visit the Book`</a>
        <div markdown="1"> </div>

4. **Normalized dyads:**{: style="color: SteelBlue  "}{: .bodyContents8 #bodyContents84}
    :   We can always normalize the dyad, by assuming that both u,v are of unit (Euclidean) norm, and using a factor to capture their scale.  
    :   That is, any dyad can be written in normalized form:  
    :   $$ A = uv^T = (\|u\|_2 \cdot |v|_2 ) \cdot (\dfrac{u}{\|u\|_2}) ( \dfrac{v}{\|v\|_2}) ^T = \sigma \tilde{u}\tilde{v}^T,$$  
    :   where $$\sigma > 0$$, and $$\|\tilde{u}\|_2 = \|\tilde{v}\|_2 = 1.$$


***

## Nine
{: #content9}

1. **Definition.**{: style="color: SteelBlue  "}{: .bodyContents9 #bodyContents91}

2. **Properties:**{: style="color: SteelBlue  "}{: .bodyContents9 #bodyContents92}

3. **Examples:**{: style="color: SteelBlue  "}{: .bodyContents9 #bodyContents93}

***

## Ten
{: #content10}

1. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents10 #bodyContents101}

2. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents10 #bodyContents102}

3. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents10 #bodyContents103}

4. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents10 #bodyContents104}

5. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents10 #bodyContents105}

6. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents10 #bodyContents106}

7. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents10 #bodyContents107}

8. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents10 #bodyContents108}
