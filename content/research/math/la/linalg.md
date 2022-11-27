---
layout: NotesPage
title: The Essence of Linear Algebra
permalink: /work_files/research/math/la/linalg
prevLink: /work_files/research/dl/nlp.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Definitions and Intuitions](#content1)
  {: .TOC1}
<!--   * [SECOND](#content2)
  {: .TOC2} -->
</div>

***
***

[Essence of LA - 3b1b](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=1)  
[LA (Stanford Review)](http://web.stanford.edu/class/cs224n/readings/cs229-linalg.pdf)  



## Definitions and Intuitions
{: #content1}

1. **Linear Algebra:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    __Linear Algebra__ is about two operations on _a list of numbers_:  
    1. Scalar Multiplication
    2. Vector Addition

2. **Vectors:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    Think of each element in the vector as a __scalar__ that scales the corresponding basis vectors.  
    > Meaning, think about how each one stretches or squishes vectors (in this case, the basis vectors $$\hat{i}, \hat{j}$$)  

    <p>$$\begin{bmatrix}
            x \\
            y 
        \end{bmatrix} = \begin{bmatrix}
            1 & 0  \\
            0 & 1 
        \end{bmatrix}   \begin{bmatrix}
                            x \\
                            y 
                        \end{bmatrix} = \color{red} x \color{red} {\underbrace{\begin{bmatrix}
            1 \\
            0 
        \end{bmatrix}}_ {\hat{i}}} + \color{red} y \color{red} {\underbrace{\begin{bmatrix}
            0 \\
            1 
        \end{bmatrix}}_ {\hat{j}}} = \begin{bmatrix}
            1\times x + 0 \times y \\
            0\times x + 1 \times y 
        \end{bmatrix}
    $$</p>  


    <button>How to think of vectors? Three Perspectives:</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    __Three Perspectives on Vectors:__{: style="color: red"}  
    * __Physics__: an arrow pointing in space defined by the _length_ and _direction_ of the arrow.   
    * __CS__: vectors are _Ordered Lists of numbers_.  
    * __Math__: a vector is any object that can be (1) _added_ to another vector  (2) _multiplied_ by a scalar value.    

    For the purposes of this treatment, think about a vector as an _arrow_ inside a __coordinate system__  with its' _tail_ at the *__origin__*:  
    <button>Vector</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](https://cdn.mathpix.com/snip/images/UmqicBBCkZNLYE0wLmgNlBF4JYfPGGE3T53OuWEmGvI.original.fullsize.png){: width="100%" hidden=""}  
    {: hidden=""}


    __The coordinates of a Vector:__{: style="color: red"}  
    Think about the coordinates as a list of numbers that tells you how to get from the _tail_ of the arrow to its _tip_.  
    Each coordinate at the $$i$$th index tells you how many _"steps"_ to take in that dimension ($$i$$th dimension).  
    E.g. for the $$2-$$D case, the first coordinate dictates the steps in the $$x$$ dimension, and the second coordinate the steps in the $$y$$ dimension.  
    <button>Example 2D</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](https://cdn.mathpix.com/snip/images/Elldl45_uEEsD-8yA1yo93ZuI3tuwa9xjzHYIcGWr9U.original.fullsize.png){: width="100%" hidden=""}  

    You can completely specify a vector with its list of coordinates (uniquely).  




    __Vector Addition:__{: style="color: red"}  
    To add the two vectors $$\overrightarrow{\mathbf{v}} + \overrightarrow{\mathbf{w}}$$:
    * Move the _tail_ of the second vector $$\overrightarrow{\mathbf{w}}$$ to the tip of the first vector $$\overrightarrow{\mathbf{v}}$$
    * Draw an arrow from the _tail_ of $$\overrightarrow{\mathbf{v}}$$ to the _new tip_ of $$\overrightarrow{\mathbf{w}}$$.  
    * This new arrow (vector) is the sum
    <button>example</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](https://cdn.mathpix.com/snip/images/TTkkCYHhSV_H6Ra3GRbOcvE1urAkBv9rYJSVJK1e4rA.original.fullsize.png){: width="100%" hidden=""}  
    * __Intuition:__{: style="color: blue"}  
        Essentially, you are taking a step in the first vector direction then another step in the second vector direction.  You can do so by moving the tail of the second vector to the tip of the first vector, essentially taking a step in that direction, then take a step in the direction (and length) of the second vector:  
        <button>Visualization</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        ![img](/main_files/math/la/linalg/vector_addition_intuition.gif){: width="60%" hidden=""}  

    * __Vector Addition Numerically:__{: style="color: blue"}  
        ![img](https://cdn.mathpix.com/snip/images/4Z81bQcbzCd4BXoTJvFvarxKzR4ySkUVrIFHmLeAVnk.original.fullsize.png){: width="80%"}  


    __Scalar Multiplication:__{: style="color: red"}  
    ![img](https://cdn.mathpix.com/snip/images/sePiRB-OEtiKddOAvgGJ9ZsUCUA8yl9h4WszpfgjHGw.original.fullsize.png){: width="80%"}  


3. **Span:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    The __Span__ of two vectors $$\mathbf{v}, \mathbf{w}$$ is the set of all linear combinations of the two vectors:  
    <p>$$a \mathbf{v} + b \mathbf{w} \: ; \: a,b \in \mathbb{R}$$</p>  

    I.E. What are all the possible vectors you can create using the two fundamental operations on two vectors: vector addition and scalar multiplication.  

    As long as each vector does __not__ point in the *__same direction__* as another vector, you can reach all possible dimensions as the number of vectors.  
    I.E. if you cannot create the vector using the other vectors you have, then it is pointing in a new direction.  




4. **Linearly Dependent Vectors:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    If one vector is in the span of the other vectors.  
    Mathematically:  
    <p>$$a \vec{v}+b \vec{w}+c \vec{u}=\overrightarrow{0} \implies a=b=c=0$$</p>  

44. **The Basis:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents144}  
    The __Basis__ of a vector space is a set of _linearly independent_ vectors that __span__ the full space.  

5. **Matrix as a Linear Transformation:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    __Note:__  A *__Transformation__* == *__Function__* $$\iff$$ *__Linear Transformation__* == *__Linear Function__*.  
    With __vectors__ as *__inputs__* and __vectors__ as *__outputs__*.  


    __Summary__:
    1. Each *__column__*  is a _transformed_ version of the *__basis vectors__* (e.g. $$\hat{i}, \hat{j}$$)  
    2. The result of a __Matrix-Vector Product__ is the __linear combination__ of the *__vectors__* with the appropriate *__transformed coordinate/basis vectors__*  
        > i.e. __Matrix-Vector Product__ is a way to compute what the corresponding linear transformation does to the given vector.  


    __Matrix as a Linear Transformation:__{: style="color: red"}  
    Always think of Matrices as __Transformations of Space__:    
    * A Matrix represents a _specific linear transformation_
        * Where the *__columns__* represent the *__coordinates__* of the _transformed_ *__basis vectors__*   
    * & __Multiplying a matrix by a vector__ is EQUIVALENT to __Applying the transformation to that vector__  
    > The word _"transformation"_ suggests that you think using *__movement__*  
    > If a transformation takes some input vector to some output vector, we imagine that input vector moving over to the output vector.  
    > Then to understand the transformation as a whole, we might imagine watching every possible input vector move over to its corresponding output vector.  
    > This transformation/_"movement"_ is __Linear__, if it keeps all the vectors *__parallel__* and *__evenly spaced__*, and *__fixes the origin__*.  


    __Matrices and Vectors | The Matrix-Vector Product:__{: style="color: red"}  
    Again, We think of _each element_ in a vector as a *__scalar__* that scales the corresponding _basis vectors_.  
    * Thus, if we know how the basis vectors get transformed, we can then just scale them (by multiplying with our vector elements).  
        __Mathematically__, we think of the vector:   
        <p>$$\mathbf{v} = \begin{bmatrix}x \\y \end{bmatrix} = x\hat{i} + y\hat{j}$$</p>
        and its _transformed_ version:  
        <p>$$\text{Transformed } \mathbf{v} = x (\text{Transformed } \hat{i}) + y (\text{Transformed }  \hat{j})$$</p>  
        $$\implies$$ we can describe where any vector $$\mathbf{v}$$ go, by describing where the *__basis vectors__* will land.  
        > If you're given a _two-by-two matrix_ describing a _linear transformation_ and some specific _vector_ and you want to know where that linear transformation takes that vector, you can (1) take the coordinates of the vector (2) multiply them by the corresponding *__columns__* of the matrix, (3) then add together what you get.  
        This corresponds with the idea of adding the scaled versions of our new basis vectors.  
            ![img](/main_files/math/la/linalg/1.png){: width="60%"}  


    <p>$$ \mathbf{v} = 
        \begin{bmatrix}
            x \\
            y 
        \end{bmatrix} = \begin{bmatrix}
            1 & 0  \\
            0 & 1 
        \end{bmatrix}   \begin{bmatrix}
                            x \\
                            y 
                        \end{bmatrix} = \color{red} x \color{red} {\underbrace{\begin{bmatrix}
            1 \\
            0 
        \end{bmatrix}}_ {\hat{i}}} + \color{red} y \color{red} {\underbrace{\begin{bmatrix}
            0 \\
            1 
        \end{bmatrix}}_ {\hat{j}}} = \begin{bmatrix}
            1\times x + 0 \times y \\
            0\times x + 1 \times y 
        \end{bmatrix} = x\hat{i} + y\hat{j}
    $$</p>  

    __The Matrix-Vector Product:__  
    ![img](/main_files/math/la/linalg/2.png){: width="100%"}  


    __Non-Square Matrices $$(N \times M)$$:__{: style="color: red"}  
    Map vectors from $$\mathbb{R}^M \rightarrow \mathbb{R}^N$$.  
    They are _transformations between **dimensions**_.   


6. **The Product of Two Matrices:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    The Product of Two Matrices Corresponds to the *__composition of the transformations__*, being applied from right to left.  
    This is __very important__ intuition:  

    > e.g. _do matrices commute?_  
    if you think of the matrices as _transformations of space_ then answer quickly is no.  
    Equivalently, _Are matrices associative?_  Yes, function composition is associative: $$f \circ(g \circ h) = (f \circ g) \circ h$$   


    ![img](https://cdn.mathpix.com/snip/images/ZYp1ZL39v2kgOdZTN5KCkcNbnuUK8rOVTeh_gBgQXAk.original.fullsize.png){: width="60%"}  


    E.g. __Applying two transformations, a Rotation and a Shear:__{: style="color: red"}  
    Instead of computing the __Matrix-Vector product__ first, then another __Matrix-Vector product__ on the resulting vector, we can *__combine/compose__* the two __transformations/matrices__ into one new *__composition matrix__* that specifies the new coordinates after applying the two transformations:  
    ![img](https://cdn.mathpix.com/snip/images/PNa-fCtDnMWPINvGu1f9QMboxBWuk9jZfCEYDXqGbzg.original.fullsize.png){: width="50%"}  
    where the *__composition matrix__* is the *__Product__* __of the two matrices__:  
    ![img](https://cdn.mathpix.com/snip/images/uRugAK1YebxnsNk_jdxvQD9wvEZtLnLSFzd2p1FczSk.original.fullsize.png){: width="50%"}  

    <button>Matrix Multiplication (Composition) Explained</button>{: .showText value="show" onclick="showTextPopHide(event);"}
     ![img](/main_files/math/la/linalg/Matrix_Multiplication.gif){: width="80%" hidden=""}   


7. **Linear Transformations:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  
    __Linear Transformations__ are transformations that preserve the following properties:  
    1. All vectors that are _parallel_ remain parallel 
    2. All vectors are _evenly spaced_ 
    3. The __origin__ remains _fixed_  

    * [**An Equivalent Definition of Linearity**](https://www.youtube.com/embed/LyGKycYT2v0?start=295){: value="show" onclick="iframePopA(event)"}
    <a href="https://www.youtube.com/embed/LyGKycYT2v0?start=295"></a>
        <div markdown="1"> </div>    


    __Visualization:__{: style="color: red"}  
    * __Examples of NON-linear Transformations:__{: style="color: blue"}  
        <button>Breaks first rule</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        ![img](https://cdn.mathpix.com/snip/images/7d1c_Jc99wme2hsbF5ZWTRQb9W-uxU3un8jJ_CNRDUY.original.fullsize.png){: width="100%" hidden=""}  

        <button>Breaks second rule</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        ![img](https://cdn.mathpix.com/snip/images/_P-t1J52sWH0fgXer96UEQ77TyCA8vIuDU5OsUhnZGs.original.fullsize.png){: width="100%" hidden=""}  

        <button>Breaks first rule but only on diagonal lines</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        ![img](https://cdn.mathpix.com/snip/images/btfvED_Np-hxPIysD2KKYi7suvwZLlRJtBzGxqAmDVc.original.fullsize.png){: width="100%" hidden=""}  

    * The new coordinates of the basis vectors $$\hat{\imath}, \hat{\jmath}$$ allow us to deduce where every vector is going to land after the transformation, by deducing the linear transformation/function on each basis vector.  
        So, if a vector $$\overrightarrow{\mathbf{v}}$$ - defined as some linear combination of $$\hat{\imath}, \hat{\jmath}$$:  
        <p>$$\overrightarrow{\mathbf{v}}=a \hat{\imath}+b \hat{\jmath}$$</p>  
        undergoes a linear transformation, it is now defined as the exact same linear combination of the transformed coordinates of $$\hat{\imath}, \hat{\jmath}$$:  
        <p>$$\text { Transformed } \overrightarrow{\mathbf{v}}=a(\text { Transformed } \hat{\imath})+b(\text { Transformed } \hat{\jmath})$$</p>  

    * We tend to combine the two basis vectors in a matrix side by side.  
        <p>$$\begin{bmatrix}
            \hat{\imath} & \hat{\jmath}
        \end{bmatrix}$$</p>  
        E.g. in 2D:  
        <p>$$\begin{bmatrix}
            1 & 0
            0 & 1
        \end{bmatrix}$$</p>  
        and after a certain linear transformation, the new coordinates of the basis vectors define the new matrix:  




    __Notes:__{: style="color: red"}  
    {: #lst-p}
    * You should interpret a __matrix__ as a *__linear transformation of space__*  
    * Each __column__ represents where the corresponding *__basis vector__* lands  
    * To Transform any vector to the new, _linearly transformed_ space, you can *__multiply__* the __matrix__ and the __vector__ to get the new coordinates of the vector in that space
    * A __Linear Transformation__ corresponds to *__Matrix-Vector multiplication__* 
        * Each _coordinate_ of your vector, *__scales__* the corresponding _basis vector_ (_column_) in the new space  
    * We only need to know the *__coordinates__* of the __basis vectors__ in the new space to *__specify a linear transformation__*  
    <br>



8. **The Determinant:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}  
    __The Determinant__ of a transformation is the _"scaling factor"_ by which the transformation changed any __area__ in the vector space.  

    __The Negative Determinant__ determines the *__orientation__*.  

    __Linearity of the Determinant__:  
    <p>$$\text{det}(AB) = \text{det}(A) \text{det}(B)$$</p>  


10. **Solving Systems of Equations:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents110}  
    The Equation $$A\mathbf{x} = \mathbf{b}$$, finds the vector $$\mathbf{x}$$ that _lands on the vector $$\mathbf{b}$$_ when the transformation $$A$$ is applied to it.    

    Again, the __intuition__: is to think of a linear system of equations, __geometrically__, as trying to find a particular vector that once transformed/moved, lands on the output vector $$\mathbf{b}$$.  
    * This becomes more important when you think of the different __properties__, of that transformation/function, encoded (now) in the *__matrix__* $$A$$:  
        * When the $$det(A) \neq 0$$ we know that space is preserved, and from the _properties of linearity_, we know there will always be one (*__unique__*) vector that would land on $$\mathbf{b}$$ once transformed (and you can find it by _"playing the transformation in reverse"_ i.e. the __inverse matrix__).  
        * When $$det(A) = 0$$ then the space is __squished__ down to a lower representation, resulting in __information loss__.  


    Usefullness of LA:

    __Linear System of Equations:__{: style="color: red"}  
    ![img](https://cdn.mathpix.com/snip/images/wQXF7sx1lI7v_rTlzqUafEia9y5WNP98vz4UarIkNqE.original.fullsize.png){: width="80%"}{: .center-image}  
    <button>Linear System of Equations</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](https://cdn.mathpix.com/snip/images/NRSwQoxtgSkk15ua28mpzYxmlupyKStNl46BF7axbTI.original.fullsize.png){: width="100%" hidden=""}  

    * __Intuition:__{: style="color: blue"}  
        * The __Matrix__ $$A$$ is a *__Linear Transformation__*  
        * _Solving the Equation_ $$A \overrightarrow{\mathrm{x}}=\overrightarrow{\mathrm{v}}$$ means:  
            We are looking for a __vector__ $$\mathrm{x}$$ which - *__after applying the linear transformation__* $$A$$ - __lands on__{: style="color: goldenrod"} $$\mathrm{v}$$.  

        <button>Visualization</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        ![img](/main_files/math/la/linalg/linear_eq_vis.gif){: width="100%" hidden=""}  

        * __Key Insight:__ you can understand the system of linear equations of $$n$$ variables by thinking of transforming space and figuring out which vector lands on another!  

        <button>Insight</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        ![img](/main_files/math/la/linalg/insight_linear_equation.gif){: width="100%" hidden=""}  


    <!-- __Solving the Linear Equation:__{: style="color: red"}   -->




9. **The Inverse of a matrix:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents19}  
    The __Inverse__ of a matrix is the matrix such that if we "algebraically" multiply the two matrices, we get back to the original coordinates (the identity).  
    It is, basically, the transformation applied in _reverse_.  

    The *__Inverse Matrix__* is the __unique__ matrix/transformation such that:  
    <p>$$A^{-1} A = I$$</p>  

    __Why inverse transformation/matrix DNE when the __det__erminant is Zero (i.e. space is squished):__{: style="color: red"}  
    To do so, is equivalent to _transforming_ a __line__ into a __plane__, which would require _mapping_ each, individual, __vector__ into a __"whole line full of vectors" (multiple vectors)__; which is _not something a **Function**_ can do.  
    Functions map *__single input__* to *__single output__*.  


    * __Solution when the Determinant is Zero:__{: style="color: blue"}  It is still possible that a solution exists even when the Determinant is Zero.  
    This happens in the (_lucky_) case that the vector $$\mathrm{v}$$ (on the RHS) "lives" on the line (squashed space):  
        ![img](https://cdn.mathpix.com/snip/images/OSMKQi7nay1Rp-8me67aq27creMilXa6rYpZbMnbUHE.original.fullsize.png){: width="60%"}  
        Note: the green and red vectors are the transformed basis vectors.  

        * This becomes more and more unlikely as the *__Rank__* of the __output__ gets lower  
            e.g. Squashing a 3D space on a line (Rank=1)  


11. **The Determinants, Inverses, & Solutions to Equations:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents111}  
    When the $$\text{det} = 0$$ the area gets _squashed to $$0$$_, and _information is lost_. Thus:   
    1. The Inverse DNE
    2. A unique solution DNE  
    > i.e. there is no function that can take a line onto a plane; info loss  


12. **The Rank:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents112}  
    __The Rank__ is the _dimensionality_ of the output of a transformation.  
    More precisely, it is the __number of dimensions__ in the *__column space__*.  
    __Viewed as a Matrix__, it is the _number of independent vectors (as columns)_ that make up the matrix.  


13. **The Column Space:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents113}  
    __The Column Space__ is the set of all possible outputs of a transformation/matrix.  
    * View each column as a _basis vector_; there span is then, all the possible outputs
    * __The Zero Vector (origin)__: is always in the column space (corresponds to preserving the origin)
        
    <span>The __Column Space__ allows us to understand *__when a solution exists__*.</span>{: style="color: goldenrod"}    
    For example, even when the matrix is not full-rank (det=0) a solution might still exist; if, when $$A$$ squishes space onto a line, the vector $$\mathbf{b}$$ lies on that line (in the __span__ of that line).  
    * Formally, solution exists if $$\mathbf{b}$$ is in the __column space__ of $$A$$.   


14. **The Null Space:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents114}  
    __The Null Space__ is the set of vectors that get _mapped to the origin_; also known as __The Kernel__.   

    <span>The **Null Space** allows us to understand *what the set of all possible solutions look like*.</span>{: style="color: goldenrod"}  

    When the vector $$\mathrm{v} = \mathrm{0}$$, the __null space__ gives all the _possible solutions_ to the equation $$A \overrightarrow{\mathrm{x}}=\overrightarrow{\mathbf{v}} = 0$$ 

     __Rank and the Zero Vector:__{: style="color: red"}  
    A __Full-Rank__ matrix maps __only the origin__ to itself.  
    A __non-full rank (det=0)__ (rank=n-1) matrix maps a whole __line__ to the origin, (rank=n-2) a __plane__ to the origin, etc.  


14. __Non-Square Matrices__:{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents1144}   
    If the matrix has shape $$m \times n$$ and $$m > n$$, then it can be viewed as transforming an $$n$$-dimensional vector to $$m$$-dimensional space.  (e.g. a line to a plane)  

    When $$m < n$$ then it is taking your vector from a lower dimension to a higher dimension.  


    Some Extra Intuition:  
    <button>Intuition</button>{: .showText value="show" onclick="showText_withParent_PopHide(event);"}
    <div hidden="" markdown="1">
    Nonsquare matrices are just an awkward, but space saving, notation for square matrices with zeros filling the empty slots.

    Not only is every matrix "actually" square, but infinite dimensional as well. When we talk about an "n×m" matrix, that's just a way of saying the framing the scope of whatever problem we are trying to solve, but it's always best to try visualizing problems in the highest dimension possible even if your computation truncates the extra zeros. 

    This view also shows that vectors themselves are "really" just matrices in disguise, but that notion isn't quite as enlightening although it does show how all mathematics is interconnected. Our clumsy symbolism is what makes concepts feels independent from one another.  
    </div>  


15. **The Dot Product/Scalar Product:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents115}  
    (for vectors $$\mathbf{u}, \mathbf{v}$$)  
    <p>$$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v}$$</p>   
    * __Geometrically__:  
        1. project 
            

16. **Cramer's Rule:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents116}  
    * [**Cramer's Rule Geometrically**](https://www.youtube.com/embed/jBsC34PxzoM?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab){: value="show" onclick="iframePopA(event)"}
    <a href="https://www.youtube.com/embed/jBsC34PxzoM?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"></a>
        <div markdown="1"> </div>    



17. **Coordinate Systems:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents117}  

    A __Coordinate System__ is just a way to formalize the basis vectors and their lengths. All coordinate systems agree on where the __origin__ is.  
    __Coordinate Systems__ are a way to _translate_ between *__vectors__* (defined w.r.t. basis vectors being scaled and added in space) and *__sets of numbers__* (the elements of the vector/list/array of numbers which we define).  

    * [**Translating between Coordinate Systems**](https://www.youtube.com/embed/P2LTAUO1TdA?start=257){: value="show" onclick="iframePopA(event)"}
    <a href="https://www.youtube.com/embed/P2LTAUO1TdA?start=257"></a>
        <div markdown="1"> </div>    
        > Imp: _6:47_  

    * [**Translating a Matrix/Transformation between Coordinate Systems**](https://www.youtube.com/embed/P2LTAUO1TdA?start=552){: value="show" onclick="iframePopA(event)"}
    <a href="https://www.youtube.com/embed/P2LTAUO1TdA?start=552"></a>
        <div markdown="1"> </div>    

    __Implicit Assumptions in a Coordinate System:__  
    * Direction of each _basis vector_ our vector is scaling
    * The unit of _distance_  


18. **Eigenstuff:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents118}  

    * [**Computing the Eigenvalues/vectors Intuition**](https://www.youtube.com/embed/PFDu9oVAE-g?start=322){: value="show" onclick="iframePopA(event)"}
    <a href="https://www.youtube.com/embed/PFDu9oVAE-g?start=322"></a>
        <div markdown="1"> </div>    

    * [**Diagonalization**](https://www.youtube.com/embed/PFDu9oVAE-g?start=881){: value="show" onclick="iframePopA(event)"}
    <a href="https://www.youtube.com/embed/PFDu9oVAE-g?start=881"></a>
        <div markdown="1"> </div>    


    __Notes:__  
    * Complex Eigenvalues, generally, correspond to some kind of rotation in the transformation/matrix (think, multiplication by $$i$$ in $$\mathbb{C}$$ is a $$90^{\deg}$$ rotation).  
    * For a __diagonal matrix__, all the *__basis vectors__* are *__eigenvectors__* and the *__diagonal entries__* are their *__eigenvalues__*.  

<!-- 19. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents119}  

20. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents120}  

21. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents121}  

22. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents122}  

23. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents123}  

24. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents124}  

25. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents125}  

26. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents126}  

27. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents127}  

28. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents128}  
29. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents129}  
30. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents130}  
31. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents131}  
32. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents132}  
33. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents133}  
34. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents134}  
35. **The:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents135}   -->

