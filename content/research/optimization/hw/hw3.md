---
layout: NotesPage
title: HW 3
permalink: /work_files/research/conv_opt/hw/hw3
prevLink: /work_files/research/conv_opt.html
---



## Q.1)
1. 
    We quote the well-known expression that minimizes $$\|Ax-y\|_2^2$$ over $$x$$:  
    <p>$$x^{(k)} = (A_k^TA_k)^{-1}A_k^Ty^{(k)}$$</p>

2. 
    First, we define  
    <p>$$H_k = A_k^TA_k$$</p>  
    Then, notice  
    <p>$$
    \begin{align}
    \left(A_{k+1}^TA_{k+1}\right) &= \left(A_k^TA_k + a_{k+1}a_{k+1}^T\right) = \left(H_k + a_{k+1}a_{k+1}^T\right) & (1) \\
    \left(A_{k+1}^T y^{k+1}\right) &= \left(A_{k}^Ty^k + a_{k+1}y^{k+1}\right) & (2) \\
    \left(H_k + a_{k+1}a_{k+1}^T\right)^{-1} &= H_k^{-1} - \dfrac{1}{1+a_{k+1}^TH_k^{-1}a_{k+1}} H_k^{-1}a_{k+1}a_{k+1}^TH_k^{-1} & (3) \\
    \end{align}$$</p>  
    > We will use these three equations as we derive the solution.  
    <p> So, the objective function we are minimizing is:</p>
    <p>$$\|A_{k+1}x-y^{k+1}\|_2^2$$</p>
    We notice, from the Least Squares solution in part 1 (and derived in Lecture) that the answer should have this form:
    <p> $$x^{(k+1)} = (A_{k+1}^TA_{k+1})^{-1}A_{k+1}^Ty^{(k+1)}$$</p>
    Now, we use the equations $$((1), (2), (3))$$ we wrote out to find a solution given the parameters stated in the question,
    <p> 
    $$
    \begin{align}
    x^{(k+1)} &= (A_{k+1}^TA_{k+1})^{-1}A_{k+1}^Ty^{(k+1)} \\
    &= \left(A_k^TA_k + a_{k+1}a_{k+1}^T\right)^{-1}A_{k+1}^Ty^{(k+1)} & \text{from part } (1) \\
    &= \left(H_k + a_{k+1}a_{k+1}^T\right)^{-1}A_{k+1}^Ty^{(k+1)} & \text{from part } (1) \\
    &= \left(H_k + a_{k+1}a_{k+1}^T\right)^{-1}\left(A_{k}^Ty^k + a_{k+1}y^{k+1}\right) & \text{from part } (2) \\
    &= \left(H_k^{-1} - \dfrac{1}{1+a_{k+1}^TH_k^{-1}a_{k+1}} H_k^{-1}a_{k+1}a_{k+1}^TH_k^{-1}\right) \left(A_{k}^Ty^k + a_{k+1}y^{k+1}\right) & \text{from part } (3)
    \end{align}
    $$
    </p>
    > Notice that the inverse of $$A_k^TA_k$$ always exists because $$A$$ had Full-Column-Rank.
    <p>






## Q.2)
```python
import numpy as np
from numpy import *
from numpy.linalg import *
ar = array

x = ar([0,1,2,3,4,5,6,7,8,9])
y = ar([5.31, 5.61, 5.28, 5.54, 3.85, 4.49, 5.99, 5.32, 4.39, 4.73])

def mse(p, x, y):
    sm = [abs(p(x[i])-y[i])**2 for i in range(len(x))]
    return sum(sm)/len(x), sm[argmax(sm)]

def q2_1():
    for n in range(5,9):
        p = poly1d(polyfit(x, y, n))
        print("n =", n, ":  p_" + str(n)," =",p)

def q2_2():   
    for n in range(5,9):
        p = poly1d(polyfit(x, y, n))
        print("n =", n, ":  Mean_Sq_Err =", round(mse(p, x, y)[0], 11), "| Max_Err =", round(mse(p, x, y)[1], 10))
```  
> Notice, all the code blocks below (Q.2) refer to this piece of code.  

1. We proceed by using the "polyfit" module in "numpy".  
    ```python
    >>> q2_1()
                                 5           4          3         2         1
    >>> n = 5 :  P_5 = 0.002876 x - 0.06977 x + 0.5933 x - 2.043 x + 2.259 x + 5.2
                                 6           5          4         3         2         1  
    >>> n = 6 :  P_6 = 0.001985 x - 0.05071 x + 0.4724 x - 1.932 x + 3.237 x - 1.593 x + 5.33
                                  7            6           5          4         3         2
    >>> n = 7 :  P_7 = 4.785e-06 x + 0.001834 x - 0.04884 x + 0.4609 x - 1.895 x + 3.181 x - 1.562 x + 5.33
                                8          7         6         5         4         3         2
    >>> n = 8 :  P_8 = -.00067 x + .02413 x - .3528 x + 2.687 x - 11.37 x + 26.32 x  - 30.5 x + 13.46 x + 5.31
    ```  
    This gives us the solutions:  
    <p>$$
        P_5(x) = 0.002876 x^{5} - 0.06977 x^{4} + 0.5933 x^{3} - 2.043 x^{2} + 2.259 x^{1} + 5.2 \\
        P_6(x) = 0.001985 x^{6} - 0.05071 x^{5} + 0.4724 x^{4} - 1.932 x^{3} + 3.237 x^{2} - 1.593 x^{1} + 5.33 \\
        P_7(x) = 4.785 \cdot 10^6 x^{7} + 0.001834 x^{6} - 0.04884 x^{5} + 0.4609 x^{4} - 1.895 x^{3} + 3.181 x^{2} - 1.562 x^{1} + 5.33 \\
        P_8(x) = -6.701 \cdot 10^{-4} x^{8} + 0.02413 x^{7} - 0.3528 x^{6} + 2.687 x^{5} - 11.37 x^{4} + 26.32 x^{3} - 30.49 x^{2} + 13.46 x^{1} + 5.311
        $$
    </p>

2. We proceed by using the found polynomials and the defined "mean-squared-loss" function.    
    ```python
    >>> q2_2()
    >>> n = 5 :  Mean_Sq_Err = 0.23202104429 | Max_Err = 0.7823959188
    >>> n = 6 :  Mean_Sq_Err = 0.10826098368 | Max_Err = 0.3902666906
    >>> n = 7 :  Mean_Sq_Err = 0.10825867873 | Max_Err = 0.3887930312
    >>> n = 8 :  Mean_Sq_Err = 0.00614953188 | Max_Err = 0.0200802073
    ```


## Q.3)
1. 
    $$ \begin{align}
        \|x-a_1\|^2 &= d_1^2 \\
        \|x-a_2\|^2 &= d_2^2 \\
        \|x-a_3\|^2 &= d_3^2 \\
        \end{align}\\
        $$
        $$
        \implies \\
        $$
        $$
        \begin{align}
        x^Tx−2a_1^Tx+\|a_1\|^2 &= d_1^2 &(1)\\ 
        x^Tx−2a_2^Tx+\|a_2\|^2 &= d_2^2 &(2)\\ 
        x^Tx−2a_3^Tx+\|a_3\|^2 &= d_3^2 &(3)
        \end{align}
        $$  
    We subtract Eq. $$(1)$$ from Eq. $$(2)$$ and again from Eq. $$(3)$$.  
    $$\implies \\$$
    $$ \begin{align}
        2(a_1 − a_2)^Tx &= \|a_1\|^2 − \|a_2\|^2 − d_1^2 + d_2^2 \\
        2(a_1 − a_3)^Tx &= \|a_1\|^2 − \|a_3\|^2 − d_1^2 + d_3^2 
        \end{align}\\
        \implies \\
        $$
    $$\left[ \begin{array}{c}   2(a_1 − a_2)^Tx \\ 
                                2(a_1 − a_3)^Tx  
     \end{array} \right]\vec{x} = \left[ \begin{array}{c}   \|a_1\|^2 − \|a_2\|^2 − d_1^2 + d_2^2 \\ 
                                \|a_1\|^2 − \|a_3\|^2 − d_1^2 + d_3^2  
     \end{array} \right]
     \\ \\
     =\\
     A\vec{x}= \vec{y}$$