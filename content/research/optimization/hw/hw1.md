---
layout: NotesPage
title: HW 1
permalink: /work_files/research/conv_opt/hw/hw1
prevLink: /work_files/research/conv_opt.html
---

## Q.2)


1. **$$\| x \|_\infty \leq $$ \| x \|_2$$:**  
    $$
    

## Q.5)

### 1. We solve the optimization problem.

* **Proof.**  
    > The Objective Function  

    $$t_i(\vec{w}) = \displaystyle {\mathrm{arg\ min}_{t} \|t\vec{w} − \vec{x}^{(i)} \|_2, \  i = 1, \ldots, m}.$$

    > We minimize using calculus

    $$
    \begin{align}
    & \ = \displaystyle {\mathrm{arg\ min}_{t} \|t\vec{w} − \vec{x}^{(i)} \|_2, \  i = 1, \ldots, m} \\
    & \ = \nabla_{t}  \displaystyle {\mathrm{arg\ min}_{t} \|t\vec{w} − \vec{x}^{(i)} \|_2, \  i = 1, \ldots, m} \\
    & \ = \nabla_{t} \left[\left(t\vec{w} - \vec{x}^{(i)}\right)^T \left(t\vec{w} - \vec{x}^{(i)}\right)\right]^{1/2} \\
    & \ = \nabla_{t} \left[\left(t\vec{w}^T - {\vec{x}^{(i)}}^T\right) \left(t\vec{w} - \vec{x}^{(i)}\right)\right]^{1/2} \\
    & \ = \nabla_{t} \left(t^2\vec{w}^T\vec{w} - 2t\vec{w}^T\vec{x}^{(i)} + {\vec{x}^{(i)}}^T\vec{x}^{(i)}\right) \\
    & \ = \dfrac{1}{2} \left[t^2\vec{w}^T\vec{w} - 2t\vec{w}^T\vec{x}^{(i)} + {\vec{x}^{(i)}}^T\vec{x}^{(i)} \right]^{-1/2} \left(2t\vec{w}^T\vec{w} - 2\vec{w}^T\vec{x}^{(i)} \right) \\
    & \ = \dfrac{1}{2} \left[t^2\vec{w}^T\vec{w} - 2t\vec{w}^T\vec{x}^{(i)} + {\vec{x}^{(i)}}^T\vec{x}^{(i)} \right]^{-1/2} \left(2t\vec{w}^T\vec{w} - 2\vec{w}^T\vec{x}^{(i)} \right) = 0 \\
    & \iff t\vec{w}^T\vec{w} - \vec{w}^T\vec{x}^{(i)} = 0 \\
    & \iff t_i(\vec{w}) =  \dfrac{\vec{w}^T\vec{x}^{(i)}}{\vec{w}^T\vec{w}} \\    
    & \iff t_i(\vec{w}) =  \dfrac{\vec{w}^T\vec{x}^{(i)}}{\|\vec{w}\|_2} \\    
    & \iff t_i(\vec{w}) =  \dfrac{\vec{w}^T\vec{x}^{(i)}}{1} \\    
    & \iff t_i(\vec{w}) =  \vec{w}^T\vec{x}^{(i)}
    \end{align}
    $$



    $$
    \begin{align}
    \end{align}
    $$
