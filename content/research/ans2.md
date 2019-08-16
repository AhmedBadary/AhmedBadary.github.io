---
layout: NotesPage
title: Answers to Prep Questions (Learning)
permalink: /work_files/research/ans2
prevLink: /work_files/research.html
---



# Maximum Margin Classifiers
* __Define Margin Classifiers:__{: style="color: red"}  
    A __margin classifier__ is a classifier which is able to give an associated distance from the decision boundary for each example.  
* __What is a Margin for a linear classifier?__{: style="color: red"}  
    The **margin** of a linear classifier is the distance from the decision boundary to the nearest sample point.  
* __Give the motivation for margin classifiers:__{: style="color: red"}  
    Non-margin classifiers (e.g. Centroid, Perceptron, LR) will converge to a correct classifier on linearly separable data; however, the classifier they converge to is **not** unique nor the best.  

* __Define the notion of the "best" possible classifier__{: style="color: red"}  
    We assume that if we can maximize the distance between the data points to be classified and the hyperplane that classifies them, then we have reached a boundary that allows for the "best-fit", i.e. allows for the most room for error.  
* __How can we achieve the "best" classifier?__{: style="color: red"}  
    We enforce a constraint that achieves a classifier that has a maximum-margin.  
* __What unique vector is orthogonal to the hp? Prove it:__{: style="color: red"}  
    The weight vector $$\mathbf{w}$$ is orthogonal to the separating-plane/decision-boundary, defined by $$\mathbf{w}^T\mathbf{x} + b = 0$$, in the $$\mathcal{X}$$ space; Reason:  
    Since if you take any two points $$\mathbf{x}^\prime$$ and $$\mathbf{x}^{\prime \prime}$$ on the plane, and create the vector $$\left(\mathbf{x}^{\prime}-\mathbf{x}^{\prime \prime}\right)$$  parallel to the plane by subtracting the two points, then the following equations must hold:  
    <p>$$\mathbf{w}^{\top} \mathbf{x}^{\prime}+b=0 \wedge \mathbf{w}^{\top} \mathbf{x}^{\prime \prime}+b=0 \implies \mathbf{w}^{\top}\left(\mathbf{x}^{\prime}-\mathbf{x}^{\prime \prime}\right)=0$$</p>  
* __What do we mean by "signed distance"? Derive its formula:__{: style="color: red"}  
    The _signed distance_ is the minimum distance from a point to a hyperplane.
    We solve for the signed distance to achieve the following formula for it:
    $$d = \dfrac{\| w \cdot x_0 + b \|}{\|w\|},$$  
    where we have an n-dimensional hyperplane: $$w \cdot x + b = 0$$ and a point $$\mathbf{x}_ n$$.  
    __Derivation:__  
    * Suppose we have an affine hyperplane defined by $$w \cdot x + b$$ and a point $$\mathbf{x}_ n$$.
    * Suppose that $$\mathbf{x} \in \mathbf{R}^n$$ is a point satisfying $$w \cdot \mathbf{x} + b = 0$$, i.e. it is a point on the plane.
    * We construct the vector $$\mathbf{x}_ n−\mathbf{x}$$ which points from $$\mathbf{x}$$ to $$\mathbf{x}_ n$$, and then, project (scalar projection==signed distance) it onto the unique vector perpendicular to the plane, i.e. $$w$$,  
        <p>$$d=| \text{comp}_{w} (\mathbf{x}_ n-\mathbf{x})| = \left| \frac{(\mathbf{x}_ n-\mathbf{x})\cdot w}{\|w\|} \right| = \frac{|\mathbf{x}_ n \cdot w - \mathbf{x} \cdot w|}{\|w\|}.$$</p>
    * Since $$\mathbf{x}$$  is a vector on the plane, it must satisfy $$w\cdot \mathbf{x}=-b$$ so we get  
        <p>$$d=| \text{comp}_{w} (\mathbf{x}_ n-\mathbf{x})| = \frac{|\mathbf{x}_ n \cdot w +b|}{\|w\|}$$</p>  

    Thus, we conclude that if $$\|w\| = 1$$ then the _signed distance_ from a datapoint $$X_i$$ to the hyperplane is $$\|wX_i + b\|$$.  
* __Given the formula for signed distance, calculate the "distance of the point closest to the hyperplane":__{: style="color: red"}  
    Let $$X_n$$ be the point that is closest to the plane, and let $$\hat{w} = \dfrac{w}{\|w\|}$$.  
    Take any point $$X$$ on the plane, and let $$\vec{v}$$ be the vector $$\vec{v} = X_n - X$$.  
    Now, the distance, $$d$$ is equal to  
    <p>$$\begin{align}
        d & \ = \vert\hat{w}^{\top}\vec{v}\vert \\
        & \ = \vert\hat{w}^{\top}(X_n - X)\vert \\
        & \ = \vert\hat{w}^{\top}X_n - \hat{w}^{\top}X)\vert \\
        & \ = \dfrac{1}{\|w\|}\vert w^{\top}X_n + b - w^{\top}X) - b\vert ,  & \text{we add and subtract the bias } b\\
        & \ = \dfrac{1}{\|w\|}\vert (w^{\top}X_n + b) - (w^{\top}X + b)\vert  \\
        & \ = \dfrac{1}{\|w\|}\vert (w^{\top}X_n + b) - (0)\vert ,  & \text{from the eq. of the plane on a point on the plane} \\
        & \ = \dfrac{\vert (w^{\top}X_n + b)\vert}{\|w\|}
        \end{align}
    $$</p>
* __Use geometric properties of the hp to Simplify the expression for the distance of the closest point to the hp, above__{: style="color: red"}  
    First, we notice that for any given plane $$w^Tx = 0$$, the equations, $$\gamma * w^Tx = 0$$, where $$\gamma \in \mathbf{R}$$ is a scalar, basically characterize the same plane and not many planes.  
    This is because $$w^Tx = 0 \iff \gamma * w^Tx = \gamma * 0 \iff \gamma * w^Tx = 0$$.  
    The above implies that any model that takes input $$w$$ and produces a margin, will have to be **_Scale Invariant_**.  
    To get around this and simplify the analysis, I am going to consider all the representations of the same plane, and I am going to pick one where we normalize (re-scale) the weight $$w$$ such that the signed distance (distance to the point closest to the margin) is equal to one:  
    <p>$$|w^Tx_n| > 0 \rightarrow |w^Tx_n| = 1$$</p>  
    , where $$x_n$$ is the point closest to the plane.  
    We constraint the hyperplane by normalizing $$w$$ to this equation $$|w^Tx_i| = 1$$ or with added bias, $$|w^Tx_i + b| = 1$$.  
    $$\implies$$  
    <p>$$\begin{align}
        d & \ = \dfrac{\vert (w^{\top}X_n + b)\vert}{\|w\|} \\
        & \ = \dfrac{\vert (1)\vert}{\|w\|} ,  & \text{from the constraint on the distance of the closest point} \\
        & \ = \dfrac{1}{\|w\|}
        \end{align}
    $$</p>
* __Characterize the margin, mathematically:__{: style="color: red"}  
    we can characterize the margin, with its size, as the distance, $$\frac{1}{\|\mathbf{w}\|}$$, between the hyperplane/boundary and the closest point to the plane $$\mathbf{x}_ n$$, in both directions (multiply by 2) $$= \frac{2}{\|\mathbf{w}\|}$$ ; given the condition we specified earlier $$\left|\mathbf{w}^{\top} \mathbf{x}_ {n} + b\right|=1$$ for the closest point $$\mathbf{x}_ n$$.  
* __Characterize the "Slab Existence":__{: style="color: red"}  
    The analysis done above allows us to conclusively prove that there exists a slab of width $$\dfrac{2}{\|w\|}$$ containing no sample points where the hyperplane runs through (bisects) its center.  
* __Formulate the optimization problem of *maximizing the margin* wrt analysis above:__{: style="color: red"}  
    We formulate the optimization problem of *__maximizing the margin__* by _maximizing the distance_, subject to the condition on how we derived the distance:  
    <p>$$\max_{\mathbf{w}} \dfrac{2}{\|\mathbf{w}\|} \:\:\: : \:\: \min _{n=1,2, \ldots, N}\left|\mathbf{w}^{\top} \mathbf{x}_{n}+b\right|=1$$</p>  
* __Reformulate the optimization problem above to a more "friendly" version (wrt optimization -> put in standard form):__{: style="color: red"}  
    We can reformulate by (1) Flipping and __Minimizing__, (2) Taking a square since it's monotonic and convex, and (3) noticing that $$\left|\mathbf{w}^{T} \mathbf{x}_ {n}+b\right|=y_{n}\left(\mathbf{w}^{T} \mathbf{x}_ {n}+b\right)$$ (since the signal and label must agree, their product will always be positive) and the $$\min$$ operator can be replaced by ensuring that for all the points the condition $$y_{n}\left(\mathbf{w}^{\top} \mathbf{x}_ {n}+b\right) \geq 1$$ holds [proof (by contradiction)](https://www.youtube.com/watch?v=eHsErlPJWUU&t=1555) as:   
    <p>$$\min_w \dfrac{1}{2} \mathbf{w}^T\mathbf{w} \:\:\: : \:\: y_{n}\left(\mathbf{w}^{\top} \mathbf{x}_ {n}+b\right) \geq 1 \:\: \forall i \in [1,N]$$</p>  
    Now when we solve the "friendly" equation above, we will get the __separating plane__ with the *__best possible margin__* (best=biggest).  

    * __Give the final (standard) formulation of the "Optimization problem for maximum margin classifiers":__{: style="color: blue"}  
        <p>$$\min_w \dfrac{1}{2} \mathbf{w}^T\mathbf{w} \:\:\: : \:\: y_{n}\left(\mathbf{w}^{\top} \mathbf{x}_ {n}+b\right) \geq 1 \:\: \forall i \in [1,N]$$</p>  
    * __What kind of formulation is it (wrt optimization)? What are the parameters?__{: style="color: blue"}  
        The above problem is a Quadratic Program, in $$d + 1$$-dimensions and $$n$$-constraints, in standard form.  

***

# Hard-Margin SVMs
* __Define:__{: style="color: red"}  
    * __SVMs:__{: style="color: blue"}  
        **Support Vector Machines** (SVMs) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis.  
        The SVM is a [_Maximum Margin Classifier_](/work_files/research/ml/1_3) that aims to find the "maximum-margin hyperplane" that divides the group of points $${\displaystyle {\vec {x}}_{i}} {\vec {x}}_{i}$$ for which $${\displaystyle y_{i}=1}$$ from the group of points for which $${\displaystyle y_{i}=-1}$$.  
    * __Support Vectors:__{: style="color: blue"}  
        **Support Vectors** are the data-points that lie exactly on the margin (i.e. on the boundary of the slab).  
        They satisfy $$\|w^TX' + b\| = 1, \forall $$ support vectors $$X'$$  
    * __Hard-Margin SVM:__{: style="color: blue"}  
        The _Hard-Margin SVM_ is just a maximum-margin classifier with features and kernels (discussed later).  
* __Define the following wrt hard-margin SVM:__{: style="color: red"}  
    * __Goal:__{: style="color: blue"}  
        Find weights '$$w$$' and scalar '$$b$$' that correctly classifies the data-points and, moreover, does so in the "_best_" possible way.  
    * __Procedure:__{: style="color: blue"}  
        (1) Use a linear classifier
        (2) But, Maximize the Margin
        (3) Do so by Minimizing $$\|w\|$$  
    * __Decision Function:__{: style="color: blue"}  
        <p>$${\displaystyle f(x)={\begin{cases}1&{\text{if }}\ w\cdot X_i+\alpha>0\\0&{\text{otherwise}}\end{cases}}}$$</p>  
    * __Constraints:__{: style="color: blue"}  
        <p>$$y_i(wX_i + b) \geq 1, \forall i \in [1,n]$$</p>  
    * __The Optimization Problem:__{: style="color: blue"}  
        Find weights '$$w$$' and scalar '$$b$$' that minimize  
        <p>$$ \dfrac{1}{2} w^Tw$$</p>  
        Subject to  
        <p>$$y_i(wX_i + b) \geq 1, \forall i \in [1,n]$$</p>  
        Formally,  
        <p>$$\min_w \dfrac{1}{2}w^Tw \:\:\: : \:\: y_i(wX_i + b) \geq 1, \forall i \in [1,n]$$</p>  
    * __The Optimization Method:__{: style="color: blue"}  
        The SVM optimization problem reduces to a [Quadratic Program](work_files/research/conv_opt/3_3).  
* __Elaborate on the generalization analysis:__{: style="color: red"}  
    We notice that, geometrically, the hyperplane (the maximum margin classifier) is completely characterized by the _support vectors_ (the vectors that lie on the margin).  
    A very important conclusion arises.  
    The maximum margin classifier (SVM) depends **only** on the number of support vectors and   **_not_** on the diminsion of the problem.  
    This implies that the computation doesn't scale up with the diminsion and, also implies, that the _kernel trick_ works very well.  
* __List the properties:__{: style="color: red"}  
    1. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.
    2. The hyperplane is determined solely by its support vectors.
    3. The SVM always converges on linearly seprable data.
    4. The Hard-Margin SVM fails if the data is not linearly separable. 
    4. The Hard-Margin SVM is quite sensetive to outliers
* __Give the solution to the optimization problem for H-M SVM:__{: style="color: red"}  
    To solve the above problem, we need something that deals with __inequality constraints__; thus, we use the __KKT method__ for solving a *__Lagrnagian under inequality constraints__*.  
    The __Lagrange Formulation__:  
    * Formulate the Lagrangian:  
        1. Take each inequality constraint and put them in the _zero-form_ (equality with Zero)  
        2. Multiply each inequality by a Lagrange Multiplier $$\alpha_n$$
        3. Add them to the objective function $$\min_w \dfrac{1}{2} \mathbf{w}^T\mathbf{w}$$  
            The sign will be $$-$$ (negative) simply because the inequality is $$\geq 0$$  
        <p>$$\min_{w, b} \max_{\alpha_n} \mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \dfrac{1}{2} \mathbf{w}^T\mathbf{w} -\sum_{n=1}^{N} \alpha_{n}\left(y_{n}\left(\mathbf{w}^{\top} \mathbf{x}_ {n}+b\right)-1\right) \:\:\: : \:\: \alpha_n \geq 0$$</p>  
    * Optimize the objective independently, for each of the unconstrained variables:  
        1. Gradient w.r.t. $$\mathbf{w}$$:   
            <p>$$\nabla_{\mathrm{w}} \mathcal{L}=\mathrm{w}-\sum_{n=1}^{N} \alpha_{n} y_{n} \mathrm{x}_ {n}=0 \\ \implies \\ \mathbf{w}=\sum_{n=1}^{N} \alpha_{n} y_{n} \mathbf{x}_ {n}$$</p>  
        2. Derivative w.r.t. $$b$$:  
            <p>$$\frac{\partial \mathcal{L}}{\partial b}=-\sum_{n=1}^{N} \alpha_{n} y_{n}=0 \\ \implies \\ \sum_{n=1}^{N} \alpha_{n} y_{n}=0$$</p>  
    * Get the *__Dual Formulation__* w.r.t. the (_tricky_) __constrained__ variable $$\alpha_n$$:  
        * Substitute with the above conditions in the original lagrangian (such that the optimization w.r.t. $$\alpha_n$$ will become free of $$\mathbf{w}$$ and $$b$$:   
            <p>$$\mathcal{L}(\boldsymbol{\alpha})=\sum_{n=1}^{N} \alpha_{n}-\frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} y_{n} y_{m} \alpha_{n} \alpha_{m} \mathbf{x}_{n}^{\mathrm{T}} \mathbf{x}_{m}$$</p>  
        * Notice that the first constraint $$\mathbf{w}=\sum_{n=1}^{N} \alpha_{n} y_{n} \mathbf{x}_ {n}$$ has-no-effect/doesn't-constraint $$\alpha_n$$ so it's a vacuous constraint. However, not the second constraint $$\sum_{n=1}^{N} \alpha_{n} y_{n}=0$$.   
        * Set the optimization objective and the constraints, a __quadratic function in $$\alpha_n$$__:  
        <p>$$\max_{\alpha} \mathcal{L}(\boldsymbol{\alpha})=\sum_{n=1}^{N} \alpha_{n}-\frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} y_{n} y_{m} \alpha_{n} \alpha_{m} \mathbf{x}_{n}^{\mathrm{T}} \mathbf{x}_{m} \\ \:\:\:\:\:\:\:\:\:\: : \:\: \alpha_n \geq 0 \:\: \forall \: n= 1, \ldots, N \:\: \wedge \:\: \sum_{n=1}^{N} \alpha_{n} y_{n}=0$$</p>  
    * Set the problem as a __Quadratic Programming__ problem:  
        * Change the _maximization_ to _minimization_ by flipping the signs:  
            <p>$$\min _{\alpha} \frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} y_{n} y_{m} \alpha_{n} \alpha_{m} \mathbf{x}_{0}^{\mathrm{T}} \mathbf{x}_{m}-\sum_{n=1}^{N} \alpha_{n}$$</p>  
        * __Isolate the Coefficients from the $$\alpha_n$$s__ and set in _matrix-form_:  
            <p>$$\min _{\alpha} \frac{1}{2} \alpha^{\top} 
                \underbrace{\begin{bmatrix}
                    y_{1} y_{1} \mathbf{x}_{1}^{\top} \mathbf{x}_{1} & y_{1} y_{2} \mathbf{x}_{1}^{\top} \mathbf{x}_{2} & \ldots & y_{1} y_{N} \mathbf{x}_{1}^{\top} \mathbf{x}_{N}  \\
                    y_{2} y_{1} \mathbf{x}_{2}^{\top} \mathbf{x}_{1} & y_{2} y_{2} \mathbf{x}_{2}^{\top} \mathbf{x}_{2} & \ldots & y_{2} y_{N} \mathbf{x}_{2}^{\top} \mathbf{x}_{N} \\
                    \ldots & \ldots & \ldots & \ldots \\
                    y_{N} y_{1} \mathbf{x}_{N}^{\top} \mathbf{x}_{1} & y_{N} y_{2} \mathbf{x}_{N}^{\top} \mathbf{x}_{2} & \ldots & y_{N} y_{N} \mathbf{x}_{N}^{\top} \mathbf{x}_{N} 
                \end{bmatrix}}_{\text{quadratic coefficients}}
            \alpha+\underbrace{\left(-1^{\top}\right)}_ {\text { linear }} \alpha \\ 
        \:\:\:\:\:\:\:\:\:\: : \:\: \underbrace{\mathbf{y}^{\top} \boldsymbol{\alpha}=0}_{\text { linear constraint }} \:\: \wedge \:\: \underbrace{0}_{\text { lower bounds }} \leq \alpha \leq \underbrace{\infty}_{\text { upper bounds }}  $$</p>  
            > The _Quadratic Programming Package_ asks you for the __Quadratic Term (Matrix)__ and the __Linear Term__, and for the __Linear Constraint__ and the __Range of $$\alpha_n$$s__; and then, gives you back an $$\mathbf{\alpha}$$.     

        Equivalently:  
        <p>$$\min _{\alpha} \frac{1}{2} \boldsymbol{\alpha}^{\mathrm{T}} \mathrm{Q} \boldsymbol{\alpha}-\mathbf{1}^{\mathrm{T}} \boldsymbol{\alpha} \quad \text { subject to } \quad \mathbf{y}^{\mathrm{T}} \boldsymbol{\alpha}=0 ; \quad \boldsymbol{\alpha} \geq \mathbf{0}$$</p>  
    * __What method does it require to be solved:__{: style="color: blue"}  
        To solve the above problem, we need something that deals with __inequality constraints__; thus, we use the __KKT method__ for solving a *__Lagrnagian under inequality constraints__*.  
    * __Formulate the Lagrangian:__{: style="color: blue"}  
        1. Take each inequality constraint and put them in the _zero-form_ (equality with Zero)  
        2. Multiply each inequality by a Lagrange Multiplier $$\alpha_n$$
        3. Add them to the objective function $$\min_w \dfrac{1}{2} \mathbf{w}^T\mathbf{w}$$  
            The sign will be $$-$$ (negative) simply because the inequality is $$\geq 0$$  
        <p>$$\min_{w, b} \max_{\alpha_n} \mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \dfrac{1}{2} \mathbf{w}^T\mathbf{w} -\sum_{n=1}^{N} \alpha_{n}\left(y_{n}\left(\mathbf{w}^{\top} \mathbf{x}_ {n}+b\right)-1\right) \:\:\: : \:\: \alpha_n \geq 0$$</p>  
    * __Optimize the objective for each variable:__{: style="color: blue"}  
        1. Gradient w.r.t. $$\mathbf{w}$$:   
            <p>$$\nabla_{\mathrm{w}} \mathcal{L}=\mathrm{w}-\sum_{n=1}^{N} \alpha_{n} y_{n} \mathrm{x}_ {n}=0 \\ \implies \\ \mathbf{w}=\sum_{n=1}^{N} \alpha_{n} y_{n} \mathbf{x}_ {n}$$</p>  
        2. Derivative w.r.t. $$b$$:  
            <p>$$\frac{\partial \mathcal{L}}{\partial b}=-\sum_{n=1}^{N} \alpha_{n} y_{n}=0 \\ \implies \\ \sum_{n=1}^{N} \alpha_{n} y_{n}=0$$</p>  
    * __Get the *Dual Formulation* w.r.t. the (_tricky_) constrained variable $$\alpha_n$$:__{: style="color: blue"}  
        * Substitute with the above conditions in the original lagrangian (such that the optimization w.r.t. $$\alpha_n$$ will become free of $$\mathbf{w}$$ and $$b$$:   
            <p>$$\mathcal{L}(\boldsymbol{\alpha})=\sum_{n=1}^{N} \alpha_{n}-\frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} y_{n} y_{m} \alpha_{n} \alpha_{m} \mathbf{x}_{n}^{\mathrm{T}} \mathbf{x}_{m}$$</p>  
        * Notice that the first constraint $$\mathbf{w}=\sum_{n=1}^{N} \alpha_{n} y_{n} \mathbf{x}_ {n}$$ has-no-effect/doesn't-constraint $$\alpha_n$$ so it's a vacuous constraint. However, not the second constraint $$\sum_{n=1}^{N} \alpha_{n} y_{n}=0$$.   
        * Set the optimization objective and the constraints, a __quadratic function in $$\alpha_n$$__:  
        <p>$$\max_{\alpha} \mathcal{L}(\boldsymbol{\alpha})=\sum_{n=1}^{N} \alpha_{n}-\frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} y_{n} y_{m} \alpha_{n} \alpha_{m} \mathbf{x}_{n}^{\mathrm{T}} \mathbf{x}_{m} \\ \:\:\:\:\:\:\:\:\:\: : \:\: \alpha_n \geq 0 \:\: \forall \: n= 1, \ldots, N \:\: \wedge \:\: \sum_{n=1}^{N} \alpha_{n} y_{n}=0$$</p>  
    * __Set the problem as a *Quadratic Programming* problem:__{: style="color: blue"}  
        * Change the _maximization_ to _minimization_ by flipping the signs:  
            <p>$$\min _{\alpha} \frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} y_{n} y_{m} \alpha_{n} \alpha_{m} \mathbf{x}_{0}^{\mathrm{T}} \mathbf{x}_{m}-\sum_{n=1}^{N} \alpha_{n}$$</p>  
        * __Isolate the Coefficients from the $$\alpha_n$$s__ and set in _matrix-form_:  
            <p>$$\min _{\alpha} \frac{1}{2} \alpha^{\top} 
                \underbrace{\begin{bmatrix}
                    y_{1} y_{1} \mathbf{x}_{1}^{\top} \mathbf{x}_{1} & y_{1} y_{2} \mathbf{x}_{1}^{\top} \mathbf{x}_{2} & \ldots & y_{1} y_{N} \mathbf{x}_{1}^{\top} \mathbf{x}_{N}  \\
                    y_{2} y_{1} \mathbf{x}_{2}^{\top} \mathbf{x}_{1} & y_{2} y_{2} \mathbf{x}_{2}^{\top} \mathbf{x}_{2} & \ldots & y_{2} y_{N} \mathbf{x}_{2}^{\top} \mathbf{x}_{N} \\
                    \ldots & \ldots & \ldots & \ldots \\
                    y_{N} y_{1} \mathbf{x}_{N}^{\top} \mathbf{x}_{1} & y_{N} y_{2} \mathbf{x}_{N}^{\top} \mathbf{x}_{2} & \ldots & y_{N} y_{N} \mathbf{x}_{N}^{\top} \mathbf{x}_{N} 
                \end{bmatrix}}_{\text{quadratic coefficients}}
            \alpha+\underbrace{\left(-1^{\top}\right)}_ {\text { linear }} \alpha \\ 
        \:\:\:\:\:\:\:\:\:\: : \:\: \underbrace{\mathbf{y}^{\top} \boldsymbol{\alpha}=0}_{\text { linear constraint }} \:\: \wedge \:\: \underbrace{0}_{\text { lower bounds }} \leq \alpha \leq \underbrace{\infty}_ {\text { upper bounds }}  $$</p>  
    * __What are the inputs and outputs to the Quadratic Program Package?__{: style="color: blue"}  
        The _Quadratic Programming Package_ asks you for the __Quadratic Term (Matrix)__ and the __Linear Term__, and for the __Linear Constraint__ and the __Range of $$\alpha_n$$s__; and then, gives you back an $$\mathbf{\alpha}$$.     
    * __Give the final form of the optimization problem in standard form:__{: style="color: blue"}  
        <p>$$\min_{\alpha} \frac{1}{2} \boldsymbol{\alpha}^{\mathrm{T}} \mathrm{Q} \boldsymbol{\alpha}-\mathbf{1}^{\mathrm{T}} \boldsymbol{\alpha} \quad \text { subject to } \quad \mathbf{y}^{\mathrm{T}} \boldsymbol{\alpha}=0 ; \quad \boldsymbol{\alpha} \geq \mathbf{0}$$</p>  

***

# Soft-Margin SVM
* __Motivate the soft-margin SVM:__{: style="color: red"}  
    The Hard-Margin SVM faces a few issues:  
    1. The Hard-Margin SVM fails if the data is not linearly separable. 
    2. The Hard-Margin SVM is quite sensitive to outliers  

    The Soft-Margin SVM aims to fix/reconcile these problems.  
* __What is the main idea behind it?__{: style="color: red"}  
    Allow some points to violate the margin, by introducing slack variables.  
* __Define the following wrt soft-margin SVM:__{: style="color: red"}  
    * __Goal:__{: style="color: blue"}  
        Find weights '$$w$$' and scalar '$$b$$' that correctly classifies the data-points and, moreover, does so in the "_best_" possible way, but allow some points to violate the margin, by introducing slack variables.  
    * __Procedure:__{: style="color: blue"}  
        (1) Use a linear classifier  
        (2) But, Maximize the Margin  
        (3) Do so by Minimizing $$\|w\|$$  
        (4) But allow some points to penetrate the margin  
    * __Decision Function:__{: style="color: blue"}  
    * __Constraints:__{: style="color: blue"}  
        <p>$$y_i(wX_i + b) \geq 1 - \zeta_i, \forall i \in [1,n]$$</p>  
        where the $$\zeta_i$$s are slack variables.  
        We, also, enforce the non-negativity constraint on the slack variables:  
        <p>$$\zeta_i \geq 0, \:\:\: \forall i \in [1, n]$$</p>  
        * __Why is there a non-negativity constraint?__{: style="color: blue"}     
            The non-negativity constraint forces the slack variables to be zero for all points that do not violate the original constraint:  
            i.e. are not inside the slab.  
    * __Objective/Cost Function:__{: style="color: blue"}  
        <p>$$ R(w) = \dfrac{1}{2} w^Tw + C \sum_{i=1}^n \zeta_i$$</p>  
    * __The Optimization Problem:__{: style="color: blue"}  
        Find weights '$$w$$', scalar '$$b$$', and $$\zeta_i$$s that minimize  
        <p>$$ \dfrac{1}{2} w^Tw + C \sum_{i=1}^n \zeta_i$$</p>  
        Subject to  
        <p>$$y_i(wX_i + b) \geq 1 - \zeta_i, \zeta_i \geq 0, \forall i \in [1,n]$$</p>  
        Formally,  
        <p>$$\min_w \dfrac{1}{2}w^Tw \:\:\: : \:\: y_i(wX_i + b) \geq 1 - \zeta_i, 
        \:\: \zeta_i \geq 0, \forall i \in [1,n]$$</p>  
    * __The Optimization Method:__{: style="color: blue"}  
        The SVM optimization problem reduces to a [Quadratic Program](work_files/research/conv_opt/3_3) in $$d + n + 1$$-dimensions and $$2n$$-constraints.  
    * __Properties:__{: style="color: blue"}  
        * The Soft-Margin SVM will converge on non-linearly separable data.  
* __Specify the effects of the regularization hyperparameter $$C$$:__{: style="color: red"}  

    | |__Small C__|__Large C__  
    | __Desire__|Maximizing Margin = $$\dfrac{1}{\|w\|}$$|keep most slack variables zero or small  
    | __Danger__|underfitting (High Misclassification)|overfitting (awesome training, awful test)  
    | __outliers__|less sensitive|very sensitive  
    | __boundary__|more "flat"|more sinuous  

    * __Describe the effect wrt over/under fitting:__{: style="color: blue"}   
        Increase $$C$$ hparam in SVM = causes overfitting  
* __How do we choose $$C$$?__{: style="color: red"}  
    We choose '$$C$$' with cross-validation.  
* __Give an equivalent formulation in the standard form objective for function estimation (what should it minimize?)__{: style="color: red"}  
    In function estimation we prefer the standard-form objective  to minimize (and trade-off); the loss + penalty form.  
    We introduce a loss function to moderate the use of the slack variables (i.e. to avoid abusing the slack variables), namely, __Hinge Loss__:  
    <p>$${\displaystyle \max \left(0, 1-y_{i}({\vec {w}}\cdot {\vec {x}}_ {i}-b)\right)}$$</p>  
    The motivation: We motivate it by comparing it to the traditional $$0-1$$ Loss function.  
    Notice that the $$0-1$$ loss is actually non-convex. It has an infinite slope at $$0$$. On the other hand, the hinge loss is actually convex.    

    Analysis wrt maximum margin:  
    This function is zero if the constraint, $$y_{i}({\vec {w}}\cdot {\vec {x}}_ {i}-b)\geq 1$$, is satisfied, in other words, if $${\displaystyle {\vec {x}}_{i}} {\vec {x}}_{i}$$ lies on the correct side of the margin.  
    For data on the wrong side of the margin, the function's value is proportional to the distance from the margin.  

    __Modified Objective Function:__  
    <p>$$R(w) = \dfrac{\lambda}{2} w^Tw +  \sum_{i=1}^n {\displaystyle \max \left(0, 1-y_{i}({\vec {w}}\cdot {\vec {x}}_ {i}-b)\right)}$$</p>  
    __Proof of equivalence:__  
    <p>$$\begin{align}
            y_if\left(x_i\right) & \ \geq 1-\zeta_i, & \text{from 1st constraint } \\
            \implies \zeta_i & \ \geq 1-y_if\left(x_i\right) \\
            \zeta_i & \ \geq 1-y_if\left(x_i\right) \geq 0, & \text{from 2nd positivity constraint on} \zeta_i \\
            \iff \zeta_i & \ \geq \max \{0, 1-y_if\left(x_i\right)\} \\
            \zeta_i & \ = \max \{0, 1-y_if\left(x_i\right)\}, & \text{minimizing means } \zeta_i \text{reach lower bound}\\
            \implies R(w) & \ = \dfrac{\lambda}{2} w^Tw +  \sum_{i=1}^n {\displaystyle \max \left(0, 1-y_{i}({\vec {w}}\cdot {\vec {x}}_ {i}-b)\right)}, & \text{plugging in and multplying } \lambda = \dfrac{1}{C}
            \end{align}$$</p>  
    __The (reformulated) Optimization Problem:__  
    <p>$$\min_{w, b}\dfrac{\lambda}{2} w^Tw +  \sum_{i=1}^n {\displaystyle \max \left(0, 1-y_{i}({\vec {w}}\cdot {\vec {x}}_ {i}-b)\right)}$$</p>  

***

# Loss Functions
* __Define:__{: style="color: red"}  
    * __Loss Functions - Abstractly and Mathematically:__{: style="color: blue"}  
        Abstractly, a __loss function__ or __cost function__ is a function that maps an event or values of one or more variables onto a real number, intuitively, representing some "cost" associated with the event.  

        Formally, a __loss function__ is a function $$L :(\hat{y}, y) \in \mathbb{R} \times Y \longmapsto L(\hat{y}, y) \in \mathbb{R}$$  that takes as inputs the predicted value $$\hat{y}$$ corresponding to the real data value $$y$$ and outputs how different they are.  
    * __Distance-Based Loss Functions:__{: style="color: blue"}  
        A Loss function $$L(\hat{y}, y)$$ is called __distance-based__ if it:  
        {: #lst-p}
        * Only depends on the __residual__:  
            <p>$$L(\hat{y}, y) = \psi(y-\hat{y})  \:\: \text{for some } \psi : \mathbb{R} \longmapsto \mathbb{R}$$</p>  
        * Loss is $$0$$ when residual is $$0$$:  
            <p>$$\psi(0) = 0$$</p>  
        * __What are they used for?__{: style="color: blue"}  
            Regression.  
        * __Describe an important property of dist-based losses:__{: style="color: blue"}  
            __Translation Invariance:__{: style="color: red"}  
            Distance-based losses are translation-invariant:  
            <p>$$L(\hat{y}+a, y+a) = L(\hat{y}, y)$$</p>  
    * __Relative Error - What does it lack?__{: style="color: blue"}  
        __Relative-Error__ $$\dfrac{\hat{y}-y}{y}$$ is a more _natural_ loss but it is NOT translation-invariant.  
* __List 3 Regression Loss Functions__{: style="color: red"}  
    1. MSE
    2. MAE
    3. Huber  

<button>Show the rest of the questions</button>{: .showText value="show"
onclick="showText_withParent_PopHide(event);"}
* __MSE__{: style="color: red"}  
    * __What does it minimize:__{: style="color: blue"}  
        The __MSE__ minimizes the sum of *__squared differences__* between the predicted values and the target values.  
    * __Formula:__{: style="color: blue"}  
        <p>$$L(\hat{y}, y) = \dfrac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\hat{y}_ {i}\right)^{2}$$</p>  
    * __Graph:__{: style="color: blue"}  
        ![img](/main_files/dl/concepts/loss_funcs/1.png){: width="30%" .center-image}  
    * __Derivation:__{: style="color: blue"}  
        <button>Derivation</button>{: .showText value="show"
        onclick="showTextPopHide(event);"}
        ![img](/main_files/dl/concepts/loss_funcs/5.png){: width="100%" hidden=""}  
* __MAE__{: style="color: red"}  
    * __What does it minimize:__{: style="color: blue"}  
        The __MAE__ minimizes the sum of *__absolute differences__* between the predicted values and the target values.  
    * __Formula:__{: style="color: blue"}  
        <p>$$L(\hat{y}, y) = \dfrac{1}{n} \sum_{i=1}^{n}\vert y_{i}-\hat{y}_ {i}\vert$$</p>  
    * __Graph:__{: style="color: blue"}  
        ![img](/main_files/dl/concepts/loss_funcs/6.png){: width="40%"}  
    * __Derivation:__{: style="color: blue"}  
    * __List properties:__{: style="color: blue"}  
        * Solution may be __Non-unique__  
        * __Robustness__ to outliers  
        * __Unstable Solutions:__{: #bodyContents22stability}    
            <button>Explanation</button>{: .showText value="show"
            onclick="showTextPopHide(event);"}
            _The instability property of the method of least absolute deviations means that, for a small horizontal adjustment of a datum, the regression line may jump a large amount. The method has continuous solutions for some data configurations; however, by moving a datum a small amount, one could “jump past” a configuration which has multiple solutions that span a region. After passing this region of solutions, the least absolute deviations line has a slope that may differ greatly from that of the previous line. In contrast, the least squares solutions is stable in that, for any small adjustment of a data point, the regression line will always move only slightly; that is, the regression parameters are continuous functions of the data._{: hidden=""}  
        * __Data-points "Latching" [ref](/work_files/research/dl/concepts/loss_funcs#bodyContents22):__  
            * __Unique Solution__:  
                If there are $$k$$ *__features__* (including the constant), then at least one optimal regression surface will pass through $$k$$ of the *__data points__*; unless there are multiple solutions.  
            * __Multiple Solutions__:  
                The region of valid least absolute deviations solutions will be __bounded by at least $$k$$ lines__, each of which __passes through at least $$k$$ data points__.  
            > [Wikipedia](https://en.wikipedia.org/wiki/Least_absolute_deviations#Other_properties)  
* __Huber Loss__{: style="color: red"}  
    * __AKA:__{: style="color: blue"}  
        __Smooth Mean Absolute Error__  
    * __What does it minimize:__{: style="color: blue"}  
    * __Formula:__{: style="color: blue"}  
        <p>$$L(\hat{y}, y) = \left\{\begin{array}{cc}{\frac{1}{2}(y-\hat{y})^{2}} & {\text { if }|(y-\hat{y})|<\delta} \\ {\delta(y-\hat{y})-\frac{1}{2} \delta} & {\text { otherwise }}\end{array}\right.$$</p>  
    * __Graph:__{: style="color: blue"}  
        ![img](/main_files/dl/concepts/loss_funcs/6.png){: width="40%"}  
    * __List properties:__{: style="color: blue"}  
        * It’s __less sensitive__{: style="color: green"} to outliers than the *MSE* as it treats error as square only inside an interval.  
* __Analyze MSE vs MAE [ref](/work_files/research/dl/concepts/loss_funcs#bodyContents26):__{: style="color: red"}  
    | __MSE__ | __MAE__ |
    | Sensitive to _outliers_ | Robust to _outliers_ |
    | Differentiable Everywhere | Non-Differentiable at $$0$$ |
    | Stable[^1] Solutions | Unstable Solutions |
    | Unique Solution | Possibly multiple[^2] solutions |

    * __Statistical Efficiency__:  
        * "For normal observations MSE is about $$12\%$$ more efficient than MAE" - Fisher  
        * $$1\%$$ Error is enough to make MAE more efficient  
        * 2/1000 bad observations, make the median more efficient than the mean  
    * Subgradient methods are slower than gradient descent  
        * you get a lot better convergence rate guarantees for MSE  
{: hidden=""}

* __List 7 Classification Loss Functions__{: style="color: red"}  
    1. $$0-1$$ Loss
    2. Square Loss
    3. Hinge Loss
    4. Logistic Loss
    5. Cross-Entropy
    6. Exponential Loss
    7. Perceptron Loss

<button>Show Questions on Classification</button>{: .showText value="show"
onclick="showText_withParent_PopHide(event);"}
* __$$0-1$$ loss__{: style="color: red"}  
    * __What does it minimize:__{: style="color: blue"}  
        It measures __accuracy__, and minimizes __mis-classification error/rate__.  
    * __Formula:__{: style="color: blue"}  
        <p>$$L(\hat{y}, y) = I(\hat{y} \neq y) = \left\{\begin{array}{ll}{0} & {\hat{y}=y} \\ {1} & {\hat{y} \neq y}\end{array}\right.$$</p>  
    * __Graph:__{: style="color: blue"}  
        ![img](/main_files/dl/concepts/loss_funcs/0.png){: width="40%"}  
* __MSE__{: style="color: red"}  
    * __Formula:__{: style="color: blue"}  
        <p>$$L(\hat{y}, y) = (1-y \hat{y})^{2}$$</p>  
    * __Graph:__{: style="color: blue"}  
        ![img](/main_files/dl/concepts/loss_funcs/0.png){: width="40%"}  
    * __Derivation (for classification) - give assumptions:__{: style="color: blue"}  
        We can write the loss in terms of the margin $$m = y\hat{y}$$:  
        $$L(\hat{y}, y)=(y - \hat{y})^{2}=(1-y\hat{y})^{2}=(1-m)^{2}$$   
        > Since $$y \in {-1,1} \implies y^2 = 1$$  
    * __Properties:__{: style="color: blue"}  
        * Convex
        * Smooth
        * Sensitive to outliers: Penalizes outliers excessively  
        * ^Slower Convergence Rate (wrt sample complexity) than logistic or hinge loss  
        * Functions which yield high values of $$f({\vec {x}})$$ for some $$x\in X$$ will perform poorly with the square loss function, since high values of $$yf({\vec {x}})$$ will be penalized severely, regardless of whether the signs of $$y$$ and $$f({\vec {x}})$$ match.  
* __Hinge Loss__{: style="color: red"}  
    * __What does it minimize:__{: style="color: blue"}  
        It minimizes missclassification wrt penetrating a margin $$\rightarrow$$ maximizes a margin.  
    * __Formula:__{: style="color: blue"}  
        <p>$$L(\hat{y}, y) = \max (0,1-y \hat{y})=|1-y \hat{y}|_ {+}$$</p>  
    * __Graph:__{: style="color: blue"}  
        ![img](/main_files/dl/concepts/loss_funcs/3.png){: width="30%" .center-image}  
    * __Properties:__{: style="color: blue"}  
        * Continuous, Convex, Non-Differentiable  
        * The hinge loss provides a relatively tight, convex upper bound on the $$0–1$$ indicator function  
    * __Describe the properties of the Hinge loss and why it is used?__{: style="color: blue"}  
        * Hinge loss upper bounds 0-1 loss  
        * It is the tightest _convex_ upper bound on the 0/1 loss  
        * Minimizing 0-1 loss is NP-hard in the worst-case  
* __Logistic Loss__{: style="color: red"}
    * __AKA:__{: style="color: blue"}    
        __Log-Loss__, __Logarithmic Loss__  
    * __What does it minimize:__{: style="color: blue"}  
        Minimizes the Kullback-Leibler divergence between the empirical distribution and the predicted distribution.  
    * __Formula:__{: style="color: blue"}  
        <p>$$L(\hat{y}, y) = \log{\left(1+e^{-y \hat{y}}\right)}$$</p>  
    * __Graph:__{: style="color: blue"}  
        ![img](/main_files/dl/concepts/loss_funcs/2.png){: width="30%" .center-image}  
    * __Derivation:__{: style="color: blue"}  
        We get the __likelihood__ of the dataset $$\mathcal{D}=\left(\mathbf{x}_{1}, y_{1}\right), \ldots,\left(\mathbf{x}_{N}, y_{N}\right)$$:  
        <p>$$\prod_{n=1}^{N} P\left(y_{n} | \mathbf{x}_{n}\right) =\prod_{n=1}^{N} \theta\left(y_{n} \mathbf{w}^{\mathrm{T}} \mathbf{x}_ {n}\right)$$</p>  

        <button>Derivation</button>{: .showText value="show"
        onclick="showText_withParent_PopHide(event);"}
        1. Maximize:  
            <p>$$\prod_{n=1}^{N} \theta\left(y_{n} \mathbf{w}^{\top} \mathbf{x}_ {n}\right)$$</p>  
        2. Take the natural log to avoid products:  
            <p>$$\ln \left(\prod_{n=1}^{N} \theta\left(y_{n} \mathbf{w}^{\top} \mathbf{x}_ {n}\right)\right)$$</p>  
            Motivation:  
            * The inner quantity is __non-negative__ and non-zero.  
            * The natural log is __monotonically increasing__ (its max, is the max of its argument)  
        3. Take the average (still monotonic):  
            <p>$$\frac{1}{N} \ln \left(\prod_{n=1}^{N} \theta\left(y_{n} \mathbf{w}^{\top} \mathbf{x}_ {n}\right)\right)$$</p>  
        4. Take the negative and __Minimize__:  
            <p>$$-\frac{1}{N} \ln \left(\prod_{n=1}^{N} \theta\left(y_{n} \mathbf{w}^{\top} \mathbf{x}_ {n}\right)\right)$$</p>  
        5. Simplify:  
            <p>$$=\frac{1}{N} \sum_{n=1}^{N} \ln \left(\frac{1}{\theta\left(y_{n} \mathbf{w}^{\tau} \mathbf{x}_ {n}\right)}\right)$$</p>  
        6. Substitute $$\left[\theta(s)=\frac{1}{1+e^{-s}}\right]$$:  
            <p>$$\frac{1}{N} \sum_{n=1}^{N} \underbrace{\ln \left(1+e^{-y_{n} \mathbf{w}^{\top} \mathbf{x}_{n}}\right)}_{e\left(h\left(\mathbf{x}_{n}\right), y_{n}\right)}$$</p>  
        7. Use this as the *__Cross-Entropy__*  __Error Measure__:  
            <p>$$E_{\mathrm{in}}(\mathrm{w})=\frac{1}{N} \sum_{n=1}^{N} \underbrace{\ln \left(1+e^{-y_{n} \mathrm{w}^{\top} \mathbf{x}_{n}}\right)}_{\mathrm{e}\left(h\left(\mathrm{x}_{n}\right), y_{n}\right)}$$</p>  
        {: hidden=""}
    * __Properties:__{: style="color: blue"}  
        * Convex
        * Grows linearly for negative values which make it less sensitive to outliers  
        * The logistic loss function does not assign zero penalty to any points. Instead, functions that correctly classify points with high confidence (i.e., with high values of $${\displaystyle \vert f({\vec {x}})\vert }$$) are penalized less. This structure leads the logistic loss function to be sensitive to outliers in the data.  
* __Cross-Entropy__{: style="color: red"}  
    * __What does it minimize:__{: style="color: blue"}  
        It minimizes the Kullback-Leibler divergence between the empirical distribution and the predicted distribution.  
    * __Formula:__{: style="color: blue"}  
        <p>$$L(\hat{y}, y) = -\sum_{i} y_i \log \left(\hat{y}_ {i}\right)$$</p>  
    * __Binary Cross-Entropy:__{: style="color: blue"}  
        <p>$$L(\hat{y}, y) = -\left[y \log \hat{y}+\left(1-y\right) \log \left(1-\hat{y}_ {n}\right)\right]$$</p>  
    * __Graph:__{: style="color: blue"}  
        ![img](/main_files/dl/concepts/loss_funcs/4.png){: width="30%" .center-image}  
    * __CE and Negative-Log-Probability:__{: style="color: blue"}  
        The __Cross-Entropy__ is equal to the __Negative-Log-Probability__ (of predicting the true class) in the case that the true distribution that we are trying to match is *__peaked at a single point__* and is *__identically zero everywhere else__*; this is usually the case in ML when we are using a _one-hot encoded vector_ with one class $$y = [0 \: 0 \: \ldots \: 0 \: 1 \: 0 \: \ldots \: 0]$$ peaked at the $$j$$-th position   
        $$\implies$$  
        <p>$$L(\hat{y}, y) = -\sum_{i} y_i \log \left(\hat{y}_ {i}\right) = - \log (\hat{y}_ {j})$$</p>  
    * __CE and Log-Loss:__{: style="color: blue"}  
        Given $$p \in\{y, 1-y\}$$ and $$q \in\{\hat{y}, 1-\hat{y}\}$$:  
        <p>$$H(p,q)=-\sum_{x }p(x)\,\log q(x) = -y \log \hat{y}-(1-y) \log (1-\hat{y}) = L(\hat{y}, y)$$</p>  
        * __Derivation:__{: style="color: blue"}  

            <button>Derivation</button>{: .showText value="show"
            onclick="showText_withParent_PopHide(event);"}
            _Given:_{: hidden=""}  
            * $$\hat{y} = \sigma(yf(x))$$,[^5]  
            * $$y \in \{-1, 1\}$$,   
            * $$\hat{y}' = \sigma(f(x))$$,  
            * $$y' = (1+y)/2 = \left\{\begin{array}{ll}{1} & {\text { for }} y' = 1 \\ {0} & {\text { for }} y = -1\end{array}\right. \in \{0, 1\}$$[^4]   
            * We start with the modified binary cross-entropy  
                $$\begin{aligned} -y' \log \hat{y}'-(1-y') \log (1-\hat{y}') &= \left\{\begin{array}{ll}{-\log\hat{y}'} & {\text { for }} y' = 1 \\ {-\log(1-\hat{y}')} & {\text { for }} y' = 0\end{array}\right. \\ \\
                &= \left\{\begin{array}{ll}{-\log\sigma(f(x))} & {\text { for }} y' = 1 \\ {-\log(1-\sigma(f(x)))} & {\text { for }} y' = 0\end{array}\right. \\ \\
                &= \left\{\begin{array}{ll}{-\log\sigma(1\times f(x))} & {\text { for }} y' = 1 \\ {-\log(\sigma((-1)\times f(x)))} & {\text { for }} y' = 0\end{array}\right. \\ \\
                &= \left\{\begin{array}{ll}{-\log\sigma(yf(x))} & {\text { for }} y' = 1 \\ {-\log(\sigma(yf(x)))} & {\text { for }} y' = 0\end{array}\right. \\ \\
                &= \left\{\begin{array}{ll}{-\log\hat{y}} & {\text { for }} y' = 1 \\ {-\log\hat{y}} & {\text { for }} y' = 0\end{array}\right. \\ \\
                &= -\log\hat{y} \\ \\
                &= \log\left[\dfrac{1}{\hat{y}}\right] \\ \\
                &= \log\left[\hat{y}^{-1}\right] \\ \\
                &= \log\left[\sigma(yf(x))^{-1}\right] \\ \\
                &= \log\left[ \left(\dfrac{1}{1+e^{-yf(x)}}\right)^{-1}\right] \\ \\
                &= \log \left(1+e^{-yf(x)}\right)\end{aligned}$$  
            {: hidden=""}
    * __CE and KL-Div:__{: style="color: blue"}  
        When comparing a distribution $${\displaystyle q}$$ against a fixed reference distribution $${\displaystyle p}$$, cross entropy and KL divergence are identical up to an additive constant (since $${\displaystyle p}$$ is fixed): both take on their minimal values when $${\displaystyle p=q}$$, which is $${\displaystyle 0}$$ for KL divergence, and $${\displaystyle \mathrm {H} (p)}$$ for cross entropy.  
        > Basically, minimizing either will result in the same solution.  
* __Exponential Loss__{: style="color: red"}  
    * __Formula:__{: style="color: blue"}  
        <p>$$L(\hat{y}, y) = e^{-\beta y \hat{y}}$$</p>  
    * __Properties:__{: style="color: blue"}  
        * Convex  
        * Grows Exponentially for negative values making it __more sensitive to outliers__  
        * It penalizes incorrect predictions more than Hinge loss and has a larger gradient.  
        * Used in __AdaBoost__ algorithm  
* __Perceptron Loss__{: style="color: red"}  
    * __Formula:__{: style="color: blue"}  
        <p>$${\displaystyle L(z, y_i) = {\begin{cases}0&{\text{if }}\ y_i\cdot z_i \geq 0\\-y_i z&{\text{otherwise}}\end{cases}}}$$</p>  
* __Analysis__{: style="color: red"}  
    * __Logistic vs Hinge Loss:__{: style="color: blue"}  
        __Logistic loss__ diverges faster than __hinge loss__ [(image)](#losses). So, in general, it will be more sensitive to outliers. [Reference. Bad info?](https://towardsdatascience.com/support-vector-machine-vs-logistic-regression-94cc2975433f)   
    * __Cross-Entropy vs MSE:__{: style="color: blue"}  
        Basically, CE > MSE because the gradient of MSE $$z(1-z)$$ leads to saturation when then output $$z$$ of a neuron is near $$0$$ or $$1$$ making the gradient very small and, thus, slowing down training.  
        CE > Class-Loss because Class-Loss is binary and doesn't take into account _"how well"_ are we actually approximating the probabilities as opposed to just having the target class be slightly higher than the rest (e.g. $$[c_1=0.3, c_2=0.3, c_3=0.4]$$).  
{: hidden=""}





***

# Information Theory
* __What is Information Theory? In the context of ML?__{: style="color: red"}  
    __Information theory__ is a branch of applied mathematics that revolves around quantifying how much information is present in a signal.    

    In the context of machine learning, we can also apply information theory to continuous variables where some of these message length interpretations do not apply, instead, we mostly use a few key ideas from information theory to characterize probability distributions or to quantify similarity between probability distributions.  
* __Describe the Motivation and Intuition for Information Theory:__{: style="color: red"}  
    The basic intuition behind information theory is that learning that an unlikely event has occurred is more informative than learning that a likely event has occurred. A message saying “the sun rose this morning” is so uninformative as to be unnecessary to send, but a message saying “there was a solar eclipse this morning” is very informative.  

    Thus, information theory quantifies information in a way that formalizes this intuition:    
    * Likely events should have low information content - in the extreme case, guaranteed events have no information at all  
    * Less likely events should have higher information content  
    * Independent events should have additive information. For example, finding out that a tossed coin has come up as heads twice should convey twice as much information as finding out that a tossed coin has come up as heads once.  
* __Measuring Information - Definitions and Formulas:__{: style="color: red"}  
    * __In Shannons Theory, how do we quantify *"transmitting 1 bit of information"*?__{: style="color: blue"}  
        To __transmit $$1$$ bit of information__ means to __divide the recipients *Uncertainty* by a factor of $$2$$__.  
    * __What is *the amount of information transmitted*?__{: style="color: blue"}  
        The __amount of information__ transmitted is the __logarithm__ (base $$2$$) of the __uncertainty reduction factor__.   
    * __What is the *uncertainty reduction factor*?__{: style="color: blue"}  
        It is the __inverse of the probability__ of the event being communicated.  
    * __What is the *amount of information in an event $$x$$*?__{: style="color: blue"}  
        The __amount of information__ in an event $$\mathbf{x} = x$$, called the *__Self-Information__*  is:  
        <p>$$I(x) = \log (1/p(x)) = -\log(p(x))$$</p>  
* __Define the *Self-Information* - Give the formula:__{: style="color: red"}  
    The __Self-Information__ or __surprisal__ is a synonym for the surprise when a random variable is sampled.  
    The __Self-Information__ of an event $$\mathrm{x} = x$$:    
    <p>$$I(x) = - \log P(x)$$</p>  
    * __What is it defined with respect to?__{: style="color: blue"}  
        Self-information deals only with a single outcome.  
* __Define *Shannon Entropy* - What is it used for?__{: style="color: red"}  
    __Shannon Entropy__ is defined as the average amount of information produced by a stochastic source of data.  
    Equivalently, the amount of information that you get from one sample drawn from a given probability distribution $$p$$.  
    
    To quantify the amount of uncertainty in an entire probability distribution, we use __Shannon Entropy__.  
    <p>$$H(x) = {\displaystyle \operatorname {E}_{x \sim P} [I(x)]} = - {\displaystyle \operatorname {E}_{x \sim P} [\log P(X)] = -\sum_{i=1}^{n} p\left(x_{i}\right) \log p\left(x_{i}\right)}$$</p>      

    * __Describe how Shannon Entropy relate to distributions with a graph:__{: style="color: blue"}  
        ![img](/main_files/math/prob/11.png){: width="100%"}    
* __Define *Differential Entropy*:__{: style="color: red"}  
    __Differential Entropy__ is Shannons entropy of a __continuous__ random variable $$x$$  
* __How does entropy characterize distributions?__{: style="color: red"}  
    Distributions that are nearly deterministic (where the outcome is nearly certain) have low entropy; distributions that are closer to uniform have high entropy.   
* __Define *Relative Entropy* - Give it's formula:__{: style="color: red"}  
    The __Kullback–Leibler divergence__ (__Relative Entropy__) is a measure of how one probability distribution diverges from a second, expected probability distribution.    

    __Mathematically:__    
    <p>$${\displaystyle D_{\text{KL}}(P\parallel Q)=\operatorname{E}_{x \sim P} \left[\log \dfrac{P(x)}{Q(x)}\right]=\operatorname{E}_{x \sim P} \left[\log P(x) - \log Q(x)\right]}$$</p>  
    * __Discrete__:    
    <p>$${\displaystyle D_{\text{KL}}(P\parallel Q)=\sum_{i}P(i)\log \left({\frac {P(i)}{Q(i)}}\right)}$$  </p>  
    * __Continuous__:    
    <p>$${\displaystyle D_{\text{KL}}(P\parallel Q)=\int_{-\infty }^{\infty }p(x)\log \left({\frac {p(x)}{q(x)}}\right)\,dx,}$$ </p>  

    * __Give an interpretation:__{: style="color: blue"}  
        * __Discrete variables__:  
            It is the extra amount of information needed to send a message containing symbols drawn from probability distribution $$P$$, when we use a code that was designed to minimize the length of messages drawn from probability distribution $$Q$$.  
    * __List the properties:__{: style="color: blue"}  
        * Non-Negativity:  
                $${\displaystyle D_{\mathrm {KL} }(P\|Q) \geq 0}$$  
        * $${\displaystyle D_{\mathrm {KL} }(P\|Q) = 0 \iff}$$ $$P$$ and $$Q$$ are:
            * *__Discrete Variables__*:  
                    the same distribution 
            * *__Continuous Variables__*:  
                    equal "almost everywhere"  
        * Additivity of _Independent Distributions_:  
                $${\displaystyle D_{\text{KL}}(P\parallel Q)=D_{\text{KL}}(P_{1}\parallel Q_{1})+D_{\text{KL}}(P_{2}\parallel Q_{2}).}$$  
        * $${\displaystyle D_{\mathrm {KL} }(P\|Q) \neq D_{\mathrm {KL} }(Q\|P)}$$  
            > This asymmetry means that there are important consequences to the choice of the ordering   
        * Convexity in the pair of PMFs $$(p, q)$$ (i.e. $${\displaystyle (p_{1},q_{1})}$$ and  $${\displaystyle (p_{2},q_{2})}$$ are two pairs of PMFs):  
                $${\displaystyle D_{\text{KL}}(\lambda p_{1}+(1-\lambda )p_{2}\parallel \lambda q_{1}+(1-\lambda )q_{2})\leq \lambda D_{\text{KL}}(p_{1}\parallel q_{1})+(1-\lambda )D_{\text{KL}}(p_{2}\parallel q_{2}){\text{ for }}0\leq \lambda \leq 1.}$$  
    * __Describe it as a distance:__{: style="color: blue"}  
        Because the KL divergence is non-negative and measures the difference between two distributions, it is often conceptualized as measuring some sort of distance between these distributions.  
        However, it is __not__ a true distance measure because it is __*not symmetric*__.  
        > KL-div is, however, a *__Quasi-Metric__*, since it satisfies all the properties of a distance-metric except symmetry  
    * __List the applications of relative entropy:__{: style="color: blue"}  
        Characterizing:  
        {: #lst-p}
        * Relative (Shannon) entropy in information systems
        * Randomness in continuous time-series 
        * Information gain when comparing statistical models of inference  
    * __How does the direction of minimization affect the optimization:__{: style="color: blue"}  
        Suppose we have a distribution $$p(x)$$ and we wish to _approximate_ it with another distribution $$q(x)$$.  
        We have a choice of _minimizing_ either:  
        1. $${\displaystyle D_{\text{KL}}(p\|q)} \implies q^\ast = \operatorname {arg\,min}_q {\displaystyle D_{\text{KL}}(p\|q)}$$  
            Produces an approximation that usually places high probability anywhere that the true distribution places high probability.  
        2. $${\displaystyle D_{\text{KL}}(q\|p)} \implies q^\ast \operatorname {arg\,min}_q {\displaystyle D_{\text{KL}}(q\|p)}$$  
            Produces an approximation that rarely places high probability anywhere that the true distribution places low probability.  
            > which are different due to the _asymmetry_ of the KL-divergence  

        <button>Choice of KL-div Direction</button>{: .showText value="show"  
         onclick="showTextPopHide(event);"}
        ![img](/main_files/math/infothry/1.png){: width="100%" hidden=""}  
* __Define *Cross Entropy* - Give it's formula:__{: style="color: red"}  
    The __Cross Entropy__ between two probability distributions $${\displaystyle p}$$ and $${\displaystyle q}$$ over the same underlying set of events measures the average number of bits needed to identify an event drawn from the set, if a coding scheme is used that is optimized for an "unnatural" probability distribution $${\displaystyle q}$$, rather than the "true" distribution $${\displaystyle p}$$.    
    <p>$$H(p,q) = \operatorname{E}_{p}[-\log q]= H(p) + D_{\mathrm{KL}}(p\|q) =-\sum_{x }p(x)\,\log q(x)$$</p>  
    * __What does it measure?__{: style="color: blue"}  
        The average number of bits that need to be transmitted using a different probability distribution $$q$$ (for encoding) than the "true" distribution $$p$$, to convey the information in $$p$$.  
    * __How does it relate to *relative entropy*?__{: style="color: blue"}  
        It is similar to __KL-Div__ but with an additional quantity - the entropy of $$p$$.  
    * __When are they equivalent?__{: style="color: blue"}  
        Minimizing the cross-entropy with respect to $$Q$$ is equivalent to minimizing the KL divergence, because $$Q$$ does not participate in the omitted term.  




***

# Recommendation Systems
* __Describe the different algorithms for recommendation systems:__{: style="color: red"}  




***

# Ensemble Learning
* __What are the two paradigms of ensemble methods?__{: style="color: red"}  
    * Parallel
    * Sequential
* __Random Forest VS GBM?__{: style="color: red"}  
    The fundamental difference is, random forest uses bagging technique to make predictions. GBM uses boosting techniques to make predictions.  




***

# Data Processing and Analysis
* __What are 3 data preprocessing techniques to handle outliers?__{: style="color: red"}  
    1. Winsorizing/Winsorization (cap at threshold).
    2. Transform to reduce skew (using Box-Cox or similar).
    3. Remove outliers if you're certain they are anomalies or measurement errors.
* __Describe the strategies to dimensionality reduction?__{: style="color: red"}  
    1. Feature Selection  
    2. Feature Projection/Extraction  
* __What are 3 ways of reducing dimensionality?__{: style="color: red"}  
    1. Removing Collinear Features
    2. Performing PCA, ICA, etc. 
    3. Feature Engineering
    4. AutoEncoder
    5. Non-negative matrix factorization (NMF)
    6. LDA
    7. MSD
* __List methods for Feature Selection__{: style="color: red"}  
    * Variance Threshold: normalize first (variance depends on scale)
    * Correlation Threshold: remove the one with larger mean absolute correlation with other features.  
    * Genetic Algorithms
    * Stepwise Search: bad performance, regularization much better, it's a greedy algorithm (can't account for future effects of each change)    
    * LASSO, Elastic-Net  
* __List methods for Feature Extraction__{: style="color: red"}  
    * PCA, ICA, CCA
    * AutoEncoders
    * LDA: LDA is a supervised linear transformation technique since the dependent variable (or the class label) is considered in the model. It Extracts the k new independent variables that __maximize the separation between the classes of the dependent variable__.  
        * Linear discriminant analysis is used to find a linear combination of features that characterizes or separates two or more classes (or levels) of a categorical variable.  
    * Latent Semantic Analysis
    * Isomap
* __How to detect correlation of "categorical variables"?__{: style="color: red"}  
    * Chi-Squared test: it is a statistical test applied to the groups of categorical features to evaluate the likelihood of correlation or association between them using their frequency distribution.  
* __Feature Importance__{: style="color: red"}  
    * Use linear regression and select variables based on p values
    * Use Random Forest, Xgboost and plot variable importance chart
    * Lasso
    * Measure information gain for the available set of features and select top n features accordingly.
    * Use Forward Selection, Backward Selection, Stepwise Selection
    * Remove the correlated variables prior to selecting important variables
    * In linear models, feature importance can be calculated by the scale of the coefficients  
    * In tree-based methods (such as random forest), important features are likely to appear closer to the root of the tree. We can get a feature's importance for random forest by computing the averaging depth at which it appears across all trees in the forest   
* __Capturing the correlation between continuous and categorical variable? If yes, how?__{: style="color: red"}  
    Yes, we can use ANCOVA (analysis of covariance) technique to capture association between continuous and categorical variables.  
* __What cross validation technique would you use on time series data set?__{: style="color: red"}  
    Forward chaining strategy with k folds.  
* __How to deal with missing features? (Imputation?)__{: style="color: red"}  
    1. Assign a unique category to missing values, who knows the missing values might decipher some trend.  
    2. Remove them blatantly
    3. we can sensibly check their distribution with the target variable, and if found any pattern we’ll keep those missing values and assign them a new category while removing others.  
* __Do you suggest that treating a categorical variable as continuous variable would result in a better predictive model?__{: style="color: red"}  
    For better predictions, categorical variable can be considered as a continuous variable only when the variable is ordinal in nature.  
* __What are collinearity and multicollinearity?__{: style="color: red"}  
    * __Collinearity__ occurs when two predictor variables (e.g., $$x_1$$ and $$x_2$$) in a multiple regression have some correlation.  
    * __Multicollinearity__ occurs when more than two predictor variables (e.g., $$x_1, x_2, \text{ and } x_3$$) are inter-correlated.  


***

# ML/Statistical Models
* __What are parametric models?__{: style="color: red"}  
    Parametric models are those with a finite number of parameters. To predict new data, you only need to know the parameters of the model. Examples include linear regression, logistic regression, and linear SVMs.
* __What is a classifier?__{: style="color: red"}  
    A function that maps... 



***

# K-NN




***

# [PCA](/work_files/research/conv_opt/pca)  
* __What is PCA?__{: style="color: red"}  
    It is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.  
* __What is the Goal of PCA?__{: style="color: red"}  
    Given points $$\mathbf{x}_ i \in \mathbf{R}^d$$, find k-directions that capture most of the variation.  
* __List the applications of PCA:__{: style="color: red"}  
    1. Find a small basis for representing variations in complex things.
        > e.g. faces, genes.  

    2. Reducing the number of dimensions makes some computations cheaper.  
    3. Remove irrelevant dimensions to reduce over-fitting in learning algorithms.
        > Like "_subset selection_" but the features are __not__ _axis aligned_.  
        > They are linear combinations of input features.  

    4. Represent the data with fewer parameters (dimensions)  
* __Give formulas for the following:__{: style="color: red"}  
    * __Assumptions on $$X$$:__{: style="color: blue"}  
        The analysis above is valid only for (1) $$X$$ w/ samples in rows and variables in columns  (2) $$X$$ is centered (mean=0)  
    * __SVD of $$X$$:__{: style="color: blue"}  
        $$X = USV^{T}$$
    * __Principal Directions/Axes:__{: style="color: blue"}  
        $$V$$ 
    * __Principal Components (scores):__{: style="color: blue"}  
        $$US$$  
    * __The $$j$$-th principal component:__{: style="color: blue"}  
        $$Xv_j = Us_j$$  
* __Define the transformation, mathematically:__{: style="color: red"}  
    Mathematically, the transformation is defined by a set of $$p$$-dimensional vectors of weights or coefficients $${\displaystyle \mathbf {v}_ {(k)}=(v_{1},\dots ,v_{p})_ {(k)}}$$ that map each row vector $${\displaystyle \mathbf {x}_ {(i)}}$$ of $$X$$ to a new vector of principal component scores $${\displaystyle \mathbf {t} _{(i)}=(t_{1},\dots ,t_{l})_ {(i)}}$$, given by:  
    <p>$${\displaystyle {t_{k}}_{(i)}=\mathbf {x}_ {(i)}\cdot \mathbf {v}_ {(k)}\qquad \mathrm {for} \qquad i=1,\dots ,n\qquad k=1,\dots ,l}$$</p>  
    in such a way that the individual variables $${\displaystyle t_{1},\dots ,t_{l}}$$  of $$t$$ considered over the data set successively inherit the maximum possible variance from $$X$$, with each coefficient vector $$v$$ constrained to be a unit vector (where $$l$$ is usually selected to be less than $${\displaystyle p}$$ to reduce dimensionality).  
* __What does PCA produce/result in?__{: style="color: red"}  
    <button>Show</button>{: .showText value="show"
    onclick="showTextPopHide(event);"}
    * __Finds a lower dimensional subspace spanned by what?:__{: style="color: blue"}  
        Finds a lower dimensional subspace spanned by PCs.  
    * __Finds a lower dimensional subspace that minimizes what?:__{: style="color: blue"}  
        Finds a lower dimensional subspace (PCs) that Minimizes the RSS of projection errors.  
    * __What does each PC have (properties)?__{: style="color: blue"}  
        Produces a vector (1st PC) with the highest possible variance, each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components.  
    * __What does the procedure find in terms of a "basis"?__{: style="color: blue"}  
        Results in an __uncorrelated orthogonal basis set__.  
    * __What does the procedure find in terms of axes? (where do they point?):__{: style="color: blue"}  
        PCA constructs new axes that point to the directions of maximal variance (in the original variable space)  
    {: hidden=""}
* __Describe the PCA algorithm:__{: style="color: red"}  
    * __Data Preprocessing__:  
        * Training set: $$x^{(1)}, x^{(2)}, \ldots, x^{(m)}$$ 
        * Preprocessing (__feature scaling__ + __mean normalization__):  
            * __mean normalization__:  
                $$\mu_{j}=\frac{1}{m} \sum_{i=1}^{m} x_{j}^{(i)}$$  
                Replace each $$x_{j}^{(i)}$$ with $$x_j^{(i)} - \mu_j$$  
            * __feature scaling__:  
                If different features on different, scale features to have comparable range  
                $$s_j = S.D(X_j)$$ (the standard deviation of feature $$j$$)  
                Replace each $$x_{j}^{(i)}$$ with $$\dfrac{x_j^{(i)} - \mu_j}{s_j}$$    
    * __Computing the Principal Components__:  
        * Compute the __SVD__ of the matrix $$X = U S V^T$$  
        * Compute the Principal Components:  
            <p>$$T = US = XV$$</p>  
            > Note: The $$j$$-th principal component is: $$Xv_j$$  
        * Choose the top $$k$$ components singular values in $$S = S_k$$  
        * Compute the Truncated Principal Components:  
            <p>$$T_k = US_k$$</p>  
    * __Computing the Low-rank Approximation Matrix $$X_k$$__:  
        * Compute the reconstruction matrix:  
            <p>$$X_k = T_kV^T = US_kV^T$$</p>  

    <button>Show specifics</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __What Data Processing needs to be done?__{: style="color: blue"}  
        * Preprocessing (__feature scaling__ + __mean normalization__):  
            * __mean normalization__:  
                $$\mu_{j}=\frac{1}{m} \sum_{i=1}^{m} x_{j}^{(i)}$$  
                Replace each $$x_{j}^{(i)}$$ with $$x_j^{(i)} - \mu_j$$  
            * __feature scaling__:  
                If different features on different, scale features to have comparable range  
                $$s_j = S.D(X_j)$$ (the standard deviation of feature $$j$$)  
                Replace each $$x_{j}^{(i)}$$ with $$\dfrac{x_j^{(i)} - \mu_j}{s_j}$$    
    * __How to compute the Principal Components?__{: style="color: blue"}  
        * Compute the __SVD__ of the matrix $$X = U S V^T$$  
        * Compute the Principal Components:  
            <p>$$T = US = XV$$</p>  
    * __How do you compute the Low-Rank Approximation Matrix $$X_k$$?__{: style="color: blue"}  
        * Choose the top $$k$$ components singular values in $$S = S_k$$  
        * Compute the Truncated Principal Components:  
            <p>$$T_k = US_k$$</p>  
        * Compute the reconstruction matrix:  
            <p>$$X_k = T_kV^T = US_kV^T$$</p>  
    {: hidden=""}
* __Describe the Optimality of PCA:__{: style="color: red"}  
    Optimal for Finding a lower dimensional subspace (PCs) that Minimizes the RSS of projection errors.  
* __List limitations of PCA:__{: style="color: red"}  
    * PCA is highly sensitive to the (relative) scaling of the data; no consensus on best scaling. 
* __Intuition:__{: style="color: red"}  
    * PCA can be thought of as fitting a $$p$$-dimensional ellipsoid to the data, where each axis of the ellipsoid represents a principal component. If some axis of the ellipsoid is small, then the variance along that axis is also small, and by omitting that axis and its corresponding principal component from our representation of the dataset, we lose only a commensurately small amount of information.  
    * Its operation can be thought of as revealing the internal structure of the data in a way that best explains the variance in the data.  

    <button>Show specifics</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __What property of the internal structure of the data does PCA reveal/explain?__{: style="color: blue"}  
        The variance in the data.  
    * __What object does it fit to the data?:__{: style="color: blue"}  
        * PCA can be thought of as fitting a $$p$$-dimensional ellipsoid to the data, where each axis of the ellipsoid represents a principal component.  
    {: hidden=""}
* __Should you remove correlated features b4 PCA?__{: style="color: red"}  
    Yes. Discarding correlated variables have a substantial effect on PCA because, in presence of correlated variables, the variance explained by a particular component gets inflated. [Discussion](https://stats.stackexchange.com/questions/50537/should-one-remove-highly-correlated-variables-before-doing-pca)  
* __How can we measure the "Total Variance" of the data?__{: style="color: red"}  
    **The Total Variance** of the data can be expressed as the sum of all the eigenvalues:  
    <p>$$\mathbf{Tr} \Sigma = \mathbf{Tr} (U \Lambda U^T) = \mathbf{Tr} (U^T U \Lambda) = \mathbf{Tr} \Lambda = \lambda_1 + \ldots + \lambda_n.$$</p>  
* __How can we measure the "Total Variance" of the *projected data*?__{: style="color: red"}  
    **The Total Variance** of the **_Projected_** data is:
    <p>$$\mathbf{Tr} (P \Sigma P^T ) = \lambda_1 + \lambda_2 + \cdots + \lambda_k. $$</p>  
* __How can we measure the *"Error in the Projection"*?__{: style="color: red"}  
    **The Error in the Projection** could be measured with respect to variance.  
    * We define the **ratio of variance** "explained" by the projected data (equivalently, the ratio of information _"retained"_) as:  
    <p>$$\dfrac{\lambda_1 + \ldots + \lambda_k}{\lambda_1 + \ldots + \lambda_n}. $$</p>  
    * __What does it mean when this ratio is high?__{: style="color: blue"}  
        If the ratio is _high_, we can say that much of the variation in the data can be observed on the projected plane.  
* __How does PCA relate to CCA?__{: style="color: red"}  
    * __CCA__ defines coordinate systems that optimally describe the cross-covariance between two datasets while  
    * __PCA__ defines a new orthogonal coordinate system that optimally describes variance in a single dataset.  
* __How does PCA relate to ICA?__{: style="color: red"}  
    __Independent component analysis (ICA)__ is directed to similar problems as principal component analysis, but finds additively separable components rather than successive approximations.  





***

# The Centroid Method

* **Define "The Centroid":**{: style="color: red"}    
    In mathematics and physics, the centroid or geometric center of a plane figure is the arithmetic mean ("average") position of all the points in the shape.   

    The definition extends to any object in n-dimensional space: its centroid is the mean position of all the points in all of the coordinate directions.  

* **Describe the Procedure:**{: style="color: red"}    
    Compute the mean ($$\mu_c$$) of all the vectors in class $$C$$ and the mean ($$\mu_x$$) of all the vectors not in $$C$$  

* **What is the Decision Function:**{: style="color: red"}    
    <p>$$f(x) = (\mu_c - \mu_x) \cdot \vec{x} - (\mu_c - \mu_x) \cdot \dfrac{\mu_c + \mu_x}{2}$$</p>  

* **Describe the Decision Boundary:**{: style="color: red"}    
    The decision boundary is a Hyperplane that bisects the line segment with endpoints $$<\mu_c, \mu_x>$$  


***





# [K-Means](/work_files/research/ml/kmeans)
* __What is K-Means?__{: style="color: red"}  
    It is a clustering algorithm. It aims to partition $$n$$ observations into $$k$$ clusters in which each observation belongs to the cluster with the nearest mean. It results in a partitioning of the data space into __Voronoi Cells__.   
* __What is the idea behind K-Means?__{: style="color: red"}  
    * Minimize the _aggregate intra-cluster distance_ 
    * Equivalent to minimizing the _variance_ 
    * Thus, it finds $$k-$$clusters with __minimum aggregate variance__
* __What does K-Mean find?__{: style="color: red"}  
    It finds $$k-$$clusters with __minimum aggregate variance__.  
* __Formal Description of the Model:__{: style="color: red"}  
    Given a set of observations $$\left(\mathbf{x}_{1}, \mathbf{x} _{2}, \ldots, \mathbf{x}_{n}\right)$$, $$\mathbf{x}_ i \in \mathbb{R}^d$$, the algorithm aims to partition the $$n$$ observations into $$k$$ sets $$\mathbf{S}=\left\{S_{1}, S_{2}, \ldots, S_{k}\right\}$$ so as to minimize the __intra-cluster Sum-of-Squares__ (i.e. __variance__).  
    * __What is the Objective?__{: style="color: blue"}  
        <p>$$\underset{\mathbf{S}}{\arg \min } \sum_{i=1}^{k} \sum_{\mathbf{x} \in S_{i}}\left\|\mathbf{x}-\boldsymbol{\mu}_{i}\right\|^{2}=\underset{\mathbf{S}}{\arg \min } \sum_{i=1}^{k}\left|S_{i}\right| \operatorname{Var} S_{i}$$</p>  
        where $$\boldsymbol{\mu}_i$$ is the mean of points in $$S_i$$. 
* __Description of the Algorithm:__{: style="color: red"}  
    * Choose two random points, call them _"Centroids"_  
    * Assign the closest $$N/2$$ points (Euclidean-wise) to each of the Centroids  
    * Compute the mean of each _"group"/class_ of points  
    * Re-Assign the centroids to the newly computed Means ↑
    * REPEAT!
* __What is the Optimization method used? What class does it belong to?__{: style="color: red"}  
    Coordinate descent. Expectation-Maximization.  
    <button>Show</button>{: .showText value="show"
    onclick="showTextPopHide(event);"}
    * __How does the optimization method relate to EM?__{: style="color: blue"}  
        The "assignment" step is referred to as the "expectation step", while the "update step" is a maximization step, making this algorithm a variant of the generalized expectation-maximization algorithm.  
    {: hidden=""}
* __What is the Complexity of the algorithm?__{: style="color: red"}  
    The original formulation of the problem is __NP-Hard__; however, __EM__ algorithms (specifically, Coordinate-Descent) can be used as efficient heuristic algorithms that converge quickly to good local minima.  
* __Describe the convergence and prove it:__{: style="color: red"}  
    Guaranteed to converge after a finite number of iterations to a local minimum.  
    * __Proof:__  
        The Algorithm Minimizes a __monotonically decreasing__, __Non-Negative__ _Energy function_ on a finite Domain:  
        By *__Monotone Convergence Theorem__* the objective Value Converges.  

        <button>Show Proof</button>{: .showText value="show"
        onclick="showTextPopHide(event);"}
        ![img](/main_files/ml/kmeans/2.png){: hidden=""}    
* __Describe the Optimality of the Algorithm:__{: style="color: red"}  
    * __Locally optimal__: due to convergence property  
    * __Non-Globally optimal:__  
        * The _objective function_ is *__non-convex__*  
        * Moreover, coordinate Descent doesn't converge to global minimum on non-convex functions.  
* __Derive the estimated parameters of the algorithm:__{: style="color: red"}  
    * __Objective Function:__{: style="color: blue"}  
        <p>$$J(S, \mu)= \sum_{i=1}^{k} \sum_{\mathbf{x} \in S_{i}} \| \mathbf{x} -\mu_i \|^{2}$$</p>  
    * __Optimization Objective:__{: style="color: blue"}  
        <p>$$\min _{\mu} \min _{S} \sum_{i=1}^{k} \sum_{\mathbf{x} \in S_{i}}\left\|\mathbf{x} -\mu_{i}\right\|^{2}$$</p>  
    * __Derivation:__{: style="color: blue"}  
    * Fix $$S = \hat{S}$$, optimize $$\mu$$:  
        <p>$$\begin{aligned} & \min _{\mu} \sum_{i=1}^{k} \sum_{\mathbf{x} \in \hat{S}_{i}}\left\|\mu_{i}-x_{j}\right\|^{2}\\
            =&  \sum_{i=1}^{k} \min _{\mu_i} \sum_{\mathbf{x} \in \hat{S}_{i}}\left\|\mathbf{x} - \mu_{i}\right\|^{2}
        \end{aligned}$$</p>  
        * __MLE__:  
            <p>$$\min _{\mu_i} \sum_{\mathbf{x} \in \hat{S}_{i}}\left\|\mathbf{x} - \mu_{i}\right\|^{2}$$</p>  
            $$ \implies $$  
            <p>$${\displaystyle \hat{\mu_i} = \dfrac{\sum_{\mathbf{x} \in \hat{S}_ {i}} \mathbf{x}}{\vert\hat{S}_ i\vert}}$$</p>  
            <button>Show Derivation</button>{: .showText value="show"
            onclick="showTextPopHide(event);"}
            ![img](/main_files/ml/kmeans/3.png){: width="75%" hidden=""}  
    * Fix $$\mu_i = \hat{\mu_i}, \forall i$$, optimize $$S$$[^1]:  
        <p>$$\arg \min _{S} \sum_{i=1}^{k} \sum_{\mathbf{x} \in S_{i}}\left\|\mathbf{x} - \hat{\mu_{i}}\right\|^{2}$$</p>  
        $$\implies$$  
        <p>$$S_{i}^{(t)}=\left\{x_{p} :\left\|x_{p}-m_{i}^{(t)}\right\|^{2} \leq\left\|x_{p}-m_{j}^{(t)}\right\|^{2} \forall j, 1 \leq j \leq k\right\}$$</p>  
        * __MLE__:  
            <button>Show Derivation</button>{: .showText value="show"
            onclick="showTextPopHide(event);"}
            ![img](/main_files/ml/kmeans/1.png){: width="75%" hidden=""}  





***

# [Naive Bayes](/work_files/research/ml/naive_bayes)
* __Define:__{: style="color: red"}  
    * __Naive Bayes:__{: style="color: blue"}  
        It is a simple technique used for constructing classifiers.  
    * __Naive Bayes Classifiers:__{: style="color: blue"}  
        A family of _simple probabilistic classifiers_ based on applying _bayes theorem_ with _strong (naive) independence assumptions_ __between the features__.  
    * __Bayes Theorem:__{: style="color: blue"}  
        $$p(x\vert y) = \dfrac{p(y\vert x) p(x)}{p(y)}$$ 
* __List the assumptions of Naive Bayes:__{: style="color: red"}  
    1. __Conditional Independence:__ the features are _conditionally independent_ from each other given a class $$C_k$$  
    2. __Bag-of-words:__ The relative importance (positions) of the features do not matter  
* __List some properties of Naive Bayes:__{: style="color: red"}  
    * __Not__ a _Bayesian method_  
    * It's a __Bayes Classifier__: minimizes the probability of misclassification  

    <button>Show</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __Is it a Bayesian Method or Frequentest Method?__{: style="color: blue"}  
        __Not__ a _Bayesian method_  
    * __Is it a Bayes Classifier? What does that mean?:__{: style="color: blue"}  
        It's a __Bayes Classifier__: minimizes the probability of misclassification 
    {: hidden=""}
* __Define the Probabilistic Model for the method:__{: style="color: red"}   
    Naive Bayes is a __conditional probability model:__  
    given a problem instance to be classified represented by a vector $$\boldsymbol{x} = (x_1, \cdots, x_n)$$ of $$n$$ features/words (independent variables), it assigns to this instance probabilities:  
    <p>$$P(C_k\vert \boldsymbol{x}) = p(C_k\vert x_1, \cdots, x_n)$$</p>  
    for each of the $$k$$ classes.  

    Using __Bayes theorem__ to decompose the conditional probability:  
    <p>$$p(C_k\vert \boldsymbol{x}) = \dfrac{p(\boldsymbol{x}\vert C_k) p(C_k)}{p(\boldsymbol{x})}$$</p>  

    Notice that the *__numerator__* is equivalent to the *__joint probability distribution__*:  
    <p>$$p\left(C_{k}\right) p\left(\mathbf{x} | C_{k}\right) = p\left(C_{k}, x_{1}, \ldots, x_{n}\right)$$</p>  
    Using the __Chain-Rule__ for repeated application of the conditional probability, the _joint probability_ model can be rewritten as:  
    <p>$$p(C_{k},x_{1},\dots ,x_{n})\, = p(x_{1}\mid x_{2},\dots ,x_{n},C_{k})p(x_{2}\mid x_{3},\dots ,x_{n},C_{k})\dots p(x_{n-1}\mid x_{n},C_{k})p(x_{n}\mid C_{k})p(C_{k})$$</p>  

    Using the __Naive Conditional Independence__ assumptions:  
    <p>$$p\left(x_{i} | x_{i+1}, \ldots, x_{n}, C_{k}\right)=p\left(x_{i} | C_{k}\right)$$</p>  
    Thus, we can write the __joint model__ as:  
    <p>$${\displaystyle {\begin{aligned}p(C_{k}\mid x_{1},\dots ,x_{n})&\varpropto p(C_{k},x_{1},\dots ,x_{n})\\&=p(C_{k})\ p(x_{1}\mid C_{k})\ p(x_{2}\mid C_{k})\ p(x_{3}\mid C_{k})\ \cdots \\&=p(C_{k})\prod _{i=1}^{n}p(x_{i}\mid C_{k})\,,\end{aligned}}}$$</p>  

    Finally, the *__conditional distribution over the class variable $$C$$__* is:  
    <p>$${\displaystyle p(C_{k}\mid x_{1},\dots ,x_{n})={\frac {1}{Z}}p(C_{k})\prod _{i=1}^{n}p(x_{i}\mid C_{k})}$$</p>   
    where, $${\displaystyle Z=p(\mathbf {x} )=\sum _{k}p(C_{k})\ p(\mathbf {x} \mid C_{k})}$$ is a __constant__ scaling factor, a __dependent only__ on the, _known_, feature variables $$x_i$$s.  

    <button>Show</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __What kind of model is it?__{: style="color: blue"}  
        Conditional Probability Model.  
    * __What is a conditional probability model?__{: style="color: blue"}  
        A model that assigns probabilities to an input $$\boldsymbol{x}$$ conditioned on being a member of each class in a set of $$k$$ classes $$C_1, \cdots, C_k$$.  
    * __Decompose the conditional probability w/ Bayes Theorem:__{: style="color: blue"}  
        <p>$$p(C_k\vert \boldsymbol{x}) \dfrac{p(\boldsymbol{x}\vert C_k) p(C_k)}{p(\boldsymbol{x})}= $$</p>
    * __How does the new expression incorporate the joint probability model?__{: style="color: blue"}  
        We notice that the *__numerator__* is equivalent to the *__joint probability distribution__*:  
        <p>$$p\left(C_{k}\right) p\left(\mathbf{x} | C_{k}\right) = p\left(C_{k}, x_{1}, \ldots, x_{n}\right)$$</p>
    * __Use the chain rule to re-write the joint probability model:__{: style="color: blue"}  
        <p>$$p(C_{k},x_{1},\dots ,x_{n})\, = p(x_{1}\mid x_{2},\dots ,x_{n},C_{k})p(x_{2}\mid x_{3},\dots ,x_{n},C_{k})\dots p(x_{n-1}\mid x_{n},C_{k})p(x_{n}\mid C_{k})p(C_{k})$$</p>  
    * __Use the Naive Conditional Independence assumption to rewrite the joint model:__{: style="color: blue"}  
        <p>$${\displaystyle {\begin{aligned}p(C_{k}\mid x_{1},\dots ,x_{n})&\varpropto p(C_{k},x_{1},\dots ,x_{n})\\&=p(C_{k})\ p(x_{1}\mid C_{k})\ p(x_{2}\mid C_{k})\ p(x_{3}\mid C_{k})\ \cdots \\&=p(C_{k})\prod _{i=1}^{n}p(x_{i}\mid C_{k})\,,\end{aligned}}}$$</p>  
    * __What is the conditional distribution over the class variable $$C_k$$:__{: style="color: blue"}  
        <p>$${\displaystyle p(C_{k}\mid x_{1},\dots ,x_{n})={\frac {1}{Z}}p(C_{k})\prod _{i=1}^{n}p(x_{i}\mid C_{k})}$$</p>  
        where, $${\displaystyle Z=p(\mathbf {x} )=\sum _{k}p(C_{k})\ p(\mathbf {x} \mid C_{k})}$$ is a __constant__ scaling factor, a __dependent only__ on the, _known_, feature variables $$x_i$$s.   
    {: hidden=""}
* __Construct the classifier. What are its components? Formally define it.__{: style="color: red"}  
    The classifier is made of (1) the conditional probability model (above) $$\:$$  and (2) A decision rule.  

    The decision rule used is the __MAP__ hypothesis: i.e. pick the hypothesis that is most probable (maximize the MAP estimate=posterior\*prior).  

    The classifier is the __function that assigns a class label $$\hat{y} = C_k$$__ for some $$k$$ as follows:  
    <p>$$\hat{y}=\underset{k \in\{1, \ldots, K\}}{\operatorname{argmax}} p\left(C_{k}\right) \prod_{i=1}^{n} p\left(x_{i} | C_{k}\right)$$</p>  
    <button>Show</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __What's the decision rule used?__{: style="color: blue"}  
        we commonly use the [__Maximum A Posteriori (MAP)__](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) hypothesis, as the decision rule; i.e. pick the hypothesis that is most probable.  
    * __List the difference between the Naive Bayes Estimate and the MAP Estimate:__{: style="color: blue"}  
        * __MAP Estimate__:  
            <p>$${\displaystyle {\hat {y}_{\text{MAP}}}={\underset {k\in \{1,\dots ,K\}}{\operatorname {argmax} }}\ p(C_{k})\ p(\mathbf {x} \mid C_{k})}$$</p>  
        * __Naive Bayes Estimate:__  
            <p>$${\displaystyle {\hat {y}_{\text{NB}}}={\underset {k\in \{1,\dots ,K\}}{\operatorname {argmax} }}\ p(C_{k})\displaystyle \prod _{i=1}^{n}p(x_{i}\mid C_{k})}$$</p>  
    {: hidden=""}
* __What are the parameters to be estimated for the classifier?:__{: style="color: red"}  
    * The posterior probability of each feature/word given a class
    * The prior probability of each class
* __What method do we use to estimate the parameters?:__{: style="color: red"}  
    Maximum Likelihood Estimation (MLE).  
* __What are the estimates for each of the following parameters?:__{: style="color: red"}  
    <button>Show</button>{: .showText value="show"
    onclick="showTextPopHide(event);"}
    * __The prior probability of each class:__{: style="color: blue"}  
        $$\hat{P}(C_k) = \dfrac{\text{doc-count}(C=C_k)}{N_\text{doc}}$$,  
    * __The conditional probability of each feature (word) given a class:__{: style="color: blue"}  
        $$\hat{P}(x_i | C_i) = \dfrac{\text{count}(x_i,C_j)}{\sum_{x \in V} \text{count}(x, C_j)}$$
    {: hidden=""}





***

# CNNs
* __What is a CNN?__{: style="color: red"}  
* __What are the layers of a CNN?__{: style="color: red"}  
* __What are the four important ideas and their benefits that the convolution affords CNNs:__{: style="color: red"}  
* __What is the inspirational model for CNNs:__{: style="color: red"}  
* __Describe the connectivity pattern of the neurons in a layer of a CNN:__{: style="color: red"}  
* __Describe the process of a ConvNet:__{: style="color: red"}  
* __Convolution Operation:__{: style="color: red"}  
    * __Define:__{: style="color: blue"}  
    * __Formula (continuous):__{: style="color: blue"}  
    * __Formula (discrete):__{: style="color: blue"}  
    * __Define the following:__{: style="color: blue"}  
        * __Feature Map:__{: style="color: blue"}  
    * __Does the operation commute?__{: style="color: blue"}  
* __Cross Correlation:__{: style="color: red"}  
    * __Define:__{: style="color: blue"}  
    * __Formulae:__{: style="color: blue"}  
    * __What are the differences/similarities between convolution and cross-correlation:__{: style="color: blue"}  
* __Write down the Convolution operation and the cross-correlation over two axes and:__{: style="color: red"}  
    * __Convolution:__{: style="color: blue"}  
    * __Convolution (commutative):__{: style="color: blue"}  
    * __Cross-Correlation:__{: style="color: blue"}  
* __The Convolutional Layer:__{: style="color: red"}  
    * __What are the parameters and how do we choose them?__{: style="color: blue"}  
    * __Describe what happens in the forward pass:__{: style="color: blue"}  
    * __What is the output of the forward pass:__{: style="color: blue"}  
    * __How is the output configured?__{: style="color: blue"}  
* __Spatial Arrangements:__{: style="color: red"}  
    * __List the Three Hyperparameters that control the output volume:__{: style="color: blue"}  
    * __How to compute the spatial size of the output volume?__{: style="color: blue"}  
    * __How can you ensure that the input & output volume are the same?__{: style="color: blue"}  
    * __In the output volume, how do you compute the $$d$$-th depth slice:__{: style="color: blue"}  
* __Calculate the number of parameters for the following config:__{: style="color: red"}  
    > Given:  
        * __Input Volume__:  $$64\times64\times3$$  
        * __Filters__:  $$15 7\times7$$  
        * __Stride__:  $$2$$  
        * __Pad__:  $$3$$  
* __Definitions:__{: style="color: red"}  
    * __Receptive Field:__{: style="color: blue"}  
* __Suppose the input volume has size  $$[ 32 × 32 × 3 ]$$  and the receptive field (or the filter size) is  $$5 × 5$$ , then each neuron in the Conv Layer will have weights to a *\_\_Blank\_\_* region in the input volume, for a total of  *\_\_Blank\_\_* weights:__{: style="color: red"}  
* __How can we achieve the greatest reduction in the spatial dims of the network (for classification):__{: style="color: red"}  
* __Pooling Layer:__{: style="color: red"}  
    * __Define:__{: style="color: blue"}  
    * __List key ideas/properties and benefits:__{: style="color: blue"}  
    * __List the different types of Pooling:__{: style="color: blue"}  
        [Answer](http://localhost:8889/work_files/research/dl/nlp/cnnsNnlp#bodyContents12)  
    * __List variations of pooling and their definitions:__{: style="color: blue"}  
        <button>Show Questions</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        * __What is "Learned Pooling":__{: style="color: blue"}  
        * __What is "Dynamical Pooling":__{: style="color: blue"}  
        {: hidden=""}
    * __List the hyperparams of Pooling Layer:__{: style="color: blue"}  
    * __How to calculate the size of the output volume:__{: style="color: blue"}  
    * __How many parameters does the pooling layer have:__{: style="color: blue"}    
    * __What are other ways to perform downsampling:__{: style="color: blue"}  
* __Weight Priors:__{: style="color: red"}  
    * __Define "Prior Prob Distribution on the parameters":__{: style="color: blue"}  
    * __Define "Weight Prior" and its types/classes:__{: style="color: blue"}  
        <button>Show Questions</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        * __Weak Prior:__{: style="color: blue"}  
        * __Strong Prior:__{: style="color: blue"}  
        * __Infinitely Strong Prior:__{: style="color: blue"}  
        {: hidden=""}
    * __Describe the Conv Layer as a FC Layer using priors:__{: style="color: blue"}  
    * __What are the key insights of using this view:__{: style="color: blue"}  
        <button>Show Questions</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        * __When is the prior imposed by convolution INAPPROPRIATE:__{: style="color: blue"}  
        * __What happens when the priors imposed by convolution and pooling are not suitable for the task?__{: style="color: blue"}  
        * __What kind of other models should Convolutional models be compared to? Why?:__{: style="color: blue"}  
        {: hidden=""}
* __When do multi-channel convolutions commute?__{: style="color: red"}  
[Answer](/work_files/research/dl/archits/convnets#bodyContents61)
* __Why do we use several different kernels in a given conv-layer?__{: style="color: red"}  
* __Strided Convolutions__{: style="color: red"}    
    * __Define:__{: style="color: blue"}  
    * __What are they used for?__{: style="color: blue"}  
    * __What are they equivalent to?__{: style="color: blue"}  
    * __Formula:__{: style="color: blue"}  
* __Zero-Padding:__{: style="color: red"}  
    * __Definition/Usage:__{: style="color: blue"}  
    * __List the types of padding:__{: style="color: blue"}  
* __Locally Connected Layers/Unshared Convolutions:__{: style="color: red"}  
* __Bias Parameter:__{: style="color: red"}  
    * __How many bias terms are used per output channel in the tradional convolution:__{: style="color: blue"}  
* __Dilated Convolutions__{: style="color: red"}    
    * __Define:__{: style="color: blue"}  
    * __What are they used for?__{: style="color: blue"}  
* __Stacked Convolutions__{: style="color: red"}    
    * __Define:__{: style="color: blue"}  
    * __What are they used for?__{: style="color: blue"}  

* [Archits](http://localhost:8889/work_files/research/dl/arcts)





***

# Theory





# RNNs
* __What is an RNN?__{: style="color: red"}  
    * __Definition:__{: style="color: blue"}  
    * __What machine-type is the standard RNN:__{: style="color: blue"}  
* __What is the big idea behind RNNs?__{: style="color: red"}  
* __Dynamical Systems:__{: style="color: red"}  
    * __Standard Form:__{: style="color: blue"}  
    * __RNN as a Dynamical System:__{: style="color: blue"}  
* __Unfolding Computational Graphs__{: style="color: red"}  
    * __Definition:__{: style="color: blue"}  
    * __List the Advantages introduced by unfolding and the benefits:__{: style="color: blue"}  
    * __Graph and write the equations of Unfolding hidden recurrence:__{: style="color: blue"}    
* __Describe the State of the RNN, its usage, and extreme cases of the usage:__{: style="color: red"}  
* __RNN Architectures:__{: style="color: red"}  
    * __List the three standard architectures of RNNs:__{: style="color: blue"}  
        * __Graph:__{: style="color: blue"}  
        * __Architecture:__{: style="color: blue"}  
        * __Equations:__{: style="color: blue"}  
        * __Total Loss:__{: style="color: blue"}  
        * __Complexity:__{: style="color: blue"}  
        * __Properties:__{: style="color: blue"}  
* __Teacher Forcing:__{: style="color: red"}  
    * __Definition:__{: style="color: blue"}  
    * __Application:__{: style="color: blue"}  
    * __Disadvantages:__{: style="color: blue"}  
    * __Possible Solutions for Mitigation:__{: style="color: blue"}  



***

# Optimization
* __Define the *sigmoid* function and some of its properties:__{: style="color: red"}  
    $$\sigma(-x) = 1 - \sigma(x)$$  
* __Backpropagation:__{: style="color: red"}  
    * __Definition:__{: style="color: blue"}  
        Backpropagation algorithms are a family of methods used to efficiently train artificial neural networks (ANNs) following a gradient descent approach that exploits the chain rule.  
    * __Derive Gradient Descent Update:__{: style="color: blue"}  
        [Answer](/work_files/research/dl/concepts/grad_opt#bodyContents22)  
    * __Explain the difference kinds of gradient-descent optimization procedures:__{: style="color: blue"}  
        1. __Batch Gradient Descent__ AKA __Vanilla Gradient Descent__, computes the gradient of the objective wrt. the parameters $$\theta$$ for the entire dataset:  
            <p>$$\theta=\theta-\epsilon \cdot \nabla_{\theta} J(\theta)$$</p>  
        2. __SGD__ performs a parameter update for each data-point:  
            <p>$$\theta=\theta-\epsilon \cdot \nabla_{\theta} J\left(\theta ; x^{(i)} ; y^{(i)}\right)$$</p>  
        3. __Mini-batch Gradient Descent__ a hybrid approach that perform updates for a, pre-specified, mini-batch of $$n$$ training examples:  
            <p>$$\theta=\theta-\epsilon \cdot \nabla_{\theta} J\left(\theta ; x^{(i : i+n)} ; y^{(i : i+n)}\right)$$</p> 
    * __List the different optimizers and their properties:__{: style="color: blue"}  
        [Answer](/work_files/research/dl/concepts/grad_opt#content4)  
* __Error-Measures:__{: style="color: red"}  
    * __Define what an error measure is:__{: style="color: blue"}  
        __Error Measures__ aim to answer the question:  
        "What does it mean for $$h$$ to approximate $$f$$ ($$h \approx f$$)?"  
        The __Error Measure__: $$E(h, f)$$  
        It is almost always defined point-wise: $$\mathrm{e}(h(\mathbf{X}), f(\mathbf{X}))$$.  
    * __List the 5 most common error measures and where they are used:__{: style="color: blue"}  

    * __Specific Questions:__{: style="color: blue"}  
        <button>Show Questions</button>{: .showText value="show" onclick="showTextPopHide(event);"}
        * __Derive MSE carefully:__{: style="color: blue"}  
        * __Derive the Binary Cross-Entropy Loss function:__{: style="color: blue"}  
            It is the log-likelihood of a Bernoulli probability model:  
            <p>$$\begin{array}{c}{L(p)=p^{y}(1-p)^{1-y}} \\ {\log (L(p))=y \log p+(1-y) \log (1-p)}\end{array}$$</p>  
        * __Explain the difference between Cross-Entropy and MSE and which is better (for what task)?__{: style="color: blue"}  
        * __Describe the properties of the Hinge loss and why it is used?__{: style="color: blue"}  
            * Hinge loss upper bounds 0-1 loss
            * It is the tightest _convex_ upper bound on the 0/1 loss  
            * Minimizing 0-1 loss is NP-hard in the worst-case  
        {: hidden=""}  
* __Show that the weight vector of a linear signal is orthogonal to the decision boundary?__{: style="color: red"}  
    The weight vector $$\mathbf{w}$$ is orthogonal to the separating-plane/decision-boundary, defined by $$\mathbf{w}^T\mathbf{x} + b = 0$$, in the $$\mathcal{X}$$ space; Reason:  
    Since if you take any two points $$\mathbf{x}^\prime$$ and $$\mathbf{x}^{\prime \prime}$$ on the plane, and create the vector $$\left(\mathbf{x}^{\prime}-\mathbf{x}^{\prime \prime}\right)$$  parallel to the plane by subtracting the two points, then the following equations must hold:  
    <p>$$\mathbf{w}^{\top} \mathbf{x}^{\prime}+b=0 \wedge \mathbf{w}^{\top} \mathbf{x}^{\prime \prime}+b=0 \implies \mathbf{w}^{\top}\left(\mathbf{x}^{\prime}-\mathbf{x}^{\prime \prime}\right)=0$$</p>  
* __What does it mean for a function to be *well-behaved* from an optimization pov?__{: style="color: red"}  
    The __well-behaved__ property from an optimization standpoint, implies that $$f''(x)$$ doesn't change too much or too rapidly, leading to a nearly quadratic function that is easy to optimize by gradient methods.  
* __Write $$\|\mathrm{Xw}-\mathrm{y}\|^{2}$$ as a summation__{: style="color: red"}  
    <p>$$\frac{1}{N} \sum_{n=1}^{N}\left(\mathbf{w}^{\mathrm{T}} \mathbf{x}_{n}-y_{n}\right)^{2} = \frac{1}{N}\|\mathrm{Xw}-\mathrm{y}\|^{2}$$</p>  
* __Compute:__{: style="color: red"}  
    * __$$\dfrac{\partial}{\partial y}\vert{x-y}\vert=$$__{: style="color: blue"}  
        $$\dfrac{\partial}{\partial y} \vert{x-y}\vert  = - \text{sign}(x-y)$$  
* __State the difference between SGD and GD?__{: style="color: red"}  
    __Gradient Descent__’s cost-function iterates over ALL training samples.  
    __Stochastic Gradient Descent__’s cost-function only accounts for ONE training sample, chosen at random.  
* __When would you use GD over SDG, and vice-versa?__{: style="color: red"}  
    GD theoretically minimizes the error function better than SGD. However, SGD converges much faster once the dataset becomes large.  
    That means GD is preferable for small datasets while SGD is preferable for larger ones.  
* __What is convex hull ?__{: style="color: red"}  
    In case of linearly separable data, convex hull represents the outer boundaries of the two group of data points. Once convex hull is created, we get maximum margin hyperplane (MMH) as a perpendicular bisector between two convex hulls. MMH is the line which attempts to create greatest separation between two groups.  
* __OLS vs MLE__{: style="color: red"}  
    They both estimate parameters in the model. They are the same in the case of normal distribution.  

***

# ML Theory
* __Explain intuitively why Deep Learning works?__{: style="color: red"}  
    __Circuit Theory:__ There are function you can compute with a "small" L-layer deep NN that shallower networks require exponentially more hidden units to compute. (comes from looking at networks as logic gates).  
    * __Example__:  
        Computing $$x_1 \text{XOR} x_2 \text{XOR} ... \text{XOR} x_n$$  takes:   
        * $$\mathcal{O}(log(n))$$ in a tree representation.  
            ![img](/main_files/concepts/7.png){: width="65%"}  
        * $$\mathcal{O}(2^n)$$ in a one-hidden-layer network because you need to exhaustively enumerate all possible $$2^N$$ configurations of the input bits that result in the $$\text{XOR}$$ being $${1, 0}$$.   
            ![img](/main_files/concepts/8.png){: width="65%"}  
* __List the different types of Learning Tasks and their definitions:__{: style="color: red"}  
    * __Multi-Task Learning__: general term for training on multiple tasks  
        * _Joint Learning:_ by choosing mini-batches from two different tasks simultaneously/alternately
        * _Pre-Training:_ first train on one task, then train on another  
            > widely used for __word embeddings__  
    * __Transfer Learning__:  
        a type of multi-task learning where we are focused on one task; by learning on another task then applying those models to our main task  
    * __Domain Adaptation__:  
        a type of transfer learning, where the output is the same, but we want to handle different inputs/topics/genres  
    * __Zero-Shot Learning__:  
    [answer](/concepts_#bodyContents64)  
* __Describe the relationship between supervised and unsupervised learning?__{: style="color: red"}  
    [answer](/concepts_#bodyContents64)  
    Many ml algorithms can be used to perform both tasks. E.g., the chain rule of probability states that for a vector $$x \in \mathbb{R}^n$$, the joint distribution can be decomposed as:  
    $$p(\mathbf{x})=\prod_{i=1}^{n} p\left(\mathrm{x}_{i} | \mathrm{x}_{1}, \ldots, \mathrm{x}_{i-1}\right)$$  
    which implies that we can solve the Unsupervised problem of modeling $$p(x)$$ by splitting it into $$n$$ supervised learning problems.  
    Alternatively, we can solve the supervised learning problem of learning $$p(y \vert x)$$ by using traditional unsupervised learning technologies to learn the joint distribution $$p(x, y)$$, then inferring:  
    $$p(y | \mathbf{x})=\frac{p(\mathbf{x}, y)}{\sum_{y} p\left(\mathbf{x}, y^{\prime}\right)}$$  
* __Describe the differences between Discriminative and Generative Models?__{: style="color: red"}  
* __Describe the curse of dimensionality and its effects on problem solving:__{: style="color: red"}  
    It is a phenomena where many machine learning problems become exceedingly difficult when the number of dimensions in the data is high.  

    The number of possible distinct configurations of a set of variables increases exponentially as the number of variables increases:  
    <button>Capacity and Bias/Variance</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](/main_files/dl_book/2.png){: hidden=""}  
    * __Statistical Challenge:__ the number of possible configurations of $$x$$ is much larger than the number of training examples  
    * __Stastical Sampling__: The sampling density is proportional to $$N^{1/p}$$, where $$p$$ is the dimension of the input space and $$N$$ is the sample size. Thus, if $$N_1 = 100$$ represents a dense sample for a single input problem, then $$N_{10} = 100^{10}$$ is the sample size required for the same sampling density with $$10$$ inputs. Thus in high dimensions all feasible training samples sparsely populate the input space.   
    * [Further Reading](/concepts_#bodyContents621)  
* __How to deal with curse of dimensionality__{: style="color: red"}  
    * Feature Selection
    * Feature Extraction  
* __Describe how to initialize a NN and any concerns w/ reasons:__{: style="color: red"}  
    * Don't initialize the weights to Zero. The symmetry of hidden units results in a similar computation for each hidden unit, making all the rows of the weight matrix to be equal (by induction).  
    * It's OK to initialize the bias term to zero.  
* __Describe the difference between Learning and Optimization in ML:__{: style="color: red"}  
    * The problem of Reducing the __training error__ on the __training set__ is one of *__optimization__*.  
    * The problem of Reducing the __training error__, as well as, the __generalization (test) error__ is one of *__learning__*.  
* __List the 12 Standard Tasks in ML:__{: style="color: red"}  
    [Answer](/work_files/research/dl/theory/dl_book_pt1#bodyContents12)  
* __What is the difference between inductive and deductive learning?__{: style="color: red"}  
    * __Inductive learning__ is the process of using observations to draw conclusions 
    * __Deductive learning__ is the process of using conclusions to form observations 


***

# Statistical Learning Theory
* __Define Statistical Learning Theory:__{: style="color: red"}  
    It is a framework for machine learning drawing from the fields of __statistics__ and __functional analysis__ that allows us, under certain assumptions, to study the question:  
    > __How can we affect performance on the test set when we can only observe the training set?__{: style="color: blue"}    

    It is a statistical approach to __Computational Learning Theory__.   
* __What assumptions are made by the theory?__{: style="color: red"}  
    1. The training and test data are generated by an *__unknown__* __data generating distribution__ (over the product space $$Z = X \times Y$$, denoted: $$p_{\text{data}}(z) = p(x,y)$$) called the __data-generating process__.  
    2. The __i.i.d__ assumptions:  
        1. The data-points in each dataset are __independent__ from each other
        2. The training and testing are both __identically distributed__ (drawn from the same distribution)  

    <button>Show</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __Define the i.i.d assumptions?__{: style="color: blue"}  
        A collection of random variables is __independent and identically distributed__ if each random variable has the same probability distribution as the others and all are mutually independent.  
    * __Why assume a *joint* probability distribution $$p(x,y)$$?__{: style="color: blue"}  
        Note that the assumption of a joint probability distribution allows us to model uncertainty in predictions (e.g. from noise in data) because $${\displaystyle y}$$ is not a deterministic function of $${\displaystyle x}$$, but rather a random variable with conditional distribution $${\displaystyle P(y|x)}$$ for a fixed $${\displaystyle x}$$.  
    * __Why do we need to model $$y$$ as a target-distribution and not a target-function?__{: style="color: red"}  
        The ‘Target Function’ is not always a function because two ‘identical’ input points can be mapped to two different outputs (i.e. they have different labels).  
            
    {: hidden=""}
* __Give the Formal Definition of SLT:__{: style="color: red"}  
    <button>Show</button>{: .showText value="show"
     onclick="showTextPopHide(event);"}
    * __The Definitions:__{: style="color: blue"}  
        * $$X$$: $$\:$$ Input (vector) Space
        * $$Y$$: $$\:$$ Output (vector) Space
        * $$Z = X \times Y$$: $$\:$$ Product space of (input, output) pairs
        * $$n$$: $$\:$$ number of data points
        * $$S = \left\{\left(\vec{x}_{1}, y_{1}\right), \ldots,\left(\vec{x}_{n}, y_{n}\right)\right\}=\left\{\vec{z}_{1}, \ldots, \vec{z}_{n}\right\}$$: $$\:$$ the __training set__   
        * $$\mathcal{H} = f : X \rightarrow Y$$: $$\:$$ the __hypothesis space__ of all functions  
        * $$V(f(\vec{x}), y)$$: $$\:$$ an __error/loss function__  
    * __The Assumptions:__{: style="color: blue"}  
        1. The training and testing sets are generated by an *__unknown__* __data-generating distribution function__ (over $$Z$$, denoted: $$p_{\text{data}} = p(z) = p(x,y)$$) called the __data-generation process__.  
        2. The __i.i.d assumptions:__  
            1. The examples in each dataset are generated independently from each other  
            2. Both datasets are identically distributed  
    * __The Inference Problem:__{: style="color: blue"}  
        Find a function $$f : X \rightarrow Y$$ such that $$f(\vec{x}) \sim y$$.  
    * __The Expected Risk:__{: style="color: blue"}  
        It is the overall average risk over the entire (data) probability-distribution:  
        <p>$$I[f] = \mathbf{E}[V(f(\vec{x}), y)]=\int_{X \times Y} V(f(\vec{x}), y) p(\vec{x}, y) d \vec{x} d y$$</p>  
    * __The Target Function:__{: style="color: blue"}  
        It is the best possible function $$f$$ that can be chosen, and is given by:  
        <p>$$f = \inf_{h \in \mathcal{H}} I[h]$$</p>  
    * __The Empirical Risk:__{: style="color: blue"}  
        It is a __proxy measure__ to the _expected risk_ based on the training set. It is _necessary_ since the probability distribution $$p(\vec{x}, y)$$ is _unknown_.  
        <p>$$I_{S}[f]=\frac{1}{n} \sum_{i=1}^{n} V\left(f\left(\vec{x}_{i}\right), y_{i}\right)$$</p>  
    {: hidden=""}
* __Define Empirical Risk Minimization:__{: style="color: red"}  
    It is a (_learning_) principle in _statistical learning theory_ that is based on _approximating_ the __Expected/True Risk (Generalization Error)__ by measuring the __Empirical Risk (Training Error)__; i.e. the performance on the training-data.  

    A __learning algorithm__ that chooses the function $$f_S$$ which _minimizes the empirical risk_ is called __Empirical Risk Minimization:__  
    <p>$$R_{\text{emp}} = I_S[f] = \dfrac{1}{n} \sum_{i=1}^n V(f(\vec{x}_ i, y_i))$$</p>  
    <p>$$f_{S} = \hat{h} = \arg \min _{h \in \mathcal{H}} R_{\mathrm{emp}}(h)$$</p>  
* __What is the Complexity of ERM?__{: style="color: red"}  
    __NP-Hard__ for *__classification__* with $$0-1$$ loss function, even for linear classifiers  
    * It can be solved _efficiently_ when the minimal _empirical risk_ is ZERO; i.e. the data is linearly separable  

    <button>Show</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __How do you Cope with the Complexity?__{: style="color: blue"}  
        * Employ a __convex approximation__ to the $$0-1$$ loss: _Hinge_, _SVM_ etc.  
        * Imposing __assumptions on the data-generating distribution__ thus, stop being an __agnostic learning algorithm__  
    {: hidden=""}
* __Definitions:__{: style="color: red"}  
    * __Generalization:__{: style="color: blue"}  
        The ability to do well on previously unobserved data:  
        <p>$$I[f] \approx I_S[f]$$</p>  
        _"Good" generalization_ is achieved when the _empirical risk_ approximates the _expected risk_ __well__.  
    * __Generalization Error:__{: style="color: blue"}  
        AKA: __Expected Risk__, __Out-of-sample Error__, __$$E_{\text{out}}$$__  
        It is a measure of how accurately an algorithm is able to predict outcome values for previously unseen data defined as the __expected value of the error__ on a new input, measured w.r.t. the data-generating probability distribution.  

    * __Generalization Gap:__{: style="color: blue"}  
        It is a measure of how accurately an algorithm is able to predict outcome values for previously unseen data, defined as the difference between the __expected risk__ and __empirical risk__:    
        <p>$$G =I\left[f_{n}\right]-I_{S}\left[f_{n}\right]$$</p>  
        * __Computing the Generalization Gap:__{: style="color: blue"}  
            Since the _empirical risk/generalization error $$I[f_n]$$_ cannot be computed for an *__unknown__* __distribution__, the generalization gap cannot be computed either.  
        * __What is the goal of SLT in the context of the Generalization Gap given that it can't be computed?__{: style="color: blue"}  
            Instead, the goal of SLT is to bound/characterize the _gap_ in __probability:__  
            <p>$$P_{G}=P\left(I\left[f_{n}\right]-I_{S}\left[f_{n}\right] \leq \epsilon\right) \geq 1-\delta_{n}$$</p>  
            I.E. goal is to characterize the probability $$1- \delta_n$$ that the generalization gap is _less_ than some __error bound__ $$\epsilon$$ (known as the __learning rate__)  
    * __Achieving ("good") Generalization:__{: style="color: blue"}  
        An _algorithm_ is said to __generalize__ when the *__expected risk__* is __well approximated__ by the *__empirical risk__*:  
        <p>$$I\left[f_{n}\right] \approx I_{S}\left[f_{n}\right]$$</p>  
        Equivalently:  
        <p>$$E_{\text {out}}(g) \approx E_{\text {in}}(g)$$</p>  
        I.E. when the __generalization gap__ approaches _zero_ in the limit of data-points:  
        <p>$$\lim _{n \rightarrow \infty} G_{n}=\lim _{n \rightarrow \infty} I\left[f_{n}\right]-I_{S}\left[f_{n}\right]=0$$</p>  
    * __Empirical Distribution:__{: style="color: blue"}  
        AKA: __Data-Generating Distribution__  
        is the __discrete, uniform, joint__ distribution $$p_{\text{data}} = p(x,y)$$ over the sample points.  

* __Describe the difference between Learning and Optimization in ML:__{: style="color: red"}  
    * __Optimization__: is concerned with the problem of *__reducing__* the __training error__ on the __training set__  
    * __Learning__: is concerned with the problem of *__reducing__* the __training error__, as well as, the __generalization (test) error__  
* __Describe the difference between Generalization and Learning in ML:__{: style="color: red"}  
    * __Generalization__ guarantee: tells us that it is likely that the following condition holds:  
        <p>$$E_{\mathrm{out}}(\hat{h}) \approx E_{\mathrm{in}}(\hat{h})$$</p>  
        I.E. that the _empirical error_ __tracks/approximates__ the _expected/generalization error_ __well__.  
    * __Learning__: corresponds to the condition that $$\hat{h} \approx f$$ the *__chosen hypothesis approximates the target function well__*, which in-turn corresponds to the condition:  
        <p>$$E_{\mathrm{out}}(\hat{h}) \approx 0$$</p>  
* __How to achieve Learning?__{: style="color: red"}          
    To achieve learning we need to achieve the condition $$E_{\mathrm{out}}(\hat{h}) \approx 0$$, which we do by:  
    1. $$E_{\mathrm{out}}(\hat{h}) \approx E_{\mathrm{in}}(\hat{h})$$  
        A __theoretical__ result achieved through *__Hoeffding Inequality__*  
    2. $$E_{\mathrm{in}}(\hat{h}) \approx 0$$  
        A __practical__ result of *__Empirical Error Minimization__*  
* __What does the (VC) Learning Theory Achieve?__{: style="color: red"}  
    1. Characterizing the __feasibility of learning__ for *__infinite hypotheses__*  
    2. Characterizing the __Approximation-Generalization Tradeoff__  
* __Why do we need the probabilistic framework?__{: style="color: red"}  
* __What is the *Approximation-Generalization Tradeoff*:__{: style="color: red"}  
    It is a tradeoff between (1) How well we can approximate the target function $$f \:\: $$ and (2) How well we can generalize to unseen data.  
    Given the __goal:__ Small __$$E_{\text{out}}$$__, good approximation of $$f$$ *__out of sample__* (not in-sample).  
    The tradeoff is characterized by the __complexity__ of the __hypothesis space $$\mathcal{H}$$__:  
    * __More Complex $$\mathcal{H}$$__: Better chance of approximating $$f$$  
    * __Less Complex $$\mathcal{H}$$__: Better chance of generalizing out-of-sample  
* __What are the factors determining how well an ML-algo will perform?__{: style="color: red"}  
    1. _Ability to_ Approximate $$f$$ well, in-sample \| Make the training error small  &    
    2. Decrease the gap between $$E_{\text{in}}$$ and $$E_{\text{out}}$$ \| Make gap between training and test error small  
* __Define the following and their usage/application & how they relate to each other:__{: style="color: red"}  
    * __Underfitting:__{: style="color: blue"}  
        Occurs when the model cannot fit the training data well; high $$E_{\text{in}}$$.  
    * __Overfitting:__{: style="color: blue"}  
        Occurs when the gap between the training error and test error is too large.  
    * __Capacity:__{: style="color: blue"}  
        a models ability to fit a high variety of functions (complexity).  
        It allows us to control the amount of overfitting and underfitting
        * Models with __Low-Capacity:__{: style="color: blue"}  
            Underfitting. High-Bias. Struggle to fit training-data.  
        * Models with __High-Capacity:__{: style="color: blue"}  
            Overfitting. High-Variance. Memorizes noise.  
    * __Hypothesis Space:__{: style="color: blue"}  
        The set of functions that the learning algorithm is allowed to select as being the target function.  
        Allows us to control the capacity of a model.  
    * __VC-Dimension:__{: style="color: red"}  
        The largest possible value of $$m$$ for which there exists a training set of $$m$$ different points that the classifier can label arbitrarily.  
        Quantifies a models capacity.  
        * __What does it measure?__{: style="color: blue"}  
            Measures the __capacity of a binary classifier__.  
    * __Graph the relation between Error, and Capacity in the ctxt of (Underfitting, Overfitting, Training Error, Generalization Err, and Generalization Gap):__{: style="color: blue"}  
        ![img](/main_files/dl_book/10.png){: width="70%"}   
* __What is the most important result in SLT that show that learning is feasible?__{: style="color: red"}  
    Shows that the discrepancy between training error and generalization error is bounded above by a quantity that grows as the model capacity grows but shrinks as the number of training examples increases.  

***





# Bias-Variance Decomposition Theory
* __What is the Bias-Variance Decomposition Theory:__{: style="color: red"}  
    It is an approach for the quantification of the __Approximation-Generalization Tradeoff__.
* __What are the Assumptions made by the theory?__{: style="color: red"}  
    * Analysis is done over the __entire data-distribution__ 
    * __Real-Valued__ inputs, targets (can be extended)
    * Target function $$f$$ is __known__
    * Uses __MSE__ (can be extended)
* __What is the question that the theory tries to answer? What assumption is important? How do you achieve the answer/goal?__{: style="color: red"}
    * "How can $$\mathcal{H}$$ approximate $$f$$ overall? not just on our sample/training-data.".  
    * We assume that the target function $$f$$ is known.  
    * By taking the __expectation over all possible realization of $$N$$ data-points__.  
* __What is the Bias-Variance Decomposition:__{: style="color: red"}  
    It is the decomposition of the __expected error__ as a sum of three concepts: __bias__, __variance__ and __irreducible error__, each quantifying an aspect of the error.  
* __Define each term w.r.t. source of the error:__{: style="color: red"}  
    * __Bias__: the error from the erroneous assumptions in the learning algorithm  
    * __Variance__: the error from sensitivity to small fluctuations in the training set  
    * __Irreducible Error__: the error resulting from the noise in the problem itself  
* __What does each of the following measure? Describe it in Words? Give their AKA in statistics?__{: style="color: red"}  
    * __Bias:__{: style="color: blue"}  
        * AKA: __Approximation Error__ 
        * Measures the error in approximating the target function with the best possible hypothesis in $$\mathcal{H}$$  
        * The expected deviation from the true value of the function (or parameter)  
        * How well can $$\mathcal{H}$$ approximate the target function $$f$$  
    * __Variance:__{: style="color: blue"}  
        * AKA: __Estimation Error__  
        * Measures the error in estimating the best possible hypothesis in $$\mathcal{H}$$ with a particular hypothesis resulting from a specific training-set 
        * The deviation from the expected estimator value that any particular sampling of the data is likely to cause  
        * How well can we zoom in on a good $$h \in \mathcal{H}$$  
    * __Irreducible Error:__{: style="color: blue"} measures the inherent noise in the target $$y$$   
* __Give the Formal Definition of the Decomposition (Formula):__{: style="color: red"}  
    Given any hypothesis $$\hat{f} = g^{\mathcal{D}}$$ we select, we can decompose its __expected risk__ on an _unseen sample_ $$x$$ as:  
    <p>$$\mathbb{E}\left[(y-\hat{f}(x))^{2}\right]=(\operatorname{Bias}[\hat{f}(x)])^{2}+\operatorname{Var}[\hat{f}(x)]+\sigma^{2}$$</p>  
    Where:  
    * __Bias__: 
        <p>$$\operatorname{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$$</p>  
    * __Variance__:  
        <p>$$\operatorname{Var}[\hat{f}(x)] = \mathbb{E}\left[(\hat{f}(x))^2\right] - \mathbb{E}\left[\hat{f}(x)\right]^2$$</p>  
            
    * __What is the Expectation over?__{: style="color: blue"}  
        The expectation is over all different samplings of the data $$\mathcal{D}$$  
* __Define the *Bias-Variance Tradeoff*:__{: style="color: red"}  
    It is the property of a set of predictive models whereby, models with a *__lower bias__* have a *__higher variance__* and vice-versa.  
    * __Effects of Bias:__{: style="color: blue"}  
        * __High Bias__: simple models $$\rightarrow$$ __underfitting__  
        * __Low Bias__: complex models $$\rightarrow$$ __overfitting__ 
    * __Effects of Variance:__{: style="color: blue"}  
        * __High Variance__: complex models $$\rightarrow$$ __overfitting__  
        * __Low Variance__: simple models $$\rightarrow$$ __underfitting__  
    * __Draw the Graph of the Tradeoff (wrt model capacity):__{: style="color: blue"}  
        ![img](/main_files/dl_book/1.png){: width="60%"}  
* __Derive the Bias-Variance Decomposition with explanations:__{: style="color: red"}  
    <p>$${\displaystyle {\begin{aligned}\mathbb{E}_{\mathcal{D}} {\big [}I[g^{(\mathcal{D})}]{\big ]}&=\mathbb{E}_{\mathcal{D}} {\big [}\mathbb{E}_{x}{\big [}(g^{(\mathcal{D})}-y)^{2}{\big ]}{\big ]}\\
    &=\mathbb{E}_{x} {\big [}\mathbb{E}_{\mathcal{D}}{\big [}(g^{(\mathcal{D})}-y)^{2}{\big ]}{\big ]}\\
    &=\mathbb{E}_{x}{\big [}\mathbb{E}_{\mathcal{D}}{\big [}(g^{(\mathcal{D})}- f -\varepsilon)^{2}{\big ]}{\big ]}
    \\&=\mathbb{E}_{x}{\big [}\mathbb{E}_{\mathcal{D}} {\big [}(f+\varepsilon -g^{(\mathcal{D})}+\bar{g}-\bar{g})^{2}{\big ]}{\big ]}\\
    &=\mathbb{E}_{x}{\big [}\mathbb{E}_{\mathcal{D}} {\big [}(\bar{g}-f)^{2}{\big ]}+\mathbb{E}_{\mathcal{D}} [\varepsilon ^{2}]+\mathbb{E}_{\mathcal{D}} {\big [}(g^{(\mathcal{D})} - \bar{g})^{2}{\big ]}+2\: \mathbb{E}_{\mathcal{D}} {\big [}(\bar{g}-f)\varepsilon {\big ]}+2\: \mathbb{E}_{\mathcal{D}} {\big [}\varepsilon (g^{(\mathcal{D})}-\bar{g}){\big ]}+2\: \mathbb{E}_{\mathcal{D}} {\big [}(g^{(\mathcal{D})} - \bar{g})(\bar{g}-f){\big ]}{\big ]}\\
    &=\mathbb{E}_{x}{\big [}(\bar{g}-f)^{2}+\mathbb{E}_{\mathcal{D}} [\varepsilon ^{2}]+\mathbb{E}_{\mathcal{D}} {\big [}(g^{(\mathcal{D})} - \bar{g})^{2}{\big ]}+2(\bar{g}-f)\mathbb{E}_{\mathcal{D}} [\varepsilon ]\: +2\: \mathbb{E}_{\mathcal{D}} [\varepsilon ]\: \mathbb{E}_{\mathcal{D}} {\big [}g^{(\mathcal{D})}-\bar{g}{\big ]}+2\: \mathbb{E}_{\mathcal{D}} {\big [}g^{(\mathcal{D})}-\bar{g}{\big ]}(\bar{g}-f){\big ]}\\
    &=\mathbb{E}_{x}{\big [}(\bar{g}-f)^{2}+\mathbb{E}_{\mathcal{D}} [\varepsilon ^{2}]+\mathbb{E}_{\mathcal{D}} {\big [}(g^{(\mathcal{D})} - \bar{g})^{2}{\big ]}{\big ]}\\
    &=\mathbb{E}_{x}{\big [}(\bar{g}-f)^{2}+\operatorname {Var} [y]+\operatorname {Var} {\big [}g^{(\mathcal{D})}{\big ]}{\big ]}\\
    &=\mathbb{E}_{x}{\big [}\operatorname {Bias} [g^{(\mathcal{D})}]^{2}+\operatorname {Var} [y]+\operatorname {Var} {\big [}g^{(\mathcal{D})}{\big ]}{\big ]}\\
    &=\mathbb{E}_{x}{\big [}\operatorname {Bias} [g^{(\mathcal{D})}]^{2}+\sigma ^{2}+\operatorname {Var} {\big [}g^{(\mathcal{D})}{\big ]}{\big ]}\\\end{aligned}}}$$</p>  
    where:  
    $$\overline{g}(\mathbf{x})=\mathbb{E}_{\mathcal{D}}\left[g^{(\mathcal{D})}(\mathbf{x})\right]$$ is the __average hypothesis__ over all realization of $$N$$ data-points $$\mathcal{D}_ i$$.  
* __What are the key Takeaways from the Tradeoff?__{: style="color: red"}  
    Match the "Model Capacity/Complexity" to the "Data Resources", NOT to the Target Complexity.  
* __What are the most common ways to negotiate the Tradeoff?__{: style="color: red"}  
    * Cross-Validation
    * MSE of the Estimates
* __How does the decomposition relate to Classification?__{: style="color: red"}  
    A similar decomposition exists for:  
    * Classification with $$0-1$$ loss
    * Probabilistic Classification with MSE
* __Increasing/Decreasing Bias&Variance__:  
    * __Adding Good Feature__: 
        * Decrease Bias
    * __Adding Bad Feature__: 
        * No effect
    * __Adding ANY Feature__: 
        * Increase Variance 
    * __Adding more Data__: 
        * Decrease Variance
        * May decrease Bias (if $$f^{\ast} \in \mathcal{H}$$)
    * __Noise in Test Set__: 
        * Affects Only Irreducible Err
    * __Noise in Training Set__: 
        * Affects Bias and Variance
    * __Dimensionality Reduction__: 
        * Decrease Variance
    * __Feature Selection__: 
        * Decrease Variance
    * __Regularization__: 
    


        * Increase Bias
        * Decrease Variance
    * __Increasing # of Hidden Units in ANNs__: 
    


        * Decrease Bias
        * Increase Variance
    * __Increasing # of Hidden Layers in ANNs__: 
        * Decrease Bias
        * Increase Variance
    * __Increasing $$k$$ in K-NN__: 
        * Increase Bias
        * Decrease Variance
    * __Increasing Depth in Decision-Trees__: 
        * Increase Variance
    * __Boosting__: 
        * Decrease Bias
    * __Bagging__: 
        * Decrease Variance

        




***

# Activation Functions
* __Describe the Desirable Properties for activation functions:__{: style="color: red"}  
    * __Non-Linearity__:  
        When the activation function is non-linear, then a two-layer neural network can be proven to be a universal function approximator. The identity activation function does not satisfy this property. When multiple layers use the identity activation function, the entire network is equivalent to a single-layer model. 
    * __Range__:  
        When the range of the activation function is finite, gradient-based training methods tend to be more stable, because pattern presentations significantly affect only limited weights. When the range is infinite, training is generally more efficient because pattern presentations significantly affect most of the weights. In the latter case, smaller learning rates are typically necessary.  
    * __Continuously Differentiable__:  
        This property is desirable for enabling gradient-based optimization methods. The binary step activation function is not differentiable at 0, and it differentiates to 0 for all other values, so gradient-based methods can make no progress with it.
    * __Monotonicity__:  
        When the activation function is monotonic, the error surface associated with a single-layer model is guaranteed to be convex.  
    * __Smoothness with Monotonic Derivatives__:  
        These have been shown to generalize better in some cases.  
    * __Approximating Identity near Origin__:  
        Equivalent to $${\displaystyle f(0)=0}$$ and $${\displaystyle f'(0)=1}$$, and $${\displaystyle f'}$$ is continuous at $$0$$.  
        When activation functions have this property, the neural network will learn efficiently when its weights are initialized with small random values. When the activation function does not approximate identity near the origin, special care must be used when initializing the weights.  
    * __Zero-Centered Range__:  
        Has effects of centering the data (zero mean) by centering the activations. Makes learning easier.   

    <button>Explain the specifics of the desirability of each of the following</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __Non-Linearity:__{: style="color: blue"}  
    * __Range:__{: style="color: blue"}  
    * __Continuously Differentiable:__{: style="color: blue"}  
    * __Monotonicity:__{: style="color: blue"}  
    * __Smoothness with Monotonic Derivatives:__{: style="color: blue"}  
    * __Approximating Identity near Origin:__{: style="color: blue"}  
    * __Zero-Centered Range:__{: style="color: blue"}  
    {: hidden=""}

* __Describe the NON-Desirable Properties for activation functions:__{: style="color: red"}  
    * __Saturation__:  
        An activation functions output, with finite range, may saturate near its tail or head (e.g. $$\{0, 1\}$$ for sigmoid). This leads to a problem called __vanishing gradient__.  
    * __Vanishing Gradients__:  
        Happens when the gradient of an activation function is very small/zero. This usually happens when the activation function __saturates__ at either of its tails.  
        The chain-rule will *__multiply__* the local gradient (of activation function) with the whole objective. Thus, when gradient is small/zero, it will "kill" the gradient $$\rightarrow$$ no signal will flow through the neuron to its weights or to its data.  
        __Slows/Stops learning completely__.  
    * __Range Not Zero-Centered__:  
        This is undesirable since neurons in later layers of processing in a Neural Network would be receiving data that is not zero-centered. This has implications on the dynamics during gradient descent, because if the data coming into a neuron is always positive (e.g. $$x>0$$ elementwise in $$f=w^Tx+b$$), then the gradient on the weights $$w$$ will during backpropagation become either all be positive, or all negative (depending on the gradient of the whole expression $$f$$). This could introduce undesirable zig-zagging dynamics in the gradient updates for the weights. However, notice that once these gradients are added up across a batch of data the final update for the weights can have variable signs, somewhat mitigating this issue. Therefore, this is an inconvenience but it has less severe consequences compared to the saturated activation problem above.  
        __Makes optimization harder.__   

    <button>Explain the specifics of the non-desirability of each of the following</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __Saturation:__{: style="color: blue"}  
    * __Vanishing Gradients:__{: style="color: blue"}  
    * __Range Not Zero-Centered:__{: style="color: blue"}  
    {: hidden=""}
* __List the different activation functions used in ML?__{: style="color: red"}  
    Identity, Sigmoid, Tanh, ReLU, L-ReLU, ELU, SoftPlus  
    ![img](/main_files/concepts/16.png){: max-width="150%"}  
    __Names, Definitions, Properties (pros&cons), Derivatives, Applications, pros/cons:__{: style="color: blue"}  
    * __Sigmoid__:  
        <p>$$S(z)=\frac{1}{1+e^{-z}} \\ S^{\prime}(z)=S(z) \cdot(1-S(z))$$</p>  
        ![img](/main_files/concepts/3.png){: width="68%" .center-image}  
        * __Properties__:  
            Never use as activation, use as an output unit for binary classification.  
            * __Pros__:  
                * Has a nice interpretation as the firing rate of a neuron  
            * __Cons__:  
                * They Saturate and kill gradients $$\rightarrow$$ Gives rise to __vanishing gradients__[^1] $$\rightarrow$$ Stop Learning  
                    * Happens when initialization weights are too large  
                    * or sloppy with data preprocessing  
                    * Neurons Activation saturates at either tail of $$0$$ or $$1$$  
                * Output NOT __Zero-Centered__ $$\rightarrow$$ Gradient updates go too far in different directions $$\rightarrow$$ makes optimization harder   
                * The local gradient $$(z * (1-z))$$ achieves maximum at $$0.25$$, when $$z = 0.5$$. $$\rightarrow$$ very time the gradient signal flows through a sigmoid gate, its magnitude always diminishes by one quarter (or more) $$\rightarrow$$ with basic SGD, the lower layers of a network train much slower than the higher one  
    * __Tanh__:  
        <p>$$\tanh (z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}} \\ \tanh ^{\prime}(z)=1-\tanh (z)^{2}$$</p>  
        ![img](/main_files/concepts/4.png){: width="68%" .center-image}  
        __Properties:__  
        Strictly superior to Sigmoid (scaled version of sigmoid \| stronger gradient). Good for activation.  
        * __Pros__:  
            * Zero Mean/Centered  
        * __Cons__:  
            * They Saturate and kill gradients $$\rightarrow$$ Gives rise to __vanishing gradients__[^1] $$\rightarrow$$ Stop Learning  
    * __Relu__:  
        <p>$$R(z)=\left\{\begin{array}{cc}{z} & {z>0} \\ {0} & {z<=0}\end{array}\right\} \\  R^{\prime}(z)=\left\{\begin{array}{ll}{1} & {z>0} \\ {0} & {z<0}\end{array}\right\}$$</p>  
        ![img](/main_files/concepts/5.png){: width="68%" .center-image}  
        __Properties:__  
        The best for activation (Better gradients).  
        * __Pros__:  
            * Non-saturation of gradients which _accelerates convergence_ of SGD  
            * Sparsity effects and induced regularization. [discussion](https://stats.stackexchange.com/questions/176794/how-does-rectilinear-activation-function-solve-the-vanishing-gradient-problem-in/176905#176905)  
            * Not computationally expensive  
        * __Cons__:  
            * __ReLU not zero-centered problem__:  
                The problem that ReLU is not zero-centered can be solved/mitigated by using __batch normalization__, which normalizes the signal before activation:  
                > From paper: We add the BN transform immediately before the nonlinearity, by normalizing $$x =  Wu + b$$; normalizing it is likely to produce activations with a stable distribution.  
                > * [WHY NORMALIZING THE SIGNAL IS IMPORTANT](https://www.youtube.com/watch?v=FDCfw-YqWTE&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=10&t=0s)
            * __Dying ReLUs (Dead Neurons):__  
                If a neuron gets clamped to zero in the forward pass (it doesn’t "fire" / $$x<0$$), then its weights will get zero gradient. Thus, if a ReLU neuron is unfortunately initialized such that it never fires, or if a neuron’s weights ever get knocked off with a large update during training into this regime (usually as a symptom of aggressive learning rates), then this neuron will remain permanently dead.  
                * [**cs231n Explanation**](https://www.youtube.com/embed/gYpoJMlgyXA?start=1249){: value="show" onclick="iframePopA(event)"}
                <a href="https://www.youtube.com/embed/gYpoJMlgyXA?start=1249"></a>
                    <div markdown="1"> </div>    
            * __Infinite Range__:  
                Can blow up the activation.  
    * __Leaky Relu__:  
        <p>$$R(z)=\left\{\begin{array}{cc}{z} & {z>0} \\ {\alpha z} & {z<=0}\end{array}\right\} \\ 
        R^{\prime}(z)=\left\{\begin{array}{ll}{1} & {z>0} \\ {\alpha} & {z<0}\end{array}\right\}$$</p>  
        ![img](/main_files/concepts/6.png){: width="68%" .center-image}  
        __Properties:__  
        Sometimes useful. Worth trying.  
        * __Pros__:  
            * Leaky ReLUs are one attempt to fix the “dying ReLU” problem by having a small negative slope (of 0.01, or so).  
        * __Cons__:  
            The consistency of the benefit across tasks is presently unclear.  

<button>Show Questions</button>{: .showText value="show" onclick="showText_withParent_PopHide(event);"}
* __Fill in the following table:__{: style="color: red"}  
    ![img](/main_files/dl/concepts/act_funcs/0.png){: width="100%"}  
* __Tanh VS Sigmoid for activation?__{: style="color: red"}  
* __ReLU:__{: style="color: red"}  
    * __What makes it superior/advantageous?__{: style="color: red"}  
    * __What problems does it have?__{: style="color: red"}  
        * __What solution do we have to mitigate the problem?__{: style="color: red"}  
* __Compute the derivatives of all activation functions:__{: style="color: red"}  
* __Graph all activation functions and their derivatives:__{: style="color: red"}  
{: hidden=""}





***

# Kernels
* __Define "Local Kernel" and give an analogy to describe it:__{: style="color: red"}  
* __Write the following kernels:__{: style="color: red"}  
    * __Polynomial Kernel of degree, up to, $$d$$:__{: style="color: blue"}  
    * __Gaussian Kernel:__{: style="color: blue"}  
    * __Sigmoid Kernel:__{: style="color: blue"}  
    * __Polynomial Kernel of degree, exactly, $$d$$:__{: style="color: blue"}  
    





***

# Math
* __What is a metric?__{: style="color: red"}  
    A __Metric (distance function)__ $$d$$  is a function that defines a distance between each pair of elements of a set $$X$$.  
    A Metric induces a _topology_ on a set; BUT, not all topologies can be generated by a metric.  
    Mathematically, it is a function:  
    $${\displaystyle d:X\times X\to [0,\infty )},$$  
    that must satisfy the following properties:  
    1.  $${\displaystyle d(x,y)\geq 0}$$ $$\:\:\:\:\:\:\:$$   non-negativity or separation axiom  
    2.  $${\displaystyle d(x,y)=0\Leftrightarrow x=y}$$ $$\:\:\:\:\:\:\:$$  identity of indiscernibles  
    3.  $${\displaystyle d(x,y)=d(y,x)}$$ $$\:\:\:\:\:\:\:$$  symmetry  
    4.  $${\displaystyle d(x,z)\leq d(x,y)+d(y,z)}$$ $$\:\:\:\:\:\:\:$$  subadditivity or triangle inequality  
    > The first condition is implied by the others.  
    [Metric](http://localhost:8889/concepts_#bodyContents31)  
* __Describe Binary Relations and their Properties?__{: style="color: red"}  
    A __binary relation__ on a set $$A$$ is a set of ordered pairs of elements of $$A$$. In other words, it is a subset of the Cartesian product $$A^2 = A ×A$$.  
    The number of binary relations on a set of $$N$$ elements is $$= 2^{N^2}$$
    __Examples:__  
    * "is greater than"  
    * "is equal to"  
    * A function $$f(x)$$  
    __Properties:__  (for a relation $$R$$ and set $$X$$)  
    * _Reflexive:_ for all $$x$$ in $$X$$ it holds that $$xRx$$  
    * _Symmetric:_ for all $$x$$ and $$y$$ in $$X$$ it holds that if $$xRy$$ then $$yRx$$  
    * _Transitive:_ for all $$x$$, $$y$$ and $$z$$ in $$X$$ it holds that if $$xRy$$ and $$yRz$$ then $$xRz$$  
    [answer](/concepts_#bodyContents32)  
* __Formulas:__{: style="color: red"}  
    * __Set theory:__{: style="color: blue"}  
        * __Number of subsets of a set of $$N$$ elements:__{: style="color: blue"}  
            $$= 2^N$$  
        * __Number of pairs $$(a,b)$$ of a set of N elements:__{: style="color: blue"}  
            $$= N^2$$  
    * __Binomial Theorem:__{: style="color: blue"}  
        <p>$$(x+y)^{n}=\sum_{k=0}^{n}{n \choose k}x^{n-k}y^{k}=\sum_{k=0}^{n}{n \choose k}x^{k}y^{n-k} \\={n \choose 0}x^{n}y^{0}+{n \choose 1}x^{n-1}y^{1}+{n \choose 2}x^{n-2}y^{2}+\cdots +{n \choose n-1}x^{1}y^{n-1}+{n \choose n}x^{0}y^{n},$$</p>  
    * __Binomial Coefficient:__{: style="color: blue"}  
        <p>$${\binom {n}{k}}={\frac {n!}{k!(n-k)!}} = N \text{choose} k = N \text{choose} (n-k)$$</p>  
    * __Expansion of $$x^n - y^n = $$__{: style="color: blue"}  
        <p>$$x^n - y^n = (x-y)(x^{n-1} + x^{n-2} y + ... + x y^{n-2} + y^{n-1})$$</p>  
    * __Number of ways to partition $$N$$ data points into $$k$$ clusters:__{: style="color: blue"}  
        * __Number of pairs (e.g. $$(a,b)$$) of a set of $$N$$ elements__ $$= N^2$$  
        * There are at most $$k^N$$ ways to partition $$N$$ data points into $$k$$ clusters - there are $$N$$ choose $$k$$ clusters, precisely  
    * __$$\log_x(y) =$$__{: style="color: blue"}  
        <p>$$\log_x(y) = \dfrac{\ln(y)}{\ln(x)}$$</p>  
    * __The length of a vector $$\mathbf{x}$$  along a direction (projection):__{: style="color: blue"}  
        1. Along a unit-length vector $$\hat{\mathbf{w}}$$:  
            $$\text{comp}_ {\hat{\mathbf{w}}}(\mathbf{x}) = \hat{\mathbf{w}}^T\mathbf{x}$$  
        2. Along an unnormalized vector $$\mathbf{w}$$:  
            $$\text{comp}_ {\mathbf{w}}(\mathbf{x}) = \dfrac{1}{\|\mathbf{w}\|} \mathbf{w}^T\mathbf{x}$$  
    * __$$\sum_{i=1}^{n} 2^{i}=$$__{: style="color: blue"}  
        $$\sum_{i=1}^{n} 2^{i}=2^{n+1}-2$$  

* __List 6 proof methods:__{: style="color: red"}  
    * Direct Proof
    * Mathematical Induction
        * Strong Induction
        * Infinite Descent
    * Contradiction
    * Contraposition ($$(p \implies q) \iff (!q \implies !p)$$)  
    * Construction
    * Combinatorial
    * Exhaustion
    * Non-Constructive proof (existence proofs)
[answer](/concepts_#bodyContents34)

* __Important Formulas__{: style="color: red"}  
    * __Projection $$\tilde{\mathbf{x}}$$ of a vector $$\mathbf{x}$$ onto another vector $$\mathbf{u}$$:__{: style="color: blue"}  
        <p>$$\tilde{\mathbf{x}} = \mathbf{x}\cdot \dfrac{\mathbf{u}}{\|\mathbf{u}\|_2} \mathbf{u}$$</p>  


***

# Statistics
* __ROC curve:__{: style="color: red"}  
    * __Definition:__{: style="color: blue"}  
        A __receiver operating characteristic curve__, or __ROC curve__, is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.  
    * __Purpose:__{: style="color: blue"}  
        A way to quantify how good a **binary classifier** separates two classes.  
    * __How do you create the plot?__{: style="color: blue"}  
        The ROC curve is created by plotting the __true positive rate (TPR)__ against the __false positive rate (FPR)__ at various threshold settings.  
    * __How to identify a good classifier:__{: style="color: blue"}  
        A Good classifier has a ROC curve that is near the top-left diagonal (hugging it).  
    * __How to identify a bad classifier:__{: style="color: blue"}  
        A Bad Classifier has a ROC curve that is close to the diagonal line.  
    * __What is its application in tuning the model?__{: style="color: blue"}  
        It allows you to set the **classification threshold**:  
        * You can minimize False-positive rate or maximize the True-Positive Rate  
* __AUC - AUROC:__{: style="color: red"}  
    * __Definition:__{: style="color: blue"}  
        When using normalized units, the area under the curve (often referred to as simply the AUC) is equal to the probability that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one (assuming 'positive' ranks higher than 'negative').  
    * __Range:__{: style="color: blue"}  
        Range $$ = 0.5 - 1.0$$, from poor to perfect, with an uninformative classifier yielding $$0.5$$    
    * __What does it measure:__{: style="color: blue"}  
        It is a measure of aggregated classification performance.  
    * __Usage in ML:__{: style="color: blue"}  
        For model comparison.  
* __Define Statistical Efficiency (of an estimator)?__{: style="color: red"}  
    Essentially, a more efficient estimator, experiment, or test needs fewer observations than a less efficient one to achieve a given performance.  
    Efficiencies are often defined using the _variance_ or _mean square error_ as the measure of desirability.  
    An efficient estimator is also the minimum variance unbiased estimator (MVUE).  

    * An Efficient Estimator has lower variance than an inefficient one  
    * The use of an inefficient estimator gives results equivalent to those obtainable from a subset of data; and is therefor, wasteful of data  
* __Whats the difference between *Errors* and *Residuals*:__{: style="color: red"}  
    The __Error__ of an observed value is the deviation of the observed value from the (unobservable) **_true_** value of a quantity of interest.  

    The __Residual__ of an observed value is the difference between the observed value and the *__estimated__* value of the quantity of interest.  
  
    * __Compute the statistical errors and residuals of the univariate, normal distribution defined as $$X_{1}, \ldots, X_{n} \sim N\left(\mu, \sigma^{2}\right)$$:__{: style="color: blue"}  
        * __Statistical Errors__:  
            <p>$$e_{i}=X_{i}-\mu$$</p>  
        * __Residuals__:  
            <p>$$r_{i}=X_{i}-\overline {X}$$</p>  
        * [**Example in Univariate Distributions**](https://en.wikipedia.org/wiki/Errors_and_residuals#In_univariate_distributions){: value="show" onclick="iframePopA(event)"}
        <a href="https://en.wikipedia.org/wiki/Errors_and_residuals#In_univariate_distributions"></a>
            <div markdown="1"> </div>    
* __Clearly Define MLE and derive the final formula:__{: style="color: red"}  
    __MLE__ is a method/principle from which we can derive specific functions that are *__good estimators__* for different models.  
    __Likelihood in Parametric Models:__{: style="color: silver"}  
    Suppose we have a parametric model $$\{p(y ; \theta) | \theta \in \Theta\}$$ and a sample $$D=\left\{y_{1}, \ldots, y_{n}\right\}$$:  
    * The likelihood of parameter estimate $$\hat{\theta} \in \Theta$$ for sample $$\mathcal{D}$$ is:  
        <p>$$\begin{aligned}  {\displaystyle {\mathcal {L}}(\theta \,;\mathcal{D} )} &= p(\mathcal{D} ; \theta) \\&=\prod_{i=1}^{n} p\left(y_{i} ; \theta\right)\end{aligned}$$</p>  
    * In practice, we prefer to work with the __log-likelihood__.  Same maximum but  
        <p>$$\log {\displaystyle {\mathcal {L}}(\theta \,;\mathcal{D} )}=\sum_{i=1}^{n} \log p\left(y_{i} ; \theta\right)$$</p>  
        and sums are easier to work with than products.  

    __MLE for Parametric Models:__{: style="color: silver"}  
    The __maximum likelihood estimator (MLE)__ for $$\theta$$ in the (parametric) model $$\{p(y, \theta) | \theta \in \Theta\}$$ is:  
    <p>$$\begin{aligned} \hat{\theta} &=\underset{\theta \in \Theta}{\arg \max } \log p(\mathcal{D}, \hat{\theta}) \\ &=\underset{\theta \in \Theta}{\arg \max } \sum_{i=1}^{n} \log p\left(y_{i} ; \theta\right) \end{aligned}$$</p>  
    * __Write MLE as an expectation wrt the Empirical Distribution:__{: style="color: blue"}  
        Because the $$\text { arg max }$$ does not change when we rescale the cost function, we can divide by $$m$$ to obtain a version of the criterion that is expressed as an __expectation with respect to the empirical distribution $$\hat{p}_ {\text { data }}$$__  defined by the training data:  
        <p>$$\boldsymbol{\theta}_{\mathrm{ML}}=\underset{\boldsymbol{\theta}}{\arg \max } \mathbb{E}_{\mathbf{x} \sim \hat{p} \text { data }} \log p_{\text { model }}(\boldsymbol{x} ; \boldsymbol{\theta}) \tag{5.59}$$</p>  
    * __Describe formally the relationship between MLE and the KL-divergence:__{: style="color: blue"}  
        __MLE as Minimizing KL-Divergence between the Empirical dist. and the model dist.:__{: style="color: silver"}  
        We can interpret maximum likelihood estimation as _minimizing the dissimilarity_ between the __empirical distribution $$\hat{p}_ {\text { data }}$$__, defined by the training set, and the __model distribution__, with the degree of dissimilarity between the two measured by the __KL divergence__.  
        * The __KL-divergence__ is given by:  
            <p>$$D_{\mathrm{KL}}\left(\hat{p}_{\text { data }} \| p_{\text { model }}\right)=\mathbb{E}_{\mathbf{x} \sim \hat{p}_{\text { data }}}\left[\log \hat{p}_{\text { data }}(\boldsymbol{x})-\log p_{\text { model }}(\boldsymbol{x})\right] \tag{5.60}$$</p>  
        The term on the left is a function only of the data-generating process, not the model. This means when we train the model to minimize the KL divergence, we need only minimize:  
        <p>$$-\mathbb{E}_{\mathbf{x} \sim \hat{p}_{\text { data }}}\left[\log p_{\text { model }}(\boldsymbol{x})\right] \tag{5.61}$$</p>  
        which is of course the same as the _maximization_ in equation $$5.59$$.  
    * __Extend the argument to show the link between MLE and Cross-Entropy. Give an example:__{: style="color: blue"}  
        Minimizing this KL-divergence corresponds exactly to __minimizing the cross-entropy between the distributions__.  
        Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution defined by the training set and the probability distribution defined by model.  
        E.g. __MSE__ is the _cross-entropy_ between the __empirical distribution__ and a __Gaussian model__.  

        Maximum likelihood thus becomes minimization of the negative log-likelihood(NLL), or equivalently, minimization of the cross-entropy.  
    * __How does MLE relate to the model distribution and the empirical distribution?__{: style="color: blue"}  
        We can thus see maximum likelihood as an attempt to _make the model distribution match the empirical distribution $$\hat{p} _ {\text { data }}$$_.  
    * __What is the intuition behind using MLE?__{: style="color: blue"}  
        If I choose a _hypothesis_ $$h$$ underwhich the _observed data_ is very *__plausible__* then the _hypothesis_ is very *__likely__*.  
    * __What does MLE find/result in?__{: style="color: blue"}  
        It finds the value of the parameter $$\theta$$ that, if used (in the model) to generate the probability of the data, would make the data most _"likely"_ to occur.  
    * __What kind of problem is MLE and how to solve for it?__{: style="color: blue"}  
        It is an optimization problem that is solved by calculus for problems with closed-form solutions or with numerical methods (e.g. SGD).    
    * __How does it relate to SLT:__{: style="color: blue"}   
        It corresponds to __Empirical Risk Minimization__.  
    * __Explain clearly why we maximize the natural log of the likelihood__{: style="color: blue"}  
        1. Numerical Stability: change products to sums  
        2. The logarithm of a member of the family of exponential probability distributions (which includes the ubiquitous normal) is polynomial in the parameters (i.e. max-likelihood reduces to least-squares for normal distributions)  
        $$\log\left(\exp\left(-\frac{1}{2}x^2\right)\right) = -\frac{1}{2}x^2$$   
        3. The latter form is both more numerically stable and symbolically easier to differentiate than the former. It increases the dynamic range of the optimization algorithm (allowing it to work with extremely large or small values in the same way).  
        4. The logarithm is a monotonic transformation that preserves the locations of the extrema (in particular, the estimated parameters in max-likelihood are identical for the original and the log-transformed formulation)  
        5. Gradient methods generally work better optimizing $$log_p(x)$$ than $$p(x)$$ because the gradient of $$log_p(x)$$ is generally more __well-scaled__. [link](https://stats.stackexchange.com/questions/174481/why-to-optimize-max-log-probability-instead-of-probability)  
            __Justification:__ the gradient of the original term will include a $$e^{\vec{x}}$$ multiplicative term that scales very quickly one way or another, requiring the step-size to equally scale/stretch in the opposite direction.  
* __What is a biased estimator?__{: style="color: red"}  
    We define the __Bias__ of an estimator as:  
    <p>$$ \operatorname{bias}\left(\hat{\boldsymbol{\theta}}_{m}\right)=\mathbb{E}\left(\hat{\boldsymbol{\theta}}_{m}\right)-\boldsymbol{\theta} $$</p>  
    A __Biased Estimator__ is an estimator $$\hat{\boldsymbol{\theta}}_ {m}$$ such that:  
    <p>$$ \operatorname{bias}\left(\hat{\boldsymbol{\theta}}_ {m}\right) \geq 0$$</p>  
    * __Why would we prefer biased estimators in some cases?__{: style="color: blue"}  
        Mainly, due to the *__Bias-Variance Decomposition__*. The __MSE__ takes into account both the _bias_ and the _variance_ and sometimes the biased estimator might have a lower variance than the unbiased one, which results in a total _decrease_ in the MSE.  
* __What is the difference between "Probability" and "Likelihood":__{: style="color: red"}  
    __Probabilities__ are the areas under a fixed distribution  
    $$pr($$data$$|$$distribution$$)$$  
    i.e. probability of some _data_ (left hand side) given a distribution (described by the right hand side)  
    __Likelihoods__ are the y-axis values for fixed data points with distributions that can be moved..  
    $$L($$distribution$$|$$observation/data$$)$$  
    It is the likelihood of the parameter $$\theta$$ for the data $$\mathcal{D}$$.  
    > Likelihood is, basically, a specific probability that can only be calculated after the fact (of observing some outcomes). It is not normalized to $$1$$ (it is __not__ a probability). It is just a way to quantify how likely a set of observation is to occur given some distribution with some parameters; then you can manipulate the parameters to make the realization of the data more _"likely"_ (it is precisely meant for that purpose of estimating the parameters); it is a _function_ of the __parameters__.  
    Probability, on the other hand, is absolute for all possible outcomes. It is a function of the __Data__.  
* __Estimators:__{: style="color: red"}  
    * __Define:__{: style="color: blue"}  
        A __Point Estimator__ or __statistic__ is any function of the data.  
    * __Formula:__{: style="color: blue"}  
        <p>$$\hat{\boldsymbol{\theta}}_{m}=g\left(\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(m)}\right)$$</p>  
    * __Whats a good estimator?__{: style="color: blue"}  
        A good estimator is a function whose output is close to the true underlying $$ \theta $$ that generated the training data.  
    * __What are the Assumptions made regarding the estimated parameter:__{: style="color: blue"}  
        We assume that the true $$\boldsymbol{\theta}$$ is fixed, and that $$\hat{\boldsymbol{\theta}}$$ is a function of the data, which is drawn from a random process, making $$\hat{\boldsymbol{\theta}}$$ a __random variable__.  
* __What is Function Estimation:__{: style="color: red"}  
    __Function Estimation/Approximation__ refers to estimation of the relationship between _input_ and _target data_.  
    I.E. We are trying to predict a variable $$y$$ given an input vector $$x$$, and we assume that there is a function $$f(x)$$ that describes the approximate relationship between $$y$$ and $$x$$.  
    If we assume that: $$y = f(x) + \epsilon$$, where $$\epsilon$$ is the part of $$y$$ that is not predictable from $$x$$; then we are interested in approximating $$f$$ with a model or estimate $$ \hat{f} $$.  
    * __Whats the relation between the Function Estimator $$\hat{f}$$ and Point Estimator:__{: style="color: blue"}  
        Function estimation is really just the same as estimating a parameter $$\boldsymbol{\theta}$$; the function estimator $$ \hat{f} $$ is simply a point estimator in function space.  
* __Define "marginal likelihood" (wrt naive bayes):__{: style="color: red"}  
    Marginal likelihood is, the probability that the word ‘FREE’ is used in any message (not given any other condition?).   

 

***

# Text-Classification \| Classical
* __List some Classification Methods:__{: style="color: red"}  
* __(Hand-Coded)Rules-Based Algorithms__: use rules based on combinations of words or other features.   
    * Can have high accuracy if the rules are carefully refined and maintained by experts.  
    * However, building and maintaining these rules is very hard.  
* __Supervised Machine Learning__: using an ML algorithm that trains on a training set of (document, class) elements to train a classifier.  
    * _Types of Classifiers_:  
        * Naive Bayes  
        * Logistic Regression
        * SVMs
        * K-NNs  
* __List some Applications of Txt Classification:__{: style="color: red"}  
    * __Spam Filtering__: discerning spam emails form legitimate emails.  
    * __Email Routing__: sending an email sento to a genral address to a specfic affress based on the topic.  
    * __Language Identification__: automatiacally determining the genre of a piece of text.  
    * Readibility Assessment__: determining the degree of readability of a piece of text.  
    * __Sentiment Analysis__: determining the general emotion/feeling/attitude of the author of a piece of text.  
    * __Authorship Attribution__: determining which author wrote which piece of text.  
    * __Age/Gender Identification__: determining the age and/or gender of the author of a piece of text.      



***

# NLP
* __List some problems in NLP:__{: style="color: red"}  
    * Question Answering (QA) 
    * Information Extraction (IE)    
    * Sentiment Analysis  
    * Machine Translation (MT)  
    * Spam Detection  
    * Parts-of-Speech (POS) Tagging  
    * Named Entity Recognition (NER)
    * Conference Resolution  
    * Word Sense Disambugation (WSD)  
    * Parsing  
    * Paraphrasing  
    * Summarization  
    * Dialog  
* __List the Solved Problems in NLP:__{: style="color: red"}  
    * Spam Detection  
    * Parts-of-Speech (POS) Tagging  
    * Named Entity Recognition (NER) 
* __List the "within reach" problems in NLP:__{: style="color: red"}  
    * Sentiment Analysis  
    * Conference Resolution    
    * Word Sense Disambugation (WSD)  
    * Parsing  
    * Machine Translation (MT)  
    * Information Extraction (IE)    
* __List the Open Problems in NLP:__{: style="color: red"}  
    * Question Answering (QA)   
    * Paraphrasing  
    * Summarization  
    * Dialog  
* __Why is NLP hard? List Issues:__{: style="color: red"}  
    * __Non-Standard English__: "Great Job @ahmed_badary! I luv u 2!! were SOO PROUD of dis."  
    * __Segmentation Issues__: "New York-New Haven" vs "New-York New-Haven"  
    * __Idioms__: "dark horse", "getting cold feet", "losing face"  
    * __Neologisms__: "unfriend", "retweet", "google", "bromance"  
    * __World Knowledge__: "Ahmed and Zach are brothers", "Ahmed and Zach are fathers"    
    * __Tricky Entity Names__: "Where is _Life of Pie_ playing tonight?", "_Let it be_ was a hit song!"  
* __Define:__{: style="color: red"}  
    * __Morphology:__{: style="color: blue"}  
        The study of words, how they are formed, and their relationship to other words in the same language.  
    * __Morphemes:__{: style="color: blue"}  
        the small meaningful units that make up words.  
    * __Stems:__{: style="color: blue"}  
        the core meaning-bearing units of words.  
    * __Affixes:__{: style="color: blue"}  
        the bits and pieces that adhere to stems (often with grammatical functions).  
    * __Stemming:__{: style="color: blue"}  
        is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form.  
        The stem need __not__ map to a valid root in the language.  
    * __Lemmatization:__{: style="color: blue"}  
        reducing inflections or variant forms to base form.  


***

# Language Modeling
* __What is a Language Model?__{: style="color: red"}  
* __List some Applications of LMs:__{: style="color: red"}  
* __Traditional LMs:__{: style="color: red"}  
    * __How are they setup?__{: style="color: blue"}  
    * __What do they depend on?__{: style="color: blue"}  
    * __What is the Goal of the LM task? (in the ctxt of the problem setup)__{: style="color: blue"}  
    * __What assumptions are made by the problem setup? Why?__{: style="color: blue"}  
    * __What are the MLE Estimates for probabilities of the following:__{: style="color: blue"}  
        * __Bi-Grams:__{: style="color: blue"}  
            <p>$$p(w_2\vert w_1) = $$</p>  
        * __Tri-Grams:__{: style="color: blue"}  
            <p>$$p(w_3\vert w_1, w_2) = $$</p>  
    * __What are the issues w/ Traditional Approaches?__{: style="color: red"}  
* __What+How can we setup some NLP tasks as LM tasks:__{: style="color: red"}  
* __How does the LM task relate to Reasoning/AGI:__{: style="color: red"}  
* __Evaluating LM models:__{: style="color: red"}  
    * __List the Loss Functions (+formula) used to evaluate LM models? Motivate each:__{: style="color: blue"}  
    * __Which application of LM modeling does each loss work best for?__{: style="color: blue"}  
    <button>Show Questions</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    * __Why Cross-Entropy:__{: style="color: blue"}  
    * __Which setting it used for?__{: style="color: blue"}  
    * __Why Perplexity:__{: style="color: blue"}  
    * __Which setting used for?__{: style="color: blue"}  
    * __If no surprise, what is the perplexity?__{: style="color: blue"}  
    * __How does having a good LM relate to Information Theory?__{: style="color: blue"}  
    {: hidden=""}
* __LM DATA:__{: style="color: red"}  
    * __How does the fact that LM is a time-series prediction problem affect the way we need to train/test:__{: style="color: blue"}  
    * __How should we choose a subset of articles for testing:__{: style="color: blue"}  
* __List three approaches to Parametrizing LMs:__{: style="color: red"}  
    <button>Show Questions</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    * __Describe "Count-Based N-gram Models":__{: style="color: blue"}  
    * __What distributions do they capture?:__{: style="color: blue"}  
    * __Describe "Neural N-gram Models":__{: style="color: blue"}  
    * __What do they replace the captured distribution with?__{: style="color: blue"}  
    * __What are they better at capturing:__{: style="color: blue"}  
    * __Describe "RNNs":__{: style="color: blue"}  
    * __What do they replace/capture?__{: style="color: blue"}  
    * __How do they capture it?__{: style="color: blue"}  
    * __What are they best at capturing:__{: style="color: blue"}  
    {: hidden=""}
* __What's the main issue in LM modeling?__{: style="color: red"}  
    <button>Show Questions</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    * __How do N-gram models capture/approximate the history?:__{: style="color: blue"}  
    * __How do RNNs models capture/approximate the history?:__{: style="color: blue"}  
    {: hidden=""}
    * __The Bias-Variance Tradeoff of the following:__{: style="color: blue"}  
        * __N-Gram Models:__{: style="color: blue"}  
        * __RNNs:__{: style="color: blue"}  
        * __An Estimate s.t. it predicts the probability of a sentence by how many times it has seen it before:__{: style="color: blue"}  
            * __What happens in the limit of infinite data?__{: style="color: blue"}  
* __What are the advantages of sub-word level LMs:__{: style="color: red"}  
* __What are the disadvantages of sub-word level LMs:__{: style="color: red"}  
* __What is a "Conditional LM"?__{: style="color: red"}  
* __Write the decomposition of the probability for the Conditional LM:__{: style="color: red"}  
* __Describe the Computational Bottleneck for Language Models:__{: style="color: red"}  
* __Describe/List some solutions to the Bottleneck:__{: style="color: red"}  
* __Complexity Comparison of the different solutions:__{: style="color: red"}  
    <button>Show Questions</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](/main_files/qs/1.png){: width="100%" hidden=""}   




***

# Regularization
* __Define Regularization both intuitively and formally:__{: style="color: red"}  
    __Regularization__ can be, loosely, defined as: any modification we make to a learning algorithm that is intended to _reduce_ its _generalization error_ but not its _training error_.  

    Formally, it is a set of techniques that impose certain restrictions on the hypothesis space (by adding information) in order to solve an __ill-posed__ problem or to prevent __overfitting__.  
* __Define "well-posedness":__{: style="color: red"}  
    Hadamard defines __Well-Posed Problems__ as having the properties (1) A Solution Exists (2) It is Unique (3) It's behavior changes continuously with the initial conditions.  

* __Give four aspects of justification for regularization (theoretical):__{: style="color: red"}  
    <button>Show</button>{: .showText value="show"
    onclick="showTextPopHide(event);"}
    * __From a philosophical pov:__{: style="color: blue"}  
        It attempts to impose Occam’s razor on the solution.  
    * __From a probabilistic pov:__{: style="color: blue"}  
        From a Bayesian point of view, many regularization techniques correspond to imposing certain prior distributions on model parameters. This is equivalent to making a __MAP Estimate__ of the function we are trying to learn:  
        <p>$$\hat{\boldsymbol{\theta}}_{\mathrm{MAP}}=\underset{\boldsymbol{\theta}}{\arg \min }-\left(\sum_{i=1}^{n} \log \left[p\left(y_{i} | \mathbf{x}_ {i}, \boldsymbol{\theta}\right)\right]\right)-\log [p(\boldsymbol{\theta})]$$</p>   
    * __From an SLT pov:__{: style="color: blue"}  
        Regularization can be motivated as imposing a restriction on the complexity of a hypothesis space; called, the "capacity". In SLT, this is called the __Approximation-Generalization Tradeoff__. Regularization, effectively, works as a way to maneuver that tradeoff by giving preference to certain solutions. This can be described in two ways:  
        1. __VC-Theory__:  
            The theory results in a very simple, yet informative, inequality that captures the tradeoff, called __The Generalization Bound__:  
            <p>$$E_{\text {out}}(h) \leq E_{\text {in}}(h)+\Omega(\mathcal{H})$$</p>  
            But, if we formalize regularization as adding a regularizer (from the MAP estimate above): $$\Omega = \Omega(h)$$ a function of the hypothesis $$h$$, to the in-sample error $$E_{\text {in}}$$, then we get a new *__"Augmented"__*  __Error__:  
            <p>$$E_{\text {aug}}(h) = E_{\text {in}} + \lambda \Omega(h)$$</p> 
            Notice the correspondence between the form of the new _augmented error_ $$E_{\text {aug}}$$ and the _VC Inequality_:  
            <p>$$\begin{aligned}E_{\text {aug}}(h) &= E_{\text {in}}(h) + \lambda \Omega(h) \\
                &\downarrow \\
            E_{\text {out}}(h) &\leq E_{\text {in}}(h)+\Omega(\mathcal{H})\end{aligned}$$</p>  
            (we can relate the complexity of a single object to the complexity of a set of objects)  
            __Interpreting the correspondence:__  
            Consider the goal of ML: "to find an *__in-sample estimate__* of the *__out-sample error__*". From that perspective, the Augmented Error $$E_{\text {aug}}(h)$$ is a better _proxy_ to (estimate of) the out-sample error. Thus, minimizing $$E_{\text {aug}}(h)$$ corresponds, better, to minimizing $$E_{\text {out}}(h)$$. Where $$\Omega(h)$$ is an estimate of $$\Omega(\mathcal{H})$$; the regularization term a minimizer of the complexity term.  
        2. __Bias-Variance Decomposition:__  
            If the variance term $$\operatorname{Var}[h]$$ corresponds to the complexity term $$\Omega(\mathcal{H})$$, and the regularization term $$\Omega(h)$$, also, corresponds to $$\Omega(\mathcal{H})$$; then, the regularization term must also correspond to the variance term.  
            From that perspective, adding the regularization term to the augmented error accounts for the variance of the hypothesis and acts as a measure for it. Thus, minimizing the regularization term $$\Omega(h)$$ corresponds to minimizing the variance $$\operatorname{Var}[h]$$. However, when the variance of a model goes down the __bias__ tends to go up. Since this inverse-relationship is __not linear__, this analysis/view gives us a way to apply regularization effectively, by allowing us to measure the effect of regularization.  
            Basically, we would like our regularization term to _reduce the variance_ more than it _increases the bias_. We can control that tradeoff by the hyperparameter $$\lambda$$.  
    * __From a practical pov (relating to the real-world):__{: style="color: blue"}  
        Most applications of DL are to domains where the true data-generating process is almost certainly outside the model family (hypothesis space). Deep learning algorithms are typically applied to extremely complicated domains such as images, audio sequences and text, for which the true generation process essentially involves simulating the entire universe.

        Thus, controlling the complexity of the mdoel is not a simple matter of finding the model of the right size, with the right number of parameters; instead, the best fitting model (wrt. generalization error) is a large model that has been regularized appropriately.
    {: hidden=""}
* __Describe an overview of regularization in DL. How does it usually work?__{: style="color: red"}  
    In the context of DL, most regularization strategies are based on __regularizing estimators__, which usually works by _trading increased bias for reduced variance_.  
    * __Intuitively, how can a regularizer be effective?__{: style="color: blue"}  
        An effective regularizer is one that makes a profitable trade, reducing variance significantly while not overly increasing the bias.  
* __Describe the relationship between regularization and capacity:__{: style="color: red"}  
    Regularization is a (more general) way of controlling a models capacity by allowing us to express preference for one function over another in the same hypothesis space; instead of including or excluding members from the hypothesis space completely.  
* __Describe the different approaches to regularization:__{: style="color: red"}  
    * Parameter Norm Penalties: $$L^p$$ norms, early stopping
    * Data Augmentation: noise robustness/injection, dropout
    * Semi-supervised Learning
    * Multi-task Learning
    * Ensemble Learning: bagging, etc.
    * Adversarial Training
    * Infinite Priors: parameter tying and sharing
* __List 9 regularization techniques:__{: style="color: red"}  
    1. $$L^2$$ regularization  
    2. $$L^1$$ regularization  
    3. Dataset Augmentation  
    4. Noise Injection
    5. Semi-supervised Learning
    6. Multi-task Learning
    7. Early Stopping
    8. Parameter Tying, Sharing
    9. Sparse Representations
    10. Ensemble Methods, Bagging etc.
    11. Dropout
    12. Adversarial Training
    13. Tangent prop and Manifold Tangent Classifier

<button>Show the rest of the questions</button>{: .showText value="show"
onclick="showText_withParent_PopHide(event);"}
* __Describe Parameter Norm Penalties (PNPs):__{: style="color: red"}  
    Many regularization approaches are based on limiting the capacity of models by adding a parameter norm penalty $$\Omega(\boldsymbol{\theta})$$ to the objective function $$J$$. We denote the regularized objective function by $$\tilde{J}$$:  
    <p>$$\tilde{J}(\boldsymbol{\theta} ; \boldsymbol{X}, \boldsymbol{y})=J(\boldsymbol{\theta} ; \boldsymbol{X}, \boldsymbol{y})+\alpha \Omega(\boldsymbol{\theta}) \tag{7.1}$$</p>  
    where $$\alpha \in[0, \infty)$$ is a HP that weights the relative contribution of the norml penalty term, $$\Omega$$, relative to the standard objective function $$J$$.  
    <button>Show</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __Define the regularized objective:__{: style="color: blue"}  
        <p>$$\tilde{J}(\boldsymbol{\theta} ; \boldsymbol{X}, \boldsymbol{y})=J(\boldsymbol{\theta} ; \boldsymbol{X}, \boldsymbol{y})+\alpha \Omega(\boldsymbol{\theta}) \tag{7.1}$$</p>  
    * __Describe the parameter $$\alpha$$:__{: style="color: blue"}  
        $$\alpha \in[0, \infty)$$ is a HP that weights the relative contribution of the norm penalty term, $$\Omega$$, relative to the standard objective function $$J$$.  
    * __How does it influence the regularization:__{: style="color: blue"}  
        * __Effects of $$\alpha$$__:  
            * $$\alpha = 0$$ results in NO regularization
            * Larger values of $$\alpha$$ correspond to MORE regularization
    * __What is the effect of minimizing the regularized objective?__{: style="color: blue"}  
        The __effect of minimizing the regularized objective function__ is that it will *__decrease__*, both, _the original objective $$J$$_ on the training data and some _measure of the size of the parameters $$\boldsymbol{\theta}$$_.  
    {: hidden=""}
* __How do we deal with the Bias parameter in PNPs? Explain.__{: style="color: red"}  
    In NN, we usually penalize __only the weights__ of the affine transformation at each layer and we leave the __biases unregularized__.  
    Biases typically require less data than the weights to fit accurately. The reason is that _each weight specifies how TWO variables interact_ so fitting the weights well, requires observing both variables in a variety of conditions. However, _each bias controls only a single variable_, thus, we dont induce too much _variance_ by leaving the biases unregularized. If anything, regularizing the bias can introduce a significant amount of _underfitting_.  
* __Describe the tuning of the $$\alpha$$ HP in NNs for different hidden layers:__{: style="color: red"}  
    In the context of neural networks, it is sometimes desirable to use a separate penalty with a different $$\alpha$$ coefficient for each layer of the network. Because it can be expensive to search for the correct value of multiple hyperparameters, it is still reasonable to use the same weight decay at all layers just to reduce the size of search space.  
* __Formally describe the $$L^2$$ parameter regularization:__{: style="color: red"}  
    It is a regularization strategy that _drives the weights closer to the origin_[^2] by adding a regularization term:  
    <p>$$\Omega(\mathbf{\theta}) = \frac{1}{2}\|\boldsymbol{w}\|_ {2}^{2}$$</p>  
    to the objective function.  
    * __AKA:__{: style="color: blue"}  
        In statistics, $$L^2$$ regularization is also known as __Ridge Regression__ or __Tikhonov Regularization__.  
        In ML, it is known as __weight decay__.  
    * __Describe the regularization contribution to the gradient in a single step.__{: style="color: blue"}  
        The addition of the weight decay term has modified the learning rule to __multiplicatively shrink the weight vector by  a constant factor on each step__, just before performing the usual gradient update.  
        <p>$$\boldsymbol{w} \leftarrow(1-\epsilon \alpha) \boldsymbol{w}-\epsilon \nabla_{\boldsymbol{w}} J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y}) \tag{7.5}$$</p>  
    * __Describe the regularization contribution to the gradient. How does it scale?__{: style="color: blue"}  
        The effect of weight decay is to rescale $$\boldsymbol{w}^{\ast}$$ along the axes defined by the eigenvector of $$\boldsymbol{H}$$. Specifically, the component of $$\boldsymbol{w}^{\ast}$$ that is aligned with the $$i$$-th eigenvector of $$\boldsymbol{H}$$ is rescaled by a factor of $$\frac{\lambda_{i}}{\lambda_{i}+\alpha}$$.  
    * __How does weight decay relate to shrinking the individual weight wrt their size? What is the measure/comparison used?__{: style="color: blue"}  
        Only directions along which the parameters contribute significantly to reducing the objective function are preserved relatively intact. In directions that do not contribute to reducing the objective function, a small eigenvalue of the Hessian tells us that movement in this direction will not significantly increase the gradient. Components of the weight vector corresponding to such unimportant directions are decayed away through the use of the regularization throughout training.  

        | __Condition__ | __Effect of Regularization__ |   
        | $$\lambda_{i}>>\alpha$$ | Not much |  
        | $$\lambda_{i}<<\alpha$$ | The weight value almost shrunk to $$0$$ |  
* __Draw a graph describing the effects of $$L^2$$ regularization on the weights:__{: style="color: red"}  
    ![img](/main_files/dl_book/regularization/1.png){: width="80%"}   
* __Describe the effects of applying weight decay to linear regression__{: style="color: red"}  
    <button>Application to Linear Regression</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    ![img](/main_files/dl_book/regularization/2.png){: width="100%" hidden=""}   
    {: hidden=""}  
* __Derivation:__{: style="color: red"}  
    We place a __Gaussian Prior__ on the weights, with __zero mean__ and __equal variance $$\tau^2$$__:  
    <p>$$\begin{aligned} \hat{\theta}_ {\mathrm{MAP}} &=\arg \max_{\theta} \log P(y | \theta)+\log P(\theta) \\ &=\arg \max _{\boldsymbol{w}}\left[\log \prod_{i=1}^{n} \dfrac{1}{\sigma \sqrt{2 \pi}} e^{-\dfrac{\left(y_{i}-\boldsymbol{w}^T\boldsymbol{x}_i\right)^{2}}{2 \sigma^{2}}}+\log \prod_{j=0}^{p} \dfrac{1}{\tau \sqrt{2 \pi}} e^{-\dfrac{w_{j}^{2}}{2 \tau^{2}}} \right] \\ &=\arg \max _{\boldsymbol{w}} \left[-\sum_{i=1}^{n} \dfrac{\left(y_{i}-\boldsymbol{w}^T\boldsymbol{x}_i\right)^{2}}{2 \sigma^{2}}-\sum_{j=0}^{p} \dfrac{w_{j}^{2}}{2 \tau^{2}}\right] \\ &=\arg \min_{\boldsymbol{w}} \dfrac{1}{2 \sigma^{2}}\left[\sum_{i=1}^{n}\left(y_{i}-\boldsymbol{w}^T\boldsymbol{x}_i\right)^{2}+\dfrac{\sigma^{2}}{\tau^{2}} \sum_{j=0}^{p} w_{j}^{2}\right] \\ &=\arg \min_{\boldsymbol{w}} \left[\sum_{i=1}^{n}\left(y_{i}-\boldsymbol{w}^T\boldsymbol{x}_i\right)^{2}+\lambda \sum_{j=0}^{p} w_{j}^{2}\right] \\ &= \arg \min_{\boldsymbol{w}} \left[ \|XW - \boldsymbol{y}\|^2 + \lambda {\|\boldsymbol{w}\|_ 2}^2\right]\end{aligned}$$</p>  
    <button>Show</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __What is $$L^2$$ regularization equivalent to?__{: style="color: blue"}  
        $$L^2$$ regularization is equivalent to __MAP Bayesian inference with a Gaussian prior on the weights__.  
    * __What are we maximizing?__{: style="color: blue"}  
        The MAP Estimate of the data.  
    * __Derive the MAP Estimate:__{: style="color: blue"}  
        <p>$$\begin{aligned} \hat{\theta}_ {\mathrm{MAP}} &=\arg \max_{\theta} P(\theta | y) \\ &=\arg \max_{\theta} \frac{P(y | \theta) P(\theta)}{P(y)} \\ &=\arg \max_{\theta} P(y | \theta) P(\theta) \\ &=\arg \max_{\theta} \log (P(y | \theta) P(\theta)) \\ &=\arg \max_{\theta} \log P(y | \theta)+\log P(\theta) \end{aligned}$$</p>  
    * __What kind of prior do we place on the weights? What are its parameters?__{: style="color: blue"}  
        We place a __Gaussian Prior__ on the weights, with __zero mean__ and __equal variance $$\tau^2$$__.   
    {: hidden=""}
* __List the properties of $$L^2$$ regularization:__{: style="color: red"}  
    * Notice that L2-regularization has a rotational invariance. This actually makes it more sensitive to irrelevant features.  
    * Adding L2-regularization to a convex function gives a strongly-convex function. So L2-regularization can make gradient descent converge much faster.  
* __Formally describe the $$L^1$$ parameter regularization:__{: style="color: red"}  
    $$L^1$$ Regularization is another way to regulate the model by _penalizing the size of its parameters_; the technique adds a regularization term:  
    <p>$$\Omega(\boldsymbol{\theta})=\|\boldsymbol{w}\|_{1}=\sum_{i}\left|w_{i}\right| \tag{7.18}$$</p>  
    which is a sum of absolute values of the individual parameters.  
    * __AKA:__{: style="color: blue"}  
        __LASSO__.  
    * __Whats the regularized objective function?__{: style="color: blue"}  
        <p>$$\tilde{J}(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})=\alpha\|\boldsymbol{w}\|_ {1}+J(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y}) \tag{7.19}$$</p>  
    * __What is its gradient?__{: style="color: blue"}  
        <p>$$\nabla_{\boldsymbol{w}} \tilde{J}(\boldsymbol{w} ; \boldsymbol{X}, \boldsymbol{y})=\alpha \operatorname{sign}(\boldsymbol{w})+\nabla_{\boldsymbol{w}} J(\boldsymbol{X}, \boldsymbol{y} ; \boldsymbol{w}) \tag{7.20}$$</p>  
    * __Describe the regularization contribution to the gradient compared to L2. How does it scale?__{: style="color: blue"}  
        The regularization contribution to the gradient __no longer scales linearly with each $$w_i$$__; instead it is a __constant factor with a sign = $$\text{sign}(w_i)$$__.  
* __List the properties and applications of $$L^1$$ regularization:__{: style="color: red"}  
    * Induces Sparser Solutions
    * Solutions may be non-unique
    <button>Show</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __How is it used as a feature selection mechanism?__{: style="color: blue"}  
        __LASSO__: The Least Absolute Shrinkage and Selection Operator integrates an $$L^1$$ penalty with a _linear model_ and a _least-squares cost function_.  
        The $$L^1$$ penalty causes a subset of the weights to become __zero__, suggesting that the corresponding features may safely be discarded.  
    {: hidden=""}
* __Derivation:__{: style="color: red"}  
    <p>$$\begin{aligned} \hat{\theta}_ {\mathrm{MAP}} &=\arg \max_{\theta} \log P(y | \theta)+\log P(\theta) \\  &=\arg \max _{\boldsymbol{w}}\left[\log \prod_{i=1}^{n} \dfrac{1}{\sigma \sqrt{2 \pi}} e^{-\dfrac{\left(y_{i}-\boldsymbol{w}^T\boldsymbol{x}_i\right)^{2}}{2 \sigma^{2}}}+\log \prod_{j=0}^{p} \dfrac{1}{2 b} e^{-\dfrac{\left|\theta_{j}\right|}{2 b}} \right] \\    &=\arg \max _{\boldsymbol{w}} \left[-\sum_{i=1}^{n} \dfrac{\left(y_{i}-\boldsymbol{w}^T\boldsymbol{x}_i\right)^{2}}{2 \sigma^{2}}-\sum_{j=0}^{p} \dfrac{\left|w_{j}\right|}{2 b}\right] \\    &=\arg \min_{\boldsymbol{w}} \dfrac{1}{2 \sigma^{2}}\left[\sum_{i=1}^{n}\left(y_{i}-\boldsymbol{w}^T\boldsymbol{x}_i\right)^{2}+\dfrac{\sigma^{2}}{b} \sum_{j=0}^{p}\left|w_{j}\right|\right] \\    &=\arg \min_{\boldsymbol{w}} \left[\sum_{i=1}^{n}\left(y_{i}-\boldsymbol{w}^T\boldsymbol{x}_i\right)^{2}+\lambda \sum_{j=0}^{p}\left|w_{j}\right|\right] \\    &= \arg \min_{\boldsymbol{w}} \left[ \|XW - \boldsymbol{y}\|^2 + \lambda \|\boldsymbol{w}\|_ 1\right]\end{aligned}$$</p>  
    <button>Show</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __What is $$L^1$$ regularization equivalent to?__{: style="color: blue"}  
        MAP Bayesian inference with an isotropic Laplace distribution prior on the weights.  
    * __What kind of prior do we place on the weights? What are its parameters?__{: style="color: blue"}  
        Isotropic Laplace distribution prior on the weights: $$\operatorname{Laplace}\left(w_{i} ; 0, \frac{1}{\alpha}\right)$$.  
    {: hidden=""}
* __Analyze $$L^1$$ vs $$L^2$$ regularization:__{: style="color: red"}  
    They both shrink the weights towards zero.  

    | __$$L^2$$__ | __$$L^1$$__ |
    | Sensitive to irrelevant features | Robust to irrelevant features |
    | Computationally Efficient | Non-Differentiable at $$0$$ |
    | No Sparse Solutions | Produces Sparse Solutions |
    | No Feature Selection | Built-in Feature Selection |
    | Unique Solution | Possibly multiple solutions |

    <button>Show</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __For Sparsity:__{: style="color: blue"}  
        The analysis above shows that $$L^1$$ produces sparse solutions by killing certain coefficients/weights. While $$L^2$$ does not have this property.  
    * __For correlated features:__{: style="color: blue"}  
        * __Identical features__:   
            * $$L^1$$ regularization spreads weight arbitrarily (all weights same sign) 
            * $$L^2$$ regularization spreads weight evenly 
        * __Linearly related features__:   
            * $$L^1$$ regularization chooses variable with larger scale, $$0$$ weight to others  
            * $$L^2$$ prefers variables with larger scale — spreads weight proportional to scale  
    * __For optimization:__{: style="color: blue"}  
        Adding $$L^2$$ regularization to a convex function gives a strongly-convex function. So L2-regularization can make gradient descent converge much faster. Moreover, it has analytic solutions and is differentiable everywhere, making it more computationally efficient. $$L^1$$ being Non-Differentiable makes it harder to optimize with gradient-based methods and requires approximations.    
    * __Give an example that shows the difference wrt sparsity:__{: style="color: blue"}  
        Let's imagine we are estimating two coefficients in a regression. In $$L^2$$ regularization, the solution $$\boldsymbol{w} =(0,1)$$ has the same weight as $$\boldsymbol{w}=(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}})$$  so they are both treated equally. In $$L^1$$ regularization, the same two solutions favor the sparse one:  
        <p>$$\|(1,0)\|_{1}=1<\left\|\left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)\right\|_{1}=\sqrt{2}$$</p>  
        So $$L^2$$ regularization doesn't have any specific built in mechanisms to favor zeroed out coefficients, while $$L^1$$ regularization actually favors these sparser solutions.  
    * __For sensitivity:__{: style="color: blue"}  
        The rotational invariance of the $$L^2$$ regularizer makes it more sensitive to irrelevant features.  
    {: hidden=""}
* __Describe Elastic Net Regularization. Why was it devised? Any properties?__{: style="color: red"}  
    <p>$$\Omega = \lambda\left(\alpha\|w\|_{1}+(1-\alpha)\|w\|_{2}^{2}\right), \alpha \in[0,1]$$</p>  
    * Combines both $$L^1$$ and $$L^2$$  
    * Used to __produce sparse solutions__, but to avoid the problem of $$L^1$$ solutions being sometimes __Non-Unique__  
        * The problem mainly arises with __correlated features__  
    * Elastic net regularization tends to have a grouping effect, where correlated input features are assigned equal weights.  
* __Motivate Regularization for ill-posed problems:__{: style="color: red"}  
    * __What is the property that needs attention?__{: style="color: blue"}  
        Under-Constrained Problems. Under-determined systems. Specifically, when $$X^TX$$ is *__singular__*.   
    * __What would the regularized solution correspond to in this case?__{: style="color: blue"}  
        Many forms of regularization correspond to solving inverting $$\boldsymbol{X}^{\top} \boldsymbol{X}+\alpha \boldsymbol{I}$$ instead.  
    * __Are there any guarantees for the solution to be well-posed? How/Why?__{: style="color: blue"}  
        This regularized matrix, $$\boldsymbol{X}^{\top} \boldsymbol{X}+\alpha \boldsymbol{I}$$, is __guaranteed to be invertible__.  

    <button>Show</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __What is the Linear Algebraic property that needs attention?__{: style="color: blue"}  
        $$X^{\top}X$$ being singular.  
    * __What models are affected by this?__{: style="color: blue"}  
        Many linear models (e.g. Linear Regression, PCA) depend on __inverting $$\boldsymbol{X}^{\top}\boldsymbol{X}$$__.  
    * __What would the sol correspond to in terms of inverting $$X^{\top}X$$:__{: style="color: blue"}  
        Many forms of regularization correspond to solving inverting $$\boldsymbol{X}^{\top} \boldsymbol{X}+\alpha \boldsymbol{I}$$ instead.  
    * __When would $$X^{\top}X$$ be singular?__{: style="color: blue"}  
        * The data-generating function truly has no variance in some direction.  
        * No Variance is _observed_ in some direction because there are fewer examples (rows of $$\boldsymbol{X}$$) than input features (columns).  
    * __Describe the Linear Algebraic Perspective. What does it correspond to? [LAP]__{: style="color: blue"}  
        Given that the __Moore-Penrose pseudoinverse__ $$\boldsymbol{X}^{+}$$ of a matrix $$\boldsymbol{X}$$ can solve underdetermined linear equations:  
        <p>$$\boldsymbol{X}^{+}=\lim_{\alpha \searrow 0}\left(\boldsymbol{X}^{\top} \boldsymbol{X}+\alpha \boldsymbol{I}\right)^{-1} \boldsymbol{X}^{\top} \tag{7.29}$$</p>  
        The equation corresponds to __performing linear regression with weight-decay__.  
    * __Can models with no closed-form solution be underdetermined? Explain. [CFS]__{: style="color: blue"}  
        Models with no closed-form solution can, also, be _underdetermined_:  
        Take __logistic regression on a linearly separable dataset__, if a weight vector $$\boldsymbol{w}$$ is able to achieve perfect classification, then so does $$2\boldsymbol{w}$$ but with even __higher likelihood__. Thus, an iterative optimization procedure (sgd) will continually increase the magnitude of $$\boldsymbol{w}$$ and, in theory, will __never halt__.  
    * __What models are affected by this? [CFS]__{: style="color: blue"}  
        Many models that use MLE, log-likelihood/cross-entropy. Namely, __Logistic Regression__.   
    {: hidden=""}

    <button>Show [LAP] Problems</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __Define the Moore-Penrose Pseudoinverse:__{: style="color: blue"}  
        <p>$$\boldsymbol{X}^{+}=\lim_{\alpha \searrow 0}\left(\boldsymbol{X}^{\top} \boldsymbol{X}+\alpha \boldsymbol{I}\right)^{-1} \boldsymbol{X}^{\top} \tag{7.29}$$</p>  
    * __What can it solve? How?__{: style="color: blue"}  
        It can solve *__underdetermined linear equations__*.  
        When applied to underdetermined systems w/ non-unique solutions; It finds the minimum norm solution to a linear system.  
    * __What does it correspond to in terms of regularization?__{: style="color: blue"}  
        The equation corresponds to __performing linear regression with weight-decay__.  
    * __What is the limit wrt?__{: style="color: blue"}  
        $$7.29$$ is the limit of eq $$7.17$$ as the _regularization coefficient $$\alpha$$ shrinks to zero_.  
    * __How can we interpret the pseudoinverse wrt regularization?__{: style="color: blue"}  
        We can thus interpret the pseudoinverse as __stabilizing underdetermined problems using regularization__.  
    {: hidden=""}

    <button>Show [CFS] Problems</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __Explain the problem with Logistic Regression:__{: style="color: blue"}  
        __Logistic regression on a linearly separable dataset__, if a weight vector $$\boldsymbol{w}$$ is able to achieve perfect classification, then so does $$2\boldsymbol{w}$$ but with even __higher likelihood__. Thus, an iterative optimization procedure (sgd) will continually increase the magnitude of $$\boldsymbol{w}$$ and, in theory, will __never halt__.  
    * __What are the possible solutions?__{: style="color: blue"}  
        Regularization: e.g. __weight decay__.  
    * __Are there any guarantees that we achieve with regularization? How?__{: style="color: blue"}  
        We can use regularization to guarantee the convergence of iterative methods applied to underdetermined problems.  
        Weight decay will cause gradient descent to _quit increasing the magnitude of the weights when the **slope of the likelihood is equal to the weight decay coefficient**_.  
    {: hidden=""}
* __Describe dataset augmentation and its techniques:__{: style="color: red"}  
    Having more data is the most desirable thing to improving a machine learning model’s performance. In many cases, it is relatively easy to artificially generate data.  
* __When is it applicable?__{: style="color: red"}  
    For certain problems like __classification__ this approach is readily usable. E.g. for a classification task, we require the model to be _invariant to certain types of transformations_, of which we can generate data by applying them on our current dataset.  
* __When is it not?__{: style="color: red"}  
    This approach is not applicable to many problems, especially those that require us to learn the true data-distribution first E.g. Density Estimation.  
* __Motivate the Noise Robustness property:__{: style="color: red"}  
    Noise Robustness is an important property for ML models:    
    * For many classification and (some) regression tasks: the task should be possible to solve even if small random noise is added to the input [(Local Constancy)](/work_files/research/dl/theory/dl_book_pt1#bodyContents32)  
    * Moreover, NNs prove not to be very robust to noise.  
* __How can Noise Robustness motivate a regularization technique?__{: style="color: red"}  
    To increase noise robustness, we will need to somehow teach our models to ignore noise. That can be done by teaching it to model data that is, perhaps, more noisy than what we actually have; thus, regularizing it from purely and completely fitting the training data.   
* __How can we enhance noise robustness in NN?__{: style="color: red"}  
    By __Noise Injection__ in different parts of the network.  

    <button>Show</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __Give a motivation for Noise Injection:__{: style="color: blue"}  
        It can be seen as a form of __data augmentation__.  
    * __Where can noise be injected?__{: style="color: blue"}  
        * Input Layer
        * Hidden Layer
        * Weight Matrices
        * Output Layer
    * __Give Motivation, Interpretation and Applications of injecting noise in the different components (from above):__{: style="color: blue"}  
        __Injecting Noise in the Input Layer:__{: style="color: red"}  
        {: #lst-p}
        * __Motivation__:  
            We have motivated the injection of noise, to the inputs, as a dataset augmentation strategy.        
        * __Interpretation__:  
            For some models, the addition of noise with infinitesimal variance at the input of the model is equivalent to __imposing a penalty on the norm of the weights__ _(Bishop, 1995a,b)_.  

        __Injecting Noise in the Hidden Layers:__{: style="color: red"}  
        {: #lst-p}
        * __Motivation__:  
            We can motivate it as a variation of data augmentation.  
        * __Interpretation__:  
            It can be seen as doing __data-augmentation__ at *__multiple levels of abstraction__*.  
        * __Applications__:  
            The most successful application of this type of noise injection is __Dropout__.  
            It can be seen as a process of constructing new inputs by _multiplying_ by noise.  

        __Injecting Noise in the Weight Matrices:__{: style="color: red"}  
        {: #lst-p}
        * __Interpretation__:  
            1. It can be interpreted as a stochastic implementation of Bayesian inference over the weights.  
                * __The Bayesian View__:  
                    The Bayesian treatment of learning would consider the model weights to be _uncertain and representable via a probability distribution that reflects this uncertainty_. Adding noise to the weights is a practical, stochastic way to reflect this uncertainty.  
            2. It can, also, be interpreted as equivalent a more traditional form of regularization, _encouraging stability of the function to be learned_.  
                * <button>Analysis</button>{: .showText value="show" onclick="showTextPopHide(event);"}
                    ![img](/main_files/dl_book/regularization/3.png){: width="100%" hidden=""}   
        * __Applications__:  
            This technique has been used primarily in the context of __recurrent neural networks__ _(Jim et al., 1996; Graves, 2011)_.  

        __Injecting Noise in the Output Layer:__{: style="color: red"}  
        {: #lst-p}
        * __Motivation__:  
            * Most datasets have some number of mistakes in the $$y$$ labels. It can be harmful to maximize $$\log p(y | \boldsymbol{x})$$ when $$y$$ is a mistake. One way to prevent this is to explicitly model the noise on the labels.  
            One can assume that for some small constant $$\epsilon$$, the training set label $$y$$ is correct with probability $$1-\epsilon$$.  
                This assumption is easy to incorporate into the cost function analytically, rather than by explicitly drawing noise samples (e.g. __label smoothing__).  
            * MLE with a softmax classifier and hard targets may never converge - the softmax can never predict a probability of exactly $$0$$ or $$1$$, so it will continue to learn larger and larger weights, making more extreme predictions forever.{: #bodyContents33mle}  
        * __Interpretation__:  
            For some models, the addition of noise with infinitesimal variance at the input of the 
        * __Applications__:  
            __Label Smoothing__ regularizes a model based on a softmax with $$k$$ output values by replacing the hard $$0$$ and $$1$$ classification targets with targets of $$\dfrac{\epsilon}{k-1}$$ and $$1-\epsilon$$, respectively.   
            * [__Applied to MLE problem:__](#bodyContents33mle) Label smoothing, compared to weight-decay, has the advantage of preventing the pursuit of hard probabilities without discouraging correct classification.  
            * Application in modern NN: _(Szegedy et al. 2015)_  
    {: hidden=""}

    <button>Show further questions</button>{: .showText value="show"
    onclick="showText_withParent_PopHide(event);"}
    * __Give an interpretation for injecting noise in the Input layer:__{: style="color: blue"}  
        For some models, the addition of noise with infinitesimal variance at the input of the model is equivalent to __imposing a penalty on the norm of the weights__ _(Bishop, 1995a,b)_.  
    * __Give an interpretation for injecting noise in the Hidden layers:__{: style="color: blue"}  
        It can be seen as doing __data-augmentation__ at *__multiple levels of abstraction__*.  
    * __What is the most successful application of this technique:__{: style="color: blue"}  
        The most successful application of this type of noise injection is __Dropout__.  
    * __Describe the Bayesian View of learning:__{: style="color: blue"}  
        The Bayesian treatment of learning would consider the model weights to be _uncertain and representable via a probability distribution that reflects this uncertainty_. Adding noise to the weights is a practical, stochastic way to reflect this uncertainty.  
    * __How does it motivate injecting noise in the weight matrices?__{: style="color: blue"}  
        It can be interpreted as a stochastic implementation of Bayesian inference over the weights.  
    * __Describe a different, more traditional, interpretation of injecting noise to matrices. What are its effects on the function to be learned?__{: style="color: blue"}  
        It can, also, be interpreted as equivalent a more traditional form of regularization, _encouraging stability of the function to be learned_.  
    * __Whats the biggest application for this kind of regularization?__{: style="color: blue"}  
        This technique has been used primarily in the context of __recurrent neural networks__ _(Jim et al., 1996; Graves, 2011)_.  
    * __Motivate injecting noise in the Output layer:__{: style="color: blue"}  
        * Most datasets have some number of mistakes in the $$y$$ labels. It can be harmful to maximize $$\log p(y | \boldsymbol{x})$$ when $$y$$ is a mistake. One way to prevent this is to explicitly model the noise on the labels.  
        One can assume that for some small constant $$\epsilon$$, the training set label $$y$$ is correct with probability $$1-\epsilon$$.  
            This assumption is easy to incorporate into the cost function analytically, rather than by explicitly drawing noise samples (e.g. __label smoothing__).  
        * MLE with a softmax classifier and hard targets may never converge - the softmax can never predict a probability of exactly $$0$$ or $$1$$, so it will continue to learn larger and larger weights, making more extreme predictions forever.{: #bodyContents33mle}  
    * __What is the biggest application of this technique?__{: style="color: blue"}  
        __Label Smoothing__ regularizes a model based on a softmax with $$k$$ output values by replacing the hard $$0$$ and $$1$$ classification targets with targets of $$\dfrac{\epsilon}{k-1}$$ and $$1-\epsilon$$, respectively.   
    * __How does it compare to weight-decay when applied to MLE problems?__{: style="color: blue"}  
        [__Applied to MLE problem:__](#bodyContents33mle) Label smoothing, compared to weight-decay, has the advantage of preventing the pursuit of hard probabilities without discouraging correct classification.  
    {: hidden=""}
* __Define "Semi-Supervised Learning":__{: style="color: red"}     
    __Semi-Supervised Learning__ is a class of ML tasks and techniques that makes use of both unlabeled examples from $$P(\mathbf{x})$$ and labeled examples from $$P(\mathbf{x}, \mathbf{y})$$ to estimate $$P(\mathbf{y} | \mathbf{x})$$ or predict $$\mathbf{y}$$ from $$\mathbf{x}$$.  
    * __What does it refer to in the context of DL:__{: style="color: blue"}  
        In the context of Deep Learning, Semi-Supervised Learning usually refers to _learning a representation $$\boldsymbol{h}=f(\boldsymbol{x})$$_.  
    * __What is its goal?__{: style="color: blue"}  
        The goal is to learn a representation such that __examples from the same class have similar representations__.   
    * __Give an example in classical ML:__{: style="color: blue"}  
        Using __unsupervised learning__ to build representations: __PCA__, as a preprocessing step before applying a classifier, is a long-standing variant of this approach.  
* __Describe an approach to applying semi-supervised learning:__{: style="color: red"}  
    Instead of separating the supervised and unsupervised criteria, we can instead have a generative model of $$P(\mathbf{x})$$ (or $$P(\mathbf{x}, \mathbf{y})$$) which shares parameters with a discriminative model $$P(\mathbf{y} \vert \mathbf{x})$$.  
    The idea is to share the unsupervised/generative criterion with the supervised criterion to _express a prior belief that the structure of $$P(\mathbf{x})$$ (or $$P(\mathbf{x}, \mathbf{y})$$) is connected to the structure of $$P(\mathbf{y} \vert \mathbf{x})$$_, which is captured by the _shared parameters_.  
    By controlling how much of the generative criterion is included in the total criterion, one can find a better trade-off than with a purely generative or a purely discriminative training criterion _(Lasserre et al., 2006; Larochelle and Bengio, 2008)_.  
* __How can we interpret dropout wrt data augmentation?__{: style="color: red"}  
    Dropout can be seen as a process of constructing new inputs by multiplying by noise.  
{: hidden=""}

* __Add Answers from link below for L2 applied to linear regression and how it reduces variance:__{: style="color: red"}  
    [Link](http://cs229.stanford.edu/notes-spring2019/addendum_bias_variance.pdf)  
* __When is Ridge regression favorable over Lasso regression? for correlated features?__{: style="color: red"}  
    If there exists a subset consisting of few variables that have medium / large sized effect, use lasso regression. In presence of many variables with small / medium sized effect, use ridge regression.?? (ESL authors)  

    If features are correlated, it is so hard to determine which variables to drop, it is often better not to drop variables. Thus, use __ridge__ over __lasso__ since the latter produces non-unique solutions and might drop random features; while former, spreads weight more evenly.  





***

# Misc.
* __Explain Latent Dirichlet Allocation (LDA)__{: style="color: red"}  
    Latent Dirichlet Allocation (LDA) is a common method of topic modeling, or classifying documents by subject matter.  
    LDA is a generative model that represents documents as a mixture of topics that each have their own probability distribution of possible words.  
    The "Dirichlet" distribution is simply a distribution of distributions. In LDA, documents are distributions of topics that are distributions of words.  
* __How to deal with curse of dimensionality__{: style="color: red"}  
    * Feature Selection
    * Feature Extraction  
* __How to detect correlation of "categorical variables"?__{: style="color: red"}  
    * Chi-Squared test
* __Define "marginal likelihood" (wrt naive bayes):__{: style="color: red"}  
    Marginal likelihood is, the probability that the word ‘FREE’ is used in any message (not given any other condition?).   
* __KNN VS K-Means__{: style="color: red"}  
    * __KNN__: Supervised, Classification/Regression Algorithm
    * __K-Means__: Unsupervised Clustering Algorithm
* __When is Ridge regression favorable over Lasso regression for correlated features?__{: style="color: red"}  
    If there are exists a subset consisting of few variables that have medium / large sized effect, use lasso regression. In presence of many variables with small / medium sized effect, use ridge regression.?? (ESL authors)  

    If features are correlated; it is so hard to determine which variables to drop, it is often better not to drop variables. Thus, use __ridge__ over __lasso__ since the latter produces non-unique solutions and might drop random features; while former, spreads weight more evenly.  
* __What is convex hull ?__{: style="color: red"}  
    In case of linearly separable data, convex hull represents the outer boundaries of the two group of data points. Once convex hull is created, we get maximum margin hyperplane (MMH) as a perpendicular bisector between two convex hulls. MMH is the line which attempts to create greatest separation between two groups.  
* __Do you suggest that treating a categorical variable as continuous variable would result in a better predictive model?__{: style="color: red"}  
    For better predictions, categorical variable can be considered as a continuous variable only when the variable is ordinal in nature.
* __OLS vs MLE__{: style="color: red"}  
    They both estimate parameters in the model. They are the same in the case of normal distribution.  
* __What are collinearity and multicollinearity?__{: style="color: red"}  
    * __Collinearity__ occurs when two predictor variables (e.g., x1 and x2) in a multiple regression have some correlation.
    * __Multicollinearity__ occurs when more than two predictor variables (e.g., x1, x2, and x3) are inter-correlated.

