---
layout: NotesPage
title: Deep Learning <br > A Complete Overview
permalink: /work_files/research/dl/nlp/dl_book
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
  * [FOURTH](#content4)
  {: .TOC4}
  * [FIFTH](#content5)
  {: .TOC5}
  * [SIXTH](#content6)
  {: .TOC6}
</div>

***
***

## Machine Learning Basics
{: #content1}

1. **Introduction:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}
    * __Two Approaches to Statistics__:  
        * Frequentest Estimators
        * Bayesian Inference
            

2. **Learning Algorithms:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}
    * __Learning__:  
        A computer program is said to learn from experienceEwith respect to some class of tasks $$T$$ and performance measure $$P$$, if its performance at tasks in $$T$$, as measured by $$P$$, improves with experience $$E$$
    * __The Task $$T$$__: 
        * *__Classification__*:  
            A task where the computer program is asked to specify which of $$k$$ categories some input belongs to.  
            To solve this task, the learning algorithm is usually asked to produce a function $$f:\mathbb{R}^n \rightarrow {1, . . . , k}$$.  
            When $$y=f(x)$$, the model assigns an input described by vector $$x$$ to a category identified by numeric code $$y$$.  
            > e.g. Object Recognition
        * *__Classification with Missing Inputs__*:  
            Classification becomes more challenging if the computer program is not guaranteed that every measurement in its input vector will always be provided.  
            To solve this task, rather than providing a single classification function (as in the normal classification case), the learning algorithm must learn a set of functions, each corresponding to classifying $$x$$ with a different subset of its inputs missing.  

            One way to efficiently define such a large set of functions is to learn a probability distribution over all the relevant variables, then solve the classification task by marginalizing out the missing variables.  
            With $$n$$ input variables, we can now obtain all $$2^n$$ different classification functions needed for each possible set of missing inputs, but the computer program needs to learn only a single function describing the joint probability distribution.  
            > e.g. Medical Diagnosis (where some tests weren't conducted for any reason)  
        * *__Regression__*:  
            A computer is asked to predict a numerical value given some input.  
            To solve this task, the learning algorithm is asked to output a function $$f:\mathbb{R}^n \rightarrow R$$  
            > e.g. Object Localization
        * *__Transcription__*:  
            In this type of task, the machine learning system is asked to observe a relatively unstructured representation of some kind of data and transcribe the information into discrete textual form.  
            > e.g. OCR
        * *__Machine Translation__*:  
            In a machine translation task, the input already consists of a sequence of symbols in some language, and the computer program must convert this into a sequence of symbols in another language.  
            > e.g. Google Translate  
        * *__Structured Output__*:  
            Structured output tasks involve any task where the output is a vector (or other data structure containing multiple values) with important relationships between the different elements.  
            This is a broad category and subsumes the transcription and translation tasks described above, as well as many other tasks.  
            These tasks are called structured output tasks because the program must output several values that are all tightly interrelated. For example, the words produced by an image captioning program must form a valid sentence.  
            > e.g. Syntax Parsing, Image Segmentation  
        * *__Anomaly Detection__*:  
            In this type of task, the computer program sifts through a set of events or objects and ﬂags some of them as being unusual or atypical.  
            > e.g. Insider Trading Detection
        * *__Synthesis and Sampling__*:  
            In this type of task, the machine learning algorithm is asked to generate new examples that are similar to those in the training data.  
            This is a kind of structured output task, but with the added qualification that there is no single correct output for each input, and we explicitly desire a large amount of variation in the output, in order for the output to seem more natural and realistic.  
            > e.g. Image Synthesis, Speech Synthesis
        * *__Imputation__*:  
            In this type of task, the machine learning algorithm is given a new example $$x \in \mathbb{R}^n$$, but with some entries $$x_i$$ of $$x$$ missing. The algorithm must provide a prediction of the values of the missing entries.  
        * *__Denoising__*:  
            In this type of task, the machine learning algorithm is given as input a corrupted example $$\tilde{x} \in \mathbb{R}^n$$ obtained by an unknown corruption process from a clean example $$x \in \mathbb{R}^n$$. The learner must predict the clean example $$x$$ from its corrupted version $$\tilde{x}$$, or more generally predict the conditional probability distribution $$p(x |\tilde{x})$$.  
            > e.g. Signal Reconstruction, Image Artifact Removal  
        * *__Density (Probability Mass Function) Estimation__*:  
            In the density estimation problem, the machine learning algorithm is asked to learn a function $$p_\text{model}: \mathbb{R}^n \rightarrow R$$, where $$p_\text{model}(x)$$ can be interpreted as a probability density function (if $$x$$ is continuous) or a probability mass function (if $$x$$ is discrete) on the space that the examples were drawn from.  
            To do such a task well, the algorithm needs to learn the structure of the data it has seen. It must know where _examples cluster tightly_ and where they are _unlikely to occur_.  
            Most of the tasks described above require the learning algorithm to at least implicitly capture the structure of the probability distribution (i.e. it can be computed but we don't have an equation for it). Density estimation enables us to explicitly capture that distribution.  
            In principle,we can then perform computations on that distribution to solve the other tasks as well.  
            For example, if we have performed density estimation to obtain a probability distribution p(x), we can use that distribution to solve the missing value imputation task. Equivalently, if a value $$x_i$$ is missing, and all the other values, denoted $$x_{−i}$$, are given, then we know the distribution over it is given by $$p(x_i| x_{−i})$$.  
                In practice, density estimation does not always enable us to solve all these related tasks, because in many cases the required operations on p(x) are computationally intractable.  
            > e.g. Language Modeling  
        * *__A Lot More__*:  
            Their are many more tasks that could be defined for and solved by Machine Learning. However, this is a list of the most common problems, which have a well-known set of methods for handling them.  

    * __The Performance Measure $$P$$__:  
        

        
                

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}

***

## SECOND
{: #content2}

0. **Derivative:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents20}
:   The derivative of a function is the amount that the value of a function changes when the input changes by an $$\epsilon$$ amount:  
:   $$f'(a)=\lim_{h\to 0}{\frac {f(a+h)-f(a)}{h}}. \\
    \text{i.e. } f(x + \epsilon)\approx f(x)+\epsilon f'(x)
    $$
:   __The Chain Rule__ is a way to compute the derivative of _composite functions_.  
    If $$y = f(x)$$ and $$z = g(y)$$:  
:   $$\dfrac{\partial z}{\partial x} = \dfrac{\partial z}{\partial y} \dfrac{\partial y}{\partial x}$$    

1. **Gradient: (Vector in, Scalar out)**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}
:   Gradients generalize derivatives to __*scalar functions*__ of several variables
:  ![Gradient](/main_files/math/calc/1.png){: width="80%"}
:   __Property:__ the gradient of a function $$\nabla f(x)$$ points in the direction of __steepest ascent__ from $$x$$. 

2. **The Jacobian: (Vector in, Vector out)**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}
:   The __Jacobian__ of $$f: \mathbb{R}^n \rightarrow \mathbb{R}^m $$ is a matrix of _first-order partial derivatives_ of a __*vector-valued function*__:  
:  ![Jacobian](/main_files/math/calc/2.png){: width="80%"}  
:   __The Chain Rule:__  
    Let $$f : \mathbb{R}^N \rightarrow \mathbb{R}^M$$ and $$g : \mathbb{R}^M \rightarrow \mathbb{R}^ K$$; and let  $$x \in \mathbb{R}^N, y \in \mathbb{R}^M$$, and $$z \in \mathbb{R}^K$$ with $$y = f(x)$$ and $$z = g(y)$$:  
:  $$\dfrac{\partial z}{\partial x} = \dfrac{\partial z}{\partial y} \dfrac{\partial y}{\partial x}$$    
:   where, $$\dfrac{\partial z}{\partial y} \in \mathbb{R}^{K \times M}$$ matrix, $$\dfrac{\partial y}{\partial x} \in \mathbb{R}^{M \times N}$$ matrix, and $$\dfrac{\partial z}{\partial x} \in \mathbb{R}^{K \times N}$$  matrix;  
    the multiplication of $$\dfrac{\partial z}{\partial y}$$  and $$\dfrac{\partial y}{\partial x}$$ is a matrix multiplication.  

2. **The Generalized Jacobian: (Tensor in, Tensor out)**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}
:   __A Tensor__ is a D-dimensional grid of number.  
:   Suppose that $$f: \mathbb{R}^{N_1 \times \cdots \times N_{D_x}} \rightarrow \mathbb{R}^{M_1 \times \cdots \times M_{D_y}} $$.  
    If $$y = f(x)$$ then the derivative $$\dfrac{\partial y}{\partial x}$$ is a __generalized Jacobian__ - an object with shape:  
:   $$(M_1 \times \cdots \times M_{D_y}) \times (N_1 \times \cdots \times N_{D_x})$$ 
:   >  we can think of the generalized Jacobian as generalization of a matrix, where each “row” has the same shape as $$y$$  and each “column” has the same shape as $$x$$.  
:    Just like the standard Jacobian, the generalized Jacobian tells us the relative rates of change between all elements of $$x$$  and all elements of $$y$$:  
:   $$(\dfrac{\partial y}{\partial x})_{i,j} = \dfrac{\partial y_i}{\partial x_j} \in \mathbb{R}$$
:   Just as the derivative, the generalized Jacobian gives us the relative change in $$y$$ given a small change in $$x$$:  
:   $$f(x + \delta x)\approx f(x)+ f'(x) \delta x = y + \dfrac{\partial y}{\partial x}\delta x$$  
:   where now, $$\delta x$$ is a tensor in $$\mathbb{R}{N_1 \cdots N_{d_x}}$$ and $$\dfrac{\partial y}{\partial x}$$ is a generalized matrix in $$\mathbb{R}^{(M_1 \times \cdots \times M_{D_y}) \times (N_1 \times \cdots \times N_{D_x})} $$.  
    The product $$\dfrac{\partial y_i}{\partial x_j} \delta x$$ is, therefore, a __*generalized matrix-vector multiply*__, which results in a tensor in $$\mathbb{R}^{M_1 \times \cdots \times M_{D_y}}$$.  
:   The __generalized matrix-vector multiply__ follows the same algebraic rules as a traditional matrix-vector multiply:  
:   ![matrix-vector mult](/main_files/math/calc/4.png){: width="80%"}  
:   ![matrix-vector mult-2](/main_files/math/calc/5.png){: width="100%"}  
:   __The Chain Rule:__  
:   ![chain rule](/main_files/math/calc/6.png){: width="100%"}  


3. **The Hessian:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}
:   The __Hessian__ Matrix of a _scalar function_ $$f: \mathbb{R}^d \rightarrow \mathbb{R} $$ is a matric of _second-order partial derivatives_:
:  ![Hessian](/main_files/math/calc/3.png){: width="80%"}
:   __Properties:__ 
        * The Hessian matrix is __*symmetric*__ - since we usually work with smooth/differentiable functions - due to _Clairauts Theorem_.  
        > __Clairauts Theorem:__ if the partial derivatives are continuous, the order of differentiation can be interchanged  
        * The Hessian is used in some optimization algorithms such as Newton’s method  
        * It is expensive to calculate but can drastically reduce the number of iterations needed to converge to a local minimum by providing information about the curvature of $$f$$


4. **Matrix Calculus:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}
:   __Important Identities:__  
:   $${\frac  {\partial {\mathbf  {a}}^{\top }{\mathbf  {x}}}{\partial {\mathbf  {x}}}}={\frac  {\partial {\mathbf  {x}}^{\top }{\mathbf  {a}}}{\partial {\mathbf  {x}}}}= a \\ 
    {\frac  {\partial {\mathbf  {x}}^{\top }{\mathbf  {A}}{\mathbf  {x}}}{\partial {\mathbf  {x}}}}=  ({\mathbf  {A}}+{\mathbf  {A}}^{\top }){\mathbf  {x}} \\ 
    {\frac  {\partial {\mathbf  {x}}^{\top }{\mathbf  {A}}{\mathbf  {x}}}{\partial {\mathbf  {x}}}}=  2{\mathbf  {A}}{\mathbf  {x}} \:\:\:\:\: \text{[Symmetric } A\text{]}$$
:   [Identities](https://en.wikipedia.org/wiki/Matrix_calculus)
:   __The Product Rule:__  
:   $${\displaystyle {\begin{aligned}\nabla (\mathbf {A} \cdot \mathbf {B} )&=(\mathbf {A} \cdot \nabla )\mathbf {B} +(\mathbf {B} \cdot \nabla )\mathbf {A} +\mathbf {A} \times (\nabla \times \mathbf {B} )+\mathbf {B} \times (\nabla \times \mathbf {A} )\\&=\mathbf {J} _{\mathbf {A} }^{\mathrm {T} }\mathbf {B} +\mathbf {J}_{\mathbf {B} }^{\mathrm {T} }\mathbf {A} \\&=\nabla \mathbf {A} \cdot \mathbf {B} +\nabla \mathbf {B} \cdot \mathbf {A} \ \end{aligned}}}\\ 
    \implies \\ 
    \nabla (fg) = (f')^T g + (g')^T f$$ 
:   Thus, we set our function $$h(x) = \langle f(x), g(x) \rangle = f(x)^T g(x)$$; then,  
:   $$\nabla h(x) = f'(x)^T g(x) + g'(x)^T f(x).$$


5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents25}

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents26}

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents27}

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents28}

***

## THIRD
{: #content3}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents31}

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents32}

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents33}

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents34}

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents35}

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents36}

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents37}

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents38}

***

## FOURTH
{: #content4}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents41}

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents42}

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents43}

4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents44}

5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents45}

6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents46}

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents47}

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents48}

***