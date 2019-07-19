---
layout: NotesPage
title: Auto-Encoders
permalink: /work_files/research/dl/aencdrs
prevLink: /work_files/research/dl.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Introduction and Architecture](#content1)
  {: .TOC1}
</div>

***
***

## Introduction and Architecture
{: #content1}
0. **From PCA to Auto-Encoders:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents10}   
    :   High dimensional data can often be represented using a much lower dimensional code.  
        This happens when  the data lies near a linear manifold in the high dimensional space.  
        Thus, if we can this _linear manifold_, we can project the data on the manifold and, then, represent the data by its position on the manifold without losing much information because in the directions orthogonal to the manifold there isn't much variation in the data.  
    :   Often, __PCA__ is used as a method to determine this _linear manifold_ to reduce the dimensionality of the data from $$N$$-dimensions to, say, $$M$$-dimensions, where $$M < N$$.  
    :   However, what if the manifold that the data is close to, is non-linear?  
        Obviously, we need someway to find this non-linear manifold.  
    :   Deep-Learning provides us with Deep __AutoEncoders__.  
        __Auto-Encoders__ allows us to deal with _curved manifolds_ un the input space by using deep layers, where the _code_ is a _non-linear function_ of the input, and the _reconstruction_ of the data from the code is, also, a _non-linear function_ of the code.  

1. **Auto-Encoders:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    :   An __AutoEncoder__ is an artificial neural network used for unsupervised learning of efficient codings.   
        It aims to learn a representation (encoding) for a set of data, typically for the purpose of _dimensionality reduction_.
    :   ![img](/main_files/cs231n/aencdrs/1.png){: width="50%"}  

2. **Architecture:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    :   An auto-encoder consists of:  
        * An Encoding Function 
        * A Decoding Function 
        * A Distance Function  
    :   We choose the _encoder_ and _decoder_ to be  parametric functions (typically neural networks), and to be differentiable with respect to the distance function, so the parameters of the encoding/decoding functions can be optimize to minimize the reconstruction loss, using Stochastic Gradient Descent.  
    :   The simplest form of an autoencoder is a feedforward neural network similar to the multilayer perceptron (MLP) – having an input layer, an output layer and one or more hidden layers connecting them –, but with the output layer having the same number of nodes as the input layer, and with the purpose of reconstructing its own inputs (instead of predicting the target value $${\displaystyle Y}$$ given inputs $${\displaystyle X}$$).  

3. **Structure and Mathematics:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    :   The _encoder_ and the _decoder_ in an auto-encoder can be defined as transitions $$\phi$$ and $$ {\displaystyle \psi ,}$$ such that:  
    :   $$ {\displaystyle \phi :{\mathcal {X}}\rightarrow {\mathcal {F}}} \\
    {\displaystyle \psi :{\mathcal {F}}\rightarrow {\mathcal {X}}} \\
    {\displaystyle \phi ,\psi =\arg \min _{\phi ,\psi }\|X-(\psi \circ \phi )X\|^{2}}$$
    :   where $${\mathcal {X} = \mathbf{R}^d}$$ is the input space, and $${\mathcal {F} = \mathbf{R}^p}$$ is the latent (feature) space, and $$ p < d$$.   
    :   The encoder takes the input $${\displaystyle \mathbf {x} \in \mathbb {R} ^{d}={\mathcal {X}}}$$ and maps it to $${\displaystyle \mathbf {z} \in \mathbb {R} ^{p}={\mathcal {F}}} $$:

    :   $${\displaystyle \mathbf {z} =\sigma (\mathbf {Wx} +\mathbf {b} )}$$  
    :   * The image $$\mathbf{z}$$ is referred to as _code_, _latent variables_, or _latent representation_.  
        *  $${\displaystyle \sigma }$$ is an element-wise activation function such as a sigmoid function or a rectified linear unit.
        * $${\displaystyle \mathbf {W} }$$ is a weight matrix
        * $${\displaystyle \mathbf {b} }$$ is the bias.
    :   The Decoder maps  $${\displaystyle \mathbf {z} }$$ to the reconstruction $${\displaystyle \mathbf {x'} } $$  of the same shape as $${\displaystyle \mathbf {x} }$$:  
    :   $${\displaystyle \mathbf {x'} =\sigma '(\mathbf {W'z} +\mathbf {b'} )}$$
    :   where $${\displaystyle \mathbf {\sigma '} ,\mathbf {W'} ,{\text{ and }}\mathbf {b'} } $$ for the decoder may differ in general from those of the encoder.  
    :   Autoencoders minimize  reconstruction errors, such as the L-2 loss:  
    :   $${\displaystyle {\mathcal {L}}(\mathbf {x} ,\psi ( \phi (\mathbf {x} ) ) ) =  {\mathcal {L}}(\mathbf {x} ,\mathbf {x'} )=\|\mathbf {x} -\mathbf {x'} \|^{2}=\|\mathbf {x} -\sigma '(\mathbf {W'} (\sigma (\mathbf {Wx} +\mathbf {b} ))+\mathbf {b'} )\|^{2}}$$
    :   where $${\displaystyle \mathbf {x} }$$ is usually averaged over some input training set.

4. **Applications:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    :   The applications of auto-encoders have changed overtime.  
        This is due to the advances in the fields that auto-encoders were applied in, or to the incompetency of the auto-encoders.  
    :   Recently, auto-encoders are applied to:  
        * __Data-Denoising__ 
        * __Dimensionality Reduction__ (for data visualization)
    :   >  With appropriate dimensionality and sparsity constraints, autoencoders can learn data projections that are more interesting than PCA or other basic techniques.
    :   > For 2D visualization specifically, t-SNE is probably the best algorithm around, but it typically requires relatively low-dimensional data. So a good strategy for visualizing similarity relationships in high-dimensional data is to start by using an autoencoder to compress your data into a low-dimensional space (e.g. 32 dimensional) (by an auto-encoder), then use t-SNE for mapping the compressed data to a 2D plane. 

5. **Types of Auto-Encoders:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    :   * Vanilla Auto-Encoder
        * Sparse Auto-Encoder
        * Denoising Auto-Encoder
        * Variational Auto-Encoder (VAE)
        * Contractive Auto-Encoder


6. **Auto-Encoders for initializing Neural-Nets:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    :   After training an auto-encoder, we can use the _encoder_ to compress the input data into it's latent representation (which we can view as _features_) and input those to the neural-net (e.g. a classifier) for prediction.  
    :   ![img](/main_files/cs231n/aencdrs/2.png){: width="70%"} 
