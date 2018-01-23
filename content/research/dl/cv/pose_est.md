---
layout: NotesPage
title: Articulated Body Pose Estimation <br /> (Human Pose Estimation)
permalink: /work_files/research/dl/pose_est
prevLink: /work_files/research/dl/cv.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Introduction](#content1)
  {: .TOC1}
  * [SECOND](#content2)
  {: .TOC2}
  * [THIRD](#content3)
  {: .TOC3}
  * [FOURTH](#content4)
  {: .TOC4}
</div>

***
***

## Introduction
{: #content1}

1. **Human (Articulated) Body Pose Estimation:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    :   __Human Pose Estimation__ is the process of estimating the configuration of the body (pose) from a single, typically monocular, image. 
    :   In computer vision, __Body Pose Estimation__ is the study of algorithms and systems that recover the pose of an articulated body, which consists of joints and rigid parts using image-based observations.

2. **Difficulties in Pose Estimation:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}  
    :   Pose estimation is hard due to many reasons including:  
        * *__High Degree of Freedom (DOF)__*: 244 DOF  
        * *__Variability of human visual appearance__*
        * *__Variability in lighting conditions__* 
        * *__Variability in human physique__*
        * *__(partial) Occlusions__*
        * *__Complexity of the human physical structure__*
        * *__high dimensionality of the pose__* 
        * *__loss of 3d information that results from observing the pose from 2d planar image projections__* 
        * *__(variability in) Clothes__*  
3. **Theory:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    :   Human pose estimation is usually formulated __Probabilistically__ to account for the variability and ambiguities that exist in the inference.  
    :   In __Probabilistic__ approaches, we are interested in estimating the *__posterior distribution__* $$p(\mathbf{x}\vert \mathbf{z})$$, where $$\mathbf{x}$$ is the pose of the body and and $$\mathbf{z}$$ is a feature set derived from the image.  
    :   * __The Key Modeling choices__ that affect the inference are:   
            * The representation of the pose – $$\mathbf{x}$$
            * The nature and encoding of image features – $$\mathbf{z}$$
            * The inference framework required to estimate the posterior – $$p(\mathbf{x}\vert \mathbf{z})$$
    :   [Further Reading](https://cs.brown.edu/~ls/Publications/SigalEncyclopediaCVdraft.pdf)   

4. **Model-based Approaches:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    :   The typical body pose estimation system involves a __model-based approach__, in which the pose estimation is achieved by _maximizing/minimizing_ a _similarity/dissimilarity_ between an _observation_ (input) and a _template model_.   
    :   Different kinds of sensors have been explored for use in making the observation.  
        * __Sensors__:   
            * Visible wavelength imagery
            * Long-wave thermal infrared imagery
            * Time-of-flight imagery
            * Laser range scanner imagery
    :   These sensors produce intermediate representations that are directly used by the model.
        * __Representations__: 
            * Image appearance
            * Voxel (volume element) reconstruction
            * 3D point clouds, and sum of Gaussian kernels
            * 3D surface meshes.

33. **The Representation:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents133}  
    :   A __Representation__ is a model to depict the configuration of the human body.  
        The _configuration of the human body_ can be represented in a variety of ways.  
    :   There are two common representations used for the human body:  
        * __Kinematic Skeleton Tree__  
        * __Part Models__  

44. **Kinematic Skeleton Tree with Quaternions:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents144}  
    :   The most direct and common representation is obtained by parameterizing the body as a kinematic tree, $$\vec{x} = \{\tau, \theta_\tau, \theta_1, \theta_2, \ldots, \theta_N\}$$, where the pose is encoded using position of the root segment (the __pelvis__ is typically used as root to minimize the height of the kinematic tree), $$\tau$$, orientation of the root segment in the world, $$\theta_\tau$$, and a set of relative joint angels, $$\{\theta_i\}_{i=1}^N$$, that represent the orientation of the body parts with respect to their parents along the tree.  
        > e.g., the orientation of the thigh with respect to the pelvis, shin with respect to the thigh, etc.  
    :   ![img](/main_files/cv/pose_est/1.png){: width="60%"}
    :   The kinematic skeleton is constructed by a tree-structured chain where each rigid body segment has its local coordinate system that can be transformed to the world coordinate system via a 4×4 transformation matrix $${\displaystyle T_{l}}$$, 
    :   $${\displaystyle T_{l}=T_{\operatorname {par} (l)}R_{l},}$$
    :   where $${\displaystyle R_{l}}$$ denotes the local transformation from body segment $${\displaystyle S_{l}}$$ to its parent $${\displaystyle \operatorname {par} (S_{l})}$$.  
    :   Kinematic tree representation can be obtained for 2d, 2.5d, and 3d body models.  
        * __2-D__:   
            * $$\tau \in \mathcal{R}^2$$, 
            * $$\theta_\tau \in \mathcal{R}^1$$,
            * $$\theta_i \in \mathcal{R}^1$$:   corresponds to pose of the cardboard person in the image plane  
        * __3-D__:   
            * $$\tau \in \mathcal{R}^3$$, 
            * $$\theta_\tau \in SO(3)$$,
            * $$\theta_i \in SO(3)$$: for spherical joints, e.g. neck  
              $$\theta_i \in \mathcal{R}^2$$: for saddle joints, e.g. wrist  
              $$\theta_i \in \mathbb{R}^1$$: for hinge joints, e.g. knee    
        * __2.5-D__: are extensions of the __2-D__ representations where the pose, $$\mathbf{x}$$, is augmented with (discrete)  variables encoding the relative depth (layering) of body parts with respect to one another in the 2-d _cardboard_ model.  
            This representation is not very common.  
    :   Each joint in the body has 3 degrees of freedom (DoF) rotation. Given a transformation matrix $${\displaystyle T_{l}}$$, the joint position at the T-pose can be transferred to its corresponding position in the world coordination.  
        The __3-D joint rotation__ is, usually, expressed as a *__normalized quaternion__* $$[x, y, z, w]$$ due to its continuity that can facilitate gradient-based optimization in the parameters estimation.  
    :   In all (dimensionality) cases, kinematic tree representation results in a __high-dimensional pose vector__, $$\mathbf{x}$$, in $$\mathbb{R}^{30} - \mathbb{R}^{70}$$, depending on the fidelity and exact parameterization of the skeleton and joints.  
    :   Another parameterization uses the (2d or 3d) locations of the major joints in the world.   
        However, this parametrization is __not invariant to the morphology__ (body segment lengths) of a given individual.   

6. **Part-based Models:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    :   The body is modeled as a __set of parts__, $$\mathbf{x} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_M\}$$, each with its own position and orientation in space, $$\mathbf{x}_i = \{\tau_i, \theta_i\}$$, that are connected by a set of statistical or physical constraints that enforce skeletal (and sometimes image) consistency.  
    :   The part model is motivated by the human skeleton, since any object having the property of articulation can be broken down into smaller parts wherein each part can take different orientations, resulting in different articulations of the same object.   
    Different scales and orientations of the main object can be articulated to scales and orientations of the corresponding parts.
    :   Mathematically, the parts are connected by springs; the model is, also, known as a __spring model__.  
        The degree of closeness between each part is accounted for by the compression and expansion of the springs. There is geometric constraint on the orientation of springs. For example, limbs of legs cannot move 360 degrees. Hence parts cannot have that extreme orientation.  This reduces the possible permutations.  
    :   The model can be formulated in 2-D or in 3-D.  
        The 2-D parameterizations are much more common.  
        In 2-D, each part’s representation is often augmented with an additional variable, $$s_i$$, that accounts for uniform scaling of the body part in the image, i.e., $$\mathbf{x}_i = \{\tau_i, \theta_i, s_i\}$$ with $$\tau_i \in \mathbb{R}^2, \theta_i \in \mathbb{R}^1$$ and $$s_i \in \mathbb{R}^1$$.  
    :   The model results in very high dimensional vectors, even higher than that of _kinematic trees_.  

5. **Applications:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    :   * Markerless motion capture for human-computer interfaces,
        * Physiotherapy 
        * 3D animation 
        * Ergonomics studies 
        * Robot control  and
        * Visual surveillance
        * Human-robot interaction
        * Gaming
        * Sports performance analysis


7. **Image Features:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  
    :   In many of the classical approaches image features that represent the salient parts of the image with respect to the human pose play a huge rule in the performance of any pose estimation approach.   
    :   ![img](/main_files/cv/pose_est/2.png){: width="70%"}
    :   * __The most common features__: 
            * *__Silhouettes__*: for effectively separating the person from background in static scenes  
            * *__Color__*: for modeling un-occluded skin or clothing
            * *__Edges__*: for modeling external and internal contours of the body    
            * *__Gradients__*: for modeling the texture over the body parts  
        Other, less common features, include, __Shading__ and __Focus__.  
    :   To __reduce dimensionality__ and __increase robustness to noise__, these raw features are often encapsulated in _image descriptors_, such as __shape context__, __SIFT__, and __histogram of oriented gradients (HoG)__.  
        Alternatively, _hierarchical multi-level image encodings_ can be used, such as __HMAX__, __spatial pyramids__, and __vocabulary trees__.   


8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}  
    :   


***

## SECOND
{: #content2}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  
    :   


2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  
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


***

## THIRD
{: #content3}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents31}  
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


***

## FOURTH
{: #content4}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents41}  
    :   


2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents42}  
    :   


3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents43}  
    :   


4. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents44}  
    :   


5. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents45}  
    :   


6. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents4 #bodyContents46}  
    :   
