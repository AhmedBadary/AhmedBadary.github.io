---
layout: NotesPage
title: Answers to Prep Questions (Learning)
permalink: /work_files/research/answers_hidden
prevLink: /work_files/research.html
---



# Data Processing and Analysis
<button>Data Processing and Analysis</button>{: .showText value="show" onclick="showText_withParent_PopHide(event);"}
1. __What are 3 data preprocessing techniques to handle outliers?__{: style="color: red"}  
    1. Winsorizing/Winsorization (cap at threshold).
    2. Transform to reduce skew (using Box-Cox or similar).
    3. Remove outliers if you're certain they are anomalies or measurement errors.
1. __Describe the strategies to dimensionality reduction?__{: style="color: red"}  
    1. Feature Selection  
    2. Feature Projection/Extraction  
1. __What are 3 ways of reducing dimensionality?__{: style="color: red"}  
    1. Removing Collinear Features
    2. Performing PCA, ICA, etc. 
    3. Feature Engineering
    4. AutoEncoder
    5. Non-negative matrix factorization (NMF)
    6. LDA
    7. MSD
1. __List methods for Feature Selection__{: style="color: red"}  
    1. Variance Threshold: normalize first (variance depends on scale)
    1. Correlation Threshold: remove the one with larger mean absolute correlation with other features.  
    1. Genetic Algorithms
    1. Stepwise Search: bad performance, regularization much better, it's a greedy algorithm (can't account for future effects of each change)    
    1. LASSO, Elastic-Net  
1. __List methods for Feature Extraction__{: style="color: red"}  
    1. PCA, ICA, CCA
    1. AutoEncoders
    1. LDA: LDA is a supervised linear transformation technique since the dependent variable (or the class label) is considered in the model. It Extracts the k new independent variables that __maximize the separation between the classes of the dependent variable__.  
        1. Linear discriminant analysis is used to find a linear combination of features that characterizes or separates two or more classes (or levels) of a categorical variable.  
        1. Unlike PCA, LDA extracts the k new independent variables that __maximize the separation between the classes of the dependent variable__. LDA is a supervised linear transformation technique since the dependent variable (or the class label) is considered in the model.  
    1. Latent Semantic Analysis
    1. Isomap
1. __How to detect correlation of "categorical variables"?__{: style="color: red"}  
    1. Chi-Squared test: it is a statistical test applied to the groups of categorical features to evaluate the likelihood of correlation or association between them using their frequency distribution.  
1. __Feature Importance__{: style="color: red"}  
    1. Use linear regression and select variables based on $$p$$ values
    1. Use Random Forest, Xgboost and plot variable importance chart
    1. Lasso
    1. Measure information gain for the available set of features and select top $$n$$ features accordingly.
    1. Use Forward Selection, Backward Selection, Stepwise Selection
    1. Remove the correlated variables prior to selecting important variables
    1. In linear models, feature importance can be calculated by the scale of the coefficients  
    1. In tree-based methods (such as random forest), important features are likely to appear closer to the root of the tree. We can get a feature's importance for random forest by computing the averaging depth at which it appears across all trees in the forest   
1. __Capturing the correlation between continuous and categorical variable? If yes, how?__{: style="color: red"}  
    Yes, we can use ANCOVA (analysis of covariance) technique to capture association between continuous and categorical variables.  
    [ANCOVA Explained](https://www.youtube.com/watch?v=a61mkzQRf6c&t=2s)  
1. __What cross validation technique would you use on time series data set?__{: style="color: red"}  
    [Forward chaining strategy](https://en.wikipedia.org/wiki/Forward_chaining) with k folds.  
1. __How to deal with missing features? (Imputation?)__{: style="color: red"}  
    1. Assign a unique category to missing values, who knows the missing values might decipher some trend.  
    2. Remove them blatantly
    3. we can sensibly check their distribution with the target variable, and if found any pattern we’ll keep those missing values and assign them a new category while removing others.  
1. __Do you suggest that treating a categorical variable as continuous variable would result in a better predictive model?__{: style="color: red"}  
    For better predictions, categorical variable can be considered as a continuous variable only when the variable is ordinal in nature.  
1. __What are collinearity and multicollinearity?__{: style="color: red"}  
    1. __Collinearity__ occurs when two predictor variables (e.g., $$x_1$$ and $$x_2$$) in a multiple regression have some correlation.  
    1. __Multicollinearity__ occurs when more than two predictor variables (e.g., $$x_1, x_2, \text{ and } x_3$$) are inter-correlated.  
1. __What is data normalization and why do we need it?__{: style="color: red"}  
    ![img](https://cdn.mathpix.com/snip/images/8aNuJetgTgCtv4pvqaI0dr96pDyUmfuX_d1aLK1lmaw.original.fullsize.png){: width="80%"}  
{: hidden=""}

***


# ML/Statistical Models
<button>ML/Statistical Models</button>{: .showText value="show" onclick="showText_withParent_PopHide(event);"}
1. __What are parametric models?__{: style="color: red"}  
    Parametric models are those with a finite number of parameters. To predict new data, you only need to know the parameters of the model. Examples include linear regression, logistic regression, and linear SVMs.
1. __What is a classifier?__{: style="color: red"}  
    A function that maps... 
{: hidden=""}

***
