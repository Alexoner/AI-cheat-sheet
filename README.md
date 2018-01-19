#Artificial Intelligence cheat sheet
===================================

This cheat sheet contains many mathematical formula, equations,inference and diagrams on Artificial Intelligence.
Most of them are based on `Pattern Recognition and Machine Learning`,`Elements of Statisitical Learning` and `Machine Learning: A Probabilistic Perspective`.


##TODO
- [ ] Linear Models for Regression
  - [x] Maximum Likelihood
  - [x] Maximum Posterior
  - [x] Bayesian Linear Regression
  - [ ] Bayesian Model Comparison
  - [ ] The Evidence Approximation

- [ ] Linear Models for Classification
  - [x] Discriminant Functions
  - [x] Generative Models
  - [x] Discriminative Models
  - [ ] Laplace Approximation
  - [ ] Bayesian Logistic Regression

- [ ] Neural Networks
  - [x] Feed-forward Network Functions
  - [ ] Network Training 
    - [ ] Parameter optimization
    - [ ] Local quadratic approxmiation
    - [x] Use of gradient information
    - [x] Gradient descent optimization
  - [x] Error BackPropagation
    - [x] Evaluation of error-function derivatives
    - [ ] Example
    - [ ] Efficiency of backpropagation
    - [ ] Jacobian matrix
  - [ ] The Hessian Matrix
  - [ ] Regularization in Neural Networks
    - [ ] ...
    - [ ] Early stopping
    - [ ] Convolutional networks
  - [ ] Mixture Density Networks
  - [ ] Bayesian Neural Networks

- [ ] Kernel Methods

- [x] Sparse Kernel Machines
  - [x] Maximum Margin Classifiers
  - [ ] Relevance Vector Machines

- [x] Graphical Models

- [x] Mixture Models and EM
  - [x] K-means
  - [x] Mixtures of Gaussians
  - [x] Complementary View
  - [x] General EM

- [ ] Approximation Inference

- [ ] Sampling Methods

- [ ] Continuous Latent Variables

- [x] Sequential Data

- [ ] Combining Models
  - [x] Boosting
  - [ ] Random Forests

- [ ] Latent Dirichlet Allocation
  - [ ] Gamma function

Sample Latex formula $$\int_{0}^{\infty}f(\vec{x},\vec{w})dx$$

## Book review
For Bayesian treatment, refer to PRML. For more data mining practical issues refer to ESL.

##Download PDF
[cheat-sheet.pdf](https://github.com/Alexoner/ai-cheat-sheet/raw/master/cheat-sheet.pdf) 

##LaTeX template
This open-source book adopts the [Springer latex template](http://www.springer.com/authors/book+authors?SGWID=0-154102-12-970131-0).

##How to compile
1. Install [Tex Live 2012](http://www.tug.org/texlive/).
2. Install [TeXstudio](http://texstudio.sourceforge.net/).
3. Configure TeXstudio.  
    Run TeXstudio, click `Options-->Configure Texstudio-->Commands`, set `XeLaTex` to `xelatex -synctex=1 -interaction=nonstopmode %.tex`.
    
    Click `Options-->Configure Texstudio-->Build`,   
    set `Build & View` to `Compile & View`,  
    set `Default Compiler` to `XeLaTex`,  
    set `PDF Viewer` to `Internal PDF Viewer(windowed)`, so that when previewing it will pop up a standalone window, which will be convenient.
4. Compile. Use Open `cheat-sheet.tex` with TeXstudio，click the green arrow on the menu bar, then it will start to compile.  
    In the messages window below we can see the compilation command that TeXstudio is using is `xelatex -synctex=1 --enable-write18 -interaction=nonstopmode "cheat-sheet".tex`
