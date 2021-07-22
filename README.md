# Dhiren Kakkar - Data Science Roadmap


## About

Made this repository to track my progress as I learn data science. Hopefully It'll be a good tool to organise my goals, develop more holistic projects and provide a clear learning roadmap. I'll try to update it regularly :)

*many concepts were directly applied while learning them or working on personal project (repo are linked in the tools/library section)

## Table of content
* [MOOCs & Relevant courses](https://github.com/dhirenkakkar/RoadToDS#moocs--relevant-courses) <br>
* [Supervised learning](https://github.com/dhirenkakkar/RoadToDS#supervised-learning) <br>
  * [Regression](https://github.com/dhirenkakkar/RoadToDS#Regression) <br>
  * [Classification](https://github.com/dhirenkakkar/RoadToDS#classification) <br>
* [Deep learning](https://github.com/dhirenkakkar/RoadToDS#deep-learning) <br>
* [Natural language processing](https://github.com/dhirenkakkar/RoadToDS#natural-language-processing) <br>
* [Unsupervised learning](https://github.com/dhirenkakkar/RoadToDS#unsupervised-learning) <br>
* [Visualization and Analytics](https://github.com/dhirenkakkar/RoadToDS#visualization-and-analytics) <br>

***

### MOOCs & Relevant courses
| Course | Topics | Instructor/Platform |
| :------------: | :---: | :-----: |
| [Machine Learning*](https://www.coursera.org/learn/machine-learning) | Linear & Logistic regression, Regularization, Support vector machines, Anomaly detection, Unsupervised learning, Dimensionality reduction, Recommender systems | Coursera |
| [Machine learning](https://www.cs.ubc.ca/~schmidtm/Courses/340-F19/) | Probablistic & linear classifiers, Stochastic gradient descent, MLE/MAP estimation, Latent-factor models, Dimentionality reduction and Multi-Deminsional scaling (PCA/MDS) | [Dr. Mark Schmidt](https://www.cs.ubc.ca/~schmidtm/), CPSC340, UBC | 
| [Deep Learning*](https://www.coursera.org/specializations/deep-learning) | Neural networks, Non-linear activation, Hyperparameter optimization, Batch-procedures, CNNs, RNNs & LSTMs | deeplearning.ai |
| [Intro to Probability](https://courses.students.ubc.ca/cs/courseschedule?pname=subjarea&tname=subj-course&dept=STAT&course=302) | Probability distributions, Bayes belief network, Jointly distributed random variables, Law of large numbers, Chebyshev's inequality | STAT302, UBC |
| [Matrix Algebra](https://courses.students.ubc.ca/cs/courseschedule?pname=subjarea&tname=subj-course&dept=MATH&course=221) | Matrices & Determinants, Linear independence & transformation, Vector-span, High dimensional subspaces, Digonalization, Eigenvalues & Eigenvectors, Orthogonal sets & projections | MATH221, UBC |
| [Statistical learning](https://ubc-stat.github.io/stat-406/) | Model Accuracy, Information criteria, Kernel smoothing, Linear/Quadratic discrimant analysis, Bagging, Boosting, Bootstrapping & Ensemble learning, Hierarchical clustering| [Daniel McDonald](https://dajmcdon.github.io/), STAT406, UBC |
| [Statistical inference](https://courses.students.ubc.ca/cs/courseschedule?pname=subjarea&tname=subj-course&dept=STAT&course=305) | Moment generating functions, Samping distribution theory, Maximum likelihood & Bayesian estimation, Fisher information,  Confidence/credible intervals, Likelihood ratio testing | STAT305, UBC | |

*Audited courses

***

### Supervised learning

<br>

| Concept | Topics | Tools/Library |
| :------------: | :---: | :-----: |
| Fundamnentals | Data cleaning, Visualization, Data aggregation, Feature selection, Feature transformation, Distances and Similarities | Pandas, NumPy |
| Fundamentals of learning | Overfitting/Underfitting, IID learning theory, Training vs testing error, Bias-variance decomposition, Cross validation, Optimization bias | |
| Ensemble learning | Random Forests, Boosting, Bagging, Bootstrapping, Stacking, Averaging | XGBoost, LightGBM & AdaBoost |||


  #### Regression
| Concept | Topics | Tools/Library |
| :------------: | :---: | :-----: |
| Evaluation | Squared error (MSE,RMSE), Information crteria (AIC,BIC), Adjusted R-squared error, Mallow's Cp |
| Linear regression | Ordinary least squares, Residuals, Bias (intercept), Multiple regression, Multicollinearity, Gradient descent, Convexity | |
| Non-linear regression | Logistic regression, Non-linear feature transformation (change of basis), Guassian RBF, Shrinkage & Sparsity |  |
| Norms & Regularization | Lasso, Ridge & ElastricNet regularization, Robust/Brittle regression, Huber loss & log-sum-exp approximation | |


  #### Classification
| Concept | Topics | Tools/Library |
| :------------: | :---: | :-----: |
| Fundamentals | Accuracy score, Hinge-loss, Entropy, Information gain, Decision theory, Precision vs recall ( F-score), PR/ROC curve  | |
| Probablistic classifiers | Naive Bayes, Laplace Smoothing, Multi-class classification| |
| Distance based classifiers | Decision stump & trees, Pruning, Greedy recursive splitting, Discriminant analysis (LDA,QDA), K-nearest neighbours, SVM | |
| 3 | | |

***

### Deep learning
| Concept | Topics | Tools/Library |
| :------------: | :---: | :-----: |
| Neural networks | Forward & Backpropagation, Batch gradient descent, Learning rate, Random initialization, Sigmoid/ReLU activation function | |
| Deep learning basics |  | Tensorflow, PyTorch |
| Classification | Multi-class classification (Softmax) | |
| Gradient Descent optimization | Vanishing gradients, Stochastic (mini-batch) gradient descent, Momentum, RMSprop, Adam, Learning-rate decay, Line search | |
| Hyperparameter tuning | Grid search, Random search, Batch normalization | |
| Convolutional neural networks | Convolutions, Padding, Fully-connected/Convolutional/Pooling layer | |

***

### Natural language processing
| Concept | Topics | Tools/Library |
| :------------: | :---: | :-----: |
| Text preprocessing | Text normalization, Bag Of Words, Stemming, Lemmatization, Stopwords, Tokenization, Regular expressions | NLTK, SpaCy |
| Feature extraction | Positive/Negative frequency, TF-IDF, bi/tri-grams, Cosine similarity, [Byte pair encoding](https://arxiv.org/abs/1508.07909), Edit distance & Fuzzy matching | |
| NLP tasks | Sentiment analysis, Part of Speech tagging, Topic modelling (LDA), Named entity recognition, Text readability | |
| Probablistic modelling | Hidden markov model & Viterbi algorithm, N-grams, Backoff, Perplexity, Word embeddings (Word2Vec) | |
| Deep learning modelling | Transformers, Attention models and BERT | |

***

### Latent-Factor models
| Topics | Tools/Library |
| :---: | :-----: |
| Dimensionality Reduction (PCA/MDS), Recommender systems (Collaborative Filtering )  | |

***

### Unsupervised learning
| Concept | Topics | Tools/Library |
| :------------: | :---: | :-----: |
| Convex clusters | K-Means clustering, Random initializations, K-Means++, Convexity, Gaussian mixture model, Outlier detection | |
| Non-convex clusters | Density-based clustering, Elbow method, Hierarchical Clustering | |
| Ensemble | Label switching problem, Bootstrapping, Biclustering | |

***

### Visualization and Analytics
| Concept | Topics | Tools/Library |
| :------------: | :---: | :-----: |
| [Predictive & Non-Predictive Analytics](https://www.forbes.com/sites/piyankajain/2012/05/01/the-power-of-non-predictive-analytics/#5cb247587909) | Descriptive/Inferential statistics, Hypothesis (A/B) testing, Predictive modelling | N/A |
| Visualization | Data storytelling, Basic 2d and time-series plots, Pearson correlation matrix, Visual encodings, Browser/BI dashboards | D3, Matplotlib, Seaborn, Plotly |

 
