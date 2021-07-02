# Dhiren Kakkar - Data Science Roadmap


## About

Made this repository to track my progress as I learn data science. Hopefully It'll be a good tool to organise my goals, develop more holistic projects and provide a clear learning roadmap. I'll try to update it regularly :)

*many concepts were directly applied while learning them or working on personal project (repo are linked in the tools/library section)

## Table of content
* [MOOCs & Relevant courses](https://github.com/dhirenkakkar/RoadToDS#moocs--relevant-courses) <br>
* [Supervised learning](https://github.com/dhirenkakkar/RoadToDS#supervised-learning) <br>
  * [Regression](https://github.com/dhirenkakkar/RoadToDS#Regression) <br>
  * [Classification](https://github.com/dhirenkakkar/RoadToDS#classification) <br>
* [Unsupervised learning](https://github.com/dhirenkakkar/RoadToDS#unsupervised-learning) <br>
* [Natural language processing](https://github.com/dhirenkakkar/RoadToDS#natural-language-processing) <br>
* [Visualization and Analytics](https://github.com/dhirenkakkar/RoadToDS#visualization-and-analytics) <br>

***

### MOOCs & Relevant courses
| Course | Topics | Instructor/Platform |
| :------------: | :---: | :-----: |
| [Machine Learning*](https://www.coursera.org/learn/machine-learning) | Linear & Logistic regression, Regularization, Support vector machines, Anomaly detection, Unsupervised learning, Dimensionality reduction, Recommender systems | Coursera |
| [Machine learning](https://www.cs.ubc.ca/~schmidtm/Courses/340-F19/) | Probablistic & linear classifiers, Stochastic gradient descent, Maximum likelihood and Maximum A Posteriori estimation, Latent-factor models, Dimentionality reduction and Multi-Deminsional scaling (PCA/MDS) | [Dr. Mark Schmidt](https://www.cs.ubc.ca/~schmidtm/), CPSC340, UBC | 
| [Deep Learning*](https://www.coursera.org/specializations/deep-learning) | Neural networks, Non-linear activation, Hyperparameter optimization, Batch-procedures, CNNs, RNNs & LSTMs (Sequence modelling) | deeplearning.ai |
| [Intro to Probability](https://courses.students.ubc.ca/cs/courseschedule?pname=subjarea&tname=subj-course&dept=STAT&course=302) | Probability distributions, Bayes belief network, Jointly distributed random variables, Law of large numbers, Chebyshev's inequality & Markov chains | STAT302, UBC |
| [Matrix Algebra](https://courses.students.ubc.ca/cs/courseschedule?pname=subjarea&tname=subj-course&dept=MATH&course=221) | Linear independence and transformation, Vector span, High dimensional subspaces, Digonalization, Eigenvectors, Orthogonal sets and projections | MATH221, UBC |
| [Statistical learning](https://ubc-stat.github.io/stat-406/) | Model Accuracy, Information criteria, Kernel smoothing, Linear/Quadratic discrimant analysis, Bagging, Boosting, Hierarchical clustering| [Daniel McDonald](https://dajmcdon.github.io/), STAT406, UBC |
| [Statistical inference](https://courses.students.ubc.ca/cs/courseschedule?pname=subjarea&tname=subj-course&dept=STAT&course=305) | Moment generating functions, Maximum likelihood estimation, Fisher information, Bayesian estimation & credible intervals, Hypothesis testing | STAT305, UBC | |

*Audited courses

***

### Supervised learning

<br>

| Concept | Topics | Tools/Library |
| :------------: | :---: | :-----: |
| Data fundamnentals | Data cleaning, Dummy variables, Feature selection and transformation, Distances (Hamming, Jaccard) and Similarities (Rank correlation) | |
| Fundamentals of learning | Overfitting and Underfitting, IID learning theory, Training vs testing error, Bias-variance tradeoff, K-fold Cross validation, Optimization bias | |
| Ensemble methods | Averaging, Boosting, Bootstrapping, Bagging and Stacking | ||



  #### Regression
| Concept | Topics | Tools/Library |
| :------------: | :---: | :-----: |
| Linear regression | Ordinary, Sparse & Total least squares, Residuals, Multicollinearity, Entropy & Information gain, Normalization and standardization | [House prices - Advanced regression techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| Non-linear regression | Logistic/Sigmoidal regression, Non-linear transformations & polynomial regression, Guassian RBF, Shrinkage & Sparsity, Segmentation |  |
| Norms & Regularization | Lasso (L1), Ridge (L2) & ElastricNet (L1+L2) regression, Robust regression, Non-convex approximation using Huber loss & log-sum-exp, M-estimator | |
| Ensemble learning | Random Forests, Hyperparameter Grid-search, Boosting & Bagging, Cross validation, Stacking & Averaging, Precision vs Recall, ROC curves | XGBoost, LightGBM & AdaBoost|
| Gradient Descent optimization | Mini-batch gradient descent, Stochastic gradient descent, Polyak’s heavy ball method, Nesterov’s accelerated gradient descent and Line search. | |


  #### Classification
| Concept | Topics | Tools/Library |
| :------------: | :---: | :-----: |
| Fundamentals | Decision stump and trees, Greedy recursive splitting, Accuracy score, Entropy and information gain, Precision vs recall  | |
| Probabilistic  classification | Naive Bayes (w/ MLE & MAP), Laplace Smoothing | |
| 3 | | |

***

### Unsupervised learning
| Concept | Topics | Tools/Library |
| :------------: | :---: | :-----: |
| Cluster Analysis | K-Means clustering, Random initializations, Vector quantization,, Gaussian mixture model & Expectation Maximization, Outlier detection | |
| Non-convex | Density-based clustering (DBSCAN), Hierarchical (Agglomerative/Divisive) Clustering | BIRCH |
| Ensemble |  Label switching problem, Bootstrapping, Biclustering, Grid-based Clustering  | |

***

### Natural language processing
| Concept | Topics | Tools/Library |
| :------------: | :---: | :-----: |
| Text pre-processing | Text normalization, Bag Of Words, Stemming, Lemmatization, Stopwords, Tokenization, Regular expressions (Treebank tokenizer) | NLTK, SpaCy |
| Word associations | TF-IDF, n-grams, [Byte pair encoding](https://arxiv.org/abs/1508.07909), Word embeddings (Word2vec) | |
| Language understanding | Sentiment analysis, Topic modelling (Latent dirichlet allocation), Part of Speech (PoS) Tagging | Gensim |
| Deep learning modelling | Transformers (bidirectional RNNs), encoders, BERT | |

***

### Visualization and Analytics
| Concept | Topics | Tools/Library |
| :------------: | :---: | :-----: |
| [Predictive & Non-Predictive Analytics](https://www.forbes.com/sites/piyankajain/2012/05/01/the-power-of-non-predictive-analytics/#5cb247587909) | Descriptive/Inferential statistics, Hypothesis (A/B) testing, Predictive modelling | N/A |
| Visualization | Data storytelling, Basic 2d and time-series plots, Pearson correlation matrix, Visual encodings, Browser/BI dashboards | D3, Matplotlib, Seaborn, Plotly |

 
