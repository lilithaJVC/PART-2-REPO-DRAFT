# PART-2-REPO-DRAFT

![Image](https://github.com/user-attachments/assets/5cea83bc-14ce-4b6d-9c5e-16ad8784c485) 

<h5>

<p>
Logistic Regression is a  statistical model that is used for classification problems where the outcome is binary, this means that it has only two possible results (like "Yes/No," "Popular/Not Popular," ).
as opposed to  standard Linear Regression, which predicts a continuous number, Logistic Regression predicts the probability of the event occurring (a value between 0 and 1). It achieves this by using an S-shaped sigmoid function to "squash" the linear combination of the input features into the probability range.
This model is prized for its high interpretability. Its coefficients can be directly translated into Odds Ratios, which tell us exactly how much a change in a feature (for exmample a higher book rating) increases or decreases the odds of the target outcome (efor example, becoming a best-seller).
</p>


<p>The main objective of this project is to develop as well as  evaluate a Logistic Regression model to predict whether a book will achieve "Popular" (Best-Seller) status. This classification task transforms book metadata, such as ratings, publication history, and publisher, into a reliable forecast for the publishing house.
for this task , i used a dataset of over 55,000 records. The project follows a systematic process which includes : Exploratory Data Analysis (EDA), statistical Feature Selection (using VIF and P-values), and Model Training. The evaluation critically assesses performance in this imbalanced classification problem, focusing on metrics like Recall to deliver a robust and interpretable model for maximizing acquisition success.</p>
</h5>

# Evaluation and Justification of the Dataset 
# Overview of the dataset 

<h5>

<p>
  The dataset I chose for this logistic task is a subset of the Goodreads Book Dataset 10M. I downloaded this dataset from an open-source Kaggle. This big dataset consists of many CSV files inside and the selected one is book400k-500k.csv. this selected CSV file consists of 55,156 records in total, and it has many attributes that are related to the book metadata as well as the engagement of readers. Each record is a representation of a single book, and it has the following attributes: 
</p>

<ul>•	Detains of the publication (PublishMonth ,PublishYear,  and PublishDay)</ul>
<ul>•	User ratings that are numeric ( Rating ) </ul>
<ul>•	The number of review (CountsOfReview)</ul>
<ul>•	The length of the book (pagesNumber)</ul>
<ul>•	Variables that are vategorical ( Authors ,Publisher,  and Language)</ul>
<ul>•	Star ratings from 1-5 ( RatingDist1–RatingDist5) </ul>
<p>
  This dataset presents a balance of categorical data and numerical  data which is important for logistic regression classification. It is an ideal choice for an activity that requires you to predict the best seller’s book according to characteristics  that are quantifiable. 
</p>
</h5>

# Justification for Dataset Selection
<h5>
<p>The reason for choosing this dataset is that it meets the requirements from the task in terms of the size. It also has meaningful variables that are relevant for a classification task. This dataset has over 50,000 records and with that it ensures:</p>
<ul>•	That the sample size that is sufficient for training, validation as well as the splits of testing. </ul>
<ul>•	Feature diversity which will allow modelling of popularity, publication trends and the interest of reader. </ul>
<ul>•	Numeric indicators that are rich, for example review counts and ratings which are suitable for logistic regression that is dependent on predictors that are numerical. </ul>
<p>
  The scale and the structure of the dataset supports a string exploratory data analysis (EDA), feature selection, correlation test which are aligned with the assignment’s requirement. 
</p>

</h5>

# Evaluation of Data Quality 

<h5>

<p>
  This data is well organised in an excel file (CSV). however, it has many real-world data problems that needs to be addressed before we can proceed to data modelling. These challenges are as follows: 
</p>
<ul>•	Missing values: there are some missing entries in the dataset, for example, the language column has some missing values. This can be addressed through imputation methods or maybe getting rid of the records that are incomplete. </ul>
<ul>•	Outliers and noise: results may be skewed due to extremely high or low values in the number of pages for example. This needs us to perform data normalisation. </ul>
<ul>•	Formatting inconsistency: we have a presence of string variables in the dataset, these variables include RatingDistTotal, RatingDist1 as well as RatingDist5. For these variables we will need to convert them and clean them before we can proceed to data analysis. </ul>
<ul>•	Categorical variables: we will need to perform encoding (one-hot or label encoding) for features like publisher and Author so that they can be effectively used in logistic regression. </ul>
<p>
  Identifying and also addressing these mentioned issues ensures that the dataset is statistically reliable and is also high in terms of quality for classification modelling. 
</p>  
</h5>

#   Alignment with Logistic Regression Requirements

<h5>

This dataset is suitable for logistic regression because of the following: 
<p>•	It has continuous predictors that includes pages, rating, as well as review counts. </p>
<ul>•	It provides an opportunity of binary target variable creation where we can say the books that  has over 1500 reviews are considered as best sellers </ul>
<ul>•	It provides sufficient sample for a coefficient estimation that is stable and model generalisation. </ul>
<ul>•	It also has enough variance throughout predictors which prevent challenges of perfect multicollinearity. 
</ul>
</h5>

# This dataset has some common pitfalls, they are as follows:  

<h5>
  
Common Pitfalls and Mitigation Strategies

<ul>•	Missing values: there are records that might have some missing information like reviews or language information, and to sort that out, there will be data imputation implementation or maybe remove the records with incomplete data. </ul>
<ul>•	Skewed distributions:  in this dataset, there are books with very high number of reviews when compared to others. this means that few popular books can be dominant in the results and that can affect model balance. To address this, I can implement log normalization on the column with the number of reviews so that the data can be distributed evenly by reducing values that are extreme. </ul>
<ul>•	Rating distribution (non-numeric): there are columns that are stored as string such as RatingDist1, RatingDist2. These columns must be converted into numeric values so that they can be used to train the logistic regression model. To solve this, I can use string manipulation methods to extract the numeric part of these strings and store then as integers, this will simplify my correlation and calculation performances. </ul>
<ul>•	Class imbalance: the creation of best seller target variable will result into best sellers that are few compared to the non-best sellers. This class imbalance results into biased model that predicts majority of the class mostly. To address this issue, I can utilize methods such as stratified sampling or Synthetic Minority oversampling technique in order to generate more synthetic examples of bestsellers so that there is a fair model training across both classes. 
</ul>
</h5>

#   Conclusion 

<h5>

To conclude, Goodreads Book Datasets 10M (subset book400k–500k.csv) is a perfect choice for logistic regression classification. It satisfies the minimum size requirements and it contain categorical and numerical predictors. It also demonstrates data challenges that are realistic that enables a meaningful data pre-processing and cleaning. Its quality makes it suitable for modelling the factors that influence whether a book becomes a bestseller which shows an excellent understanding of data selection, modelling as well as quality considerations. 

</h5>

