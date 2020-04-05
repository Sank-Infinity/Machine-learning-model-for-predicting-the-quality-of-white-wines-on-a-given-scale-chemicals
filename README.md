# Machine-learning-model-for-predicting-the-quality-of-white-wines-on-a-given-scale-chemicals

- Main agenda of this machine learning model is to predict what should be the ideal proportion of different ingredients to make better quality of white wine.
- On the basis of proportion of various ingredients we have to predict quality of wine in range of 0 to 10 with the help of given dataset.
- In the given dataset we have near about 5000 observations which are more than enough for train the model.
- For this problem we are going to use Random Forest regressor algorithm.
- This is very popular and efficient model with high dimentional dataset.
- Sometimes it overfits the data which is its drawback overall it's perfect.

# Overview of the Dataset 

- In the dataset total 12 variables out of which 11 are independent variables and last one is dependent variable.

1.Fixed acidity.
2.Volatile acidity.
3.Citric acid.
4.Residual sugar.
5.Chlorides.
6.Free sulfur dioxide.
7.Total sulfur dioxide.
8.Density.
9.pH.
10.Sulphates.
11.Alcohol.
12.Quality (score between 0 and 10).

# Overview of the process

- First of all we will Import required libraries and dataset.
- Split dataset into training set and test set.
- Import Random Forest Regressor.
- predict results.
- Applying Backward Elimination method for removing independent variables which are not affecting on results.
- Predict optimal results.

