
# Dimensionality Reduction With PCA

## Objective

In this lab, we'll explore how we can use PCA to reduce the dimensionality of our dataset by dropping Principal Components with that don't explain much of the variance in our dataset. 

## 1. A Quick Recap

Let's go back to the food price data example that we used in the first lab. We pasted the necessary code below.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
% matplotlib inline
```

Now, read in the dataset stored in `foodusa.csv` and store it in a pandas DataFrame.  Be sure to set the `index_col` parameter to 0 for this dataset.  Then, display the head of the dataset to ensure everything loaded correctly. 


```python
data = None
```

In this lab, we'll be optimizing for using as little code as possible, so let's use scikit-learns `StandardScaler()` to get to the standardized data.  

In the cell below:

* Import `StandardScaler` from sklearn's `preprocessing` module
* Create a `StandardScaler()` object
* Use the scaler object's `fit_transform()` method on our `data` to create a scaled version of the dataset. 
* Store the transformed data in a new pandas DataFrame. 
* Our new DataFrame doesn't have column names, so fix this by setting `.columns` equal to a `list(data)`.


```python
from None import None
scaler = None
data_std = None
data_std = None
data_std.columns = None
```

Now, create a correlation matrix of our scaled DataFrame by using its `.corr()` method.  Then, display this correlation matrix in the cell under it. 


```python
corr_mat = None
```


```python
corr_mat
```

Now that we have a correlation matrix, we can easily compute the eigenvectors and eigenvalues for this dataset.  

In the cell below, use the correct function from numpy's `.linalg` module to get the eigenvalues and eigenvectors of the `corr_mat` variable.  Then, inspect the eigenvalues and eigenvectors in the following cells. 


```python
eig_values, eig_vectors = None
```


```python
eig_values
```


```python
eig_vectors
```

## 2. Visualizing how much information our PCs store.

Recall from the lecture how the sum of the eigenvalues

$$ \lambda_1 + \lambda_2 + \ldots + \lambda_5 $$ 

is equal to the sum of the variance of the variables, applied to this case:

$$ var(bread) + var(burger) + var(Milk) + var(Oranges) + var(Tomatoes)$$

with standardized data, we know that this sum is equal to the number of variables in the data set, which is 5 in this case. Now let's see if our eig_values add up to 5.

In the cell below, display the `sum()` of our eigenvalues. 

Looks great!

Now we go into the essentials of PCA. for each principal component (defined by the eigenvectors), the amount of variance represented in this PC is reflected by its respective eigenvalue. So what we'll want to do is *keep* PCs with a high eigenvalue, and drop the ones with a low eigenvalue.

In the cell below, get a sorted version of the `eig_values` array, sorted from highest to loweset (hint: use the `reverse` keyword!)


```python
eig_val_sorted = None
```

Now, let's `.plot()` our sorted array.


```python
plt.plot(eig_val_sorted)
```

We can plot a cumulative plot representing how much variance is explained with each new eigenvalue or principal components.

In the cell below:

* calculate the total of `eig_val_sorted`
* get the `variance_explained` by dividing `eig_val_sorted` by the `total` we computed.
* get the cumulative variance by using numpy's `.cumsum()` method and passing in `variance explained`.
* inspect our `cum_variance` array to ensure that it is monotonically increasing, and that the final value is 1. 


```python
total = None
variance_explained = None
cum_variance = None
cum_variance
```

Now, run the cell below to plot `cum_variance`, so that we can easily see how much of the variance is the data is explained with each additional principal component considered:


```python
index = None
plt.bar(index, cum_variance);
```

Let's say we decided to keep the the PCs with an eigenvalue bigger than 1. This is a popular decision rule, which in this case leads to having 2 principal components. Let's see how we can do that and how to interpret all this, but first, let's move over to scikit learn. This library has very easy-to-use PCA capabilities, and very easy to use. Let's look at how it's done!

## 3. PCA in scikit learn

Sklearn makes it simple to use PCA on our data! 

In the cell below:

* Import `PCA` from sklearn's `.decomposition` module. 
* Create a `PCA()` object and set `n_components` equal to 5 (the number of columns in our dataset).
* `fit()` our pca object on our scaled data, which we stored in `data_std`.


```python
from None import None
pca = None
pca.fit(None)
```

Now, get the eigenvalues from the `.explained_variance_` attribute.

Additionally, you can get the percentage of the variance explained by each component from the  `pca.explained_variance_ratio` attribute. Do this now in the cell below, and then inspect the array. 


```python
exp_var_ratio = None
```

And finally, let's get the cumulative sum by using the array's `.cumsum()` method.

Let's say we've decided to keep 2 Principal components. Conveniently, there is the arguments `n_components` in the `PCA` function. Additionally, as you saw, the eigenvalues are already sorted according to their size!

Run the cells below to get the first two principal components of our dataset using sklearn. 


```python
pca_2 = PCA(n_components=2)
pca_2.fit(data_std)
```


```python
eig_values = pca_2.explained_variance_
```


```python
eig_vectors = pca_2.components_
eig_vectors
```

As seen before, 70% of the variance in the data can still be explained by just having the 2 principal components that replace the 5 variables!

## 4. Relationship with the original variables

Let's store the loadings of the principal components in `pc1` and `pc2`.

The components themselves are stored sequentially in an array within the pca object's `components_` attribute. 

In the cell below, slice the first and second principal components and store them within the appropriate variables in the cell below.


```python
pc1 = None
pc2 = None
```

Now, let's get the structure loadings for each component, and store them in a pandas series, so that we can add our original labels back in. 

In the cell below:

* compute `structure_loading_1` by multiplying `pc1` times the square root of the corresponding eigenvalue, which can be found in `eig_values[0]`.  (hint: use `np.sqrt()` to make this easy!)
* Store `structure_loading_1` in a pandas Series, and set the `index` parameter to `data.columns`.
* Run the following cell to inspect the loadings for pc1 in descending order. 


```python
structure_loading_1 = None
str_loading_1 = None
```


```python
str_loading_1.sort_values(ascending=False)
```

Now, repeat the process to get the structure loadings for the second principal component.


```python
structure_loading_2 = None
str_loading_2 = None
```


```python
str_loading_2.sort_values(ascending=False)
```

**_NOTE:_** It doesn't really matter if a value is positive or negative--the magnitude is what matters when determining the importance.  Consequently, when we sort `str_loading_2` by value, we see "Oranges" at the very bottom.  although when we compare the magnitude of this to the others, we see that it's actually the most important category in this variable, not the least!

Let's see if we can interpret our principal components, and figure out what real-world things they correspond to!

Run the cell below to print the head of the `data` DataFrame again. 


```python
data.head()
```

Let's try to interpret what each principal component stands for. You can argue that PC1 represents bread and burgers, let's say "processed foods". PC2 represents oranges and milk, let's call that PC the "non-processed foods". Again, note that both strongly positive and strongly negative PCs are the most important--it's the strength, not the sign, that matters here!

# 5. The new data

What you'll usually want to do is replace your 5 columns with the new data. Now, what does the new data look like? You'll need to transform your columns using the loadings used in your eigenvectors. Let's try and find the PC1 value for Atlanta!

First, let's select the attribute values for Atlanta. We'll need to use the **standardized** values as this is the data we used to generate the PCs.


```python
atlanta_var = data_std.loc[0]
```


```python
atlanta_var
```

Now we have to multiply this with our loadings vectors! Let's do this for both PCs.

In the cell below, get the `np.sum()` of `atlanta_var` multiplied by `pc1`.

Now, do this again, but for `pc2`.

If we wanted to transform all of our data this way, we'd be here all day! Luckily, sklearn provides functionality that makes short work of this task. There is a method called `transform` in the PCA library that does this for all the observations in one step. 

Run the cells below to transform our city data!


```python
PC_df = pd.DataFrame(pca_2.transform(data_std), index=data.index, columns=['PC1','PC2'])
```


```python
PC_df
```

## Conclusion

You did it! PCA provides a great, intuitive way for us to transform our dataset into a format where the predictors are uncorrelated, as well as to reduce dimensionality by dropping less important components. 
