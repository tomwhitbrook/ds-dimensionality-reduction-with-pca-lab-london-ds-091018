
# Dimensionality Reduction With PCA

## Objective

In this lab, we'll explore how we can use PCA to reduce the dimensionality of our dataset by dropping Principal Components with that don't explain much of the variance in our dataset. 

## 1. A Quick Recap

Let's go back to the food price data example that we used in the first lab. We pasted the necessary code below.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline
```

Now, read in the dataset stored in `foodusa.csv` and store it in a pandas DataFrame.  Be sure to set the `index_col` parameter to 0 for this dataset.  Then, display the head of the dataset to ensure everything loaded correctly. 


```python
data = pd.read_csv('foodusa.csv', index_col=0)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bread</th>
      <th>Burger</th>
      <th>Milk</th>
      <th>Oranges</th>
      <th>Tomatoes</th>
    </tr>
    <tr>
      <th>City</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ATLANTA</th>
      <td>24.5</td>
      <td>94.5</td>
      <td>73.9</td>
      <td>80.1</td>
      <td>41.6</td>
    </tr>
    <tr>
      <th>BALTIMORE</th>
      <td>26.5</td>
      <td>91.0</td>
      <td>67.5</td>
      <td>74.6</td>
      <td>53.3</td>
    </tr>
    <tr>
      <th>BOSTON</th>
      <td>29.7</td>
      <td>100.8</td>
      <td>61.4</td>
      <td>104.0</td>
      <td>59.6</td>
    </tr>
    <tr>
      <th>BUFFALO</th>
      <td>22.8</td>
      <td>86.6</td>
      <td>65.3</td>
      <td>118.4</td>
      <td>51.2</td>
    </tr>
    <tr>
      <th>CHICAGO</th>
      <td>26.7</td>
      <td>86.7</td>
      <td>62.7</td>
      <td>105.9</td>
      <td>51.2</td>
    </tr>
  </tbody>
</table>
</div>



In this lab, we'll be optimizing for using as little code as possible, so let's use scikit-learns `StandardScaler()` to get to the standardized data.  

In the cell below:

* Import `StandardScaler` from sklearn's `preprocessing` module
* Create a `StandardScaler()` object
* Use the scaler object's `fit_transform()` method on our `data` to create a scaled version of the dataset. 
* Store the transformed data in a new pandas DataFrame. 
* Our new DataFrame doesn't have column names, so fix this by setting `.columns` equal to a `list(data)`.


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_std = scaler.fit_transform(data)
data_std = pd.DataFrame(data_std)
data_std.columns = list(data)
```

Now, create a correlation matrix of our scaled DataFrame by using its `.corr()` method.  Then, display this correlation matrix in the cell under it. 


```python
corr_mat = data_std.corr()
```


```python
corr_mat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bread</th>
      <th>Burger</th>
      <th>Milk</th>
      <th>Oranges</th>
      <th>Tomatoes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bread</th>
      <td>1.000000</td>
      <td>0.681700</td>
      <td>0.328239</td>
      <td>0.036709</td>
      <td>0.382241</td>
    </tr>
    <tr>
      <th>Burger</th>
      <td>0.681700</td>
      <td>1.000000</td>
      <td>0.333422</td>
      <td>0.210937</td>
      <td>0.631898</td>
    </tr>
    <tr>
      <th>Milk</th>
      <td>0.328239</td>
      <td>0.333422</td>
      <td>1.000000</td>
      <td>-0.002779</td>
      <td>0.254417</td>
    </tr>
    <tr>
      <th>Oranges</th>
      <td>0.036709</td>
      <td>0.210937</td>
      <td>-0.002779</td>
      <td>1.000000</td>
      <td>0.358061</td>
    </tr>
    <tr>
      <th>Tomatoes</th>
      <td>0.382241</td>
      <td>0.631898</td>
      <td>0.254417</td>
      <td>0.358061</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have a correlation matrix, we can easily compute the eigenvectors and eigenvalues for this dataset.  

In the cell below, use the correct function from numpy's `.linalg` module to get the eigenvalues and eigenvectors of the `corr_mat` variable.  Then, inspect the eigenvalues and eigenvectors in the following cells. 


```python
eig_values, eig_vectors = np.linalg.eig(corr_mat)
```


```python
eig_values
```




    array([2.42246795, 1.10467489, 0.2407653 , 0.73848053, 0.49361132])




```python
eig_vectors
```




    array([[ 0.49614868,  0.30861972,  0.49989887,  0.38639398, -0.50930459],
           [ 0.57570231,  0.04380176, -0.77263501,  0.26247227,  0.02813712],
           [ 0.33956956,  0.43080905, -0.00788224, -0.83463952, -0.0491    ],
           [ 0.22498981, -0.79677694,  0.0059668 , -0.29160659, -0.47901574],
           [ 0.50643404, -0.28702846,  0.39120139,  0.01226602,  0.71270629]])



## 2. Visualizing how much information our PCs store.

Recall from the lecture how the sum of the eigenvalues

$$ \lambda_1 + \lambda_2 + \ldots + \lambda_5 $$ 

is equal to the sum of the variance of the variables, applied to this case:

$$ var(bread) + var(burger) + var(Milk) + var(Oranges) + var(Tomatoes)$$

with standardized data, we know that this sum is equal to the number of variables in the data set, which is 5 in this case. Now let's see if our eig_values add up to 5.

In the cell below, display the `sum()` of our eigenvalues. 


```python
eig_values.sum()
```




    4.999999999999999



Looks great!

Now we go into the essentials of PCA. for each principal component (defined by the eigenvectors), the amount of variance represented in this PC is reflected by its respective eigenvalue. So what we'll want to do is *keep* PCs with a high eigenvalue, and drop the ones with a low eigenvalue.

In the cell below, get a sorted version of the `eig_values` array, sorted from highest to loweset (hint: use the `reverse` keyword!)


```python
eig_val_sorted = sorted(eig_values, reverse=True)
```

Now, let's `.plot()` our sorted array.


```python
plt.plot(eig_val_sorted);
```


![png](index_files/index_26_0.png)


We can plot a cumulative plot representing how much variance is explained with each new eigenvalue or principal components.

In the cell below:

* calculate the total of `eig_val_sorted`
* get the `variance_explained` by dividing `eig_val_sorted` by the `total` we computed.
* get the cumulative variance by using numpy's `.cumsum()` method and passing in `variance explained`.
* inspect our `cum_variance` array to ensure that it is monotonically increasing, and that the final value is 1. 


```python
total = sum(eig_val_sorted)
variance_explained = eig_val_sorted / total
cum_variance = np.cumsum(variance_explained)
cum_variance
```




    array([0.48449359, 0.70542857, 0.85312468, 0.95184694, 1.        ])



Now, run the cell below to plot `cum_variance`, so that we can easily see how much of the variance is the data is explained with each additional principal component considered:


```python
index = np.arange(len(cum_variance))+1
plt.bar(index, cum_variance);
```


![png](index_files/index_30_0.png)


Let's say we decided to keep the the PCs with an eigenvalue bigger than 1. This is a popular decision rule, which in this case leads to having 2 principal components. Let's see how we can do that and how to interpret all this, but first, let's move over to scikit learn. This library has very easy-to-use PCA capabilities, and very easy to use. Let's look at how it's done!

## 3. PCA in scikit learn

Sklearn makes it simple to use PCA on our data! 

In the cell below:

* Import `PCA` from sklearn's `.decomposition` module. 
* Create a `PCA()` object and set `n_components` equal to 5 (the number of columns in our dataset).
* `fit()` our pca object on our scaled data, which we stored in `data_std`.


```python
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(data_std)
```




    PCA(copy=True, iterated_power='auto', n_components=5, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)



Now, get the eigenvalues from the `.explained_variance_` attribute.


```python
pca.explained_variance_
```




    array([2.53258013, 1.15488739, 0.77204783, 0.5160482 , 0.25170918])



Additionally, you can get the percentage of the variance explained by each component from the  `pca.explained_variance_ratio` attribute. Do this now in the cell below, and then inspect the array. 


```python
exp_var_ratio = pca.explained_variance_ratio_
exp_var_ratio
```




    array([0.48449359, 0.22093498, 0.14769611, 0.09872226, 0.04815306])



And finally, let's get the cumulative sum by using the array's `.cumsum()` method.


```python
exp_var_ratio.cumsum()
```




    array([0.48449359, 0.70542857, 0.85312468, 0.95184694, 1.        ])



Let's say we've decided to keep 2 Principal components. Conveniently, there is the arguments `n_components` in the `PCA` function. Additionally, as you saw, the eigenvalues are already sorted according to their size!

Run the cells below to get the first two principal components of our dataset using sklearn. 


```python
from sklearn.decomposition import PCA
pca_2 = PCA(n_components=2)
pca_2.fit(data_std)
```




    PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)




```python
eig_values = pca_2.explained_variance_
```


```python
eig_vectors = pca_2.components_
eig_vectors
```




    array([[ 0.49614868,  0.57570231,  0.33956956,  0.22498981,  0.50643404],
           [ 0.30861972,  0.04380176,  0.43080905, -0.79677694, -0.28702846]])



As seen before, 70% of the variance in the data can still be explained by just having the 2 principal components that replace the 5 variables!

## 4. Relationship with the original variables

Let's store the loadings of the principal components in `pc1` and `pc2`.

The components themselves are stored sequentially in an array within the pca object's `components_` attribute. 

In the cell below, slice the first and second principal components and store them within the appropriate variables in the cell below.


```python
pc1 = pca_2.components_[0]
pc2 = pca_2.components_[1]
```

Now, let's get the structure loadings for each component, and store them in a pandas series, so that we can add our original labels back in. 

In the cell below:

* compute `structure_loading_1` by multiplying `pc1` times the square root of the corresponding eigenvalue, which can be found in `eig_values[0]`.  (hint: use `np.sqrt()` to make this easy!)
* Store `structure_loading_1` in a pandas Series, and set the `index` parameter to `data.columns`.
* Run the following cell to inspect the loadings for pc1 in descending order. 


```python
structure_loading_1 = pc1* np.sqrt(eig_values[0])
str_loading_1 = pd.Series(structure_loading_1, index=data.columns)
```


```python
str_loading_1.sort_values(ascending=False)
```




    Burger      0.916177
    Tomatoes    0.805943
    Bread       0.789575
    Milk        0.540394
    Oranges     0.358051
    dtype: float64



Now, repeat the process to get the structure loadings for the second principal component.


```python
structure_loading_2 = pc2* np.sqrt(eig_values[1])
str_loading_2 = pd.Series(structure_loading_2, index=data.columns)
```


```python
str_loading_2.sort_values(ascending=False)
```




    Milk        0.462972
    Bread       0.331660
    Burger      0.047072
    Tomatoes   -0.308457
    Oranges    -0.856262
    dtype: float64



**_NOTE:_** It doesn't really matter if a value is positive or negative--the magnitude is what matters when determining the importance.  Consequently, when we sort `str_loading_2` by value, we see "Oranges" at the very bottom.  although when we compare the magnitude of this to the others, we see that it's actually the most important category in this variable, not the least!

Let's see if we can interpret our principal components, and figure out what real-world things they correspond to!

Run the cell below to print the head of the `data` DataFrame again. 


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bread</th>
      <th>Burger</th>
      <th>Milk</th>
      <th>Oranges</th>
      <th>Tomatoes</th>
    </tr>
    <tr>
      <th>City</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ATLANTA</th>
      <td>24.5</td>
      <td>94.5</td>
      <td>73.9</td>
      <td>80.1</td>
      <td>41.6</td>
    </tr>
    <tr>
      <th>BALTIMORE</th>
      <td>26.5</td>
      <td>91.0</td>
      <td>67.5</td>
      <td>74.6</td>
      <td>53.3</td>
    </tr>
    <tr>
      <th>BOSTON</th>
      <td>29.7</td>
      <td>100.8</td>
      <td>61.4</td>
      <td>104.0</td>
      <td>59.6</td>
    </tr>
    <tr>
      <th>BUFFALO</th>
      <td>22.8</td>
      <td>86.6</td>
      <td>65.3</td>
      <td>118.4</td>
      <td>51.2</td>
    </tr>
    <tr>
      <th>CHICAGO</th>
      <td>26.7</td>
      <td>86.7</td>
      <td>62.7</td>
      <td>105.9</td>
      <td>51.2</td>
    </tr>
  </tbody>
</table>
</div>



Let's try to interpret what each principal component stands for. You can argue that PC1 predominantly represents bread, tomatoes and burgers, while PC2 is mainly influenced by (and is therefore representative of) oranges and milk. Again, note that both strongly positive and strongly negative PCs are the most important--it's the strength, not the sign, that matters here!

# 5. The new data

What you'll usually want to do is replace your 5 columns with the new data. Now, what does the new data look like? You'll need to transform your columns using the loadings used in your eigenvectors. Let's try and find the PC1 value for Atlanta!

First, let's select the attribute values for Atlanta. We'll need to use the **standardized** values as this is the data we used to generate the PCs.


```python
atlanta_var = data_std.loc[0]
```


```python
atlanta_var
```




    Bread      -0.322747
    Burger      0.357765
    Milk        1.707156
    Oranges    -1.643751
    Tomatoes   -0.963643
    Name: 0, dtype: float64



Now we have to multiply this with our loadings vectors! Let's do this for both PCs.

In the cell below, get the `np.sum()` of `atlanta_var` multiplied by `pc1`.


```python
np.sum(atlanta_var * pc1)
```




    -0.2323147488693299



Now, do this again, but for `pc2`.


```python
np.sum(atlanta_var * pc2)
```




    2.2378185664644548



If we wanted to transform all of our data this way, we'd be here all day! Luckily, sklearn provides functionality that makes short work of this task. There is a method called `transform` in the PCA library that does this for all the observations in one step. 

Run the cells below to transform our city data!


```python
PC_df = pd.DataFrame(pca_2.transform(data_std), index=data.index, columns=['PC1','PC2'])
```


```python
PC_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
    </tr>
    <tr>
      <th>City</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ATLANTA</th>
      <td>-0.232315</td>
      <td>2.237819</td>
    </tr>
    <tr>
      <th>BALTIMORE</th>
      <td>0.288023</td>
      <td>1.926235</td>
    </tr>
    <tr>
      <th>BOSTON</th>
      <td>2.298492</td>
      <td>0.075243</td>
    </tr>
    <tr>
      <th>BUFFALO</th>
      <td>-0.348852</td>
      <td>-1.129927</td>
    </tr>
    <tr>
      <th>CHICAGO</th>
      <td>0.116322</td>
      <td>-0.088027</td>
    </tr>
    <tr>
      <th>CINCINNATI</th>
      <td>0.605998</td>
      <td>0.461222</td>
    </tr>
    <tr>
      <th>CLEVELAND</th>
      <td>-1.242714</td>
      <td>-1.335505</td>
    </tr>
    <tr>
      <th>DALLAS</th>
      <td>-1.121562</td>
      <td>-0.859501</td>
    </tr>
    <tr>
      <th>DETROIT</th>
      <td>-0.280792</td>
      <td>-1.347375</td>
    </tr>
    <tr>
      <th>HONALULU</th>
      <td>4.168851</td>
      <td>-0.505083</td>
    </tr>
    <tr>
      <th>HOUSTON</th>
      <td>-1.316582</td>
      <td>-0.151809</td>
    </tr>
    <tr>
      <th>KANSAS CITY</th>
      <td>-0.324461</td>
      <td>0.615497</td>
    </tr>
    <tr>
      <th>LOS ANGELES</th>
      <td>-1.211956</td>
      <td>1.362077</td>
    </tr>
    <tr>
      <th>MILWAUKEE</th>
      <td>-1.118232</td>
      <td>-1.882291</td>
    </tr>
    <tr>
      <th>MINNEAPOLIS</th>
      <td>-0.452064</td>
      <td>-0.990662</td>
    </tr>
    <tr>
      <th>NEW YORK</th>
      <td>3.779884</td>
      <td>0.259320</td>
    </tr>
    <tr>
      <th>PHILADELPHIA</th>
      <td>0.894790</td>
      <td>-0.031577</td>
    </tr>
    <tr>
      <th>PITTSBURGH</th>
      <td>0.619647</td>
      <td>-0.825288</td>
    </tr>
    <tr>
      <th>ST LOUIS</th>
      <td>0.233133</td>
      <td>-0.533188</td>
    </tr>
    <tr>
      <th>SAN DIEGO</th>
      <td>-1.932777</td>
      <td>0.741302</td>
    </tr>
    <tr>
      <th>SAN FRANCISCO</th>
      <td>-0.880164</td>
      <td>0.194150</td>
    </tr>
    <tr>
      <th>SEATTLE</th>
      <td>-2.137999</td>
      <td>0.375538</td>
    </tr>
    <tr>
      <th>WASHINGTON DC</th>
      <td>-0.404672</td>
      <td>1.431830</td>
    </tr>
  </tbody>
</table>
</div>



## Conclusion

You did it! PCA provides a great, intuitive way for us to transform our dataset into a format where the predictors are uncorrelated, as well as to reduce dimensionality by dropping less important components. 
