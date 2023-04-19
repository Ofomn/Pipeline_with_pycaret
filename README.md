![banks](https://user-images.githubusercontent.com/122539866/232967760-98fa0fc5-f832-4d6a-b0c4-95731bcfacf6.jpg)

----
## About

The main goal of this dataset is to build a predictive model that can accurately predict whether a customer will subscribe to a term deposit or not, based on their historical data. This dataset is often used to showcase the various features and capabilities of PyCaret, including data preprocessing, model training and selection, hyperparameter tuning, and model interpretation.


## Creating a machine learning pipeline with Pycaret
I will be exploring the following steps to achieve a ML_pipeline using pycaret:

💡 Set up the environment, Get dataset and Check for nulls

```bash python
from pycaret.classification import *
from pycaret.datasets import get_data
import pandas as pd
```

```bash python
#get dataset
dataset = get_data('bank')

#check the shape of data
dataset.shape

dataset.isna().sum()


## sample returns a random sample from an axis of the object. That would be 38,429 samples, not 45211
data = dataset.sample(frac=0.85, random_state=456)

data

# remove from the original dataset this random data
data_unseen = dataset.drop(data.index)
data_unseen

# Reseting the index of both datasets
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

model_setup = setup(data=data, target='deposit', session_id=321)

```


💡 Compare Models

```bash python

```

💡 Create the Model

```bash python

```

💡 Tune the Model

```bash python

```

💡 Plot the Model


```bash python

```

💡 Evaluate the model

```bash python

```

💡 Finalize the Model

```bash python

```

💡 Predict with the model

```bash python

```

💡 Save/Load Model


```bash python

```



-----

## Things to Note
To be able to run this analysis, you will need the following:

- Jupyter Notebook
- Pycaret 3.0
- Python 3.x


## Installation
To install Pycaret 3.0, run the following command:

```bash python
!pip install pycaret[full]
```
-------

## Dataset
The dataset used was imported from pycaret using the code below

```bash python
from pycaret.classification import *
from pycaret.datasets import get_data
dataset = get_data('bank')
```
------

 ## License 
 Licensed under GPL-3.0 license


© Ofomnbuk 2023 🇨🇦😘🇳🇬


