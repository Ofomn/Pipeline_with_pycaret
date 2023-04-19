![banks](https://user-images.githubusercontent.com/122539866/232967760-98fa0fc5-f832-4d6a-b0c4-95731bcfacf6.jpg)

----
## About

The main goal of this dataset is to build a predictive model that can accurately predict whether a customer will subscribe to a term deposit or not, based on their historical data. This dataset is often used to showcase the various features and capabilities of PyCaret, including data preprocessing, model training and selection, hyperparameter tuning, and model interpretation.


## Dataset
The dataset used was imported from pycaret using the code below

```bash python
from pycaret.classification import *
from pycaret.datasets import get_data
dataset = get_data('bank')
```
------


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

#Compare Model
best_model = compare_models()

print(best_model)

```

💡 Create the Model

```bash python

models()

gbc = create_model('gbc')

#trained model object is stored in the variable 'gbc'. 
print(gbc)

```

💡 Tune the Model

```bash python

# Accuracy in previous model is 0.9056 and in the tuned model accuracy is 0.9065
tuned_gbc = tune_model(gbc)

print(tuned_gbc)

```

💡 Plot the Model


```bash python

## AUC Plot
plot_model(tuned_gbc, plot = 'auc')


## Consufion matrix
plot_model(tuned_gbc, plot = 'confusion_matrix')

```

💡 Evaluate the model

```bash python

## model performance is to use the evaluate_model()
evaluate_model(tuned_gbc)

```

💡 Finalize the Model

```bash python

final_gbc = finalize_model(tuned_gbc)
final_gbc

#Final gbc parameters for deployment
print(final_gbc)

```

💡 Predict with the model

```bash python

predict_model(final_gbc)

unseen_predictions = predict_model(final_gbc, data=data_unseen)
unseen_predictions.head()

```

💡 Save/Load Model


```bash python

save_model(final_gbc, './Pycaret/Final_gbc')

saved_final_gbc = load_model('./Pycaret/Final_gbc')

new_prediction = predict_model(saved_final_gbc, data=data_unseen)

new_prediction.head()

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

## [Click To View My Notebook](https://nbviewer.org./github/Ofomn/Pipeline_with_pycaret/blob/e5e4231dc03814e6b4fa583c765681f07b603edd/Pipeline_with_pycarat.ipynb)

-----


 ## License 
 Licensed under GPL-3.0 license


© Ofomnbuk 2023 🇨🇦😘🇳🇬


