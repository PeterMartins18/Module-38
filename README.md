# Module-38

Credit Scoring for Credit Card Application

webrender: https://module-38-5.onrender.com
Video: https://github.com/PeterMartins18/Module-38/blob/main/Screen%20Recording%202024-06-24%20at%2018.31.43.mov



# Project Overview

This project aims to build a credit scoring model for credit card applications using a dataset with 15 cohorts, utilizing 12 months of performance data. The primary objective is to predict the likelihood of default (inadimplência) using various explanatory variables.

## Steps and Methodology

### Data Loading
- Load the dataset `credit_scoring.ftr`.

### Sampling
- Separate the last three months as out-of-time (oot) validation cohorts.

### Variables
- `data_ref`: Indicator of the cohort, not to be used as an explanatory variable.
- `index`: Customer identifier, not to be used as an explanatory variable.
- Remaining variables: Can be used to predict default, including income.

### Univariate Descriptive Analysis
- Describe the dataset in terms of the number of rows and rows for each month in `data_ref`.
- Provide a basic univariate descriptive analysis of each variable, considering both qualitative and quantitative natures.

### Bivariate Descriptive Analysis
- Perform a bivariate descriptive analysis of each variable.

### Model Development
- Develop a credit scoring model using logistic regression.
- Handle missing values and outliers.
- Address 'structural zeros'.
- Group categories as discussed in class.
- Propose a predictive equation for 'default'.
- Justify any non-significant categories.

### Model Evaluation
- Evaluate the model's discriminative power using accuracy, KS, and Gini metrics.
- Assess these metrics on both the development and out-of-time datasets.

### Pipeline Creation
- Create a pipeline using `sklearn.pipeline`.

## Preprocessing Steps

### Handling Missing Values
- Identify if there are missing values in the dataset. Determine if they are numeric or categorical and choose an appropriate replacement method (mean, mode, etc.).

### Outlier Removal
- Identify and address outliers. Decide whether to replace outliers with a specific value or remove the row.

### Variable Selection
- Select variables using techniques such as Boruta or feature importance.

### Dimensionality Reduction
- Apply suitable methods to reduce dimensionality if necessary.

