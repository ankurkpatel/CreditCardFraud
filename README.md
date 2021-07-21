# Credit Card Fraud

Dataset is imbalanced as there is only 0.17% of transactions are marked as fraud. Even if we blindly assign all transcations as not fraud still we will be right 99.93% of the times. Hence, accuracy score is not useful here. As false positive and false negative ones we care about.

So we need sample that represent both the class equally so our model can have enough data to learn from and generalize for unseen data.

How do we create subsample that has 50/50 representation of both classes? Here are some sampling techniques.

- SMOTE 
    - KNN 
- sklearn.resample (upsample and downsample)
    - avoid mistake when upsample or downsample 
        - do only after train-test-split 

- Outlier Sensitivity

- GridSearchCV

