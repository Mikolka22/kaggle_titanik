import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

def preprocess_data(train_raw, test_raw, config):
    train = train_raw.copy()
    test = test_raw.copy()

    train["Age"] = train.groupby(["Sex", "Pclass"])["Age"].transform(lambda x: x.fillna(x.median()))
    test["Age"] = test.groupby(["Sex", "Pclass"])["Age"].transform(lambda x: x.fillna(x.median()))
    test["Fare"] = test["Fare"].fillna(test["Fare"].median())
    train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])

    train = train.drop(columns=config['data']['drop_cols'])
    test = test.drop(columns=config['data']['drop_cols'])

    ohe_list = ["Sex", "Embarked", "Pclass"]
    ohe = OneHotEncoder(sparse_output=False, dtype=int)
    for col in ohe_list:
        train_enc = ohe.fit_transform(train[[col]])
        df_ohe_tr = pd.DataFrame(train_enc, columns=ohe.get_feature_names_out([col]), index=train.index)
        train = pd.concat([train.drop(columns=col), df_ohe_tr], axis=1)

        test_enc = ohe.transform(test[[col]])
        df_ohe_ts = pd.DataFrame(test_enc, columns=ohe.get_feature_names_out([col]), index=test.index)
        test = pd.concat([test.drop(columns=col), df_ohe_ts], axis=1)

    list_norm = ["Age", "SibSp", "Parch"]
    normal = MinMaxScaler()
    train[list_norm] = normal.fit_transform(train[list_norm])
    test[list_norm] = normal.transform(test[list_norm])

    stand = StandardScaler()
    train[["Fare"]] = stand.fit_transform(train[["Fare"]])
    test[["Fare"]] = stand.transform(test[["Fare"]])

    return train, test

def compute_accuracy(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()
        true_pred = (pred == y_val).sum().item()
        all_pred = y_val.shape[0]
    return true_pred / all_pred