import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from utils import preprocess_data, compute_accuracy

warnings.filterwarnings("ignore", message="X does not have valid feature names")

results = {}
def train_model(model, X_train, y_train, X_val, y_val, optimizer, loss_fn, config, scheduler=None):
    n_epoch = config['dl_model']['n_epoch']
    batch_size = config['dl_model']['batch_size']
    
    for epoch in range(n_epoch):
        model.train()
        loss_history = []
        tp = 0  
        all_p = 0  
        indices = torch.randperm(X_train.shape[0])
        for start in range(0, X_train.shape[0], batch_size):
            batch_idx = indices[start:start+batch_size]
            x_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            optimizer.zero_grad()
            logits = model(x_batch)
            loss_val = loss_fn(logits, y_batch)
            loss_val.backward()
            optimizer.step()
            
            loss_history.append(loss_val.item())
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()
            tp += (pred == y_batch).sum().item()
            all_p += y_batch.shape[0]
            
        if scheduler is not None:
            scheduler.step()
        
        ave_loss = np.mean(loss_history)
        train_acc = tp / all_p
        val_acc = compute_accuracy(model, X_val, y_val)
        if epoch == n_epoch -1: 
            results["Dl model"] = np.round(val_acc, 5)
        print(f"Epoch {epoch}: Loss = {ave_loss:.4f} | Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f}")

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_raw = pd.read_csv(config['data']['train_path'])
    test_raw = pd.read_csv(config['data']['test_path'])

    train_p, test_p = preprocess_data(train_raw, test_raw, config)
    
    X_np = train_p.drop(columns=[config['data']['target'], "PassengerId"]).values
    y_np = train_p[config['data']['target']].values
    X_test_np = test_p.drop(columns=["PassengerId"]).values

    models = [
        ("LogReg Basic", LogisticRegression(**config['models']['logistic_regression'])),
        ("LogReg L2", LogisticRegression(**config['models']['logistic_regression_l2'])),
        ("LogReg L1", LogisticRegression(**config['models']['logistic_regression_l1'])),
        ("LogReg Elastic", LogisticRegression(**config['models']['logistic_regression_elastic'])),
        ("Decision Tree", DecisionTreeClassifier(**config['models']['decision_tree'])),
        ("Random Forest", RandomForestClassifier(**config['models']['random_forest'])),
        ("XGBoost", XGBClassifier(**config['models']['xgboost'])),
        ("LightGBM", LGBMClassifier(**config['models']['lightgbm'])),
        ("CatBoost", CatBoostClassifier(**config['models']['catboost']))
    ]

    sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)
    print("Оценка классических моделей...")
    for name, model in models:
        scores = []
        for t_idx, v_idx in sk.split(X_np, y_np):
            if name in ["XGBoost", "LightGBM", "CatBoost"]:
                model.fit(X_np[t_idx], y_np[t_idx], eval_set=[(X_np[v_idx], y_np[v_idx])])
            else:
                model.fit(X_np[t_idx], y_np[t_idx])
            probs = model.predict_proba(X_np[v_idx])[:, 1]
            scores.append(roc_auc_score(y_np[v_idx], probs))
        
        results[name] = np.round(np.mean(scores), 5)
        # print(f"{name}: ROC-AUC = {results[name]}")

    # Deep Learning
    print("\nОбучение PyTorch модели...")
    X_tr, X_val, y_tr, y_val = train_test_split(X_np, y_np, test_size=0.2, random_state=21)
    
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    dl_model = nn.Sequential(
        nn.Linear(X_np.shape[1], 10),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    
    optimizer = optim.Adam(dl_model.parameters(), lr=config['dl_model']['lr'], weight_decay=config['dl_model']['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss()
    
    train_model(dl_model, X_tr_t, y_tr_t, X_val_t, y_val_t, optimizer, loss_fn, config)

    print(results)
if __name__ == "__main__":
    main()