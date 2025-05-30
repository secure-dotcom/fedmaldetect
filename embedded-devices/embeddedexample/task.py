from collections import OrderedDict
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix

class FeatureLearningModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim):
        super(FeatureLearningModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.fc1 = nn.Linear(input_dim, 32)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32, proj_dim)

    def forward(self, x, compute_losses=False):
        x1 = self.fc1(x)
        x = F.leaky_relu(x1)
        # x = self.bn1(x)
        x = F.dropout(x, p=0.2)
        # x = F.leaky_relu(self.fc2(x))
        x = self.bn2(x)
        if compute_losses:
            projected_features = F.normalize(self.fc(x), p=2, dim=1)
            return x, projected_features
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(32, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.bn1(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.fc2(x)
        return x

class FEDMALDETECT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, global_feature_learning_model, classifier):
        super(FEDMALDETECT, self).__init__()
        self.global_feature_learning_model = global_feature_learning_model
        if classifier == None:
            self.classifier = Classifier(hidden_dim, hidden_dim, output_dim)
        else:
            self.classifier = classifier
    def forward(self, x):
        features = self.global_feature_learning_model(x)
        output = self.classifier(features)
        return output, self.classifier, features

def get_weights(featurelearningmodule):
    return [val.cpu().numpy() for _, val in featurelearningmodule.state_dict().items()]

def set_weights(featurelearningmodule, parameters):
    params_dict = zip(featurelearningmodule.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    featurelearningmodule.load_state_dict(state_dict, strict=True)

def preprocess(df):
    # scaler = QuantileTransformer()
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return df

def load_data_from_disk(path: str, batch_size: int, labeled_data: bool, test_required: bool):
    df = pd.read_csv(path, nrows=5000)
    if labeled_data:
        y = torch.tensor(df['label'].values, dtype=torch.long)
        X = preprocess(df.drop(columns=['label']))
        del df
        X = torch.tensor(X, dtype=torch.float32)
        if test_required:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, shuffle=True, random_state=110, stratify=y
            )
            train_dataset = TensorDataset(X_train, y_train)
            del X_train, y_train
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            del train_dataset
            return train_loader, X_test, y_test
        else:
            return X, y
    else:
        X = preprocess(df)
        del df
        X = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        return loader


def train(featurelearningmodule, trainloader, epochs, learning_rate):
    optimizer = optim.AdamW(featurelearningmodule.parameters(), lr=learning_rate, weight_decay=0.001)
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in trainloader:
            batch_data = batch[0]
            optimizer.zero_grad()        
            features, projected_features = featurelearningmodule(batch_data, compute_losses=True)
            corr_loss = correlation_loss(features)
            latent_loss = latent_space_equalization_loss(projected_features)
            loss = corr_loss + latent_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")
    end_time = time.time()
    training_time = end_time-start_time
    print(f"Training time: {training_time:.2f} seconds")

def server_finetuning(model, train_loader, X_test, y_test, epochs=30, lr=0.001):
    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=0.01)
    model.eval()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            batch_data, batch_labels = batch[0], batch[1]
            optimizer.zero_grad()
            output, classifier, _ = model(batch_data)
            batch_labels = F.one_hot(batch_labels, num_classes=2).float()
            loss = F.cross_entropy(output, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Fine-tuning Epoch {epoch}, Loss: {total_loss}")
    precision, recall, f1, roc_auc, pr_auc, cm = test(model, X_test, y_test)
    return classifier

def test(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs, _, features = model(X_test)
        probabilities = F.softmax(outputs, dim=-1)
        prob = probabilities[:, 1].numpy()
        y_true = y_test.numpy()
        y_pred = (prob > 0.5).astype(int)
        if len(set(y_true)) == 1:
            roc_auc = None
        else:
            roc_auc = roc_auc_score(y_true, prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, prob)
        pr_auc = auc(recall_curve, precision_curve)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'ROC AUC: {roc_auc}')
        print(f'PR AUC: {pr_auc:.4f}')
        print(f'Confusion Matrix:\n{cm}')
        return precision, recall,f1, roc_auc, pr_auc, cm.flatten().tolist()

def correlation_loss(features):
    batch_mean = torch.mean(features, dim=0, keepdim=True)
    batch_std = torch.std(features, dim=0, keepdim=True) + 1e-8
    normalized_features = (features - batch_mean) / batch_std
    cov_matrix = torch.matmul(normalized_features.T, normalized_features) / (features.size(0) - 1)
    off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
    loss = torch.sum(off_diag ** 2)
    return loss

def latent_space_equalization_loss(features):
    features = F.normalize(features, p=2, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    batch_size = features.size(0)
    probs = F.softmax(similarity_matrix / 0.1, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
    return -entropy

def feddyn(models, clients_data_len, global_model, beta):
    total_data = sum(clients_data_len)
    agg_sd = {
        k: torch.zeros_like(v, dtype=torch.float32)
        for k, v in global_model.state_dict().items()
    }
    dynamic_reg = {
        k: torch.zeros_like(v, dtype=torch.float32)
        for k, v in global_model.state_dict().items()
    }
    for model, n in zip(models, clients_data_len):
        w = n / total_data
        csd = model.state_dict()
        for k in agg_sd:
            agg_sd[k] += w * (csd[k].float() - beta * dynamic_reg[k].float())
    global_model.load_state_dict(agg_sd)
    return global_model
