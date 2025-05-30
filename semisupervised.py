import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch import tensor
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix
import torch.nn.functional as F
import os
import copy
import csv
from rich import box
from rich.table import Table
from rich.console import Console
import argparse
import glob
import warnings
warnings.filterwarnings('ignore')

random_no = 110
np.random.seed(random_no)
random.seed(random_no)
torch.manual_seed(random_no)
torch.cuda.manual_seed(random_no)
torch.cuda.manual_seed_all(random_no)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

all_metrics = []
# FLM model params
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.001
OPTIMIZER = 'AdamW'
LOSS = 'LatentSpaceEqualizationLoss & Correlation Loss'

# FCL params
CLASSIFIER_EPOCHS = 50
CLASSIFIER_BATCH_SIZE = 512
CLASSIFIER_LEARNING_RATE = 0.001
CLASSIFIER_WEIGHT_DECAY = 0.01
CLASSIFIER_CRITERION = 'CrossEntropyLoss'
CLASSIFIER_OPTIMIZER = 'AdamW'

# fl params
NUM_CLIENTS = 10
NUM_ROUNDS = 20
AGGREGATION = 'FedDyn'
PERCENTAGE = 50 #percentage of clients participation in training

TRAINING_TYPE = 'Decentralized'
SAVE = False
DATASET = "nabiot"
parser = argparse.ArgumentParser()
parser.add_argument('--training_type', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--optimizer', type=str)

parser.add_argument('--classifier_epochs', type=int)
parser.add_argument('--classifier_batch_size', type=int)
parser.add_argument('--classifier_lr', type=float)
parser.add_argument('--classifier_weight_decay', type=float)
parser.add_argument('--classifier_optimizer', type=str)

parser.add_argument('--num_clients', type=int)
parser.add_argument('--num_rounds', type=int)
parser.add_argument('--aggregation', type=str)
parser.add_argument('--percentage', type=float)

parser.add_argument('--save', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--gpu', type=int, help='GPU device index')
args, _ = parser.parse_known_args()

gpu_index = args.gpu if args.gpu is not None else 0
device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")

if args.training_type is not None: TRAINING_TYPE = args.training_type
if args.epochs is not None: EPOCHS = args.epochs
if args.batch_size is not None: BATCH_SIZE = args.batch_size
if args.lr is not None: LEARNING_RATE = args.lr
if args.weight_decay is not None: WEIGHT_DECAY = args.weight_decay
if args.optimizer is not None: OPTIMIZER = args.optimizer

if args.classifier_epochs is not None: CLASSIFIER_EPOCHS = args.classifier_epochs
if args.classifier_batch_size is not None: CLASSIFIER_BATCH_SIZE = args.classifier_batch_size
if args.classifier_lr is not None: CLASSIFIER_LEARNING_RATE = args.classifier_lr
if args.classifier_weight_decay is not None: CLASSIFIER_WEIGHT_DECAY = args.classifier_weight_decay
if args.classifier_optimizer is not None: CLASSIFIER_OPTIMIZER = args.classifier_optimizer

if args.num_clients is not None: NUM_CLIENTS = args.num_clients
if args.num_rounds is not None: NUM_ROUNDS = args.num_rounds
if args.aggregation is not None: AGGREGATION = args.aggregation
if args.percentage is not None: PERCENTAGE = args.percentage
if args.save is not None: SAVE = args.save
if args.dataset is not None: DATASET = args.dataset

print(f"Using device: {device}")

def clean_filename_component(value):
    if value == 'CrossEntropyLoss':
        value = 'CE'
    elif value == 'LatentSpaceEqualizationLoss & Correlation Loss':
        value = 'Lse_CorrL'
    return value

if TRAINING_TYPE == "Decentralized":
    filename = (
        f"result_"
        f"{EPOCHS}_{BATCH_SIZE}_{str(LEARNING_RATE).replace('.', '')}_"
        f"{str(WEIGHT_DECAY).replace('.', '')}_{OPTIMIZER}_"
        f"{clean_filename_component(LOSS)}_"
        f"{CLASSIFIER_EPOCHS}_{CLASSIFIER_BATCH_SIZE}_{str(CLASSIFIER_LEARNING_RATE).replace('.', '')}_"
        f"{str(CLASSIFIER_WEIGHT_DECAY).replace('.', '')}_{CLASSIFIER_OPTIMIZER}_"
        f"{clean_filename_component(CLASSIFIER_CRITERION)}_"
        f"{NUM_CLIENTS}_{NUM_ROUNDS}_{AGGREGATION}_{PERCENTAGE}_{DATASET}_{str(SAVE)}.csv"
    )
else:
    DATASET = "nabiot"
    filename = (
        f"result_"
        f"{EPOCHS}_{BATCH_SIZE}_{str(LEARNING_RATE).replace('.', '')}_"
        f"{str(WEIGHT_DECAY).replace('.', '')}_{OPTIMIZER}_"
        f"{clean_filename_component(LOSS)}_"
        f"{CLASSIFIER_EPOCHS}_{CLASSIFIER_BATCH_SIZE}_{str(CLASSIFIER_LEARNING_RATE).replace('.', '')}_"
        f"{str(CLASSIFIER_WEIGHT_DECAY).replace('.', '')}_{CLASSIFIER_OPTIMIZER}_"
        f"{clean_filename_component(CLASSIFIER_CRITERION)}_.csv"
    )
if DATASET == "nabiot":
    PARTITIONED_DATA_DIRECTORY = "./nonIID-50-Client_Data/"
elif DATASET == "radar":
    PARTITIONED_DATA_DIRECTORY = './radar/partitioned_data/'
elif DATASET == "iot23":
    PARTITIONED_DATA_DIRECTORY = './iot-23/partitioned_data/'

console = Console()

class Data_Loader:
    def __init__(self, data_directory, console, partitioned):
        self.console = console
        self.partitioned = partitioned
        if partitioned:
            self.data_directory = data_directory
        else:
            self.loaded_files = set()
            self.files = []
            for root, _, filenames in os.walk(data_directory):
                for filename in filenames:
                    self.files.append(os.path.join(root, filename))

    def read_partitions_to_dataframe(self, file_name):
        client_files = glob.glob(os.path.join(self.data_directory, '**', f'{file_name}.csv'), recursive=True)
        return pd.concat([pd.read_csv(file, nrows=50000) for file in client_files], ignore_index=True)

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim):
        super(FeatureExtractor, self).__init__()
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

class FeatureDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim):
        super(FeatureDecoder, self).__init__()
        # self.fc1 = nn.Linear(proj_dim, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32, input_dim)

    def forward(self, x):
        # print(x.shape)
        # x = self.fc1(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.fc(x)
        return x

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
        return self.fc2(x)

class FEDMALDETECT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, global_client, global_classifier):
        super(FEDMALDETECT, self).__init__()
        self.feature_extractor = global_client
        if global_classifier == None:
            self.classifier = Classifier(hidden_dim, hidden_dim, output_dim)
        else:
            self.classifier = global_classifier
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features), self.classifier, features

def client_training(model, data, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LEARNING_RATE):  
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    model.train()
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch_data = batch[0].to(device)
            optimizer.zero_grad()        
            features, projected_features = model(batch_data, compute_losses=True)
            corr_loss = correlation_loss(features)
            latent_loss = latent_space_equalization_loss(projected_features)
            loss = abs(latent_loss) + corr_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")
    return model

def server_finetuning(model, data, labels, batch_size=CLASSIFIER_BATCH_SIZE, epochs=CLASSIFIER_EPOCHS, lr=CLASSIFIER_LEARNING_RATE):
    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=CLASSIFIER_WEIGHT_DECAY)
    model.eval()
    dataset = torch.utils.data.TensorDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch_data, batch_labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            output, global_classifier, _ = model(batch_data)
            batch_labels = F.one_hot(batch_labels, num_classes=2).float()
            loss = F.cross_entropy(output, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Fine-tuning Epoch {epoch}, Loss: {total_loss}")
    return global_classifier

def evaluate_model(model, X_test, y_test, save=False):
    model.eval()
    X_test, y_test = X_test.to(device), y_test.to(device)
    with torch.no_grad():
        outputs, _, features = model(X_test)
        probabilities = F.softmax(outputs, dim=-1)
        prob = probabilities[:, 1].cpu().numpy()
        y_true = y_test.cpu().numpy()
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
        if save:
            all_metrics.append({
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc,
                'PR AUC': pr_auc,
                'Confusion Matrix': cm
            })

def feddyn(models, clients_data_len, global_model, beta):
    total_data = sum(clients_data_len)
    aggregated_state_dict = {key: torch.zeros_like(value, dtype=torch.float32) for key, value in global_model.state_dict().items()}
    dynamic_reg = {key: torch.zeros_like(value, dtype=torch.float32) for key, value in global_model.state_dict().items()}
    for model, data_len in zip(models, clients_data_len):
        weight = data_len / total_data
        model_state_dict = model[0].state_dict()
        for key in aggregated_state_dict:
            client_tensor = model_state_dict[key].float()
            dynamic_reg_tensor = dynamic_reg[key].float()
            aggregated_state_dict[key] += weight * (client_tensor - beta * dynamic_reg_tensor)
    global_model.load_state_dict(aggregated_state_dict)
    return global_model

def fed_avg(client_models, clients_data_len, global_model):
    total_data_points = sum(clients_data_len)
    for k in global_model.state_dict().keys():
        accumulated = torch.zeros_like(global_model.state_dict()[k], dtype=torch.float32)
        for i in range(len(client_models)):
            accumulated += client_models[i][0].state_dict()[k].float() * float(clients_data_len[i] / total_data_points)
        global_model.state_dict()[k] = accumulated.type(global_model.state_dict()[k].dtype)
    return global_model

def load_client_data(client_folder):
    normal_path = os.path.join(client_folder, "normal", "data.csv")
    abnormal_path = os.path.join(client_folder, "abnormal", "data.csv")    
    test_normal_path = os.path.join(client_folder, "test_normal", "data.csv")
    print(normal_path)
    normal_data = pd.read_csv(normal_path, header=None)
    abnormal_data = pd.read_csv(abnormal_path, header=None)    
    test_normal_data = pd.read_csv(test_normal_path, header=None)
    normal_data['label'] = 0
    abnormal_data['label'] = 1
    test_normal_data['label'] = 0
    combined_data = pd.concat([normal_data, abnormal_data, test_normal_data], ignore_index=True)
    return combined_data

def preprocess(df, console):
    if DATASET == "iot23":
        # iot23
        df = df.drop(columns=['ts','uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p',
                                'service','local_orig','local_resp','detailed-label'])
        le = LabelEncoder()
        le.fit(['tcp', 'udp', 'icmp'])
        df['proto'] = le.transform(df['proto'])

        le = LabelEncoder()
        le.fit(['ShADdfF', 'ShAdDfFr', 'ShAdDfFa', 'ShAdDaFf', 'ShADadFf', 'Ar', 'ShAFa', '^dtt', 'ShADadtctfF', 'ShADFadR', 'ShAF', 'ShAdDatfr', 'D^', '^hR', 'Aa', 'ShADadTFf', 'ShAdDatrfR', 'ShADFafRdt', 'ShAdDafrFr', 'ShADdTafF', 'ShADFdRt', 'ShADFadfRR', 'ShADacdtttfF', 'ShADaCGdtfF', 'ShADaF', 'ShADadFRfR', 'ShAdDaFRf', 'ShAdDaTFRf', 'ShAdDafFR', 'ShADadtfFr', 'ShADFadfR', 'A', 'HaDdTAFf', 'ShADafF', 'ShAFfar', 'F', 'ShADdaFf', '-', 'ShwAadDfF', 'ShADdacFf', '^aA', 'ShADFdfR', 'ShAaww', 'ShADaR', 'Sr', 'ShADdFfa', 'DaFfA', 'ShAadDFR', 'ShAD', 'ShADFaT', 'ShAa', 'ShADdtaFf', 'ShwAadDftF', 'ShADdFaf', 'ShAdDtaFr', 'ShADadftFR', 'ShADFaTr', 'ShADdfFa', 'DAd', 'ShAdDaTFR', 'ShAdDFar', 'ShAdfDF', 'DFafA', 'ShADFadRfR', 'ShAdDaFRRRf', 'ShAdDaFRRf', 'SahAdDFf', 'ShAdFaf', 'D', 'ShADadfrr', 'ShADdCacFf', 'ShADFdRtf', 'ShADFfr', 'ShADadCFf', 'ShADFaTdftR', 'ShADdfr', 'ShAdDatFrR', 'Hr', 'ShAdDatfF', 'ShAaw', 'ShADa', 'ShAdDaFr', 'HaADdFf', 'ShADadttFf', 'ShAdDarr', 'ShAdDaTRft', 'ShAFafR', '^ha', 'SaR', 'ShAdDaFRR', 'ShADafFr', 'ShAFfR', 'ShADFdRafR', 'ShADdaCFf', 'ShADFafdtRR', 'ShAdDa', 'SI', 'ShAdtfFa', 'HaDdAFf', 'ShAdDaTRf', 'D^d', 'ShADaTdR', 'ShAdDaFR', 'Ffa', 'ShADadfF', 'ShADFadtRf', 'ShwAadDfr', 'ShAdDatFf', 'ShAdDaTFf', 'SahAdDFRf', 'SahAdDtFf', 'ShADdfFr', 'DdAFaf', 'C', 'ShADadtRf', 'DdAtaFf', 'ShADfFa', 'R', 'ShADadtR', 'ShAfFa', 'ShAdDaFfr', 'Dr', 'ShAdr', 'ShAdDaRRR', 'ShAdDtafFr', 'SahAdDrfR', 'ShAdDafFrR', '^hADr', 'ShwAr', 'HaR', 'ShAdDTafF', 'ShAfF', 'ShADadf', 'ShADFar', 'ShADadttFRfR', 'ShADacdFf', 'ShAdtDaFrR', 'ShAdDaFRfRR', 'ShAdDafrR', 'ShADFaTdfR', '^aR', 'ShwA', 'ShAdDar', 'ShADFadRftr', 'ShADFadfRt', 'ShADacfF', '^hwAadDfF', 'ShAdfr', 'DTT', 'SAD', 'ShADFarR', 'FaAr', 'ShAdDFf', 'ShAdDafF', 'ShADfFr', 'ShADFaTf', 'HaDdR', 'ShAdfDr', 'ShADadttfF', 'ShAdDtaR', 'ShADF', 'ShAdDfr', 'ShADFrRfaR', 'ShAdDaft', 'ShA', 'ShAdDaFT', 'ShAFdRfR', 'ShADFdRf', 'ShADFfa', '^hADafF', 'ShADFafdtR', 'ShAfdtDFr', 'ShAdDaTfF', 'ShADdaTFf', 'ShADadF', 'HaDdAfF', 'ShAfdtDr', 'ShR', 'HaDdAr', 'ShAdaFr', 'ShAdfFa', '^d', 'ShADFa', 'ShADaTdtR', 'ShADfFrr', 'ShAdDafFr', 'ShAdDaFrR', 'ShADadFRR', '^c', 'Fr', 'ShADFafR', 'S', 'ShADdaf', 'ShAdDaTFfR', 'DdAttfrF', 'ShADfdtFaR', 'ShADfaF', 'ShAdDFfR', 'ShAdDaFRfR', 'ShAadDFf', 'ShAdDatFr', 'ShADadtcfF', 'CCC', 'ShADadFfR', 'ShAdDaTRr', 'ShADadfFr', 'ShADaFr', 'ShADdFf', 'CCCC', 'ShADadtCFf', 'ShwR', 'ShADFaf', 'ShADdtatFfR', 'ShADadCcFf', 'ShAdDafr', 'ShAFafr', 'ShADadCfF', 'ShAFfa', 'ShAdDaTfRr', 'ShAdDatR', 'ShADadRf', 'ShADdfR', 'ShAdDaRr', 'ShADadtFf', 'ShADCaGcgd', 'DadA', '^hA', 'ShADafdtF', 'ShADdafR', 'ShADadtctfFR', 'ShADda', 'ShADdattFfR', 'ShADad', 'ShADadFRf', 'ShADaCGdt', 'ShADdatcFf', 'ShADfrr', 'ShADfrF', 'ShrA', 'HaDdAFTf', 'ShAdfDFr', 'ShADadfFR', 'ShAdDaFRr', 'DdA', 'ShAdDaT', 'ShADfR', 'ShADFaTdRf', 'HadfDrArR', 'ShADaCGr', 'ShAdDaFfRR', 'ShADFfdtaRR', 'ShADadFfRR', 'ShADdaFr', 'ShwAaFdfR', 'ShADadftF', 'ShAFar', 'ShADFrfR', 'HafFr', 'ShADadfR', 'ShADdFfaRR', 'FaR', 'ShAr', 'ShArR', 'ShAdDaTR', 'ShADFfR', 'ShAadDr', 'ShADfrFr', 'ShADFfRaR', 'DFr', 'ShAFdfRt', 'ShAdDafR', 'ShAafF', 'ShAdfF', 'HaFfA', 'ShADfdtR', 'ShADFadftR', 'FfA', 'ShADacdtfr', 'I', 'ShAdDtaFf', 'DrF', 'ShAfdtF', 'ShAdDaFTf', 'ShAdDaFRRfR', 'ShADaTfF', 'ShADacdtfF', '^r', 'ShADCaGdfF', 'ShADfF', 'ShADaCGcgdF', 'ShAFaf', 'ShADFadRf', 'ShDadAf', 'ShADFaR', 'ShAadDFRf', 'ShAdDFaf', 'ShADFdafRR', '^dDA', 'ShAdDaFfR', 'Fa', 'ShAdDafFrr', 'ShAdDarfR', 'ShADdf', 'ShAdDaTF', 'ShADFaTfdtR', 'ShADadR', 'ShAdDaR', 'ShADar', 'ShAdtDaFr', 'ShAdDaftFR', 'ShAdDr', 'ShAdDatf', 'ShADafr', 'DdAf', 'ShADadcFf', 'ShAdDtafF', '^hADFr', 'DT', 'ShADFadRtf', 'SahAdDF', 'ShADadttcfF', 'ShADdaftF', 'ShAdDaftF', 'ShAdaw', 'ShADarfF', 'ShADrfR', '^fA', 'ShAadDfF', 'ShADadtTfFr', 'Dd', 'ShADadtfF', 'ShADdR', 'ShAdD', '^hADadfR', 'H', 'ShADFr', 'ShAdtDafF', 'ShADadCtFf', 'ShAdDaf', 'ShAdDaF', 'ShAdF', 'ShAFr', 'DdAa', 'ShAdaDR', 'ShADadTFTf', 'ShADacdftF', 'ShADfr', 'ShADFdfaRR', 'DdAaFf', 'ShADFfaRR', 'ShADdafF', 'ShADr', 'ShAFf'])
        df['history'] = le.transform(df['history'])

        le = LabelEncoder()
        le.fit(['S0','OTH','SF','REJ','S3','RSTR','RSTO','RSTOS0','S1','S2','RSTRH','SH','SHR'])
        df['conn_state'] = le.transform(df['conn_state'])
        df['duration'] = df['duration'].str.replace('-','0')
        df['duration'] = df['duration'].astype(float)
        df['orig_bytes'] = df['orig_bytes'].str.replace('-','0')
        df['orig_bytes'] = df['orig_bytes'].astype(int)
        df['resp_bytes'] = df['resp_bytes'].str.replace('-','0')
        df['resp_bytes'] = df['resp_bytes'].astype(int)
        df['label'] = df['label'].str.replace('Benign','0')
        df['label'] = df['label'].str.replace('Malicious','1')
        df['label'] = df['label'].astype(int)
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
        df['orig_bytes'] = pd.to_numeric(df['orig_bytes'], errors='coerce')
        df['resp_bytes'] = pd.to_numeric(df['resp_bytes'], errors='coerce')
        df.fillna(-1, inplace=True)

    elif DATASET == "radar":
        # radar
        df = df.drop(columns=['Unnamed: 0', 'filename', 'url_query_names', 'ssl_ratio', 'average_public_key', 
        'tls_version_ratio', 'average_of_certificate_length', 'standart_deviation_cert_length', 
        'is_valid_certificate_during_capture', 'amount_diff_certificates', 'number_of_domains_in_certificate',
        'url_query_values', 'path', 'Src_P', 'Dest_IP', 'Dest_P', 'DestIP', 'infosteal_encoded', 
        'family_encoded', 'method_encoded', 'rootkits', 'infosteal', 'bkdoor', 'keylog', 'family_gene', 
        'method', 'binarylabel', 'label', 'full_label', 'family_label', 'cert_issuer', 'cert_subject', 'sni', 
        'hostname', 'url', 'Filename', '#Src_IP', 'Dest_IP', 'Dport', 'goal_encoded','dns_success_digitratio', 'dns_success_alpharatio', 'dns_success_specialcharratio', 'dns_success_caseratio', 'dns_status_ratio', 'dns_success_vowelchangeratio',
        'number_of_certificate_path', 'x509_ssl_ratio', 'SNI_ssl_ratio', 'self_signed_ratio', 'is_SNIs_in_SNA_dns', 'SNI_equal_DstIP', 'is_CNs_in_SNA_dns','url_query_names', 'url_query_values', 'path', 'url_path_length', 'number_of_URL_query_parameters',
        'filename_length', 'interarrival_time', 'number_of_url_flows', 'number_of_downloaded_bytes', 'number_of_uploaded_bytes',
        'noOffiles', 'filename_digitratio', 'filename_caseratio'])
        df.rename(columns={'goal': 'label'}, inplace=True)
        df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        df.fillna(0, inplace=True)
    return df

def data_split(df, req_initial):
    y = df['label']
    X = df.drop(columns=['label'])
    del df
    if DATASET == "iot23":
        # iot-23
        categorical_columns = ['proto','conn_state','history']
        numerical_columns = X.columns.difference(categorical_columns)
        X_numerical = X[numerical_columns]
        X_categorical = X[categorical_columns]
        del X, categorical_columns, numerical_columns

        scaler = StandardScaler()
        X_numerical = pd.DataFrame(scaler.fit_transform(X_numerical),columns=X_numerical.columns)
        X = pd.concat([X_categorical, X_numerical], axis=1)
        del X_numerical, X_categorical
    elif DATASET == "radar":
        # radar
        categorical_columns = ['Protocol', 'service']
        numerical_columns = X.columns.difference(categorical_columns)
        X_numerical = X[numerical_columns]
        X_categorical = X[categorical_columns]
        del X, categorical_columns, numerical_columns

        scaler = StandardScaler()
        X_numerical = pd.DataFrame(scaler.fit_transform(X_numerical),columns=X_numerical.columns)
        X = pd.concat([X_categorical, X_numerical], axis=1)
        del X_numerical, X_categorical
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, shuffle=True, random_state = 110)
    if(req_initial):
        X_train, initial_X, y_train, initial_y = train_test_split(X_train, y_train, test_size= 0.20, shuffle=True, random_state = 110, stratify=y_train)

    X_train = tensor(X_train.values, dtype=torch.float32)
    y_train = tensor(y_train.values, dtype=torch.long)
    X_test = tensor(X_test.values, dtype=torch.float32)
    y_test = tensor(y_test.values, dtype=torch.long)

    if(req_initial):
        initial_X = tensor(initial_X.values, dtype=torch.float32)
        initial_y = tensor(initial_y.values, dtype=torch.long)

    del X, y
    if(req_initial):
        return X_train, y_train, X_test, y_test, initial_X, initial_y
    return X_train, y_train, X_test, y_test

if DATASET == "radar" or DATASET == "iot23":
    NUM_CLIENTS = 10
    data_loader = Data_Loader(PARTITIONED_DATA_DIRECTORY, console, True)
    df = data_loader.read_partitions_to_dataframe("Server")
    df = preprocess(df, console)
    X_train, y_train, X_test, y_test, initial_X, initial_y = data_split(df, True)
    del df
    server_data = []
    server_data.append(X_train)
    server_data.append(y_train)
    server_data.append(X_test)
    server_data.append(y_test)

    feature_extractor = FeatureExtractor(input_dim=X_train.shape[1], hidden_dim=128, proj_dim=64)
    feature_extractor = feature_extractor.to(device)
    global_client = client_training(feature_extractor, initial_X)
    del initial_X, initial_y

    client_folders = [f"Client{i}" for i in range(1, NUM_CLIENTS+1)]
    clients_data = []
    for client_folder in client_folders:
        temp_df = data_loader.read_partitions_to_dataframe(file_name=client_folder)
        temp_df = preprocess(temp_df, console)
        X_train, y_train, X_test, y_test = data_split(temp_df, False)
        clients_data.append([X_train, y_train, X_test, y_test])

    print("\nClient training")
    total_clients = len(client_folders)
    num_to_pick = int((PERCENTAGE / 100) * total_clients)

    for round in range(NUM_ROUNDS):
        client_models = []
        clients_data_len = []
        selected_clients_folders = random.sample(client_folders, num_to_pick)
        print(selected_clients_folders)
        for client_folder in selected_clients_folders:
            client_number = int(client_folder[6:])-1
            print("Client_number", client_number)
            client = copy.deepcopy(global_client)
            client = client_training(client, clients_data[client_number][0])
            client_models.append([client])
            print("")
            clients_data_len.append(clients_data[client_number][0].size(0))
        print("Aggregation")
        global_client = fed_avg(client_models, clients_data_len, global_client)
        # global_client = feddyn(client_models, clients_data_len, global_client, beta=0.1)
        if round == 0:
            global_classifier = None
        final_model = FEDMALDETECT(input_dim=X_train.shape[1], hidden_dim=128, output_dim=len(torch.unique(y_train)), global_client=global_client, global_classifier=global_classifier).to(device)

        print("Server training")
        global_classifier = server_finetuning(final_model, server_data[0], server_data[1])

        evaluate_model(final_model, X_test, y_test)

        for i in range(len(clients_data)):
            evaluate_model(final_model, clients_data[i][2], clients_data[i][3])

        if(round==NUM_ROUNDS-1):
            SAVE = True
        evaluate_model(final_model, server_data[2], server_data[3], SAVE)
        for i in range(NUM_CLIENTS):
            evaluate_model(final_model, clients_data[i][2], clients_data[i][3], SAVE)

elif DATASET == "nabiot" and TRAINING_TYPE == "Decentralized":
    round = 0
    global_client = None
    client_folders = [f"Client-{i}" for i in range(1, NUM_CLIENTS+1)]
    client_folders.insert(0, "Client-50")
    clients_data = []
    server_data = []
    for client_folder in client_folders:
        df = load_client_data(os.path.join(PARTITIONED_DATA_DIRECTORY,client_folder))
        y = df['label']
        X = df.drop(columns=['label'])
        del df
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, shuffle=True, random_state = 110, stratify=y)
        if(client_folder == "Client-50"):
            X_train, initial_X, y_train, initial_y = train_test_split(X_train, y_train, test_size= 0.20, shuffle=True, random_state = 110, stratify=y_train)
            initial_X = tensor(initial_X.values, dtype=torch.float32)
            initial_y = tensor(initial_y.values, dtype=torch.long)
        X_train = tensor(X_train.values, dtype=torch.float32)
        y_train = tensor(y_train.values, dtype=torch.long)
        X_test = tensor(X_test.values, dtype=torch.float32)
        y_test = tensor(y_test.values, dtype=torch.long)
        if(client_folder == "Client-50"):
            server_data.append(X_train)
            server_data.append(y_train)
            server_data.append(X_test)
            server_data.append(y_test)
            feature_extractor = FeatureExtractor(input_dim=X_train.shape[1], hidden_dim=128, proj_dim=64)
            feature_extractor = feature_extractor.to(device)
            global_client = client_training(feature_extractor, initial_X)
            del initial_X, initial_y
        else:
            clients_data.append([X_train, y_train, X_test, y_test])
        del X, y

    print(len(clients_data))
    client_folders.remove("Client-50")
    for round in range(NUM_ROUNDS):
        clients_data_len = []
        total_clients = len(client_folders)
        num_to_pick = int((PERCENTAGE / 100) * total_clients)
        selected_clients_folders = random.sample(client_folders, num_to_pick)
        print(round)
        print(selected_clients_folders)
        
        client_models = []
        i = 0
        for client_folder in selected_clients_folders:
            client = copy.deepcopy(global_client)
            client_number = int(client_folder.split('-')[1])-1
            print("Client_number", client_number)
            client = client_training(client, clients_data[client_number][0])
            client_models.append([client])
            print("")
            clients_data_len.append(clients_data[client_number][0].size(0))
            i+=1

        print("Aggregation")
        print(len(clients_data_len), len(client_models))
        if AGGREGATION == 'FedAvg':
            global_client = fed_avg(client_models, clients_data_len, global_client)
        elif AGGREGATION == 'FedDyn':
            global_client = feddyn(client_models, clients_data_len, global_client, beta=0.1)
        if round == 0:
            global_classifier = None
        final_model = FEDMALDETECT(input_dim=clients_data[client_number][0].shape[1], hidden_dim=128, output_dim=len(torch.unique(clients_data[client_number][1])), global_client=global_client, global_classifier=global_classifier).to(device)

        print("Server training")
        global_classifier = server_finetuning(final_model, server_data[0], server_data[1])

        if(round==NUM_ROUNDS-1):
            SAVE = True
        evaluate_model(final_model, server_data[2], server_data[3], SAVE)
        for i in range(NUM_CLIENTS):
            evaluate_model(final_model, clients_data[i][2], clients_data[i][3], SAVE)

elif TRAINING_TYPE == "Centralized":
    client_folders = [f"Client-{i}" for i in range(1, NUM_CLIENTS+1)]
    dfs = []
    for client_folder in client_folders:
        df = load_client_data(os.path.join(PARTITIONED_DATA_DIRECTORY,client_folder))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    y = df['label']
    X = df.drop(columns=['label'])
    del df
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, shuffle=True, random_state = 110, stratify=y)
    X_train = tensor(X_train.values, dtype=torch.float32)
    y_train = tensor(y_train.values, dtype=torch.long)
    X_test = tensor(X_test.values, dtype=torch.float32)
    y_test = tensor(y_test.values, dtype=torch.long)

    feature_extractor = FeatureExtractor(input_dim=X_train.shape[1], hidden_dim=128, proj_dim=64)
    feature_extractor = feature_extractor.to(device)
    global_client = client_training(feature_extractor, X_train)
    global_classifier = None
    final_model = FEDMALDETECT(input_dim=X_train.shape[1], hidden_dim=128, output_dim=len(torch.unique(y_train)), global_client=global_client, global_classifier=global_classifier).to(device)
    global_classifier = server_finetuning(final_model, X_train, y_train)
    evaluate_model(final_model, X_test, y_test, True)

from calflops import calculate_flops
batch_size = 1
input_shape = (batch_size, server_data[0].shape[1])
flops, macs, params = calculate_flops(model=final_model, 
                                    input_shape=input_shape,
                                    output_as_string=True,
                                    output_precision=4)
print("Full deployable model : FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

flops, macs, params = calculate_flops(model=global_client, 
                                    input_shape=input_shape,
                                    output_as_string=True,
                                    output_precision=4)
print("FLM model : FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))


table = Table(box=box.ASCII_DOUBLE_HEAD, padding=(0, 0))
table.add_column("[bold bright_yellow]Model Params[/bold bright_yellow]",justify="center", width=150)
pms = Table(box=box.ASCII2, padding=(0, 0), show_edge=False, show_lines=True)
pms.add_column("[bold bright_blue]Feature Learning Module [/bold bright_blue]",justify="center", width=50)
pms.add_column("[bold bright_blue]Classifier [/bold bright_blue]",justify="center", width=50)
pms.add_column("[bold bright_blue]Federated learning [/bold bright_blue]",justify="center", width=50)

flm = Table(box=box.ASCII2, show_edge=False, show_lines=True, show_header=False)
flm.add_column(width=25)
flm.add_column(width=25)
flm.add_row("[bold white]EPOCHS[/bold white]", "[bold white]"+str(EPOCHS)+"[/bold white]")
flm.add_row("[bold white]BATCH_SIZE[/bold white]", "[bold white]"+str(BATCH_SIZE)+"[/bold white]")
flm.add_row("[bold white]OPTIMIZER[/bold white]", "[bold white]"+OPTIMIZER+"[/bold white]")
flm.add_row("[bold white]LEARNING_RATE[/bold white]", "[bold white]"+str(LEARNING_RATE)+"[/bold white]")
flm.add_row("[bold white]WEIGHT_DECAY[/bold white]", "[bold white]"+str(WEIGHT_DECAY)+"[/bold white]")
flm.add_row("[bold white]LOSS[/bold white]", "[bold white]"+str(LOSS)+"[/bold white]")

fcl = Table(box=box.ASCII2, show_edge=False, show_lines=True, show_header=False)
fcl.add_column(width=25)
fcl.add_column(width=25)
fcl.add_row("[bold white]EPOCHS[/bold white]", "[bold white]"+str(CLASSIFIER_EPOCHS)+"[/bold white]")
fcl.add_row("[bold white]BATCH_SIZE[/bold white]", "[bold white]"+str(CLASSIFIER_BATCH_SIZE)+"[/bold white]")
fcl.add_row("[bold white]OPTIMIZER[/bold white]", "[bold white]"+str(CLASSIFIER_OPTIMIZER)+"[/bold white]")
fcl.add_row("[bold white]LEARNING_RATE[/bold white]", "[bold white]"+str(CLASSIFIER_LEARNING_RATE)+"[/bold white]")
fcl.add_row("[bold white]WEIGHT_DECAY[/bold white]", "[bold white]"+str(CLASSIFIER_WEIGHT_DECAY)+"[/bold white]")
fcl.add_row("[bold white]CRITERION[/bold white]", "[bold white]"+str(CLASSIFIER_CRITERION)+"[/bold white]")

fl = Table(box=box.ASCII2, show_edge=False, show_lines=True, show_header=False)
fl.add_column(width=25)
fl.add_column(width=25)
fl.add_row("[bold white]NUM_CLIENTS[/bold white]", "[bold white]"+str(NUM_CLIENTS)+"[/bold white]")
fl.add_row("[bold white]NUM_ROUNDS[/bold white]", "[bold white]"+str(NUM_ROUNDS)+"[/bold white]")
fl.add_row("[bold white]AGGREGATION[/bold white]", "[bold white]"+str(AGGREGATION)+"[/bold white]")
fl.add_row("[bold white]PARTICIPATING_CLIENTS (%)[/bold white]", "[bold white]"+str(PERCENTAGE)+"[/bold white]")

pms.add_row(flm, fcl, fl)
table.add_row(pms)
console.print(table)

table = Table(box=box.ASCII_DOUBLE_HEAD, padding=(0, 0))
table.add_column("[bold bright_yellow]Result[/bold bright_yellow]",justify="center", width=90)
pms = Table(box=box.ASCII2, padding=(0, 0), show_edge=False, show_lines=True)
pms.add_column("[bold bright_blue]Clients[/bold bright_blue]",justify="center", width=10)
pms.add_column("[bold bright_blue]Precision[/bold bright_blue]",justify="center", width=10)
pms.add_column("[bold bright_blue]Recall[/bold bright_blue]",justify="center", width=10)
pms.add_column("[bold bright_blue]F1 Score[/bold bright_blue]",justify="center", width=10)
pms.add_column("[bold bright_blue]ROC AUC[/bold bright_blue]",justify="center", width=10)
pms.add_column("[bold bright_blue]PR AUC[/bold bright_blue]",justify="center", width=10)
pms.add_column("[bold bright_blue]Confusion Matrix[/bold bright_blue]",justify="center", width=30)

for idx, metrics in enumerate(all_metrics):
    pms.add_row(
        str(idx),
        f"{metrics['Precision']:.4f}",
        f"{metrics['Recall']:.4f}",
        f"{metrics['F1 Score']:.4f}",
        f"{metrics['ROC AUC']:.4f}",
        f"{metrics['PR AUC']:.4f}",
        str(metrics['Confusion Matrix']).replace('\n', ' ')
    )
table.add_row(pms)
console.print(table)

with open(filename, mode='w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=all_metrics[0].keys())
    writer.writeheader()
    writer.writerows(all_metrics)