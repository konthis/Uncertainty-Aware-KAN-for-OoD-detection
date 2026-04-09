import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.utils.data import Dataset

from sklearn.datasets import load_iris,load_breast_cancer,load_wine

BATCH_SIZE = 16
OOD_FEATURE_IDXS = [0, 1, 2]

# 0: Non-infectious SIRS
# 1: Sepsis
# 2: Septic Shock

def load_D1(seed,path='../datasets', only_biomarkers=True, binary=False):

    df = pd.read_excel(f"{path}/ambrosia.xlsx")
    df = pd.get_dummies(df, columns=['Sex'], prefix='Sex')
    df.drop(columns=['Blood culture microorganism 1', 'Blood culture microorganism 2'], inplace=True)

    df.dropna(inplace=True)
    df = pd.get_dummies(df, columns=['Blood culture result'], prefix='Blood_culture_result')

    label_encoder = LabelEncoder()
    df['AMBROSSIA'] = label_encoder.fit_transform(df['AMBROSSIA'])


    if only_biomarkers:
        df.drop(columns=['Age', 'SOFA', 'Blood_culture_result_Positive', 'Blood_culture_result_Negative', 'Sex_Female', 'Sex_Male', 'CSG  '], inplace=True)

    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)

    X, y = df.drop(columns=['AMBROSSIA']), df['AMBROSSIA']

    # Combine the two classes (1: Sepsis, 2: Septic Shock) into a single class (1)
    if binary:
        y = y.replace(2, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    scalers = []
    for column in ['PCR (mg/dL)', 'PCT  (ng/mL)', 'IL6 (pg/mL)', 'Age', 'SOFA']:
        if column in X_train.columns:
            scaler = StandardScaler()
            X_train[column] = scaler.fit_transform(X_train[[column]])
            X_test[column] = scaler.transform(X_test[[column]])
            scalers.append(scaler)

    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32),
                                 torch.tensor(y_test.values, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    dimensions = X_train.shape[1]
    return test_dataset, train_loader, test_loader, dimensions

def download_heart_disease(path='../datasets'):
    dest = os.path.join(path, 'heart_cleveland_upload.csv')
    if os.path.exists(dest):
        return
    try:
        import kagglehub, shutil
    except ImportError:
        raise ImportError("Please: pip install kagglehub")

    print("Downloading Heart Disease Cleveland...")
    cache_path = kagglehub.dataset_download("cherngs/heart-disease-cleveland-uci")
    src = os.path.join(cache_path, 'heart_cleveland_upload.csv')
    shutil.move(src, dest)
    print(f"Moved to {dest}")

def load_heart_disease(seed, path='../datasets'):
    download_heart_disease(path)
    df = pd.read_csv(os.path.join(path, 'heart_cleveland_upload.csv'))

    df.dropna(inplace=True)

    X = df.drop(columns=['condition'])
    y = df['condition']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    return test_dataset, train_loader, test_loader, X_train.shape[1]


def createSklearnDataloader(dataset, feature_idxs: list) -> DataLoader:
    X = dataset['data'][:, feature_idxs]
    X = StandardScaler().fit_transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(dataset['target'], dtype=torch.long)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=1000, shuffle=True)

class GaussianNoisedDataset(Dataset):
    def __init__(self, dataset, mean=3.0, std=5):
        self.dataset = dataset
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        noisy_data = data + torch.randn_like(data) * self.std + self.mean
        noisy_data = torch.clamp(noisy_data, data.min(), data.max()) # keep range
        return noisy_data, target

def load_noisedD1(seed: int, root: str, binary: bool) -> DataLoader:
    _, train_loader, _, _ = load_D1(seed, root, binary=binary)
    noised = GaussianNoisedDataset(train_loader.dataset)
    return DataLoader(noised, batch_size=BATCH_SIZE, shuffle=True)


def loadAllDataloaders(root: str = './datasets', binary: bool = False):
    seed = 1
    _, train_loader, test_loader, _ = load_D1(seed, root, binary=binary)
    false_loaders = [
        createSklearnDataloader(load_wine(),          OOD_FEATURE_IDXS),
        createSklearnDataloader(load_iris(),          OOD_FEATURE_IDXS),
        createSklearnDataloader(load_breast_cancer(), OOD_FEATURE_IDXS),
        load_noisedD1(seed, root, binary),
    ]
    return train_loader, test_loader, *false_loaders


