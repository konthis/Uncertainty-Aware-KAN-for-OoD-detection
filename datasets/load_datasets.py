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
    dest = os.path.join(path, 'heart_disease_uci.csv')
    if os.path.exists(dest):
        return
    try:
        import kagglehub, shutil
    except ImportError:
        raise ImportError("Please: pip install kagglehub")
    print("Downloading Heart Disease UCI...")
    cache_path = kagglehub.dataset_download("redwankarimsony/heart-disease-data")
    src = os.path.join(cache_path, 'heart_disease_uci.csv')
    shutil.move(src, dest)
    print(f"Moved to {dest}")


def load_heart_disease(seed, path='../datasets'):
    download_heart_disease(path)
    df = pd.read_csv(os.path.join(path, 'heart_disease_uci.csv'))

    # drop identifier/source columns
    df = df.drop(columns=['id', 'dataset'], errors='ignore')

    df = df.rename(columns={'num': 'target'})

    df = df.dropna()

    for col in ['sex', 'cp', 'thal', 'restecg', 'slope']:
        if col in df.columns and df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col])

    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    for col in X_train.columns:
        scaler = StandardScaler()
        X_train[col] = scaler.fit_transform(X_train[[col]])
        X_test[col]  = scaler.transform(X_test[[col]])

    train_dataset = TensorDataset(
        torch.tensor(X_train.values.copy(), dtype=torch.float32),
        torch.tensor(y_train.values.copy(), dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test.values.copy(), dtype=torch.float32),
        torch.tensor(y_test.values.copy(), dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    return test_dataset, train_loader, test_loader, X_train.shape[1]

def load_ctg(seed, path='../datasets'):
    import numpy as np, zipfile
    CTG_FEATURES = [
    'LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP',
    'ASTV', 'MSTV', 'ALTV', 'MLTV',
    'Width', 'Min', 'Max', 'Nmax', 'Nzeros',
    'Mode', 'Mean', 'Median', 'Variance', 'Tendency',
    ]
    dest = os.path.join(path, 'fetal_health.csv')
    if not os.path.exists(dest):
        zip_path = os.path.join(path, 'archive.zip')
        if not os.path.exists(zip_path):
            raise FileNotFoundError("Place archive.zip in ./datasets/")
        print("Extracting archive.zip...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            found = None
            for name in z.namelist():
                if not name.endswith('.csv'):
                    continue
                with z.open(name) as f:
                    header = f.readline().decode('utf-8', errors='ignore')
                    if 'NSP' in header:
                        found = name
                        break
            if found is None:
                raise FileNotFoundError(f"No CSV with NSP column in archive. Files: {z.namelist()}")
            with z.open(found) as src, open(dest, 'wb') as dst:
                dst.write(src.read())
        print(f"Extracted '{found}' to {dest}")

    df = pd.read_csv(dest)
    df.columns = df.columns.str.strip()
    df = df.replace('?', np.nan)

    X = df[CTG_FEATURES].copy()
    mask = X.notna().all(axis=1)
    X = X[mask].reset_index(drop=True)
    y = df.loc[mask, 'NSP'].astype(int).values - 1  # 1,2,3 → 0,1,2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    for col in X_train.columns:
        scaler = StandardScaler()
        X_train[col] = scaler.fit_transform(X_train[[col]])
        X_test[col]  = scaler.transform(X_test[[col]])

    train_dataset = TensorDataset(
        torch.tensor(X_train.values.copy(), dtype=torch.float32),
        torch.tensor(y_train.copy(),        dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test.values.copy(), dtype=torch.float32),
        torch.tensor(y_test.copy(),        dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    return test_dataset, train_loader, test_loader, X_train.shape[1]


def createSklearnDataloader(dataset, n_features: int) -> DataLoader:
    import numpy as np
    X = dataset['data']
    n_avail = X.shape[1]
    if n_avail >= n_features:
        X = X[:, :n_features]
    else:
        X = np.hstack([X, np.zeros((X.shape[0], n_features - n_avail))])
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

def loadAllDataloaders(root: str = './datasets', binary: bool = False, dataset='ambrosia'):
    seed = 1
    if dataset == 'ambrosia':
        _, train_loader, test_loader, dimensions = load_D1(seed, root, binary=binary)
    elif dataset == 'heart':
        _, train_loader, test_loader, dimensions = load_heart_disease(seed, root)
    noised = GaussianNoisedDataset(train_loader.dataset)
    noised_loader = DataLoader(noised, batch_size=BATCH_SIZE, shuffle=True)

    false_loaders = [
        createSklearnDataloader(load_wine(),          dimensions),
        createSklearnDataloader(load_iris(),          dimensions),
        createSklearnDataloader(load_breast_cancer(), dimensions),
        noised_loader,
    ]
    return train_loader, test_loader, *false_loaders


