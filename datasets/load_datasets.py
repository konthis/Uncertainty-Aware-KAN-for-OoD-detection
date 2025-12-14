import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.utils.data import Dataset

from sklearn.datasets import load_iris,load_breast_cancer,load_wine


# 0: Non-infectious SIRS
# 1: Sepsis
# 2: Septic Shock

BATCH_SIZE = 16

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

def createSklearnDataloader(dataset, featureIdxs):
    datasetX = dataset['data'][:,featureIdxs]
    scaler = StandardScaler()
    datasetX = scaler.fit_transform(datasetX)
    datasetX = torch.autograd.Variable(torch.tensor(datasetX,dtype=torch.float))
    datasetY = torch.autograd.Variable(torch.tensor(dataset['target'], dtype=torch.long))
    dataloader = DataLoader(TensorDataset(datasetX,datasetY), batch_size=1000, shuffle=True)
    return dataloader

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

def load_noisedD1(seed,root,binary):
    _,trainloader,_,_ = load_D1(seed,root,binary=binary)
    noiseddataset = GaussianNoisedDataset(trainloader.dataset)
    return DataLoader(noiseddataset, batch_size=BATCH_SIZE, shuffle=True)

def loadAllDataloaders(root='./datasets',binary=False):
    seed = 1
    _, ambTrainLoader, ambTestLoader, _= load_D1(seed,root,binary=binary)
    falseloader  = createSklearnDataloader(load_wine(),[0,1,2])
    falseloader2 = createSklearnDataloader(load_iris(),[0,1,2]) 
    falseloader3 = createSklearnDataloader(load_breast_cancer(),[0,1,2])
    falseloader4 = load_noisedD1(seed,root,binary)

    return ambTrainLoader,ambTestLoader,falseloader,falseloader2,falseloader3,falseloader4
