import torch

from torch.functional import F
from tqdm import tqdm
from utils.functions import gradPenalty2sideCalc
from utils.oodEvaluation import get_auroc_ood

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def networkTrainStep(netType, model,optimizer,lossFunction,trainLoader,numClasses,gradPenaltyL):
    model.train()
    totalLoss  = 0
    correct    = 0
    for x,y in trainLoader:
        x = x.to(device)
        y = y.type(torch.LongTensor).to(device).squeeze()
        if gradPenaltyL:
            x.requires_grad_(True)
        optimizer.zero_grad()
        output = model(x)
        loss = lossFunction(output,y)
        if gradPenaltyL:
            loss += gradPenaltyL * gradPenalty2sideCalc(x,output)
        loss.backward()
        correct += (torch.argmax(output,dim=1) == y).sum().item()
        totalLoss += loss.item()
        optimizer.step()

        if 'duq' in netType.lower():
            with torch.no_grad():
                y = F.one_hot(y,num_classes=numClasses).type(torch.float)
                model.update_embeddings(x,y)
    totalLoss /= len(trainLoader)
    accuracy = correct / len(trainLoader.dataset)
    return accuracy, totalLoss

def networkTest(model,lossFunction,testLoader):
    model.eval()
    valLoss = 0
    valAccuracy = 0
    with torch.no_grad():
        for x, y in testLoader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            valLoss += lossFunction(output, y).item()
            valAccuracy += (output.argmax(dim=1) == y).float().mean().item()
    valLoss /= len(testLoader)
    valAccuracy /= len(testLoader)
    return valAccuracy, valLoss

def networkTrain(netType,model,optimizer,scheduler,lossFunction,trainLoader,testLoader,falseLoaders,numClasses,gradPenaltyL,epochs):
    trainAccs   = []
    testAccs    = []
    trainLosses = []
    testLosses  = []
    aurocs      = []

    pbar = tqdm(range(epochs),desc="Epochs")
    
    for _ in pbar:
        trainAcc,trainLoss = networkTrainStep(netType,model,optimizer,lossFunction,trainLoader,numClasses,gradPenaltyL)
        testAcc,testLoss = networkTest(model,lossFunction,testLoader)
        currentAurocs = [] 
        for falseloader in falseLoaders:
            currentAurocs.append(get_auroc_ood(true_dataset=testLoader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, model_type=netType))
        aurocs.append(currentAurocs)


        scheduler.step(testLoss)
        trainAccs.append(trainAcc)
        trainLosses.append(trainLoss)
        testAccs.append(testAcc)
        testLosses.append(testLoss)

        pbar.set_postfix({'test_acc': f'{testAcc:.2f}%', 'AUROC1': f'{currentAurocs[0]:.2f}'})

    return trainAccs[-1],trainLosses[-1],testAccs[-1],testLosses[-1],aurocs[-1]

def DeepEnsambleTest(models, lossFunction, testLoader):
    for model in models:
        model.eval()
    preds = []
    totallabels = []
    with torch.no_grad():
        for x, y in testLoader:
            batchprobs = []
            for model in models:
                output = model(x.to(device))
                batchprobs.append(output)

            avg_probs = torch.stack(batchprobs).mean(dim=0)  
            preds.append(torch.argmax(avg_probs, dim=1))
            totallabels.append(y)
    preds = torch.cat(preds)
    totallabels= torch.cat(totallabels)
    valLoss = lossFunction(avg_probs, y.to(device)).item()
    valAccuracy = ((preds == totallabels.to(device)).float().mean().item())
    return valAccuracy,valLoss

def DeepEnsambleTrain(models,optimizers,schedulers,lossFunction,trainLoader,
                      testLoader,falseLoaders,numClasses,gradPenaltyL,epochs):

    demodelsNum = len(models)
    trainAccs   = []
    testAccs    = []
    trainLosses = []
    testLosses  = []
    aurocs      = []

    pbar = tqdm(range(epochs),desc="Epochs")
    
    for _ in pbar:
        for i in range(demodelsNum):
            trainAcc,trainLoss = networkTrainStep('mlp',models[i],optimizers[i],lossFunction,trainLoader,numClasses,gradPenaltyL)
            _, testLoss = networkTest(models[i],lossFunction,testLoader) 
            schedulers[i].step(testLoss)
        currentAurocs = [] 
        for falseloader in falseLoaders:
            currentAurocs.append(get_auroc_ood(true_dataset=testLoader.dataset, ood_dataset=falseloader.dataset, model=models,
                                                device=device, model_type='de'))
        aurocs.append(currentAurocs)
        trainAcc,trainLoss = DeepEnsambleTest(models,lossFunction,trainLoader)
        testAcc,testLoss = DeepEnsambleTest(models,lossFunction,testLoader)


        trainAccs.append(trainAcc)
        trainLosses.append(trainLoss)
        testAccs.append(testAcc)
        testLosses.append(testLoss)

        pbar.set_postfix({'test_acc': f'{testAcc:.2f}%', 'AUROC1': f'{currentAurocs[0]:.2f}'})

    return trainAccs[-1],trainLosses[-1],testAccs[-1],testLosses[-1],aurocs[-1]