import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def getDataloaderTargets(dataloader):
    targets = []
    for x,y in dataloader:
        targets.extend(y.tolist())
    return np.array(targets)

def saveToEXCEL(list, colNames, filename):
    #df = pd.DataFrame(list)
    df = pd.DataFrame.from_dict(list)
    df.columns = colNames
    writer = pd.ExcelWriter(f'{filename}.xlsx', engine='xlsxwriter')
    df.to_excel(writer, index=False)
    writer._save()

def topResultPerformersFromEXCEL(inFilename, outFilename,attributes):
    # attributes = [(col position, threshold)]
    df = pd.read_excel(f'{inFilename}.xlsx')
    columns = df.columns.tolist()
    rows = [row.tolist() for index, row in df.iterrows()]
    results = []
    for row in rows:
        lowerThanThreshold = False
        for attribute in attributes:
            if row[attribute[0]] <= attribute[1]:
                lowerThanThreshold = True
                break
        if not lowerThanThreshold:
            results.append(row)
    saveToEXCEL(results, columns, outFilename)

def plot(xaxis,xlabel,yaxis,ylabel,color,funcNames,xlim=None,ylim=None,xlogscale=False,ylogscale=False):
    
    if isinstance(yaxis[0],list): # if multiple funcs
        for i in range(len(yaxis)):
            plt.plot(xaxis,yaxis[i],c=color[i],label=funcNames[i])
    else:
        plt.plot(xaxis,yaxis,c=color,label=funcNames)

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlogscale:
        plt.xscale('log')
    if ylogscale:
        plt.yscale('log')
    plt.legend()
    plt.show()

