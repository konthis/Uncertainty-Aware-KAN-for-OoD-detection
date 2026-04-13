import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.functions import gradPenalty2sideCalc
from utils.oodEvaluation import get_auroc_ood

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def networkTrainStep(net_type, model, optimizer, loss_fn, train_loader, num_classes, grad_penalty_l):
    model.train()
    total_loss, correct = 0, 0

    for x, y in train_loader:
        x = x.to(device)
        #y = y.type(torch.LongTensor).to(device).squeeze()
        y = y.type(torch.LongTensor).to(device).view(-1)
        if grad_penalty_l:
            x.requires_grad_(True)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        if grad_penalty_l:
            loss = loss + grad_penalty_l * gradPenalty2sideCalc(x, output)
        loss.backward()

        correct   += (torch.argmax(output, dim=1) == y).sum().item()
        total_loss += loss.item()
        optimizer.step()

        if 'duq' in net_type.lower():
            with torch.no_grad():
                model.update_embeddings(x, F.one_hot(y, num_classes=num_classes).float())

    return correct / len(train_loader.dataset), total_loss / len(train_loader)


def networkTest(model, loss_fn, test_loader):
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            total_loss += loss_fn(output, y).item()
            total_acc  += (output.argmax(dim=1) == y).float().mean().item()

    return total_acc / len(test_loader), total_loss / len(test_loader)


def networkTrain(net_type, model, optimizer, scheduler, loss_fn,
                 train_loader, test_loader, false_loaders, num_classes, grad_penalty_l, epochs):
    train_accs, train_losses, test_accs, test_losses, aurocs = [], [], [], [], []

    pbar = tqdm(range(epochs), desc="Epochs")
    for _ in pbar:
        train_acc,  train_loss  = networkTrainStep(net_type, model, optimizer, loss_fn, train_loader, num_classes, grad_penalty_l)
        test_acc,   test_loss   = networkTest(model, loss_fn, test_loader)
        current_aurocs = [
            get_auroc_ood(test_loader.dataset, fl.dataset, model, device, net_type)
            for fl in false_loaders
        ]

        # scheduler.step(test_loss)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(test_loss)
        else:
            scheduler.step()

        train_accs.append(train_acc);   train_losses.append(train_loss)
        test_accs.append(test_acc);     test_losses.append(test_loss)
        aurocs.append(current_aurocs)

        pbar.set_postfix({'test_acc': f'{test_acc:.2f}', 'AUROC1': f'{current_aurocs[0]:.2f}'})

    return train_accs[-1], train_losses[-1], test_accs[-1], test_losses[-1], aurocs[-1]


def DeepEnsambleTest(models, loss_fn, loader):
    for m in models:
        m.eval()

    preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            outputs = torch.stack([m(x.to(device)) for m in models]).mean(dim=0)
            preds.append(torch.argmax(outputs, dim=1))
            all_labels.append(y)

    preds      = torch.cat(preds)
    all_labels = torch.cat(all_labels)
    loss       = loss_fn(outputs, y.to(device)).item()
    accuracy   = (preds == all_labels.to(device)).float().mean().item()
    return accuracy, loss


def DeepEnsambleTrain(models, optimizers, schedulers, loss_fn,
                      train_loader, test_loader, false_loaders, num_classes, grad_penalty_l, epochs):
    train_accs, train_losses, test_accs, test_losses, aurocs = [], [], [], [], []

    for _ in tqdm(range(epochs), desc="Epochs"):
        for model, optimizer, scheduler in zip(models, optimizers, schedulers):
            _, test_loss = networkTest(model, loss_fn, test_loader)
            networkTrainStep('mlp', model, optimizer, loss_fn, train_loader, num_classes, grad_penalty_l)
            scheduler.step(test_loss)

        current_aurocs = [
            get_auroc_ood(test_loader.dataset, fl.dataset, models, device, 'de')
            for fl in false_loaders
        ]

        train_acc, train_loss = DeepEnsambleTest(models, loss_fn, train_loader)
        test_acc,  test_loss  = DeepEnsambleTest(models, loss_fn, test_loader)

        train_accs.append(train_acc);   train_losses.append(train_loss)
        test_accs.append(test_acc);     test_losses.append(test_loss)
        aurocs.append(current_aurocs)

    return train_accs[-1], train_losses[-1], test_accs[-1], test_losses[-1], aurocs[-1]
