import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader


def _get_uncertainty_scores(model, dataloader, model_type: str, device: torch.device) -> np.ndarray:
    is_ensemble = isinstance(model, list)
    models = model if is_ensemble else [model]
    for m in models:
        m.eval()

    scores = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)

            if model_type.lower() in ('duq', 'kanduq'):
                output = model(data)
                kernel_distance, _ = output.max(1)
                uncertainty = -kernel_distance

            elif model_type.lower() == 'ua_kan':
                _, layer_us = model.forward_with_layer_uncertainty(data)
                uncertainty = sum(w * u for w, u in zip(model.layer_weights, layer_us))
            # elif model_type.lower() == 'ua_kan':
            #     logits, layer_us = model.forward_with_layer_uncertainty(data)
            #     probs = F.softmax(logits, dim=1)
            #     output_entropy = torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            #     layer_cert = sum(w * u for w, u in zip(model.layer_weights, layer_us))
            #     uncertainty = output_entropy + layer_cert

            elif model_type.lower() == 'kan':
                output = model.forwardSoftmax(data)
                uncertainty = torch.sum(output * torch.log(output + 1e-10), dim=1)

            elif is_ensemble:
                outputs = torch.stack([m(data) for m in models]).mean(dim=0)
                uncertainty = torch.sum(outputs * torch.log(outputs + 1e-10), dim=1)

            else:  # mlp and others
                output = model(data)
                uncertainty = torch.sum(output * torch.log(output + 1e-10), dim=1)

            scores.append(uncertainty.cpu().numpy())

    return np.concatenate(scores)


def get_auroc_ood(true_dataset, ood_dataset, model, device: torch.device, model_type: str) -> float:
    concat = ConcatDataset([true_dataset, ood_dataset])
    dataloader = DataLoader(concat, batch_size=500, shuffle=False, num_workers=4, pin_memory=False)
    labels = np.concatenate([np.zeros(len(true_dataset)), np.ones(len(ood_dataset))])

    scores = _get_uncertainty_scores(model, dataloader, model_type, device)
    return roc_auc_score(labels, scores)

