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

            # SCORE IS CALCULATED AS WEIGHTED UNC PER LAYER 
            elif model_type.lower() == 'ua_kan':
                _, layer_us = model.forward_with_layer_uncertainty(data)
                uncertainty = sum(w * u for w, u in zip(model.layer_weights, layer_us))

            # ADD WEIGHTED UNCERTAINTIES
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

            elif model_type.lower() == 'mlp_energy':
                output = model(data)  
                uncertainty = -torch.logsumexp(output, dim=1)  # enegy
            
            elif model_type.lower() == 'ft_transformer':
                probs = F.softmax(model(data), dim=1)
                uncertainty = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            elif model_type.lower() == 'mc_dropout':
                samples = model.mc_forward(data, n_samples=20)   # [n_samples, batch, n_classes]
                mean_probs = samples.mean(dim=0)                  # [batch, n_classes]
                uncertainty = torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)

            else:  # mlp and others
                output = model(data)
                uncertainty = torch.sum(output * torch.log(output + 1e-10), dim=1)

            scores.append(uncertainty.cpu().numpy())

    return np.concatenate(scores)


def get_auroc_ood(true_dataset, ood_dataset, model, device: torch.device, model_type: str) -> float:
    concat = ConcatDataset([true_dataset, ood_dataset])
    dataloader = DataLoader(concat, batch_size=500, shuffle=False, num_workers=0, pin_memory=False)
    labels = np.concatenate([np.zeros(len(true_dataset)), np.ones(len(ood_dataset))])

    scores = _get_uncertainty_scores(model, dataloader, model_type, device)
    auroc = roc_auc_score(labels, scores)
    # always reports the true discriminative power regardless of sign convention, heart disease is flipped against ambrosia
    return max(auroc, 1.0 - auroc)


