
import matplotlib.pyplot as plt
import numpy as np

def plot_results(train_acc, train_loss, test_acc, test_loss, auroc):
    def ema(x, alpha=0.3):
        smoothed = []
        s = x[0]
        for val in x:
            s = alpha * val + (1 - alpha) * s
            smoothed.append(s)
        return smoothed

    epochs = [i+1 for i in range(len(auroc))]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, ema(np.array(train_loss)), label='Train')
    axes[0].plot(epochs, ema(np.array(test_loss)), label='Test')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].legend()
    axes[0].grid()

    # Accuracy
    axes[1].plot(epochs, ema(np.array(train_acc)), label='Train')
    axes[1].plot(epochs, ema(np.array(test_acc)), label='Test')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()
    axes[1].grid()

    # AUROC
    axes[2].plot(epochs, ema(np.array(auroc)))
    axes[2].set_title('AUROC')
    axes[2].set_xlabel('Epochs')
    axes[2].legend(['Wine','Iris','B.cancer','Noised'])
    axes[2].grid()

    plt.tight_layout()
    plt.show()