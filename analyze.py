import math
import torch
import os
import re
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from datasets import getdataset
from model import Autoencoder, Classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


def get_args():
    # define some parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--model_path', type=str, default='./ckpt/finetune/100epoch.pth', help='Model path')
    parser.add_argument('--model_dir', type=str, default='./ckpt/finetune/', help='Model directory')
    args = parser.parse_args()
    return args


def load_model(model_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    autoencoder = Autoencoder()
    model = Classifier(autoencoder.encoder, autoencoder.latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=False)["model_state"])
    model.eval()

    return model, device


def sample(test_loader, num_examples, args):
    os.makedirs('./figure', exist_ok=True)
    
    with torch.no_grad():
        model, device = load_model(args.model_path)

        #Load examples
        dataiter = iter(test_loader)
        batch = next(dataiter)
        images = batch["image"].to(device)
        labels = batch["target"].to(device)

        logits = model(images)                      # [B]
        probs = torch.sigmoid(logits)               # convert logits to probabilities
        predicted = (probs > 0.5).long()            # threshold at 0.5

        num_examples = min(num_examples, images.size(0))

        cols = 4
        rows = math.ceil(num_examples / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flat  # flatten iterator
    
        for i in range(rows * cols):
            ax = axes[i]
            if i < num_examples:
                img = images[i].cpu().squeeze(0).numpy()
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Target: {labels[i].item()}, Pred: {predicted[i].item()}')
            ax.axis('off')  # hide unused axes as well

        plt.savefig(f'./figure/samples.png')
        plt.close()

def plot_roc(test_loader, args):
    os.makedirs('./figure', exist_ok=True)
    model, device = load_model(args.model_path)

    y_true = []
    y_prob = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["target"].to(device)

            logits = model(images)            # [B]
            probs  = torch.sigmoid(logits)    # [B]

            y_prob.extend(probs.cpu().tolist())
            y_true.extend(labels.cpu().long().tolist())

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--', label='Random chance') 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./figure/roc_curve.png')
    plt.close()

def score_analyze(val_loader, args):
    model_list = [
        fname for fname in os.listdir(args.model_dir)
        if re.match(r'^\d+epoch\.pth$', fname)
    ]
    # now safely extract the leading number
    model_list.sort(key=lambda fname: int(fname.split('epoch')[0]))
    
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    epoch_list = []  # New list to store the epoch of each model
    
    for model_path in tqdm(model_list):
        model, device = load_model(os.path.join(args.model_dir, model_path))
        y_pred = []
        y_true = []

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["target"].to(device)

                logits = model(images)                      # [B]
                probs = torch.sigmoid(logits)               # convert logits to probabilities
                preds = (probs > 0.5).long()                # threshold at 0.5

                y_pred.extend(preds.cpu().tolist())
                y_true.extend(labels.cpu().long().tolist())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        
        epoch_list.append(int(re.findall(r'\d+', model_path)[0]))  
        # Calculate the epoch of the model and add it to the list
    
    os.makedirs('./figure', exist_ok=True)
    plt.plot(epoch_list, accuracy_list, label='accuracy')
    plt.plot(epoch_list, precision_list, label='precision')
    plt.plot(epoch_list, recall_list, label='recall')
    plt.plot(epoch_list, f1_list, label='f1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Score Analysis')
    plt.legend()
    plt.savefig('./figure/score_analysis.png')
    plt.close()


if __name__ == '__main__':
    args = get_args()
    _, test_set = getdataset(args.batch_size)
    
    print("Taking samples...")
    sample(test_set, 16, args) # Visualize the samples
    print("Calculating score...")
    score_analyze(test_set, args) # Analyze the scores
    print("Calculating AUROC...")
    plot_roc(test_set, args)
    print("Done")