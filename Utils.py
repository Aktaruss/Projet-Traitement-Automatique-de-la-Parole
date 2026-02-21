import pickle
from torch.utils import data
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

def get_basic_dataset(filename):
    with open(filename, "rb") as f:
        dataset = pickle.load(f)
    return dataset

class SpeechCommandDataset(data.Dataset):
    def __init__(self, signals, labels, metadata, transform_type=None):
        self.labels = torch.tensor(labels, dtype=torch.long)

        self.transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={
                "n_fft": 512,
                "n_mels": 40, 
                "win_length": int(16000 * 0.03),
                "hop_length": int(16000 * 0.01),
                "center": False 
            }
        )
        signals = [torch.tensor(sig, dtype=torch.float32) for sig in signals]

        processed_signals = []

        if transform_type == "MFCC":
            for sig in signals:
                if len(sig) != 16000 : print('DIFFERNRECE')
                if sig.ndim == 1:
                    sig = sig.unsqueeze(0)
                mfcc = self.transform(sig)
                processed_signals.append(mfcc)

        self.signals = torch.stack(processed_signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.labels)

def model_summary(model):
    total = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        total += param
        print(f"Couche({name}) : {param} (Paramètres)")
    print(f"Total : {total} (Paramètres)")

def evaluate(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion=nn.CrossEntropyLoss()
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    loss = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss += criterion(logits,labels).item()
            probs = torch.nn.functional.softmax(logits, dim=1)
            _, prediction = torch.max(logits, 1)
            all_preds.extend(prediction.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    correct = sum([all_preds[i] == all_labels[i] for i in range(len(all_preds))])
    accuracy = 100 * correct / len(all_labels)
    return accuracy, all_preds, all_labels, all_probs, loss/len(loader)

def train(model, train_loader, validation_loader, nb_steps=33000, val_step=400):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    lr_drop = nb_steps * (5/6)
    
    model.to(device)
    model.train()
    train_loss = []
    val_acc = []
    val_loss = []
    step = 0

    while step < nb_steps:
        for inputs, labels in train_loader:
            if step >= nb_steps:
                break

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        step += 1

        if step == lr_drop:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        
        if step % val_step == 0:
            acc, _, _, _, loss = evaluate(model,validation_loader)
            val_acc.append(acc)
            val_loss.append(loss)

    return model, train_loss, val_acc, val_loss

def plot_data(ax,data_list,title,xtitle,ytitle,y0,y1,coeff=1):
    save_dict = {}
    for data,label,color in data_list:
      x = np.arange(0,len(data)) * coeff
      y = data
      ax.plot(x,y,label=label,color=color)
      save_dict[f"{label}_x"] = x
      save_dict[f"{label}_y"] = data
    np.savez("plot_data.npz", **save_dict)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.grid(True, which='both', linestyle='-', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, edgecolor='gray', fancybox=False)
    ax.set_xlim(0, 35000)
    ax.set_ylim(y0, y1)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 4))

def plot_all_data(data_list):
    n = len(data_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), constrained_layout=True)
    if n == 1 :
      axes = [axes]
    for ax, (data,title,xtitle,ytitle,coeff,y0,y1) in  zip(axes,data_list):
      plot_data(ax,data,title,xtitle,ytitle,y0,y1,coeff=coeff)

def plot_custom_confusion_matrix(all_preds, all_labels, class_names=None):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    plt.yticks(rotation=0)
    plt.ylabel('Vrais Labels (Ground Truth)', fontweight='bold')
    plt.xlabel('Prédictions du Modèle', fontweight='bold')
    plt.title('Matrice de Confusion', fontsize=14, pad=15)
    plt.show()
    return cm

def plot_roc_curve(all_labels, all_probs, n_classes):
  # On transforme les labels en format binaire (One-vs-Rest)
  y_test_bin = label_binarize(all_labels, classes=range(n_classes))
  y_score = np.array(all_probs) 

  # 2. Calcul de la Micro-moyenne
  # .ravel() permet d'aplatir les matrices pour traiter toutes les classes ensemble
  fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
  roc_auc_micro = auc(fpr_micro, tpr_micro)
  np.savez(f'roc_curve.npz', fpr=fpr_micro, tpr=tpr_micro, auc=roc_auc_micro)

  # 3. Affichage
  plt.figure(figsize=(6, 6))

  plt.plot(fpr_micro, tpr_micro,
          label='Courbe ROC micro-moyenne (aire = {0:0.2f})'.format(roc_auc_micro),
          color='red', linestyle='-')

  # Ligne de chance
  plt.plot([0, 1], [0, 1], 'b--', lw=2)

  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Taux de Faux Positifs (FPR)')
  plt.ylabel('Taux de Vrais Positifs (TPR)')
  plt.title('Performance Globale : Courbe ROC Moyenne')
  plt.legend(loc="lower right")
  plt.grid(alpha=0.3)
  plt.show()

def plot_precision_recall_curve(all_labels, all_probs, n_classes):
  # 1. Préparation des données (on reprend tes variables)
  n_classes = 6
  y_test_bin = label_binarize(all_labels, classes=range(n_classes))
  y_score = np.array(all_probs)

  # 2. Calcul de la Micro-moyenne pour Précision-Rappel
  # .ravel() transforme les matrices (N, 6) en vecteurs plats (N*6,)
  precision_micro, recall_micro, _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
  average_precision_micro = average_precision_score(y_test_bin.ravel(), y_score.ravel())
  np.savez('precision_recall.npz', precision=precision_micro, recall=recall_micro, ap=average_precision_micro)

  # 3. Tracé du graphique
  plt.figure(figsize=(7, 6))

  plt.plot(recall_micro, precision_micro, color='blue',
          label='Micro-average Precision-recall (AP = {0:0.2f})'.format(average_precision_micro))

  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Rappel (Recall)')
  plt.ylabel('Précision (Precision)')
  plt.title('Courbe Rappel-Précision Moyenne (Globale)')
  plt.legend(loc="lower left")
  plt.grid(True, alpha=0.3)
  plt.show()