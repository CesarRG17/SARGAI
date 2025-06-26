import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import matplotlib.pyplot as plt

# Configuraciones
BATCH_SIZE = 32
NUM_EPOCHS = 40
IMG_SIZE = 224
IMG_DIR = "images_public"
CSV_FILE = "Entrenamiento.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ['nada', 'bajo', 'moderado', 'abundante', 'excesivo']
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}

# Dataset personalizado
class SargazoDataset(Dataset):
    def __init__(self, dataframe, images_path, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = images_path
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.loc[idx, 'Id'])
        image = Image.open(img_name).convert("RGB")
        label = LABEL2ID[self.dataframe.loc[idx, 'label']]
        if self.transform:
            image = self.transform(image)
        return image, label

def entrenar():
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Cargar y separar datos
    df = pd.read_csv(CSV_FILE)
    df = df[df['label'].isin(LABELS)]
    train_df, val_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)

    # Pesos por clase
    train_labels = train_df['label'].map(LABEL2ID).values
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

    # Datasets y Loaders
    train_dataset = SargazoDataset(train_df, IMG_DIR, transform=train_transform)
    val_dataset = SargazoDataset(val_df, IMG_DIR, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Modelo
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    model = model.to(DEVICE)

    # Entrenamiento
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_acc, val_acc = [], []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        correct = total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
        acc = correct / total
        train_acc.append(acc)
        print(f"Train Accuracy: {acc:.4f}")

        # Validación
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)
        val_epoch_acc = correct / total
        val_acc.append(val_epoch_acc)
        print(f"Validation Accuracy: {val_epoch_acc:.4f}")

    # Guardar modelo
    torch.save(model.state_dict(), "modelo_sargazo_final.pt")
    print("✅ Modelo guardado como modelo_sargazo_final.pt")

    # Evaluación
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Matriz de Confusión - Validación")
    plt.savefig("matriz_confusion_val.png")

    # Gráfica de precisión
    plt.figure()
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Precisión por Época")
    plt.savefig("grafica_accuracy_val.png")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    entrenar()
