import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import sys
import time  
from torch.optim.lr_scheduler import StepLR  # 🔹 Agendador de taxa de aprendizado
import numpy as np  # 🔹 Necessário para Early Stopping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import create_model  # ✅ Importa a arquitetura da CNN

# 🔹 Aplicação de Data Augmentation e Normalização
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),  # 🔹 Rotação aleatória de até 15 graus
    transforms.RandomHorizontalFlip(),  # 🔹 Espelhamento aleatório
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 🔹 Carregar dataset
dataset = datasets.ImageFolder(root='data', transform=transform_train)
test_dataset = datasets.ImageFolder(root='data', transform=transform_test)

# 🔹 Dividir em treino (80%) e teste (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, _ = random_split(dataset, [train_size, test_size])

# 🔹 Criar DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 🔹 Criar o modelo
model = create_model()

# 🔹 Definir função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 🔹 Adicionando weight decay para regularização L2

# 🔹 Agendador de taxa de aprendizado (reduz LR a cada 5 épocas)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# 🔹 Enviar modelo para GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🔹 Iniciar contagem do tempo
start_time = time.time()
print("\n🚀 Iniciando treinamento...")

num_epochs = 100
best_accuracy = 0
early_stopping_counter = 0
early_stopping_patience = 10  # 🔹 Para interromper se não houver melhora por 3 épocas

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 🔹 Atualiza o scheduler da taxa de aprendizado
    scheduler.step()

    # 🔹 Avaliar no conjunto de teste
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Obtém a classe predita
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_train_loss = running_loss / len(train_loader)
    avg_test_loss = test_loss / len(test_loader)

    print(f"✅ Epoch {epoch+1}/{num_epochs}, 🎯 Precisão: {accuracy:.2f}%, 🔽 Loss Treino: {avg_train_loss:.4f}, 🔽 Loss Teste: {avg_test_loss:.4f}")

    # 🔹 Implementação do Early Stopping
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        early_stopping_counter = 0  # 🔹 Reseta o contador de paciência
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        print("\n⏳ Early Stopping ativado: Nenhuma melhora por 10 épocas consecutivas. Encerrando treinamento.")
        break

# 🔹 Finalizar contagem do tempo
end_time = time.time()
execution_time = end_time - start_time
print(f"\n⏳ Tempo total de treinamento: {execution_time:.2f} segundos")

# 🔹 Salvar o modelo treinado temporariamente
MODEL_PATH = 'model/model.pth'
torch.save(model.state_dict(), MODEL_PATH)

# 🔹 Verificar tamanho do modelo
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Convertendo para MB

if model_size_mb > 100:
    print(f"\n❌ O modelo ({model_size_mb:.2f} MB) excede 100 MB e não será salvo.")
    os.remove(MODEL_PATH)  # Remove o arquivo grande
else:
    print(f"\n✅ Modelo treinado ({model_size_mb:.2f} MB) e salvo em {MODEL_PATH}")
