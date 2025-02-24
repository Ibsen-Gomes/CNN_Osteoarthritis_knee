# ==============================
# Importação das bibliotecas necessárias
# ==============================
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

# Adiciona o caminho do diretório raiz para importação do modelo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import create_model  # ✅ Importa a arquitetura da CNN personalizada baseada na ResNet-18

# ==============================
# 1. Pré-processamento e Aumento de Dados (Data Augmentation)
# ==============================
"""
A normalização e o aumento de dados (data augmentation) são etapas cruciais no treinamento de redes neurais,
especialmente para melhorar a capacidade de generalização do modelo e reduzir overfitting.
"""

# 🔹 Transformações para o conjunto de treinamento
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 🔹 Converte imagens para escala de cinza (1 canal)
    transforms.Resize((224, 224)),  # 🔹 Redimensiona para 224x224 (entrada padrão da ResNet-18)
    transforms.RandomRotation(15),  # 🔹 Rotação aleatória para aumentar diversidade
    transforms.RandomHorizontalFlip(),  # 🔹 Espelhamento aleatório
    transforms.ToTensor(),  # 🔹 Converte para tensor PyTorch
    transforms.Normalize(mean=[0.5], std=[0.5])  # 🔹 Normaliza os valores dos pixels (-1 a 1)
])

# 🔹 Transformações para o conjunto de teste (sem data augmentation)
transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ==============================
# 2. Carregar o Dataset e Criar Conjuntos de Treino/Teste
# ==============================
"""
O dataset é carregado utilizando a classe `ImageFolder`, que permite organizar os dados em pastas com rótulos.
A divisão 80%-20% garante que o modelo tenha uma quantidade adequada de dados para aprender e avaliar seu desempenho.
"""

# 🔹 Carregar dataset
dataset = datasets.ImageFolder(root='data', transform=transform_train)
test_dataset = datasets.ImageFolder(root='data', transform=transform_test)

# 🔹 Dividir dataset em 80% treino e 20% teste
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, _ = random_split(dataset, [train_size, test_size])

# 🔹 Criar DataLoaders para facilitar a leitura em lotes (batchs)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==============================
# 3. Criar o Modelo e Definir Hiperparâmetros
# ==============================
"""
A arquitetura da rede neural é baseada na ResNet-18 modificada.
A função `create_model()` retorna um modelo customizado, já adaptado para imagens em escala de cinza.
"""

# 🔹 Criar modelo CNN baseado na ResNet-18 modificada
model = create_model()

# 🔹 Definir função de perda e otimizador
criterion = nn.CrossEntropyLoss()  # 🔹 Perda utilizada para classificação multi-classe
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 🔹 Adam com L2 regularization (weight decay)

# 🔹 Agendador de taxa de aprendizado (reduz LR a cada 5 épocas)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# 🔹 Enviar modelo para GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ==============================
# 4. Treinamento do Modelo
# ==============================
"""
O modelo é treinado por 100 épocas, utilizando uma abordagem iterativa.
- `loss.backward()`: Computa os gradientes para cada parâmetro treinável.
- `optimizer.step()`: Atualiza os pesos do modelo com base nos gradientes.
- `scheduler.step()`: Ajusta a taxa de aprendizado para melhorar a convergência.

O **early stopping** interrompe o treinamento caso a precisão não melhore após 10 épocas consecutivas.
Isso evita que o modelo continue treinando desnecessariamente, poupando tempo e recursos computacionais.
"""

# 🔹 Iniciar contagem do tempo
start_time = time.time()
print("\n🚀 Iniciando treinamento...")

num_epochs = 100
best_accuracy = 0
early_stopping_counter = 0
early_stopping_patience = 10  # 🔹 Número de épocas sem melhora para interromper o treinamento

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # 🔹 Zera os gradientes acumulados
        outputs = model(inputs)  # 🔹 Forward pass
        loss = criterion(outputs, labels)  # 🔹 Calcula a perda
        loss.backward()  # 🔹 Calcula gradientes
        optimizer.step()  # 🔹 Atualiza pesos da rede
        running_loss += loss.item()  # 🔹 Acumula a perda total

    # 🔹 Atualiza o scheduler da taxa de aprendizado
    scheduler.step()

    # ==============================
    # 5. Avaliação do Modelo no Conjunto de Teste
    # ==============================
    """
    O modelo é avaliado no conjunto de teste sem atualização de gradientes (`torch.no_grad()`).
    - `torch.max(outputs, 1)`: Obtém a classe predita com maior probabilidade.
    - `correct / total`: Cálculo da precisão do modelo no conjunto de teste.
    """

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
            _, predicted = torch.max(outputs, 1)  # 🔹 Obtém a classe predita
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

# ==============================
# 6. Salvamento e Gerenciamento do Modelo Treinado
# ==============================
"""
O modelo treinado é salvo no diretório `model/` e seu tamanho é verificado.
Se for maior que 100 MB, ele será excluído para evitar consumo excessivo de armazenamento.
"""

MODEL_PATH = 'model/model.pth'
torch.save(model.state_dict(), MODEL_PATH)

# 🔹 Verificar tamanho do modelo salvo
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # 🔹 Convertendo para MB

if model_size_mb > 100:
    print(f"\n❌ O modelo ({model_size_mb:.2f} MB) excede 100 MB e não será salvo.")
    os.remove(MODEL_PATH)  # 🔹 Remove o arquivo grande
else:
    print(f"\n✅ Modelo treinado ({model_size_mb:.2f} MB) salvo em {MODEL_PATH}")

