# ==============================
# Importação das bibliotecas necessárias
# ==============================
import torch
import torch.nn as nn
from torchvision import models

# ==============================
# Definição da classe ModifiedResNet18
# ==============================
"""
A classe ModifiedResNet18 é baseada na arquitetura ResNet-18, um modelo de deep learning amplamente utilizado
para tarefas de classificação de imagens. A ResNet-18 faz parte da família de Redes Residuais (ResNet), 
introduzida por He et al. em 2015, e é caracterizada pelo uso de conexões residuais (skip connections), 
que permitem um treinamento mais eficiente e profundo.

Modificações feitas na ResNet-18 original:
1. Substituição da primeira camada convolucional para aceitar imagens em escala de cinza (1 canal ao invés de 3).
2. Remoção da camada totalmente conectada (FC) original para permitir um ajuste personalizado da arquitetura.
3. Adição de camadas convolucionais extras para aumentar a capacidade de extração de características.
4. Cálculo dinâmico da entrada da nova camada totalmente conectada (FC), garantindo compatibilidade com diferentes entradas.
5. Nova camada totalmente conectada para realizar a classificação binária (osteoartrite vs normal).

A ResNet-18 é uma escolha comum devido ao seu bom equilíbrio entre desempenho e eficiência computacional,
sendo composta por 18 camadas treináveis, incluindo blocos residuais que ajudam a mitigar problemas de desaparecimento
do gradiente (vanishing gradient).
"""
class ModifiedResNet18(nn.Module):
    def __init__(self):
        super(ModifiedResNet18, self).__init__()
        
        # ==============================
        # 1. Carregar e modificar a ResNet-18
        # ==============================
        # Carrega a arquitetura ResNet-18 sem pesos pré-treinados
        self.model = models.resnet18(pretrained=False)
        
        # Modifica a primeira camada convolucional para aceitar imagens em escala de cinza (1 canal)
        self.model.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Obtém o número de características de entrada da camada totalmente conectada original
        num_ftrs = self.model.fc.in_features

        # Remove a camada totalmente conectada original para ser substituída posteriormente
        self.model.fc = nn.Identity()

        # ==============================
        # 2. Adicionar novas camadas convolucionais
        # ==============================
        self.additional_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Conv2D que expande os canais de 16 para 32
            nn.ReLU(inplace=True),  # Função de ativação ReLU
            nn.MaxPool2d(kernel_size=2, stride=2),  # Redução de dimensionalidade pela metade
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Outra convolução, expandindo para 64 canais
            nn.ReLU(inplace=True),  # Ativação ReLU
            nn.MaxPool2d(kernel_size=2, stride=2)  # Redução de dimensionalidade
        )
        
        # ==============================
        # 3. Determinar automaticamente o tamanho da entrada da camada Linear
        # ==============================
        self._calculate_fc_input_size()

        # ==============================
        # 4. Definir a nova camada totalmente conectada (FC)
        # ==============================
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),  # Primeira camada totalmente conectada (FC)
            nn.ReLU(),  # Função de ativação ReLU
            nn.Dropout(0.3),  # Dropout para evitar overfitting
            nn.Linear(128, 2)  # Camada final com 2 neurônios (classificação binária: osteoartrite vs. normal)
        )

    def _calculate_fc_input_size(self):
        """
        Método para calcular dinamicamente o tamanho da entrada da camada totalmente conectada.
        Ele passa uma imagem fictícia pela rede para determinar a dimensão da saída após as convoluções.
        """
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 224, 224)  # Simulação de uma imagem 224x224 em escala de cinza
            x = self.model.conv1(dummy_input)  # Passa pela primeira convolução da ResNet modificada
            x = self.additional_conv(x)  # Passa pelas camadas adicionais
            self.fc_input_size = x.view(1, -1).size(1)  # Calcula a dimensão após achatamento

    def forward(self, x):
        """
        Método de passagem direta (forward) que define o fluxo de dados na rede neural.
        """
        x = self.model.conv1(x)  # Passa pela primeira convolução da ResNet modificada
        x = self.additional_conv(x)  # Passa pelas camadas convolucionais adicionais
        
        x = x.view(x.size(0), -1)  # Achata os dados antes da camada totalmente conectada (FC)
        x = self.fc(x)  # Passa pelas camadas FC para obter a saída final
        return x

# ==============================
# 5. Função auxiliar para criar o modelo
# ==============================
def create_model():
    """
    Função para instanciar e retornar o modelo ModifiedResNet18.
    """
    model = ModifiedResNet18()
    return model

