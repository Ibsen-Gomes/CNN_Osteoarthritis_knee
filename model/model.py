import torch
import torch.nn as nn
from torchvision import models

class ModifiedResNet18(nn.Module):
    def __init__(self):
        super(ModifiedResNet18, self).__init__()
        # Usar a ResNet18 e modificar a camada de entrada para 1 canal (escala de cinza)
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()  # Remover a camada totalmente conectada original

        # Camadas adicionais convolucionais
        self.additional_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Determinar o tamanho da entrada da camada Linear automaticamente
        self._calculate_fc_input_size()

        # Camada totalmente conectada
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),  # Ajuste dinâmico com base no tamanho calculado
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 2 classes: osteoartrite vs normal
        )

    def _calculate_fc_input_size(self):
        # Passar uma imagem fictícia pela rede para calcular a forma da saída após convoluções
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 224, 224)  # Supondo que a imagem de entrada seja 224x224
            x = self.model.conv1(dummy_input)
            x = self.additional_conv(x)
            self.fc_input_size = x.view(1, -1).size(1)  # Quantidade de valores após o flatten

    def forward(self, x):
        # Passar pela ResNet sem a camada fc original
        x = self.model.conv1(x)
        x = self.additional_conv(x)
        
        # Flatten para a camada totalmente conectada
        x = x.view(x.size(0), -1)  # Achatar
        x = self.fc(x)
        return x

def create_model():
    model = ModifiedResNet18()
    return model