# tests/test_train.py
import torch
import sys
import os

# Adicionar o caminho da pasta "model" ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

# Agora a importação funcionará corretamente
from model import create_model  # ✅ Correto

# Adiciona o diretório raiz do projeto ao caminho de importação
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 🔹 Importa a função create_model do módulo model.train
#from model import train as custom_model

def test_model_output():
    """
    Testa a saída do modelo para garantir que ele retorna o formato esperado.
    """
    # 🔹 Cria uma instância do modelo com a arquitetura correta
    model = create_model()

    # 🔹 Gera uma entrada aleatória (dummy_input) simulando uma imagem de raio-X
    dummy_input = torch.randn(1, 1, 224, 224)  # (batch=1, canal=1, altura=224, largura=224)

    # 🔹 Passa a entrada pelo modelo para obter a saída
    output = model(dummy_input)

    # 🔹 Verifica se a forma da saída é (1, 2) -> 2 classes (normal e osteoporose)
    assert output.shape == (1, 2), f"❌ Erro: Formato de saída esperado (1,2), mas recebeu {output.shape}"

    print("✅ Teste passou: O modelo retorna a saída no formato esperado!")

# 🔹 Permite rodar o teste manualmente (caso queira testar localmente)
if __name__ == "__main__":
    test_model_output()