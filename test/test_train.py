# ============================== 
# 1. Importação das Bibliotecas Necessárias
# ==============================

import torch  # 🔹 Biblioteca principal para operações com tensores e redes neurais
import sys    # 🔹 Usada para modificar o caminho de importação
import os     # 🔹 Utilizada para manipulação de arquivos e diretórios

# ============================== 
# 2. Ajustes no Caminho de Importação
# ==============================

# Adiciona o caminho da pasta "model" ao sys.path para que o código da pasta model possa ser importado corretamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

# Agora a importação funcionará corretamente, permitindo acessar as funções da pasta "model"
from model import create_model  # ✅ Correto: importando a função que cria o modelo

# ============================== 
# 3. Função de Teste para Validar a Saída do Modelo
# ==============================

def test_model_output():
    """
    Testa a saída do modelo para garantir que ele retorna o formato esperado.
    """
    # 🔹 Criação de uma instância do modelo
    # Aqui, estamos chamando a função create_model() para instanciar o modelo com a arquitetura configurada
    model = create_model()

    # 🔹 Geração de uma entrada de teste simulando uma imagem de raio-X
    # Usamos torch.randn para criar uma entrada de 1x1x224x224 com valores aleatórios,
    # simulando uma imagem de 1 canal (escala de cinza) e tamanho 224x224
    dummy_input = torch.randn(1, 1, 224, 224)  # (batch=1, canal=1, altura=224, largura=224)

    # 🔹 Passando a entrada através do modelo para obter a saída
    output = model(dummy_input)

    # 🔹 Verificando se a forma da saída está correta: Esperamos um tensor de tamanho (1, 2), indicando 2 classes
    # O modelo deve gerar a probabilidade de cada uma das duas classes: normal e osteoporose
    assert output.shape == (1, 2), f"❌ Erro: Formato de saída esperado (1,2), mas recebeu {output.shape}"

    # 🔹 Se o teste passar, imprime uma mensagem de sucesso
    print("✅ Teste passou: O modelo retorna a saída no formato esperado!")

# ============================== 
# 4. Execução do Teste Manual
# ==============================

# Permite rodar o teste manualmente quando o script for executado diretamente
if __name__ == "__main__":
    test_model_output()
