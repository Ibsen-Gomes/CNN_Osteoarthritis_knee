# ============================== 
# 1. Importação das bibliotecas necessárias 
# ==============================

from fastapi import FastAPI, File, UploadFile  # FastAPI para criar a API
import torch  # Para manipulação de redes neurais
from torchvision import transforms  # Para transformação de imagens
from PIL import Image  # Para abertura e manipulação de imagens
import io  # Para leitura de imagens em formato binário
from model.train import SimpleCNN  # Importação do modelo treinado

# ============================== 
# 2. Inicialização da API 
# ==============================

# Criar uma instância da API com FastAPI
app = FastAPI()

# Criar o modelo e carregar os pesos do modelo previamente treinado
model = SimpleCNN()  # Instância da arquitetura do modelo
model.load_state_dict(torch.load("model/model.pth", map_location=torch.device('cpu')))  # Carregar os pesos
model.eval()  # Coloca o modelo em modo de avaliação, desativando o cálculo de gradientes

# ============================== 
# 3. Definir transformações para o pré-processamento de imagens 
# ==============================

# Definir a transformação para pré-processar as imagens de entrada
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Converte para escala de cinza
    transforms.Resize((224, 224)),  # Redimensiona para 224x224 (tamanho esperado pelo modelo)
    transforms.ToTensor(),  # Converte a imagem para tensor PyTorch
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normaliza as imagens para o intervalo [-1, 1]
])

# ============================== 
# 4. Definir a rota para previsão 
# ==============================

@app.post("/predict/")  # Rota para receber a imagem e retornar a previsão
async def predict(file: UploadFile = File(...)):
    """
    Recebe uma imagem enviada pelo usuário, faz a previsão usando o modelo treinado e retorna a classe prevista.
    """
    try:
        # Ler a imagem enviada como um arquivo binário
        image = Image.open(io.BytesIO(await file.read()))  # Converte a imagem binária em objeto Image
        
        # Aplica as transformações para pré-processamento
        image = transform(image).unsqueeze(0)  # Adiciona uma dimensão extra para representar o batch (tamanho 1)

        # Faz a previsão com o modelo sem calcular os gradientes
        with torch.no_grad():
            output = model(image)  # Faz a inferência no modelo
            prediction = torch.argmax(output, dim=1).item()  # Obtém a classe com maior probabilidade

        # Mapear a previsão para as classes correspondentes
        classes = ["Normal", "Osteoartrite"]  # Classes de diagnóstico
        return {"prediction": classes[prediction]}  # Retorna a classe prevista

    except Exception as e:
        # Em caso de erro, retorna a mensagem de erro
        return {"error": str(e)}
