from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms
from PIL import Image
import io
from model.train import SimpleCNN  # Certifique-se de importar a mesma arquitetura usada no treinamento

# Inicializar a API
app = FastAPI()

# Criar o modelo e carregar os pesos corretamente
model = SimpleCNN()  # Usa a mesma arquitetura definida em train.py
model.load_state_dict(torch.load("model/model.pth", map_location=torch.device('cpu')))
model.eval()

# Definir transformações para preprocessamento da imagem
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@app.post("https://deep-learning-pytorch-ci-cd-1.onrender.com/predict/") ###MUDAR ISSSOOOOO!!!!
async def predict(file: UploadFile = File(...)):
    """Recebe uma imagem e retorna a previsão do modelo"""
    try:
        # Ler a imagem enviada pelo usuário
        image = Image.open(io.BytesIO(await file.read()))
        
        # Converter para escala de cinza e aplicar transformações
        image = transform(image).unsqueeze(0)  # Adiciona dimensão batch

        # Fazer a previsão
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()

        # Mapear saída do modelo para as classes
        classes = ["Normal", "Osteoartrite"]
        return {"prediction": classes[prediction]}

    except Exception as e:
        return {"error": str(e)}