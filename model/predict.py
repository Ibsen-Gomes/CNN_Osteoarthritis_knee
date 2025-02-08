# model/predict.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import os
import sys
import tkinter as tk
from tkinter import filedialog

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import create_model  # ✅ Agora a importação funcionará

# 🔹 URL do modelo armazenado no GitHub Actions, agora na branch 'main'
GITHUB_MODEL_URL = "https://github.com/Ibsen-Gomes/Osteo-CNN/raw/main/model/model.pth"

# 🔹 Caminho para salvar o modelo baixado localmente
MODEL_PATH = "model/model.pth"

# 🔹 Baixar modelo treinado do GitHub Actions
def download_model():
    """ Faz o download do modelo treinado da branch 'main' do GitHub. """
    if not os.path.exists(MODEL_PATH):  # Evita baixar se já existir
        print("🔽 Baixando modelo treinado da branch 'main' no GitHub...")
        response = requests.get(GITHUB_MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("✅ Modelo baixado com sucesso!")
        else:
            print("❌ Erro ao baixar o modelo. Verifique a URL do GitHub Actions e a branch 'main'.")
            sys.exit(1)
    else:
        print("✅ Modelo já disponível localmente.")

# 🔹 Baixar o modelo antes de carregar
download_model()

# Criar o modelo idêntico ao usado no treinamento
model = create_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")), strict=False)
model.eval()

# 🔹 Definir transformações para imagens de entrada
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_image(image_path):
    """ Realiza previsão de uma única imagem """
    image = Image.open(image_path).convert("L")  # Converter para tons de cinza
    image = transform(image).unsqueeze(0)  # Aplicar transformações e adicionar dimensão batch

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    classes = ['Raio-x Normal', 'Raio-x com Osteoartrite']
    print(f"📌 Resultado: {classes[prediction]}")

if __name__ == "__main__":
    # Criar janela oculta do tkinter
    root = tk.Tk()
    root.withdraw()  # Ocultar a janela principal

    # Abrir caixa de diálogo para selecionar imagem
    image_path = filedialog.askopenfilename(
        title="Selecione uma imagem de raio-X",
        filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp")]
    )

    # Verificar se o usuário escolheu um arquivo
    if image_path:
        predict_image(image_path)
    else:
        print("❌ Nenhuma imagem selecionada. Encerrando...")
