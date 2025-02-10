# Osteo-CNN

**Osteo-CNN** é um projeto open-source que utiliza redes neurais convolucionais (CNNs) para identificar osteoartrite em imagens de raio-X. O modelo foi treinado utilizando um banco de dados de imagens do Kaggle e está integrado com uma infraestrutura de CI/CD via GitHub Actions.

## Visão Geral

O objetivo deste projeto é utilizar deep learning para auxiliar no diagnóstico da osteoartrite por meio da análise de imagens de raio-X. O modelo foi treinado em PyTorch, com a utilização de transformações de imagens e técnicas avançadas de aprendizado de máquina.

## Funcionalidades

- **Treinamento e Inferência**: O modelo pode ser treinado e utilizado para realizar previsões sobre imagens de raio-X, classificando-as como "Raio-X Normal" ou "Raio-X com Osteoartrite".
- **Pipeline de CI/CD com GitHub Actions**: Utiliza GitHub Actions para automação do pipeline de treinamento e teste, facilitando a reprodutibilidade e o controle de versões.
- **Interface de Predição Local**: A aplicação permite a realização de previsões de imagens localmente, utilizando o modelo hospedado no GitHub.

## Tecnologias

- **Deep Learning**: PyTorch
- **Modelo**: Redes Neurais Convolucionais (CNN)
- **CI/CD**: GitHub Actions
- **Banco de Dados**: Kaggle (imagens de raio-X)
- **Bibliotecas**:
  - `torch` e `torchvision` para treinamento e inferência
  - `PIL` para manipulação de imagens
  - `tkinter` para interface de seleção de imagens (local)
  - `requests` para download do modelo hospedado no GitHub

## Instalação

### Pré-requisitos

Certifique-se de que o Python 3.x está instalado. Você também precisará de um ambiente de virtualização, como o **venv** ou **conda**, para instalar as dependências.

1. Clone o repositório:

   ```bash
   git clone https://github.com/Ibsen-Gomes/Osteo-CNN.git
   cd Osteo-CNN
2. Crie e ative um ambiente virtual:

   - Para **venv**:

     ```bash
     python3 -m venv venv
     source venv/bin/activate  # Linux/Mac
     venv\Scripts\activate     # Windows
     ```

   - Para **conda**:

     ```bash
     conda create --name osteo-cnn python=3.8
     conda activate osteo-cnn
     ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt

## Uso:

1. Baixe o modelo treinado do GitHub;
2. O modelo será automaticamente baixado ao executar o script predict.py. 


## Próximos Passos:
Hospedagem de Modelo na Nuvem: Futuramente, o projeto será adaptado para usar cloud computing para armazenar um modelo mais complexo, superando a limitação de armazenamento no GitHub.
Integração com Aplicação Web: Em breve, o modelo será integrado a uma aplicação web para facilitar o uso por parte de médicos e profissionais da saúde.
