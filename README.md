# Previsão de Ondas Quadradas com Redes Neurais Multi-Layer Perceptron

## Descrição

Este repositório consiste em um script para gerar um conjunto de dados sintético, um notebook para treinar um modelo usando PyTorch e outro notebook para otimizar modelos com Optuna. Dois conjuntos de dados são fornecidos: `dataset.json` para treinamento e `newData.json` para validação visual ao final do treinamento, ambos gerados pelo script aqui fornecido.

## Estrutura do Repositório

- `dataset_generator.py`: Script em Python para gerar um conjunto de dados sintético.
- `model_trainning.ipynb`: Notebook Jupyter para treinar um modelo usando PyTorch.
- `tunning.ipynb`: Notebook Jupyter para otimizar modelos com Optuna.
- `dataset.json`: Conjunto de dados para treinamento.
- `newData.json`: Conjunto de dados para validação visual.

## Uso

1. Caso deseje gerar um novo dataset, execute o script `dataset_generator.py` para gerar um conjunto de dados sintético.
2. Abra o notebook `model_trainning.ipynb` para treinar o modelo com o conjunto de dados gerado.
3. Ainda no notebook, utilize o conjunto de dados `new_data.json` para uma validação visual ao final do treinamento Sinta-se livre para gerar novos dados para visualização.
4. Utilize o notebook `tunning.ipynb` para explorar os potenciais do seu modelo.

## Requisitos

- Python 3.8
- PyTorch
- Numpy
- Scikit-Learn
- Matplotlib
- Optuna

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir problemas ou enviar pull requests.

## Licença

Este projeto é distribuído sob a [Licença MIT](LICENSE). Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

