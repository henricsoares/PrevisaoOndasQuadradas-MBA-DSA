# Previsão de Ondas Quadradas com Redes Neurais Multi-Layer Perceptron

## Descrição

Este repositório consiste em um script para gerar um conjunto de dados sintético e um notebook para treinar um modelo usando PyTorch. Dois conjuntos de dados são fornecidos: `dataset.json` para treinamento e `new_data.json` para validação visual ao final do treinamento, ambos gerados pelo script aqui fornecido.

## Estrutura do Repositório

- `dataset_generator.py`: Script em Python para gerar um conjunto de dados sintético.
- `model_trainning.ipynb`: Notebook Jupyter para treinar um modelo usando PyTorch.
- `dataset.json`: Conjunto de dados para treinamento.
- `new_data.json`: Conjunto de dados para validação visual.

## Uso

1. Caso deseje gerar um novo dataset, execute o script `dataset_generator.py` para gerar um conjunto de dados sintético.
2. Abra o notebook `model_trainning.ipynb` para treinar o modelo com o conjunto de dados gerado.
3. Ainda no notebook, utilize o conjunto de dados `new_data.json` para uma validação visual ao final do treinamento Sinta-se livre para gerar novos dados para visualização.

## Requisitos

- Python 3.8
- PyTorch
- Numpy
- Scikit-Learn
- Matplotlib

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir problemas ou enviar pull requests.

## Licença

Este projeto é distribuído sob a [Licença MIT](LICENSE). Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

