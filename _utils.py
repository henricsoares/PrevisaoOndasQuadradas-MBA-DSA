import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):

    def __init__(self, num_layers: int, units_fc: list):
        """Retorna uma NN com os parâmetros fornecidos.\
        Recebe: num_layers, units_fc."""
        super(NeuralNetwork, self).__init__()
        self.num_layers = num_layers
        self.num_coords = 128
        self.num_dims = 2
        self.num_params = 4
        self.layers = nn.ModuleList(
            [
                nn.Flatten(),
                nn.Linear(self.num_coords * self.num_dims, self.num_coords),
                nn.Linear(self.num_coords, units_fc[0]),
            ]
        )
        for i in range(1, num_layers):
            self.layers.append(nn.Linear(units_fc[i - 1], units_fc[i]))
        self.layers.append(nn.Linear(units_fc[-1], self.num_params))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def train_and_validate_with_kfold(
    modelo, train_data, train_target, num_epochs, num_folds, batch_size
):
    """Treina e valida com k-fold, recebe: modelo,
    train_data, train_target, num_epochs, num_folds, batch_size"""
    train_losses = []
    val_losses = []
    r2_train_values = []
    r2_val_values = []
    k_fold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=0.001)
    for fold, (train_indices, val_indices) in enumerate(
        k_fold.split(range(len(train_data)))
    ):
        # Configure os conjuntos de dados e rótulos
        x_train_set = np.array(Subset(train_data, train_indices))
        y_train_set = np.array(Subset(train_target, train_indices))
        x_val_set = np.array(Subset(train_data, val_indices))
        y_val_set = np.array(Subset(train_target, val_indices))

        x_train_tensor = torch.Tensor(x_train_set)
        y_train_tensor = torch.Tensor(y_train_set)
        x_val_tensor = torch.Tensor(x_val_set)
        y_val_tensor = torch.Tensor(y_val_set)

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False)

        running_loss = 0.0
        running_val_loss = 0.0

        # Treine e valide o modelo
        for epoch in range(int(num_epochs / num_folds)):
            modelo.train()
            all_predictions = []
            all_targets = []

            # Loop de treinamento
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = modelo(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                all_predictions.extend(outputs.cpu().detach().numpy())
                all_targets.extend(batch_y.cpu().detach().numpy())

            train_losses.append(loss.item())

            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            r2 = r2_score(all_targets, all_predictions)
            r2_train_values.append(r2)

            # Loop de validação
            modelo.eval()
            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = modelo(batch_x)
                    val_loss = criterion(outputs, batch_y)
                    running_val_loss += val_loss.item()
                    all_predictions.extend(outputs.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
                val_losses.append(val_loss.item())
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            r2 = r2_score(all_targets, all_predictions)
            r2_val_values.append(r2)

        average_train_loss = np.mean(train_losses)
        average_val_loss = np.mean(val_losses)
        average_train_r2 = np.mean(r2_train_values)
        average_val_r2 = np.mean(r2_val_values)
    print(
        f"average train_loss: {average_train_loss}, \n"
        f"average val_loss: {average_val_loss}, \n"
        f"average train R²: {average_train_r2}, \n"
        f"average val R²: {average_val_r2}\n"
    )
    return modelo, [
        train_losses,
        val_losses,
        r2_train_values,
        r2_val_values,
    ]


def train_and_validade(
        modelo, train_data, train_target, test_data,
        test_target, num_epochs, batch_size):

    """Treina e valida sem k-fold, recebe: modelo,
    train_data, train_target, test_data,
    test_target, num_epochs, batch_size"""

    train_losses = []
    val_losses = []
    r2_train_values = []
    r2_val_values = []
    criterion = nn.MSELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=0.001)
    x_train_tensor = torch.Tensor(train_data)
    y_train_tensor = torch.Tensor(train_target)
    x_val_tensor = torch.Tensor(test_data)
    y_val_tensor = torch.Tensor(test_target)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    running_loss = 0.0
    running_val_loss = 0.0

    # Treine e valide o modelo
    for _ in range(num_epochs):
        modelo.train()
        all_predictions = []
        all_targets = []

        # Loop de treinamento
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = modelo(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            all_predictions.extend(outputs.cpu().detach().numpy())
            all_targets.extend(batch_y.cpu().detach().numpy())

        train_losses.append(loss.item())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        r2 = r2_score(all_targets, all_predictions)
        r2_train_values.append(r2)

        # Loop de validação
        modelo.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = modelo(batch_x)
                val_loss = criterion(outputs, batch_y)
                running_val_loss += val_loss.item()
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
            val_losses.append(val_loss.item())
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        r2 = r2_score(all_targets, all_predictions)
        r2_val_values.append(r2)

    average_train_loss = np.mean(train_losses)
    average_val_loss = np.mean(val_losses)
    average_train_r2 = np.mean(r2_train_values)
    average_val_r2 = np.mean(r2_val_values)
    print(
        f"average train_loss: {average_train_loss}, \n"
        f"average val_loss: {average_val_loss}, \n"
        f"average train R²: {average_train_r2}, \n"
        f"average val R²: {average_val_r2}\n"
    )
    return modelo, [
        train_losses,
        val_losses,
        r2_train_values,
        r2_val_values,
    ]


def plot_curvas_aprendizado(metricasA, metricasB):
    # Criar figura
    plt.figure(figsize=(8, 10))

    # Gráfico 1 (MSE e R² de Treino)
    plt.subplot(2, 1, 1)
    plt.plot(
        range(len(metricasA[1])),
        metricasA[1],
        marker="o",
        linestyle="-",
        label="MSE",
    )
    plt.plot(
        range(len(metricasA[3])),
        metricasA[3],
        marker="o",
        linestyle="-",
        label="R²",
    )
    plt.xlabel("Época", fontsize=12)
    plt.ylabel("Métricas", fontsize=12)
    plt.title(
        "A.   Curva de Validação Para o Modelo B",
        fontsize=12,
        loc="left",
    )
    plt.yscale("log")  # Escala logarítmica no eixo y para MSE
    plt.axhline(
        y=np.mean(metricasA[1]),
        color="r",
        linestyle="--",
        label=f"Média de MSE: {np.mean(metricasA[1]):.3f}",
    )
    plt.axhline(
        y=np.mean(metricasA[3]),
        color="b",
        linestyle="--",
        label=f"Média de R²: {np.mean(metricasA[3]):.3f}",
    )
    plt.legend(fontsize=12)
    plt.grid(True)

    # Gráfico 2 (MSE e R² de Validação)
    plt.subplot(2, 1, 2)
    plt.plot(
        range(len(metricasB[1])),
        metricasB[1],
        marker="o",
        linestyle="-",
        label="MSE (Validação)",
    )
    plt.plot(
        range(len(metricasB[3])),
        metricasB[3],
        marker="o",
        linestyle="-",
        label="R² (Validação)",
    )
    plt.xlabel("Época", fontsize=12)
    plt.ylabel("Métricas", fontsize=12)
    plt.title(
        "B.   Curva de Validação Para o Modelo C",
        fontsize=12,
        loc="left",
    )
    plt.yscale("log")  # Escala logarítmica no eixo y para MSE
    plt.axhline(
        y=np.mean(metricasB[1]),
        color="r",
        linestyle="--",
        label=f"Média de MSE (Validação): {np.mean(metricasB[1]):.3f}",
    )
    plt.axhline(
        y=np.mean(metricasB[3]),
        color="b",
        linestyle="--",
        label=f"Média de R² (Validação): {np.mean(metricasB[3]):.3f}",
    )
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("curva_prendizado_metricas_treino_validacao.png")
    plt.show()


def plot_pesos(modelo):
    pesos_fc1 = modelo.layers[1].weight.data.numpy().flatten()
    pesos_fc2 = modelo.layers[2].weight.data.numpy().flatten()
    pesos_fcn = modelo.layers[3].weight.data.numpy().flatten()

    plt.figure(figsize=(16, 6))

    # Gráfico 1
    plt.subplot(1, 3, 1)
    plt.hist(pesos_fc1, bins=50, color="blue", alpha=0.7)
    plt.title("Pesos na Camada Oculta 1", loc="left")

    # Gráfico 2
    plt.subplot(1, 3, 2)
    plt.hist(pesos_fc2, bins=50, color="green", alpha=0.7)
    plt.title("Pesos na Camada Oculta 2", loc="left")

    # Gráfico 3
    plt.subplot(1, 3, 3)
    plt.hist(pesos_fcn, bins=50, color="red", alpha=0.7)
    plt.title("Pesos na Camada de Saída", loc="left")

    plt.tight_layout()
    plt.savefig("pesos.png")
    plt.show()


def test_model_with_new_data(model, testLoader):
    """Testa o modelo com dados não utilziados para treinamento e validação."""
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    test_losses = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in testLoader:
            outputs = model(batch_x)
            test_loss += criterion(outputs, batch_y).item()
            test_losses.append(criterion(outputs, batch_y).item())
            all_predictions.extend(outputs.cpu().detach().numpy())
            all_targets.extend(batch_y.cpu().detach().numpy())
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    r2 = r2_score(all_targets, all_predictions)

    average_test_loss = test_loss / len(testLoader)
    print(f"average_test_loss: {average_test_loss}, average r2: {r2}")
    return test_losses


def previsao_without_normalization(modelo, dados):
    """Gera previsão para dados não normalizados"""
    pontos = [(d["coords"]) for d in dados["dados"]][0]
    params = [d["params"] for d in dados["dados"]]
    pontos_tensor = torch.Tensor(pontos)

    previsao = modelo(pontos_tensor.unsqueeze(0))
    mse = calcula_mse(previsao.detach().numpy(), params)
    print(f"MSE: {mse}")
    return previsao, pontos, params, mse


def previsao_with_normalization(modelo, dados, target, scaler):
    """Gera previsão para dados normalizados, requer o target e seu scaler"""
    pontos = [(d["coords"]) for d in dados["dados"]][0]
    params = [d["params"] for d in dados["dados"]]

    scaler.fit(pontos)

    pontos_n = scaler.transform(pontos)

    pontos_tensor = torch.Tensor(pontos_n)

    previsao = modelo(pontos_tensor.unsqueeze(0))
    scaler.fit(target)
    oParams = scaler.transform(params)

    mse = calcula_mse(previsao.detach().numpy(), oParams)
    print(f"MSE: {mse}")
    return previsao, pontos, params, mse


def calcula_mse(predictions, labels):
    mse = ((predictions - labels[0]) ** 2).mean()
    return mse.item()
