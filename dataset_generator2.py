from math import ceil, floor
import numpy as np
import json

# Defina o tamanho do gráfico
x_min, x_max = 0, 6000
y_min, y_max = 0, 8
num_coords = int(input("Digite o número de subconjuntos: "))

# Defina a área onde os pontos serão plotados e sua densidade
num_points = 128
good_points = floor(num_points * 0.953125)
outliers = ceil(num_points * 0.046875)


# Defina os intervalos para os subsets
def gen_ranges(x_min, x_max, y_min, y_max):
    width = np.random.uniform(0.33 * x_max, 0.42 * x_max)
    height = np.random.uniform(0.62 * y_max, 0.87 * y_max)
    location = np.random.uniform(0.1 * x_max, (0.9 * x_max) - width)

    centerRange = [
        location,
        location + width,
        y_min,
        height,
    ]

    leftRange = [
        x_min,
        location * 0.99,
        y_min,
        np.random.uniform(0.15, 0.25) * y_max,
    ]

    rightRange = [
        location + width * 1.01,
        x_max,
        y_min,
        np.random.uniform(0.15, 0.25) * y_max,
    ]

    return (
        leftRange,
        centerRange,
        rightRange,
    )


def gen_oRanges(thrshldLow, thrshldHigh, distMin, distMax):
    olCenterRange = [
        distMin,
        distMax,
        thrshldHigh * 1.1,
        y_max * 0.99,
    ]

    olLeftRange = [x_min * 1.01, distMin * 0.9, thrshldLow * 1.1, y_max * 0.99]

    olRightRange = [
        distMax * 1.1,
        x_max * 0.99,
        thrshldLow * 1.1,
        y_max * 0.99,
    ]
    return (
        olLeftRange,
        olCenterRange,
        olRightRange,
    )


ranges1, ranges2, ranges3 = [], [], []

for _ in range(num_coords):
    lr, cr, rr = gen_ranges(x_min, x_max, y_min, y_max)
    ranges1.append(lr)
    ranges2.append(cr)
    ranges3.append(rr)

# Gere coordenadas x e y aleatórias
xy1_list = []
for rxy in ranges1:
    np.random.seed(42)
    xy1_list.append(
        [
            np.round(np.random.uniform(rxy[0], rxy[1], int(ceil(good_points / 4))), 2),
            np.round(np.random.uniform(rxy[2], rxy[3], int(good_points / 4)), 2),
        ]
    )

xy2_list = []

for rxy in ranges2:
    np.random.seed(42)
    xy2_list.append(
        [
            np.round(np.random.uniform(rxy[0], rxy[1], int(good_points / 2)), 2),
            np.round(np.random.uniform(rxy[2], rxy[3], int(good_points / 2)), 2),
        ]
    )

xy3_list = []

for rxy in ranges3:
    np.random.seed(42)
    xy3_list.append(
        [
            np.round(np.random.uniform(rxy[0], rxy[1], int(good_points / 4)), 2),
            np.round(np.random.uniform(rxy[2], rxy[3], int(ceil(good_points / 4))), 2),
        ]
    )

# coordenadas dos pontos
xy_list = []

for c1, c2, c3 in zip(xy1_list, xy2_list, xy3_list):
    x = np.hstack((c1[0], c2[0], c3[0]))
    y = np.hstack((c1[1], c2[1], c3[1]))
    xy_list.append([x, y])

# parametros das linhas
x2min = [xy[0].min() for xy in xy2_list]
x2max = [xy[0].max() for xy in xy2_list]
y1max = [xy[1].max() for xy in xy1_list]
y2max = [xy[1].max() for xy in xy2_list]
y3max = [xy[1].max() for xy in xy3_list]

thrshldLow = np.round(
    [y1 * 1.1 if y1 > y3 else y3 * 1.1 for y1, y3 in zip(y1max, y3max)], 2
)

thrshldHigh = np.round([y2 * 1.03 for y2 in y2max], 2)
distMin = np.round([x2 * 0.95 for x2 in x2min], 2)
distMax = np.round([x2 * 1.05 for x2 in x2max], 2)

x_line = [
    [x_min, dmin, dmin, dmax, dmax, x_max] for dmin, dmax in zip(distMin, distMax)
]

y_line = [[tl, tl, th, th, tl, tl] for tl, th in zip(thrshldLow, thrshldHigh)]

# Coordenadas dos outliers

oRanges1, oRanges2, oRanges3 = [], [], []

for _ in range(num_coords):
    lr, cr, rr = gen_oRanges(thrshldLow, thrshldHigh, distMin, distMax)
    oRanges1.append(lr)
    oRanges2.append(cr)
    oRanges3.append(rr)

xy1_list_ = []
for rxy in oRanges1:
    np.random.seed(42)
    xy1_list_.append(
        [
            np.round(np.random.uniform(rxy[0], rxy[1], int(outliers / 3)), 2),
            np.round(np.random.uniform(rxy[2], rxy[3], int(outliers / 3)), 2),
        ]
    )

xy2_list_ = []

for rxy in oRanges2:
    np.random.seed(42)
    xy2_list_.append(
        [
            np.round(np.random.uniform(rxy[0], rxy[1], int(outliers / 3)), 2),
            np.round(np.random.uniform(rxy[2], rxy[3], int(outliers / 3)), 2),
        ]
    )

xy3_list_ = []

for rxy in oRanges3:
    np.random.seed(42)
    xy3_list_.append(
        [
            np.round(np.random.uniform(rxy[0], rxy[1], int(outliers / 3)), 2),
            np.round(np.random.uniform(rxy[2], rxy[3], int(outliers / 3)), 2),
        ]
    )


xy_list_ = []
for c1, c2, c3 in zip(xy1_list_, xy2_list_, xy3_list_):
    x = np.hstack((c1[0], c2[0], c3[0]))
    y = np.hstack((c1[1], c2[1], c3[1]))
    xy_list_.append([x, y])


# Junte os outliers aos demais pontos

for xyl, xyl_ in zip(xy_list, xy_list_):
    xyl[0] = np.concatenate((xyl[0], xyl_[0]))
    xyl[1] = np.concatenate((xyl[1], xyl_[1]))

# Formate as coordenadas para o formato desejado
coords = []
for c in xy_list:
    coords.append([(x, y) for x, y in zip(c[0].tolist(), c[1].tolist())])

# Crie um dicionário para armazenar os dados
# no formato { dados: [ coordenadas: [...], parametros: [...] ], ... }

dados: dict = {"dados": []}
for c, tl, th, dmn, dmx in zip(coords, thrshldLow, thrshldHigh, distMin, distMax):
    dados["dados"].append(
        {
            "coords": c,
            "params": (tl, th, dmn, dmx),
        }
    )

# Salve os dados em um arquivo JSON
file = input("Digite o nome do conjunto de dados: ")
with open(f"{file}.json", "w") as arquivo:
    json.dump(dados, arquivo)
