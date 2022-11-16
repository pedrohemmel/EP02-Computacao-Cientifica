import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



notas = pd.read_csv('https://raw.githubusercontent.com/celsocrivelaro/simple-datasets/main/notas-estudantes.csv')
print(notas)
#Principais variáveis que armazenam dados de entidades do BD
x1 = notas["nota_1"]
x2 = notas["nota_2"]
y = notas["resultado"]

#Função de criação do sigmoide
def sigmoide(x1, x2, a, b, c):
    return 1 / (1 + np.e ** -(a * x1 + b * x2 + c))

#Função que gera a perda com base nos dados inseridos
def cross_entropy(n, p, y):
    return (1 / n) * (np.sum(-y * np.log(p)) - np.sum((1 - y) * np.log(1 - p)))


def gradiente(x1, x2, y, qtd_iteracao=5000, alpha=1e-6, limite_parada=1e-6):
    a = 0.02
    b = 0.03
    c = 4
    n = float(len(x1))

    perda_ant = float('inf')
    arr_perdas = []
    arr_variancia_a = []
    arr_variancia_b = []
  
    for i in range(qtd_iteracao):
        p = sigmoide(x1, x2, a, b, c)

        perda_atual = cross_entropy(n, p, y)
      
        if (abs(perda_atual - perda_ant) <= limite_parada):
            return a, b, c, arr_perdas, arr_variancia_a, arr_variancia_b

        perda_ant = perda_atual

        arr_perdas.append(perda_atual)
        arr_variancia_a.append(a)
        arr_variancia_b.append(b)

        a_da = (1 / n) * np.sum(x1 * (p - y))
        b_db = (1 / n) * np.sum(x2 * (p - y))
        c_dc = (1 / n) * np.sum(p - y)

        a -= (alpha * a_da)
        b -= (alpha * b_db)
        c -= (alpha * c_dc)
    return a, b, c, arr_perdas, arr_variancia_a, arr_variancia_b

#Inicializando variáveis para plotagem
fig, (pri_graph, seg_graph, ter_graph, qrt_graph, qui_graph, sex_graph) = plt.subplots(6)


print("\nInciando analises\n")

print("\n\n\n#### Alpha = 0.1 e limite_parada = 1e-6 ####\n\n\n")

pri_a, pri_b, pri_c, pri_perdas, pri_variacao_a, pri_variacao_b = gradiente(x1, x2, y, qtd_iteracao=5000, alpha=0.1, limite_parada=1e-6)
print(f"\n\nPara Iteração 5000 - \nA: {pri_a}\nB: {pri_b}\nC: {pri_c}")

seg_a, seg_b, seg_c, seg_perdas, seg_variacao_a, seg_variacao_b = gradiente(x1, x2, y, qtd_iteracao=10000, alpha=0.1, limite_parada=1e-6)
print(f"\n\nPara Iteração 10000 - \nA: {seg_a}\nB: {seg_b}\nC: {seg_c}")

ter_a, ter_b, ter_c, ter_perdas, ter_variacao_a, ter_variacao_b = gradiente(x1, x2, y, qtd_iteracao=20000, alpha=0.1, limite_parada=1e-6)
print(f"\n\nPara Iteração 20000 - \nA: {ter_a}\nB: {ter_b}\nC: {ter_c}")

pri_index = np.arange(len(ter_perdas))
pri_graph.plot(pri_index, ter_perdas)
pri_graph.set_xlabel(" Iterações ")
pri_graph.set_ylabel(" Erro ")



print("\n\n\n#### Alpha = 0.01 e limite_parada = 1e-6 ####\n\n\n")

pri_a, pri_b, pri_c, pri_perdas, pri_variacao_a, pri_variacao_b = gradiente(x1, x2, y, qtd_iteracao=5000, alpha=0.01, limite_parada=1e-6)
print(f"\n\nPara Iteração 5000 - \nA: {pri_a}\nB: {pri_b}\nC: {pri_c}")

seg_a, seg_b, seg_c, seg_perdas, seg_variacao_a, seg_variacao_b = gradiente(x1, x2, y, qtd_iteracao=10000, alpha=0.01, limite_parada=1e-6)
print(f"\n\nPara Iteração 10000 - \nA: {seg_a}\nB: {seg_b}\nC: {seg_c}")

ter_a, ter_b, ter_c, ter_perdas, ter_variacao_a, ter_variacao_b = gradiente(x1, x2, y, qtd_iteracao=20000, alpha=0.01, limite_parada=1e-6)
print(f"\n\nPara Iteração 20000 - \nA: {ter_a}\nB: {ter_b}\nC: {ter_c}")

seg_index = np.arange(len(seg_perdas))
seg_graph.plot(seg_index, seg_perdas)
seg_graph.set_xlabel(" Iterações ")
seg_graph.set_ylabel(" Erro ")

print("\n\n\n#### Alpha = 0.001 e limite_parada = 1e-6 ####\n\n\n")

pri_a, pri_b, pri_c, pri_perdas, pri_variacao_a, pri_variacao_b = gradiente(x1, x2, y, qtd_iteracao=5000, alpha=0.001, limite_parada=1e-6)
print(f"\n\nPara Iteração 5000 - \nA: {pri_a}\nB: {pri_b}\nC: {pri_c}")

seg_a, seg_b, seg_c, seg_perdas, seg_variacao_a, seg_variacao_b = gradiente(x1, x2, y, qtd_iteracao=10000, alpha=0.001, limite_parada=1e-6)
print(f"\n\nPara Iteração 10000 - \nA: {seg_a}\nB: {seg_b}\nC: {seg_c}")

ter_a, ter_b, ter_c, ter_perdas, ter_variacao_a, ter_variacao_b = gradiente(x1, x2, y, qtd_iteracao=20000, alpha=0.001, limite_parada=1e-6)
print(f"\n\nPara Iteração 20000 - \nA: {ter_a}\nB: {ter_b}\nC: {ter_c}")

ter_index = np.arange(len(ter_perdas))
ter_graph.plot(ter_index, ter_perdas)
ter_graph.set_xlabel(" Iterações ")
ter_graph.set_ylabel(" Erro ")

print("\n\n\n#### Alpha = 0.1 e limite_parada = 1e-3 ####\n\n\n")

pri_a, pri_b, pri_c, pri_perdas, pri_variacao_a, pri_variacao_b = gradiente(x1, x2, y, qtd_iteracao=5000, alpha=0.1, limite_parada=1e-3)
print(f"\n\nPara Iteração 5000 - \nA: {pri_a}\nB: {pri_b}\nC: {pri_c}")

seg_a, seg_b, seg_c, seg_perdas, seg_variacao_a, seg_variacao_b = gradiente(x1, x2, y, qtd_iteracao=10000, alpha=0.1, limite_parada=1e-3)
print(f"\n\nPara Iteração 10000 - \nA: {seg_a}\nB: {seg_b}\nC: {seg_c}")

ter_a, ter_b, ter_c, ter_perdas, ter_variacao_a, ter_variacao_b = gradiente(x1, x2, y, qtd_iteracao=20000, alpha=0.1, limite_parada=1e-3)
print(f"\n\nPara Iteração 20000 - \nA: {ter_a}\nB: {ter_b}\nC: {ter_c}")

qrt_index = np.arange(len(ter_perdas))
qrt_graph.plot(qrt_index, ter_perdas)
qrt_graph.set_xlabel(" Iterações ")
qrt_graph.set_ylabel(" Erro ")



print("\n\n\n#### Alpha = 0.01 e limite_parada = 1e-3 ####\n\n\n")

pri_a, pri_b, pri_c, pri_perdas, pri_variacao_a, pri_variacao_b = gradiente(x1, x2, y, qtd_iteracao=5000, alpha=0.01, limite_parada=1e-3)
print(f"\n\nPara Iteração 5000 - \nA: {pri_a}\nB: {pri_b}\nC: {pri_c}")

seg_a, seg_b, seg_c, seg_perdas, seg_variacao_a, seg_variacao_b = gradiente(x1, x2, y, qtd_iteracao=10000, alpha=0.01, limite_parada=1e-3)
print(f"\n\nPara Iteração 10000 - \nA: {seg_a}\nB: {seg_b}\nC: {seg_c}")

ter_a, ter_b, ter_c, ter_perdas, ter_variacao_a, ter_variacao_b = gradiente(x1, x2, y, qtd_iteracao=20000, alpha=0.01, limite_parada=1e-3)
print(f"\n\nPara Iteração 20000 - \nA: {ter_a}\nB: {ter_b}\nC: {ter_c}")

qui_index = np.arange(len(ter_perdas))
qui_graph.plot(qui_index, ter_perdas)
qui_graph.set_xlabel(" Iterações ")
qui_graph.set_ylabel(" Erro ")

print("\n\n\n#### Alpha = 0.001 e limite_parada = 1e-3 ####\n\n\n")

pri_a, pri_b, pri_c, pri_perdas, pri_variacao_a, pri_variacao_b = gradiente(x1, x2, y, qtd_iteracao=5000, alpha=0.001, limite_parada=1e-3)
print(f"\n\nPara Iteração 5000 - \nA: {pri_a}\nB: {pri_b}\nC: {pri_c}")

seg_a, seg_b, seg_c, seg_perdas, seg_variacao_a, seg_variacao_b = gradiente(x1, x2, y, qtd_iteracao=10000, alpha=0.001, limite_parada=1e-3)
print(f"\n\nPara Iteração 10000 - \nA: {seg_a}\nB: {seg_b}\nC: {seg_c}")

ter_a, ter_b, ter_c, ter_perdas, ter_variacao_a, ter_variacao_b = gradiente(x1, x2, y, qtd_iteracao=20000, alpha=0.001, limite_parada=1e-3)
print(f"\n\nPara Iteração 20000 - \nA: {ter_a}\nB: {ter_b}\nC: {ter_c}")

sex_index = np.arange(len(ter_perdas))
sex_graph.plot(sex_index, ter_perdas)
sex_graph.set_xlabel(" Iterações ")
sex_graph.set_ylabel(" Erro ")

print("\nAnálise concluída com sucesso\n")


fig.show()


