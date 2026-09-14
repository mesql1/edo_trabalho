import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------
# 1. RESOLUÇÃO MATEMÁTICA (SymPy)
# ---------------------------------------------------------
t = sp.symbols('t')
q = sp.Function('q')

# EDO de 1ª Ordem: q' + 2q = 0
edo = sp.Eq(q(t).diff(t) + 2*q(t), 0)

# Condição inicial: Carga de 5 Coulombs no instante zero
condicao_inicial = {q(0): 5}

# Resolvendo a equação
solucao = sp.dsolve(edo, ics=condicao_inicial)
carga_exata = solucao.rhs

# A corrente é a derivada da carga (q')
corrente_exata = carga_exata.diff(t)

print(f"Solução da Carga: q(t) = {carga_exata}")
print(f"Solução da Corrente: i(t) = {corrente_exata}")

# ---------------------------------------------------------
# 2. GERAÇÃO DE DADOS NUMÉRICOS (NumPy)
# ---------------------------------------------------------
# Convertendo as fórmulas para funções executáveis
funcao_carga = sp.lambdify(t, carga_exata, "numpy")
funcao_corrente = sp.lambdify(t, corrente_exata, "numpy")

# Criando a linha do tempo (0 a 3 segundos é suficiente para o decaimento)
valores_t = np.linspace(0, 3, 200)
valores_q = funcao_carga(valores_t)
valores_i = funcao_corrente(valores_t)

# ---------------------------------------------------------
# 3. VISUALIZAÇÃO GRÁFICA (Matplotlib)
# ---------------------------------------------------------
# Criando uma figura larga para caber os 3 gráficos lado a lado
fig = plt.figure(figsize=(18, 6))

# --- GRÁFICO 1: Evolução no Tempo (2D) ---
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(valores_t, valores_q, color='#00a8ff', linewidth=3, label="Carga $q(t)$")
ax1.plot(valores_t, valores_i, color='#ff4757', linewidth=3, label="Corrente $i(t)$")
ax1.set_title("1. Domínio do Tempo (2D)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Tempo [segundos]")
ax1.set_ylabel("Amplitude")
ax1.axhline(0, color='black', linewidth=1)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# --- GRÁFICO 2: Retrato de Fase (2D) ---
ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(valores_q, valores_i, color='purple', linewidth=3, label="Trajetória RC")
ax2.scatter([valores_q[0]], [valores_i[0]], color='green', s=100, zorder=5, label="Início (t=0)")
ax2.scatter([0], [0], color='red', s=100, zorder=5, label="Fim (Equilíbrio)")
ax2.set_title("2. Retrato de Fase (2D)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Carga $q(t)$ [Coulombs]")
ax2.set_ylabel("Corrente $i(t)$ [Amperes]")
ax2.axhline(0, color='black', linewidth=1)
ax2.axvline(0, color='black', linewidth=1)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

# --- GRÁFICO 3: Espaço de Estados Estendido (3D) ---
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot(valores_q, valores_i, valores_t, color='#4a0072', linewidth=3)
ax3.scatter([valores_q[0]], [valores_i[0]], [valores_t[0]], color='green', s=100)
ax3.scatter([0], [0], [valores_t[-1]], color='red', s=100)
ax3.set_title("3. Visão 3D Espaço-Tempo", fontsize=14, fontweight='bold')
ax3.set_xlabel('Carga $q(t)$')
ax3.set_ylabel('Corrente $i(t)$')
ax3.set_zlabel('Tempo $t$')
ax3.view_init(elev=20, azim=45) # Ângulo de visão

# Ajuste de layout e exibição
plt.tight_layout()
plt.show()