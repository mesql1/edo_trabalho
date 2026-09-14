import customtkinter as ctk
from tkinter import messagebox
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

# ==========================================
# CONFIGURAÇÕES GLOBAIS DE TEMA (MODERNO)
# ==========================================
matplotlib.use('TkAgg')
plt.style.use('dark_background')

# Configuração do CustomTkinter
ctk.set_appearance_mode("dark")  
ctk.set_default_color_theme("blue") 

# Paleta de Cores Integrada
BG_COLOR = "#2b2b2b"      # Cor exata do fundo dos frames do CustomTkinter no modo escuro
FG_TEXT = "#d4d4d4"
ACCENT_BLUE = "#00a8ff"   # Ciano neon
ACCENT_PINK = "#ff3366"   # Rosa neon
ACCENT_PURPLE = "#b824ff" # Roxo neon

# ==========================================
# 1. VARIÁVEIS GLOBAIS E SYM-PY
# ==========================================
t = sp.symbols('t')
q = sp.Function('q')

t_vals, q_vals, i_vals = [], [], []
frame_atual = 1
animando = False

# ==========================================
# 2. FUNÇÃO MOTOR: RECALCULAR O MODELO
# ==========================================
def atualizar_modelo():
    global t_vals, q_vals, i_vals, frame_atual, animando
    
    try:
        r_val = float(entry_r.get())
        c_val = float(entry_c.get())
        q0_val = float(entry_q0.get())
        if r_val <= 0 or c_val <= 0: raise ValueError
    except ValueError:
        messagebox.showerror("Erro", "Insira valores numéricos maiores que zero.")
        return

    animando = False
    frame_atual = 1
    btn_play.configure(state="normal")

    termo = 1 / (r_val * c_val)
    edo = sp.Eq(q(t).diff(t) + termo * q(t), 0)
    solucao = sp.dsolve(edo, ics={q(0): q0_val})
    
    funcao_carga = sp.lambdify(t, solucao.rhs, "numpy")
    funcao_corrente = sp.lambdify(t, solucao.rhs.diff(t), "numpy")
    
    tau = r_val * c_val
    t_max = 6 * tau
    t_vals = np.linspace(0, t_max, 250) 
    q_vals = funcao_carga(t_vals)
    i_vals = funcao_corrente(t_vals)
    
    rodape_equacao.configure(text=f"EDO: q' + {termo:.2f}q = 0   |   Solução: q(t) = {q0_val:.2f} * e^(-{termo:.2f}t)")

    desenhar_graficos_estaticos()
    configurar_limites_animacao(t_max, q0_val)
    desenhar_frame_animacao()
    canvas1.draw_idle()
    canvas2.draw_idle()

# ==========================================
# 3. FUNÇÕES DE DESENHO (GRÁFICOS ESTÁTICOS)
# ==========================================
def formatar_eixos(ax, titulo, xlabel, ylabel):
    ax.set_title(titulo, color='white', fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color=FG_TEXT)
    ax.set_ylabel(ylabel, color=FG_TEXT)
    ax.set_facecolor(BG_COLOR) # Sincroniza com o fundo do app
    ax.grid(True, color='#444444', ls=':', alpha=0.5)
    ax.tick_params(colors=FG_TEXT)
    
    # Remove as bordas do gráfico para um visual mais "Dashboard"
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')

def desenhar_graficos_estaticos():
    ax1.clear(); ax2.clear(); ax3.clear()
    
    ax1.plot(t_vals, q_vals, color=ACCENT_BLUE, lw=2.5, label="Carga q(t)")
    ax1.plot(t_vals, i_vals, color=ACCENT_PINK, lw=2.5, label="Corrente i(t)")
    formatar_eixos(ax1, "Evolução no Tempo", "Tempo (s)", "Amplitude")
    ax1.legend(facecolor=BG_COLOR, edgecolor='#555', framealpha=0.8)
    
    ax2.plot(q_vals, i_vals, color=ACCENT_PURPLE, lw=2.5)
    formatar_eixos(ax2, "Retrato de Fase 2D", "Carga (Coulombs)", "Corrente (Amperes)")
    
    ax3.plot(q_vals, i_vals, t_vals, color=ACCENT_PURPLE, lw=2.5)
    ax3.set_title("Espaço de Estados 3D", color='white', fontweight='bold', pad=10)
    ax3.set_xlabel("Carga"); ax3.set_ylabel("Corrente"); ax3.set_zlabel("Tempo")
    ax3.set_facecolor(BG_COLOR)
    ax3.xaxis.set_pane_color((0.17, 0.17, 0.17, 1.0))
    ax3.yaxis.set_pane_color((0.17, 0.17, 0.17, 1.0))
    ax3.zaxis.set_pane_color((0.17, 0.17, 0.17, 1.0))
    ax3.view_init(elev=20, azim=45)

def configurar_limites_animacao(t_max, q0_val):
    min_i = min(i_vals) * 1.1 if min(i_vals) < 0 else -1
    max_q = max(q_vals) * 1.1 if max(q_vals) > 0 else 1
    
    ax_anim_1.set_xlim(0, t_max)
    ax_anim_1.set_ylim(min_i, max_q)
    ax_anim_2.set_xlim(-0.5, max_q)
    ax_anim_2.set_ylim(min_i, 1)
    ax_anim_3.set_xlim(-0.5, max_q)
    ax_anim_3.set_ylim(min_i, 1)
    ax_anim_3.set_zlim(0, t_max)

# ==========================================
# 4. CONTROLES DE ANIMAÇÃO E CÂMERA
# ==========================================
def desenhar_frame_animacao():
    x_q = t_vals[:frame_atual]
    y_q = q_vals[:frame_atual]
    y_i = i_vals[:frame_atual]
    
    linha_anim_q.set_data(x_q, y_q)
    linha_anim_i.set_data(x_q, y_i)
    linha_anim_2d.set_data(y_q, y_i)
    
    linha_anim_3d.set_data(y_q, y_i)
    linha_anim_3d.set_3d_properties(x_q)

    if len(y_q) > 0:
        ponto_anim_t_q.set_data([x_q[-1]], [y_q[-1]])
        ponto_anim_2d.set_data([y_q[-1]], [y_i[-1]])
        ponto_anim_3d.set_data([y_q[-1]], [y_i[-1]])
        ponto_anim_3d.set_3d_properties([x_q[-1]])
        
    canvas2.draw_idle()

def loop_animacao():
    global frame_atual, animando
    if animando and frame_atual < len(t_vals):
        desenhar_frame_animacao()
        frame_atual += 4
        root.after(16, loop_animacao)
    elif frame_atual >= len(t_vals):
        animando = False
        btn_play.configure(state="normal")

def play():
    global animando
    if not animando and frame_atual < len(t_vals):
        animando = True
        btn_play.configure(state="disabled")
        loop_animacao()

def pause():
    global animando
    animando = False
    btn_play.configure(state="normal")

def reset():
    global frame_atual, animando
    animando = False
    frame_atual = 1
    desenhar_frame_animacao()
    btn_play.configure(state="normal")

def reset_camera_3d():
    ax3.view_init(elev=20, azim=45)
    ax_anim_3.view_init(elev=20, azim=45)
    canvas1.draw_idle()
    canvas2.draw_idle()

# ==========================================
# 5. CONSTRUÇÃO DA INTERFACE MODERNA (CustomTkinter)
# ==========================================
root = ctk.CTk()
root.title("Simulador Interativo: EDO RC (Dashboard)")
root.geometry("1350x850")

# --- PAINEL SUPERIOR (Inputs) ---
painel_inputs = ctk.CTkFrame(root, corner_radius=10)
painel_inputs.pack(fill="x", padx=15, pady=10)

def add_input(parent, label_text, default_val):
    container = ctk.CTkFrame(parent, fg_color="transparent")
    container.pack(side="left", padx=15, pady=15)
    ctk.CTkLabel(container, text=label_text, font=("Roboto", 12, "bold")).pack(side="left", padx=(0, 10))
    entry = ctk.CTkEntry(container, width=60, justify="center")
    entry.insert(0, default_val)
    entry.pack(side="left")
    return entry

entry_r = add_input(painel_inputs, "Resistência (R):", "5")
entry_c = add_input(painel_inputs, "Capacitância (C):", "0.1")
entry_q0 = add_input(painel_inputs, "Carga Inicial (q0):", "5")

btn_atualizar = ctk.CTkButton(painel_inputs, text="⚡ Simular Circuito", fg_color=ACCENT_PURPLE, hover_color="#9419d1", font=("Roboto", 13, "bold"), command=atualizar_modelo)
btn_atualizar.pack(side="left", padx=30, pady=15)

btn_reset_cam = ctk.CTkButton(painel_inputs, text="🎥 Resetar Câmera 3D", fg_color="#4b4b4b", hover_color="#636363", font=("Roboto", 13, "bold"), command=reset_camera_3d)
btn_reset_cam.pack(side="left", padx=10, pady=15)

# --- SISTEMA DE ABAS MODERNO ---
tabview = ctk.CTkTabview(root, corner_radius=10)
tabview.pack(fill="both", expand=True, padx=15, pady=5)

aba_estatica = tabview.add("📊 Visão Analítica")
aba_animada = tabview.add("▶️ Simulação em Tempo Real")

# Textos Explicativos
texto_estatico = "▶  TEMPO: Carga (Ciano) sofre decaimento exponencial | Corrente (Rosa) tem pico negativo e perde força.  |  ▶  FASE 2D: Relação linear (reta) típica de EDO de 1ª ordem.  |  ▶  3D: Espaço de estados prova consumo total de energia."
texto_animado = "▶  DINÂMICA: Note o escoamento rápido inicial. À medida que o capacitor esvazia, o fluxo perde força. A EDO homogênea consome apenas a energia inicial."

ctk.CTkLabel(aba_estatica, text=texto_estatico, font=("Roboto", 12), text_color="#aaaaaa").pack(side="bottom", pady=10)
ctk.CTkLabel(aba_animada, text=texto_animado, font=("Roboto", 12), text_color="#aaaaaa").pack(side="bottom", pady=10)

# --- GRÁFICOS ABA 1 (Estática) ---
fig1 = plt.Figure(figsize=(15, 4.5), dpi=100, facecolor=BG_COLOR)
fig1.subplots_adjust(wspace=0.3, bottom=0.15)
ax1 = fig1.add_subplot(1, 3, 1)
ax2 = fig1.add_subplot(1, 3, 2)
ax3 = fig1.add_subplot(1, 3, 3, projection='3d')
canvas1 = FigureCanvasTkAgg(fig1, master=aba_estatica)
canvas1.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

# --- PAINEL DE ANIMAÇÃO (Aba 2) ---
painel_botoes = ctk.CTkFrame(aba_animada, fg_color="transparent")
painel_botoes.pack(fill="x", pady=5)

btn_play = ctk.CTkButton(painel_botoes, text="▶️ Iniciar Trajetória", fg_color=ACCENT_BLUE, hover_color="#008ecc", font=("Roboto", 13, "bold"), command=play)
btn_play.pack(side="left", padx=15)
ctk.CTkButton(painel_botoes, text="⏸️ Pausar", fg_color="#f39c12", hover_color="#d68910", font=("Roboto", 13, "bold"), command=pause).pack(side="left", padx=15)
ctk.CTkButton(painel_botoes, text="🔄 Reiniciar", fg_color="#e74c3c", hover_color="#c0392b", font=("Roboto", 13, "bold"), command=reset).pack(side="left", padx=15)

# --- GRÁFICOS ABA 2 (Animada) ---
fig2 = plt.Figure(figsize=(15, 4.5), dpi=100, facecolor=BG_COLOR)
fig2.subplots_adjust(wspace=0.3, bottom=0.15)
ax_anim_1 = fig2.add_subplot(1, 3, 1)
ax_anim_2 = fig2.add_subplot(1, 3, 2)
ax_anim_3 = fig2.add_subplot(1, 3, 3, projection='3d')

formatar_eixos(ax_anim_1, "Evolução no Tempo", "Tempo (s)", "Amplitude")
formatar_eixos(ax_anim_2, "Retrato de Fase 2D", "Carga q(t)", "Corrente i(t)")
ax_anim_3.set_title("Espaço de Estados 3D", color='white', fontweight='bold', pad=10)
ax_anim_3.set_facecolor(BG_COLOR)
ax_anim_3.xaxis.set_pane_color((0.17, 0.17, 0.17, 1.0))
ax_anim_3.yaxis.set_pane_color((0.17, 0.17, 0.17, 1.0))
ax_anim_3.zaxis.set_pane_color((0.17, 0.17, 0.17, 1.0))
ax_anim_3.view_init(elev=20, azim=45)

# Elementos Animados
linha_anim_q, = ax_anim_1.plot([], [], color=ACCENT_BLUE, lw=3, label="Carga")
linha_anim_i, = ax_anim_1.plot([], [], color=ACCENT_PINK, lw=3, label="Corrente")
ponto_anim_t_q, = ax_anim_1.plot([], [], color="#ffffff", marker='o', markersize=6, ls='', zorder=5)
ax_anim_1.legend(facecolor=BG_COLOR, edgecolor='#555', framealpha=0.8)

linha_anim_2d, = ax_anim_2.plot([], [], color=ACCENT_PURPLE, lw=3)
ponto_anim_2d, = ax_anim_2.plot([], [], color="#ffffff", marker='o', markersize=6, ls='', zorder=5)

linha_anim_3d, = ax_anim_3.plot([], [], [], color=ACCENT_PURPLE, lw=3)
ponto_anim_3d, = ax_anim_3.plot([], [], [], color="#ffffff", marker='o', markersize=6, ls='', zorder=5)

canvas2 = FigureCanvasTkAgg(fig2, master=aba_animada)
canvas2.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

# --- RODAPÉ MATEMÁTICO ---
rodape_equacao = ctk.CTkLabel(root, text="Calculando...", font=("Consolas", 15, "bold"), text_color=ACCENT_BLUE)
rodape_equacao.pack(side="bottom", pady=15)

atualizar_modelo()
root.mainloop()