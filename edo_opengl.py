import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame, 
                             QTabWidget, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# ==========================================
# DESIGN SYSTEM: VERCEL / LINEAR AESTHETIC
# ==========================================
# Paleta Zinc (Tailwind-inspired)
ZINC_950 = "#09090b"  # Fundo principal
ZINC_900 = "#18181b"  # Painéis secundários
ZINC_800 = "#27272a"  # Bordas e divisores
ZINC_400 = "#a1a1aa"  # Texto secundário
ZINC_50  = "#fafafa"  # Texto primário e botões primários
BLUE_500 = "#3b82f6"  # Accent functional

# Configuração global do PyQtGraph para casar com a UI
pg.setConfigOption('background', ZINC_950)
pg.setConfigOption('foreground', ZINC_400)
pg.setConfigOptions(antialias=True)

class SimuladorRC(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RC Circuit ODE Simulator")
        self.resize(1280, 800)
        
        # Aplicando stylesheet raiz
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {ZINC_950}; }}
            QLabel {{ color: {ZINC_400}; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 13px; }}
            QLabel#h1 {{ color: {ZINC_50}; font-weight: 600; font-size: 14px; margin-bottom: 8px; }}
            QLabel#h2 {{ color: {ZINC_50}; font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; padding-top: 16px; }}
            
            QLineEdit {{ 
                background-color: {ZINC_950}; border: 1px solid {ZINC_800}; 
                border-radius: 6px; color: {ZINC_50}; padding: 6px 12px; 
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px;
            }}
            QLineEdit:focus {{ border: 1px solid {ZINC_400}; }}
            
            QPushButton {{ 
                background-color: {ZINC_900}; border: 1px solid {ZINC_800}; 
                border-radius: 6px; color: {ZINC_50}; padding: 6px 12px; font-weight: 500; 
            }}
            QPushButton:hover {{ background-color: {ZINC_800}; }}
            QPushButton:pressed {{ background-color: {ZINC_950}; border: 1px solid {ZINC_800}; }}
            
            QPushButton#primary {{ background-color: {ZINC_50}; color: {ZINC_950}; border: none; font-weight: 600; }}
            QPushButton#primary:hover {{ background-color: #e4e4e7; }}
            
            QFrame#sidebar {{ background-color: {ZINC_900}; border-right: 1px solid {ZINC_800}; }}
            
            QTabWidget::pane {{ border: none; border-top: 1px solid {ZINC_800}; background: transparent; }}
            QTabBar::tab {{ background: transparent; color: {ZINC_400}; padding: 10px 16px; border: none; border-bottom: 2px solid transparent; font-weight: 500; font-size: 13px; }}
            QTabBar::tab:selected {{ color: {ZINC_50}; border-bottom: 2px solid {ZINC_50}; }}
            QTabBar::tab:hover:!selected {{ color: #d4d4d8; }}
        """)

        # Dados da simulação
        self.t_vals = np.array([])
        self.q_vals = np.array([])
        self.i_vals = np.array([])
        self.frame_atual = 0
        
        self.timer = QTimer()
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.atualizar_animacao)

        self.setup_ui()
        self.calcular_modelo()

    def setup_ui(self):
        widget_central = QWidget()
        layout_principal = QHBoxLayout(widget_central)
        layout_principal.setContentsMargins(0, 0, 0, 0)
        layout_principal.setSpacing(0)
        
        # ==========================================
        # SIDEBAR (ESQUERDA)
        # ==========================================
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(260)
        layout_sidebar = QVBoxLayout(sidebar)
        layout_sidebar.setContentsMargins(20, 24, 20, 24)
        layout_sidebar.setSpacing(12)
        
        lbl_title = QLabel("RC Circuit Simulator")
        lbl_title.setObjectName("h1")
        layout_sidebar.addWidget(lbl_title)
        
        lbl_params = QLabel("Parameters")
        lbl_params.setObjectName("h2")
        layout_sidebar.addWidget(lbl_params)
        
        # Inputs (Densidade alta, textos diretos)
        layout_sidebar.addWidget(QLabel("Resistance (Ω)"))
        self.input_r = QLineEdit("5.0")
        layout_sidebar.addWidget(self.input_r)
        
        layout_sidebar.addWidget(QLabel("Capacitance (F)"))
        self.input_c = QLineEdit("0.1")
        layout_sidebar.addWidget(self.input_c)
        
        layout_sidebar.addWidget(QLabel("Initial Charge (C)"))
        self.input_q0 = QLineEdit("5.0")
        layout_sidebar.addWidget(self.input_q0)
        
        btn_apply = QPushButton("Apply parameters")
        btn_apply.setObjectName("primary")
        btn_apply.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_apply.clicked.connect(self.calcular_modelo)
        layout_sidebar.addWidget(btn_apply)
        
        layout_sidebar.addSpacing(16)
        
        lbl_controls = QLabel("Simulation")
        lbl_controls.setObjectName("h2")
        layout_sidebar.addWidget(lbl_controls)
        
        # Controles operacionais
        btn_run = QPushButton("Run")
        btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_run.clicked.connect(self.play)
        layout_sidebar.addWidget(btn_run)
        
        btn_pause = QPushButton("Pause")
        btn_pause.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_pause.clicked.connect(self.pause)
        layout_sidebar.addWidget(btn_pause)
        
        btn_reset = QPushButton("Reset")
        btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_reset.clicked.connect(self.reset)
        layout_sidebar.addWidget(btn_reset)
        
        layout_sidebar.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Meta info no rodapé da sidebar
        lbl_meta = QLabel("v1.2.0-rc\nODE: q' + (1/RC)q = 0")
        lbl_meta.setStyleSheet(f"color: {ZINC_800}; font-size: 11px;")
        layout_sidebar.addWidget(lbl_meta)
        
        layout_principal.addWidget(sidebar)

        # ==========================================
        # CANVAS PRINCIPAL (DIREITA)
        # ==========================================
        canvas_area = QWidget()
        layout_canvas = QVBoxLayout(canvas_area)
        layout_canvas.setContentsMargins(24, 16, 24, 24)
        
        self.tabs = QTabWidget()
        
        # --- TAB 1: 2D Analytics ---
        aba_2d = QWidget()
        layout_2d = QHBoxLayout(aba_2d)
        layout_2d.setContentsMargins(0, 16, 0, 0)
        layout_2d.setSpacing(16)
        
        # Helper de estilo para os gráficos
        def format_plot(plot, title):
            plot.setTitle(title, color=ZINC_50, size='13px')
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.getAxis('bottom').setPen(pg.mkPen(color=ZINC_800, width=1))
            plot.getAxis('left').setPen(pg.mkPen(color=ZINC_800, width=1))
            plot.getAxis('bottom').setTextPen(ZINC_400)
            plot.getAxis('left').setTextPen(ZINC_400)

        # Gráfico 1: Time Series
        self.plot_tempo = pg.PlotWidget()
        format_plot(self.plot_tempo, "Time Series (q, i × t)")
        
        self.linha_q = self.plot_tempo.plot(pen=pg.mkPen(color=ZINC_50, width=1.5), name="Charge")
        self.linha_i = self.plot_tempo.plot(pen=pg.mkPen(color=ZINC_400, width=1.5, style=Qt.PenStyle.DashLine), name="Current")
        self.ponto_tempo = self.plot_tempo.plot(pen=None, symbol='o', symbolBrush=ZINC_50, symbolSize=6)
        layout_2d.addWidget(self.plot_tempo)

        # Gráfico 2: Phase Portrait
        self.plot_fase = pg.PlotWidget()
        format_plot(self.plot_fase, "Phase Portrait (q × i)")
        self.linha_fase = self.plot_fase.plot(pen=pg.mkPen(color=BLUE_500, width=1.5))
        self.ponto_fase = self.plot_fase.plot(pen=None, symbol='o', symbolBrush=BLUE_500, symbolSize=6)
        layout_2d.addWidget(self.plot_fase)
        
        self.tabs.addTab(aba_2d, "Analytics")

        # --- TAB 2: 3D State Space ---
        aba_3d = QWidget()
        layout_3d = QVBoxLayout(aba_3d)
        layout_3d.setContentsMargins(0, 16, 0, 0)
        
        self.view_3d = gl.GLViewWidget()
        self.view_3d.opts['distance'] = 18
        self.view_3d.setBackgroundColor(ZINC_950)
        
        # Eixos discretos, sem cores saturadas
        eixos = gl.GLAxisItem()
        eixos.setSize(x=6, y=6, z=6)
        self.view_3d.addItem(eixos)
        
        grade_chao = gl.GLGridItem(color=(255, 255, 255, 30))
        self.view_3d.addItem(grade_chao)
        
        # Trajetória num azul sólido, sem neon
        self.linha_3d = gl.GLLinePlotItem(color=pg.glColor(BLUE_500), width=1.5, antialias=True)
        self.view_3d.addItem(self.linha_3d)
        
        self.ponto_3d = gl.GLScatterPlotItem(color=pg.glColor(ZINC_50), size=6)
        self.view_3d.addItem(self.ponto_3d)
        
        layout_3d.addWidget(self.view_3d)
        self.tabs.addTab(aba_3d, "3D State Space")
        
        layout_canvas.addWidget(self.tabs)
        layout_principal.addWidget(canvas_area, stretch=1)
        
        self.setCentralWidget(widget_central)

    # ==========================================
    # CORE LÓGICA E RENDER (Mantidos Intactos)
    # ==========================================
    def calcular_modelo(self):
        try:
            r = float(self.input_r.text())
            c = float(self.input_c.text())
            q0 = float(self.input_q0.text())
        except ValueError:
            return

        self.pause()
        self.frame_atual = 0
        
        tau = r * c
        t_max = 6 * tau
        
        self.t_vals = np.linspace(0, t_max, 300) 
        self.q_vals = q0 * np.exp(-self.t_vals / tau)
        self.i_vals = -(q0 / tau) * np.exp(-self.t_vals / tau)

        self.plot_tempo.setXRange(0, t_max)
        self.plot_tempo.setYRange(min(self.i_vals)*1.1, max(self.q_vals)*1.1)
        
        self.plot_fase.setXRange(min(self.q_vals)*1.1, max(self.q_vals)*1.1)
        self.plot_fase.setYRange(min(self.i_vals)*1.1, max(self.i_vals)*1.1)

        self.atualizar_animacao() 

    def atualizar_animacao(self):
        if self.frame_atual < len(self.t_vals):
            t = self.t_vals[:self.frame_atual + 1]
            q = self.q_vals[:self.frame_atual + 1]
            i = self.i_vals[:self.frame_atual + 1]

            self.linha_q.setData(t, q)
            self.linha_i.setData(t, i)
            self.linha_fase.setData(q, i)

            self.ponto_tempo.setData([t[-1], t[-1]], [q[-1], i[-1]]) 
            self.ponto_fase.setData([q[-1]], [i[-1]])

            pontos_3d = np.vstack([q, i, t]).transpose()
            self.linha_3d.setData(pos=pontos_3d)
            self.ponto_3d.setData(pos=np.array([[q[-1], i[-1], t[-1]]]))

            self.frame_atual += 1
        else:
            self.pause()

    def play(self):
        if self.frame_atual >= len(self.t_vals):
            self.frame_atual = 0
        self.timer.start()

    def pause(self):
        self.timer.stop()

    def reset(self):
        self.timer.stop()
        self.frame_atual = 0
        self.atualizar_animacao()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    simulador = SimuladorRC()
    simulador.show()
    sys.exit(app.exec())