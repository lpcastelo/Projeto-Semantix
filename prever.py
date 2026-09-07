import pandas as pd
import joblib
import os
import sys
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinterdnd2 import TkinterDnD, DND_FILES
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Essas importações não são usadas diretamente neste arquivo, mas são as classes
# que compõem o Pipeline salvo em modelo_xgb_final.pkl. O joblib.load só consegue
# reconstruir o objeto se essas classes estiverem disponíveis; como o PyInstaller
# só enxerga imports explícitos (e não os módulos usados dentro de um arquivo .pkl),
# sem essas linhas o executável falha com "No module named 'sklearn.pipeline'".
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# No executável "congelado" pelo PyInstaller, o modelo fica ao lado do .exe.
# Rodando como script comum, fica ao lado deste arquivo .py.
def localizar_pasta():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

# Cria uma janela auxiliar só para poder exibir a caixa de erro,
# caso ainda não exista nenhuma janela do tkinter aberta.
def mostrar_erro(titulo, mensagem):
    auxiliar = tk.Tk()
    auxiliar.withdraw()
    messagebox.showerror(titulo, mensagem)
    auxiliar.destroy()


def processar_csv(arquivo_csv):
    """Carrega o modelo, roda a previsão sobre o CSV e mostra o resultado."""

    if not arquivo_csv.lower().endswith('.csv'):
        mostrar_erro("Arquivo inválido", "O arquivo selecionado não é um CSV.")
        return

    arquivo_modelo = os.path.join(localizar_pasta(), 'modelo_xgb_final.pkl')

    try:
        modelo = joblib.load(arquivo_modelo)
        df = pd.read_csv(arquivo_csv)

        if 'admitidos_desligados' not in df.columns:
            raise ValueError(
                "O arquivo CSV não possui a coluna "
                "'admitidos_desligados'.\n\n"
                "Essa coluna é necessária para calcular "
                "as métricas de desempenho."
            )

        X = df.drop(columns=['admitidos_desligados'])
        y_real = df['admitidos_desligados']

        y_pred = modelo.predict(X)

        acuracia = accuracy_score(y_real, y_pred)
        report = classification_report(y_real, y_pred)
        conf_matrix = confusion_matrix(y_real, y_pred)

        df['previsao'] = y_pred

        nome_arquivo = os.path.splitext(os.path.basename(arquivo_csv))[0]
        arquivo_saida = os.path.join(os.path.dirname(arquivo_csv), nome_arquivo + '_resultado.csv')
        df.to_csv(arquivo_saida, index=False)

        mostrar_janela_resultado(acuracia, report, conf_matrix, arquivo_saida)

    except Exception as erro:
        mostrar_erro("Erro durante a previsão", str(erro))


def mostrar_janela_resultado(acuracia, report, conf_matrix, arquivo_saida):
    root = tk.Tk()
    root.title("Resultado da previsão")
    root.resizable(True, True)

    # Título
    titulo = tk.Label(root, text="Previsão concluída!", font=("Arial", 18, "bold"))
    titulo.pack(pady=15)

    # Acurácia
    texto_acuracia = tk.Label(root, text=f"Acurácia: {acuracia:.4f}", font=("Arial", 14, "bold"))
    texto_acuracia.pack(pady=5)

    # Breve análise do resultado. O texto muda de acordo com a acurácia obtida.
    if acuracia == 1:
        texto_analise = (
            "O modelo acertou 100% das previsões, mesmo diante de "
            "dados que nunca tinha visto durante o treinamento. Esse resultado se repete "
            "porque a variável 'tipo_movimentacao_desagregado' tem uma correlação muito "
            "forte com o que está sendo previsto, o que torna a separação entre admissão "
            "e desligamento praticamente trivial para o modelo.\n\n"
            "Ainda assim, é um bom sinal: o desempenho se manteve o mesmo em dados "
            "completamente novos, e não só nos dados usados no treinamento. Isso mostra "
            "que o modelo está bem calibrado e generaliza bem, em vez de simplesmente "
            "ter decorado os exemplos de treino."
        )
    else:
        if acuracia >= 0.95:
            texto_analise = (
                "O modelo acertou praticamente 100% das previsões, mesmo diante de "
                "dados que nunca tinha visto durante o treinamento. Esse resultado se repete "
                "porque a variável 'tipo_movimentacao_desagregado' tem uma correlação muito "
                "forte com o que está sendo previsto, o que torna a separação entre admissão "
                "e desligamento praticamente trivial para o modelo.\n\n"
                "Ainda assim, é um bom sinal: o desempenho se manteve o mesmo em dados "
                "completamente novos, e não só nos dados usados no treinamento. Isso mostra "
                "que o modelo está bem calibrado e generaliza bem, em vez de simplesmente "
                "ter decorado os exemplos de treino."
            )
        else:
            texto_analise = (
                f"O modelo acertou {acuracia:.1%} das previsões neste conjunto de dados."
            )

    # IMPORTANTE: o pack() reserva espaço na ordem em que é chamado. Por isso,
    # tudo que fica "embaixo" na janela (botão, caminho do arquivo, análise) é
    # empacotado com side="bottom" e ANTES da caixa de texto - assim esses
    # widgets sempre garantem seu espaço primeiro. A caixa de texto, com
    # expand=True, é empacotada por último e fica só com o que sobrar, então é
    # ela (e não o botão) quem encolhe e ganha rolagem em telas menores.

    # Botão fechar. Atenção: em um tk.Button,"height" conta em
    # LINHAS DE TEXTO, não em pixels.
    botao = tk.Button(root, text="Fechar", command=root.destroy, width=20, height=2, font=("Arial", 12, "bold"))
    botao.pack(side="bottom", pady=15)

    # Arquivo gerado
    texto_arquivo = tk.Label(root, text=f"Resultado salvo em:\n{arquivo_saida}", font=("Arial", 12))
    texto_arquivo.pack(side="bottom", pady=10)

    analise = tk.Label(root, text=texto_analise, font=("Arial", 12), justify="left", wraplength=650)
    analise.pack(side="bottom", padx=20, pady=(0, 10), anchor="w")

    # Relatório + matriz
    texto_resultados = (
        "RELATÓRIO DE CLASSIFICAÇÃO\n"
        "===========================\n\n"
        f"{report}\n\n"
        "MATRIZ DE CONFUSÃO\n"
        "==================\n\n"
        f"{conf_matrix}")

    # Caixa de texto com barra de rolagem: com o espaço já reservado pelos
    # widgets acima, ela ocupa o que sobrar (e nunca menos que o necessário
    # para navegar o conteúdo com a rolagem).
    frame_texto = tk.Frame(root)
    frame_texto.pack(side="top", fill="both", expand=True, padx=20, pady=10)

    barra_rolagem = tk.Scrollbar(frame_texto)
    barra_rolagem.pack(side="right", fill="y")

    caixa_texto = tk.Text(frame_texto, font=("Courier New", 10), wrap="none", height=18, yscrollcommand=barra_rolagem.set)
    caixa_texto.insert("1.0", texto_resultados)
    caixa_texto.config(state="disabled")
    caixa_texto.pack(side="left", fill="both", expand=True)
    barra_rolagem.config(command=caixa_texto.yview)

    # Tamanho e posição iniciais: o conteúdo real, sem passar da altura da tela.
    # É preciso fixar também a posição (não só a altura) porque, se o Windows
    # abrir a janela mais para baixo na tela, ela pode ultrapassar a borda
    # inferior mesmo com uma altura "dentro do limite".
    root.update_idletasks()
    largura_janela = 700
    margem = 160  # espaço reservado para barra de tarefas e borda/título da janela
    altura_janela = min(root.winfo_reqheight(), root.winfo_screenheight() - margem)
    pos_x = (root.winfo_screenwidth() - largura_janela) // 2
    pos_y = 20
    root.geometry(f"{largura_janela}x{altura_janela}+{pos_x}+{pos_y}")
    root.minsize(largura_janela, 450)

    root.mainloop()


def mostrar_janela_espera():
    """Janela inicial: pede para o usuário arrastar o CSV para cima dela."""

    janela = TkinterDnD.Tk()
    janela.title("Previsão XGBoost")
    janela.geometry("480x280")
    janela.resizable(False, False)

    titulo = tk.Label(janela, text="Previsão de Admissões/Desligamentos", font=("Arial", 14, "bold"),
        wraplength=440, justify="center")
    titulo.pack(pady=(20, 10))

    area_soltar = tk.Label(janela, text="Arraste o arquivo CSV\npara esta janela", font=("Arial", 13),
        fg="#555555", bg="#f0f0f0", relief="ridge", borderwidth=2, justify="center",)
    area_soltar.pack(padx=20, pady=10, fill="both", expand=True)

    def ao_soltar_arquivo(event):
        # event.data traz o(s) caminho(s) no formato de lista do Tcl; quando o
        # caminho tem espaços, ele vem entre chaves. splitlist trata os dois casos
        # e, se mais de um arquivo for solto, usamos apenas o primeiro.
        caminhos = janela.tk.splitlist(event.data)
        janela.destroy()
        processar_csv(caminhos[0])

    area_soltar.drop_target_register(DND_FILES)
    area_soltar.dnd_bind('<<Drop>>', ao_soltar_arquivo)

    def selecionar_arquivo():
        caminho = filedialog.askopenfilename(title="Selecione o arquivo CSV", filetypes=[("Arquivos CSV", "*.csv")])
        if caminho:
            janela.destroy()
            processar_csv(caminho)

    botao_selecionar = tk.Button(janela, text="ou selecione o arquivo...", command=selecionar_arquivo, width=25)
    botao_selecionar.pack(pady=(0, 20))

    janela.mainloop()


if __name__ == '__main__':
    # Se um arquivo foi arrastado direto sobre o executável (ou passado como
    # argumento), processa direto. Caso contrário, abre a janela de espera.
    if len(sys.argv) >= 2:
        processar_csv(sys.argv[1])
    else:
        mostrar_janela_espera()
