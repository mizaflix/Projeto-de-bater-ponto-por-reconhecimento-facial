import cv2
import os
import face_recognition
import tkinter as tk
from tkinter import messagebox
import threading
import numpy as np
import sys
import time
import csv
from datetime import datetime


class FaceApp:

    def resource_path(relative_path):
        """ Get absolute path para o arquivo, independente se rodando do exe ou do script """
        try:
            base_path = sys._MEIPASS  # PyInstaller cria essa variável temporária
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecimento Facial - Batida de Ponto")

        self.known_encodings = []
        self.known_names = []
        self.treinar_modelo()

        # Botões
        self.btn_iniciar = tk.Button(root, text="Bater Ponto", command=self.iniciar_thread)
        self.btn_iniciar.pack(pady=10)

        self.btn_cadastrar = tk.Button(root, text="Cadastrar Pessoa", command=self.cadastrar_pessoa)
        self.btn_cadastrar.pack(pady=10)

        # Status
        self.lbl_status = tk.Label(root, text="Status: Aguardando ação...")
        self.lbl_status.pack(pady=10)

    # --------------------------
    # Utilitários de registro
    # --------------------------
    def ja_registrou(self, nome, data_str, registro_path="registro_ponto.csv"):
        """Verifica se já existe registro do mesmo nome e data no CSV"""
        if not os.path.exists(registro_path):
            return False
        with open(registro_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # pula cabeçalho
            for row in reader:
                if row[0] == nome and row[1] == data_str:
                    return True
        return False

    def registrar_ponto(self, nome):
        """Registra o ponto da pessoa, evitando duplicação"""
        agora = datetime.now()
        data_str = agora.strftime("%Y-%m-%d")
        hora_str = agora.strftime("%H:%M:%S")

        registro_path = "registro_ponto.csv"
        cabecalho = ["Nome", "Data", "Hora"]
        existe = os.path.exists(registro_path)

        if not self.ja_registrou(nome, data_str, registro_path):
            with open(registro_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                if not existe:
                    writer.writerow(cabecalho)
                writer.writerow([nome, data_str, hora_str])

            print(f"Ponto registrado para {nome}")
            self.root.after(100, lambda: messagebox.showinfo("Sucesso", f"Ponto registrado para {nome}!"))
        else:
            print(f"{nome} já registrou ponto hoje.")
            self.root.after(100, lambda: messagebox.showinfo("Aviso", f"{nome} já registrou ponto hoje."))

    # --------------------------
    # Lógica de interface
    # --------------------------
    def iniciar_thread(self):
        self.lbl_status.config(text="🔍 Iniciando reconhecimento...")
        threading.Thread(target=self.rodar_reconhecimento).start()

    def cadastrar_pessoa(self):
        def salvar_face(nome):
            cap = cv2.VideoCapture("teste.mp4")
            self.lbl_status.config(text="📷 Aguardando captura da webcam...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)

                for (top, right, bottom, left) in face_locations:
                    margem = 60
                    top = max(0, top - margem)
                    right = min(frame.shape[1], right + margem)
                    bottom = min(frame.shape[0], bottom + margem)
                    left = max(0, left - margem)

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                cv2.imshow("Cadastro - Pressione S para salvar | Q para sair", frame)
                key = cv2.waitKey(1)

                if key == ord('s') and face_locations:
                    top, right, bottom, left = face_locations[0]
                    face_image = frame[top:bottom, left:right]

                    if not os.path.exists("pessoas_cadastradas"):
                        os.makedirs("pessoas_cadastradas")

                    caminho = os.path.join("pessoas_cadastradas", f"{nome}.jpg")
                    cv2.imwrite(caminho, face_image)
                    self.lbl_status.config(text=f"✅ {nome} cadastrado com sucesso!")

                    # Re-treina modelo automaticamente
                    self.treinar_modelo()
                    break

                elif key == ord('q'):
                    self.lbl_status.config(text="❌ Cadastro cancelado.")
                    break

            cap.release()
            cv2.destroyAllWindows()

        def pedir_nome():
            nome = entry_nome.get().strip()
            if nome:
                cadastro_window.destroy()
                salvar_face(nome)
            else:
                messagebox.showwarning("Atenção", "Digite um nome válido.")

        cadastro_window = tk.Toplevel(self.root)
        cadastro_window.title("Cadastro de Pessoa")

        tk.Label(cadastro_window, text="Digite o nome da pessoa:").pack(pady=5)
        entry_nome = tk.Entry(cadastro_window)
        entry_nome.pack(pady=5)
        tk.Button(cadastro_window, text="Cadastrar", command=pedir_nome).pack(pady=5)

    def treinar_modelo(self):
        self.known_encodings = []
        self.known_names = []
        pasta = "pessoas_cadastradas"

        if not os.path.exists(pasta):
            os.makedirs(pasta)

        arquivos = os.listdir(pasta)
        for arquivo in arquivos:
            if arquivo.endswith(".jpg"):
                caminho = os.path.join(pasta, arquivo)
                imagem = face_recognition.load_image_file(caminho)
                encs = face_recognition.face_encodings(imagem)
                if encs:
                    self.known_encodings.append(encs[0])
                    nome = os.path.splitext(arquivo)[0]
                    self.known_names.append(nome)

        print(f"Treinamento concluído: {len(self.known_names)} pessoas cadastradas.")

    def rodar_reconhecimento(self):
        try:
            cap = cv2.VideoCapture("teste5.mp4")
            if not cap.isOpened():
                self.lbl_status.config(text="❌ Erro ao acessar a câmera.")
                return

            tolerance = 0.45  # quanto menor, mais rígido
            ja_registrados = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, encodings):
                    distancias = face_recognition.face_distance(self.known_encodings, face_encoding)
                    if len(distancias) > 0:
                        min_idx = np.argmin(distancias)
                        if distancias[min_idx] < tolerance:
                            nome = self.known_names[min_idx]
                        else:
                            nome = "Desconhecido"
                    else:
                        nome = "Desconhecido"

                    if nome != "Desconhecido" and nome not in ja_registrados:
                        ja_registrados.append(nome)
                        self.registrar_ponto(nome)
                        cap.release()
                        time.sleep(2)
                        cv2.destroyAllWindows()
                        return

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, nome, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Reconhecimento Facial", frame)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair manualmente
                    break

        except Exception as e:
            self.lbl_status.config(text=f"Erro: {e}")

        cap.release()
        cv2.destroyAllWindows()


# Criar janela
root = tk.Tk()
app = FaceApp(root)
root.mainloop()
