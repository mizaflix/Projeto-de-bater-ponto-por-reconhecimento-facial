import cv2
import dlib
import os
import face_recognition
import tkinter as tk
from tkinter import messagebox
import threading
import numpy as np
import sys
import time
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

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}
        self.label_map_invertido = {}
        self.treinar_modelo()

        # Botões
        self.btn_iniciar = tk.Button(root, text="Bater Ponto", command=self.iniciar_thread)
        self.btn_iniciar.pack(pady=10)

        self.btn_cadastrar = tk.Button(root, text="Cadastrar Pessoa", command=self.cadastrar_pessoa)
        self.btn_cadastrar.pack(pady=10)

        # Status
        self.lbl_status = tk.Label(root, text="Status: Aguardando ação...")
        self.lbl_status.pack(pady=10)

    def iniciar_thread(self):
        self.lbl_status.config(text="🔍 Iniciando reconhecimento...")
        threading.Thread(target=self.rodar_reconhecimento).start()

    def cadastrar_pessoa(self):
        def salvar_face(nome):
            cap = cv2.VideoCapture(0)
            self.lbl_status.config(text="📷 Aguardando captura da webcam...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)

                for (top, right, bottom, left) in face_locations:
                    # Adiciona margem
                    margem = 60
                    top = max(0, top - margem)
                    right = min(frame.shape[1], right + margem)
                    bottom = min(frame.shape[0], bottom + margem)
                    left = max(0, left - margem)

                    # Retângulo de visualização
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

        # Janela popup para digitar o nome
        cadastro_window = tk.Toplevel(self.root)
        cadastro_window.title("Cadastro de Pessoa")

        tk.Label(cadastro_window, text="Digite o nome da pessoa:").pack(pady=5)
        entry_nome = tk.Entry(cadastro_window)
        entry_nome.pack(pady=5)
        tk.Button(cadastro_window, text="Cadastrar", command=pedir_nome).pack(pady=5)
    
    def treinar_modelo(self):
        imagens = []
        labels = []
        pasta = "pessoas_cadastradas"

        if not os.path.exists(pasta):
            os.makedirs(pasta)

        arquivos = os.listdir(pasta)
        nomes_unicos = list(set(nome.split("_")[0] for nome in arquivos if nome.endswith(".jpg")))

        for idx, nome in enumerate(nomes_unicos):
            self.label_map[nome] = idx
            self.label_map_invertido[idx] = nome

        for arquivo in arquivos:
            if arquivo.endswith(".jpg"):
                caminho = os.path.join(pasta, arquivo)
                img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
                nome = arquivo.split("_")[0]
                label = self.label_map[nome]
                imagens.append(img)
                labels.append(label)

        if imagens:
            self.recognizer.train(imagens, np.array(labels))
        else:
            print("⚠️ Nenhuma imagem encontrada para treinar o modelo.")


    def rodar_reconhecimento(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.lbl_status.config(text="❌ Erro ao acessar a câmera.")
                return

            face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

            ja_registrados = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face_gray = gray[y:y+h, x:x+w]

                    label_pred, conf = self.recognizer.predict(face_gray)
                    nome = self.label_map_invertido.get(label_pred, "Desconhecido")

                    if nome not in ja_registrados and conf < 70:
                        ja_registrados.append(nome)

                        from datetime import datetime
                        import csv

                        agora = datetime.now()
                        data_str = agora.strftime("%Y-%m-%d")
                        hora_str = agora.strftime("%H:%M:%S")

                        registro_path = "registro_ponto.csv"
                        cabecalho = ["Nome", "Data", "Hora"]
                        existe = os.path.exists(registro_path)

                        with open(registro_path, mode="a", newline="", encoding="utf-8") as file:
                            writer = csv.writer(file)
                            if not existe:
                                writer.writerow(cabecalho)
                            writer.writerow([nome, data_str, hora_str])

                    if nome != "Desconhecido":
                        print(f"Ponto registrado para {nome}")
                            
                        # Opcional: Mostra um popup com tkinter (fora da thread de vídeo)
                        self.root.after(100, lambda: messagebox.showinfo("Sucesso", f"Ponto registrado para {nome}!"))
                        return

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, nome, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Reconhecimento Facial", frame)

                # ESC fecha manualmente
                if cv2.waitKey(1) & 0xFF == 27:
                    break


        except Exception as e:
            self.lbl_status.config(text=f"Erro: {e}")
        
        cap.release()
        cv2.destroyAllWindows()



            
# Criar janela
root = tk.Tk()
app = FaceApp(root)
root.mainloop()