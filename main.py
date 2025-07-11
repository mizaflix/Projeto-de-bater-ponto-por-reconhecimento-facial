import cv2
import dlib
import os
import face_recognition
import tkinter as tk
from tkinter import messagebox
import threading
import glob
import sys
import time
class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecimento Facial - Batida de Ponto")

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

    def rodar_reconhecimento(self):
        try:
            self.encerrar_janela = False  # Sinalizador

            net = cv2.dnn.readNetFromCaffe(
                "deploy.prototxt",
                "res10_300x300_ssd_iter_140000.caffemodel"
            )

            source = cv2.VideoCapture(0)

            if not source.isOpened():
                self.lbl_status.config(text="❌ Erro ao acessar a câmera.")
                return

            in_width = 300
            in_height = 300
            mean = [104, 117, 123]
            conf_threshold = 0.7
            win_name = "Reconhecimento Facial"

            # Carregar rostos cadastrados
            pasta_pessoas = "pessoas_cadastradas"
            conhecidos = []
            nomes_conhecidos = []

            for caminho_imagem in glob.glob(os.path.join(pasta_pessoas, "*.jpg")):
                imagem = face_recognition.load_image_file(caminho_imagem)
                encs = face_recognition.face_encodings(imagem)
                if encs:
                    conhecidos.append(encs[0])
                    nomes_conhecidos.append(os.path.splitext(os.path.basename(caminho_imagem))[0])

            # Lista de nomes já registrados na sessão
            ja_registrados = []

            while True:
                has_frame, frame = source.read()
                if not has_frame:
                    break

                frame = cv2.flip(frame, 1)
                frame_height, frame_width = frame.shape[:2]

                # Detectar rostos
                blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
                net.setInput(blob)
                detections = net.forward()

                face_locations = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > conf_threshold:
                        x1 = int(detections[0, 0, i, 3] * frame_width)
                        y1 = int(detections[0, 0, i, 4] * frame_height)
                        x2 = int(detections[0, 0, i, 5] * frame_width)
                        y2 = int(detections[0, 0, i, 6] * frame_height)
                        face_locations.append((y1, x2, y2, x1))  # (top, right, bottom, left)

                if face_locations:
                    encodings = face_recognition.face_encodings(frame, face_locations)

                    for (top, right, bottom, left), encoding in zip(face_locations, encodings):
                        distancias = face_recognition.face_distance(conhecidos, encoding)
                        if len(distancias) == 0:
                            continue

                        menor_dist = min(distancias)
                        indice = distancias.tolist().index(menor_dist)

                        if menor_dist <= 0.5:
                            nome = nomes_conhecidos[indice]

                            if nome not in ja_registrados:
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

                                self.lbl_status.config(text=f"✅ {nome} reconhecido! Ponto registrado.")
                                self.encerrar_janela = True  # <-- Agora sim! Fecha depois
                                break

                        else:
                            nome = "Desconhecido"

                        # Desenha
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, nome, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow(win_name, frame)

                # Saída automática
                if self.encerrar_janela:
                    cv2.waitKey(2000)  # Dá tempo de mostrar o texto na janela
                    break

                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                if cv2.waitKey(1) == 27:
                    break

            source.release()
            cv2.destroyAllWindows()

        except Exception as e:
            self.lbl_status.config(text=f"Erro: {e}")

            
# Criar janela
root = tk.Tk()
app = FaceApp(root)
root.mainloop()
