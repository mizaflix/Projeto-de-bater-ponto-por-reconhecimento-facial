import face_recognition
import cv2
import os

# Caminho da imagem cadastrada
imagem_path = "pessoas_cadastradas/mizael.jpg"  # <- Troque 'seu_nome' pelo nome da imagem cadastrada

if not os.path.exists(imagem_path):
    print("❌ Imagem cadastrada não encontrada.")
    exit()

imagem = face_recognition.load_image_file(imagem_path)
enc_cadastrado = face_recognition.face_encodings(imagem)

if not enc_cadastrado:
    print("❌ Nenhum rosto detectado na imagem cadastrada.")
    exit()

encoding_cadastrado = enc_cadastrado[0]

# Captura ao vivo
cap = cv2.VideoCapture(0)
print("🎥 Olhe para a câmera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    enc_entrada = face_recognition.face_encodings(rgb)

    if enc_entrada:
        encoding_entrada = enc_entrada[0]
        resultado = face_recognition.compare_faces([encoding_cadastrado], encoding_entrada, tolerance=0.6)
        distancia = face_recognition.face_distance([encoding_cadastrado], encoding_entrada)[0]

        msg = f"✅ Match: {resultado[0]} | Distância: {distancia:.4f}"
        print(msg)
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Teste de Reconhecimento", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
