# Reconhecimento-Facial
🧠 Reconhecimento Facial - Batida de Ponto
Este projeto é um sistema simples de reconhecimento facial para controle de ponto. Ele utiliza Python, OpenCV, face_recognition e Tkinter para criar uma interface gráfica onde é possível:

✅ Cadastrar o rosto de uma pessoa

🕵️‍♂️ Fazer reconhecimento ao vivo pela câmera

🗂 Comparar com rostos salvos

💬 Registrar o ponto quando a pessoa é reconhecida

📸 Funcionalidades
Interface gráfica intuitiva com Tkinter

Detecção facial em tempo real (usando dnn do OpenCV)

Reconhecimento facial com a biblioteca face_recognition

Cadastro de novos rostos com nome

Armazenamento em pasta local pessoas_cadastradas/

Registro de ponto simples com feedback visual e textual

📁 Estrutura de Pastas
Copiar
Editar
📂 seu_projeto/
├── reconhecimento_facial_ponto.py
├── pessoas_cadastradas/
│   ├── joao.jpg
│   └── maria.jpg
├── README.md
└── requirements.txt
🚀 Como Executar

![image](https://github.com/user-attachments/assets/899e2534-ed6b-41d3-a63e-cffae2fccf85)

![image](https://github.com/user-attachments/assets/dc49789b-9956-4fce-9596-0151eef44830)

Ou manualmente:

![image](https://github.com/user-attachments/assets/dfcd7b66-857f-4e90-a571-39c2a74b4fc5)

Obs: a biblioteca face_recognition exige o cmake e o dlib. No Windows, o ideal é usar Python 3.8 ou 3.9 para facilitar a instalação.

3. Execute o programa

![image](https://github.com/user-attachments/assets/f1e1e0a2-15bc-42a3-a70f-4c4bf335113d)



💾 Cadastro de Pessoas
Clique em "Cadastrar Pessoa"

Digite o nome

A webcam será aberta. Posicione o rosto.

Pressione S para salvar ou Q para cancelar

A imagem será salva na pasta pessoas_cadastradas/

🔍 Reconhecimento Facial
Clique em "Bater Ponto"

A câmera será aberta

O sistema buscará correspondência com as imagens cadastradas

Se reconhecido, o nome aparecerá com confirmação ✅

🧰 Tecnologias Usadas
Python 3.8+

OpenCV

face_recognition (baseado em dlib)

Tkinter (GUI)

Caffe model para detecção de rosto: res10_300x300_ssd_iter_140000.caffemodel

📦 Requisitos
Webcam funcional

Python instalado

Sistema operacional compatível (Windows, Linux ou Mac)

🔒 Observações
O reconhecimento é offline, sem envio de dados para a nuvem.

Para maior precisão, recomenda-se rostos bem iluminados e centralizados.

Pode ser expandido para registrar horários em arquivos .csv.

✨ Possíveis melhorias
Registro de data e hora do ponto

Geração de relatórios

Integração com banco de dados

Versão executável (.exe)

Versão web ou Android

👤 Autor
Mizael Lopes

