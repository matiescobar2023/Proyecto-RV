import socket
from threading import Thread
from EcOnda_function import ecuacion_onda


class Server:
    def __init__(self, HOST, PORT):
        self.Clients = []
        self.HOST = HOST
        self.PORT = PORT
        self.server = None
        self.start()

    def start(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.HOST, self.PORT))
        self.server.listen(100)
        print(f"Servidor iniciado y escuchando en {self.HOST}:{self.PORT}")
        Thread(target=self.listen, daemon=True).start()  # no bloquea

    def listen(self):
        while True:
            client_socket, client_address = self.server.accept()
            print(f"Conexión establecida con {client_address}")
            Thread(target=self.handle_client, args=(client_socket, client_address)).start()

    def handle_client(self, client_socket, client_address):
        client_socket.send(b"Bienvenido al servidor. Por favor, ingresa tu nombre, apellido y legajo:\n")

        client_message = client_socket.recv(1024).decode('utf-8').strip()
        client_name = client_message if client_message else "ClienteDesconocido"
        client = {'name': client_name, 'socket': client_socket, 'address': client_address}
        self.Clients.append(client)

        client_socket.send(b"Hola " + client_message.encode('utf-8') + b", ahora estas conectado al servidor.\n")
        client_socket.send(b"Para ejecutar el calculo numerico envia 'START-L-T-c'\n")

        while True:
            try:
                message = client_socket.recv(1024).decode('utf-8').strip()
                if not message:
                    break

                if message.startswith("START"):
                    try:
                        _, L, T, c = message.split('-')
                        L = int(L)
                        T = int(T)
                        c = int(c)

                        result = ecuacion_onda(L, T, c)
                        client_socket.send(b"Calculo completado. Enviando resultados...\n")
                        for chunk in str(result).encode('utf-8'):
                            client_socket.send(chunk.to_bytes(1, 'big'))
                        client_socket.send(b"\n")
                    except Exception as e:
                        client_socket.send(b"Error en los parametros o en el calculo: " + str(e).encode('utf-8') + b"\n")
                else:
                    client_socket.send(b"Comando no reconocido. Usa 'START-L-T-c' para iniciar el calculo.\n")
            except ConnectionResetError:
                break

    def conectados(self):
        return self.Clients

    def kick(self, client_name):
        for client in self.Clients:
            if client['name'] == client_name:
                client['socket'].close()
                client['socket'].send(b"Has sido expulsado del servidor.\n")
                self.Clients.remove(client)
                return f"Cliente {client_name} expulsado."
        return f"Cliente {client_name} no encontrado."



