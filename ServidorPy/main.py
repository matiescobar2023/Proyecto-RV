import socket
from threading import Thread
from EcOnda_function import ecuacion_onda as eco
import matplotlib.pyplot as plt

class Server:

    def __init__(self, host, port):
        self.HOST = host
        self.PORT = port
        self.clientes = None
        self.sock = None
        self.start()

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.HOST, self.PORT))
        self.sock.listen(50)
        print(f"Servidor escuchando en {self.HOST}:{self.PORT}")
        Thread(target=self.listen).start()

    def listen(self):
        while True:
            client_socket, client_address = self.sock.accept()
            print(f"Conexión establecida con {client_address}")

            Thread(target=self.handle_client, args=(client_socket, client_address), daemon=True).start()

    def handle_client(self, client_socket, client_address):
        client_socket.send(b"Conexion establecida con el servidor.\n")
        client_socket.send(b"Ingresa nombre, apellido y legajo: ")

        client_input = client_socket.recv(1024).decode().strip().split('-')
        client_role = client_input[0]
        client_legajo = client_input[-1]
        client_name_array = client_input[1:-1]
        client_name = ' '.join(client_name_array)
        client_dict = {"role": client_role, "name": client_name, "legajo": client_legajo, "address": client_address, "socket": client_socket}

        if self.clientes is None:
            self.clientes = [client_dict]
        else:
            for client in self.clientes:

                if client['legajo'] == client_legajo:
                    client_socket.send(b"Legajo ya registrado. Conexion terminada.\n")
                    client_socket.close()
                    print(f"Conexion terminada con {client_address} por legajo duplicado.")
                    return

        self.clientes.append(client_dict)
        print(f"{client_address} se ha identificado exitosamente como {client_name}:{client_legajo}. Rol {client_role}.")

        self.wait_for_request(client_socket, client_legajo)

    def wait_for_request(self, client_socket, client_legajo):
        client_socket.send(b"Ingresa L-T-c para ecuacion de onda.\n")

        while True:
            req = client_socket.recv(1024).decode().strip().split('-')
            L, T, c = map(int, req)
            print(f"Solicitud recibida de {client_legajo}: L={L}, T={T}, c={c}")
            result = eco(L, T, c)
            client_socket.send(b"Ecuacion de onda resuelta. Enviando datos...\n")
            client_socket.send(str(result).encode() + b'\n')




if __name__ == "__main__":
    server = Server("0.0.0.0", 5000)
