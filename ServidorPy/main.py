import cmd
from server import Server


class CLI(cmd.Cmd):

    intro = "Proyecto RV - CLI Servidor - Utilizar 'help' para ver comandos"
    prompt = '-> '

    def __init__(self):
        super().__init__()
        self.server = None

    # Comando para iniciar el servidor
    def do_start(self, arg):
        'Inicia el servidor: start'
        print('Iniciando servidor...')
        self.server = Server("0.0.0.0", 5000)

    # Comando para detener el servidor
    def do_stop(self, arg):
        'Detiene el servidor: stop'
        print('Servidor detenido...')

    def do_clients(self, arg):
        "Muestra los clientes conectados"
        print(self.server.conectados())

    # Comando para salir del CLI
    def do_exit(self, arg):
        'Salir del CLI: exit'
        print('Saliendo...')
        return True

    def do_kick(self, arg):
        'Desconecta a un cliente: kick nombre_cliente'
        if not self.server:
            print("El servidor no está iniciado.")
            return
        if not arg:
            print("Por favor, proporciona el nombre del cliente a desconectar.")
            return
        self.server.kick(arg)

    def do_help(self, arg):
        'Muestra la ayuda: help'
        super().do_help(arg)

    def emptyline(self):
        pass


if __name__ == '__main__':
    MiCli = CLI()
    MiCli.cmdloop()