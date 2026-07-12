"""
Nube de puntos interactiva
--------------------------
Clic izquierdo  -> añade un punto
Clic derecho    -> elimina el último punto añadido
Tecla 'c'       -> limpia todos los puntos
Tecla 's'       -> guarda los puntos en 'puntos.csv'
"""

import matplotlib.pyplot as plt
import csv


class NubeDePuntos:
    def __init__(self, xlim=(0, 10), ylim=(0, 10)):
        self.puntos_x = []
        self.puntos_y = []

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_title(
            "Clic izq: añadir | Clic der: deshacer | 'c': limpiar | 's': guardar"
        )
        self.ax.grid(True, linestyle="--", alpha=0.4)

        # scatter vacío al principio, se actualiza con set_offsets
        self.scatter = self.ax.scatter([], [], color="tab:blue", s=40)

        # Conectamos los eventos de ratón y teclado
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def on_click(self, event):
        """Callback for mouse clicks."""
        # Ignoramos clics fuera del área de los ejes
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # clic izquierdo
            self.puntos_x.append(event.xdata)
            self.puntos_y.append(event.ydata)
        elif event.button == 3:  # clic derecho
            if self.puntos_x:
                self.puntos_x.pop()
                self.puntos_y.pop()

        self.actualizar()

    def on_key(self, event):
        """Callback for key presses."""
        if event.key == "c":
            self.puntos_x.clear()
            self.puntos_y.clear()
            self.actualizar()
        elif event.key == "s":
            self.guardar_csv()

    def actualizar(self):
        datos = list(zip(self.puntos_x, self.puntos_y))
        self.scatter.set_offsets(datos)
        self.fig.canvas.draw_idle()

    def add_point(self, xdata, ydata):
        self.puntos_x.append(xdata)
        self.puntos_y.append(ydata)

    def dibujar_linea(self, corte_y, pendiente, color="tab:red"):
        """
        Dibuja la recta y = pendiente * x + corte_y sobre los ejes actuales.

        Parámetros:
            corte_y   -- punto de corte con el eje y (ordenada en el origen, b)
            pendiente -- pendiente de la recta (m)
        """
        self.ax.axline((0, corte_y), slope=pendiente, color=color, linewidth=1.5)
        self.fig.canvas.draw_idle()

    def guardar_csv(self, ruta="puntos.csv"):
        with open(ruta, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerows(zip(self.puntos_x, self.puntos_y))
        print(f"Puntos guardados en {ruta}")

    def plot(self):
        plt.show()


if __name__ == "__main__":
    nube = NubeDePuntos(xlim=(0, 250000), ylim=(0, 10000))
    nube.dibujar_linea(corte_y=2, pendiente=0.5)  # y = 0.5x + 2
    plt.show()
