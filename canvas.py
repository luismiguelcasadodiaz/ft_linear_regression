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

from ols import calculate_ols


class NubeDePuntos:
    def __init__(self, xlim=(0, 10), ylim=(0, 10)):
        self.puntos_x = []
        self.puntos_y = []
        self.puntos_x_sum = 0
        self.puntos_y_sum = 0
        self.puntos_count = 0

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_title(
            "Clic izq: añadir | Clic der: deshacer | 'c': limpiar | 's': guardar"
        )
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.line = None  # Regression line that updates dinamically

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
            self.puntos_x_sum += event.xdata
            self.puntos_y_sum += event.ydata
            self.puntos_count += 1
        elif event.button == 3:  # clic derecho
            if self.puntos_x:
                self.puntos_x_sum -= self.puntos_x[-1]
                self.puntos_y_sum -= self.puntos_y[-1]
                self.puntos_x.pop()
                self.puntos_y.pop()
                self.puntos_count -= 1

        self.actualizar()

    def on_key(self, event):
        """Callback for key presses."""
        if event.key == "c":
            self.puntos_x.clear()
            self.puntos_y.clear()
            self.puntos_count = 0
            self.actualizar()
        elif event.key == "s":
            self.guardar_csv()

    def actualizar(self):
        if self.puntos_x:
            datos = list(zip(self.puntos_x, self.puntos_y))
            self.scatter.set_offsets(datos)
            if self.puntos_count > 1:  # Solo dibujar la línea si hay al menos 2 puntos
                slope, intercept_y = calculate_ols(self.puntos_x, self.puntos_y)
                #slope, intercept_y = calculate_gd(self.puntos_x, self.puntos_y)
                self.unset_line()  # Remove previous regression line if exists
                self.set_line(intercept_y, slope, color="tab:red")
        else:
            self.scatter.remove()
            self.unset_line()  # Remove previous regression line if exists
            self.scatter = self.ax.scatter([], [], color="tab:blue", s=40)


        self.fig.canvas.draw_idle()

    def add_point(self, xdata, ydata):
        self.puntos_x.append(xdata)
        self.puntos_y.append(ydata)

    def set_line(self, intercept_y, slope, color="tab:red"):
        """
        Dibuja la recta y = slope * x + intercept_y sobre los ejes actuales.

        Parámetros:
            intercept_y   -- punto de corte con el eje y (ordenada en el origen, b)
            slope -- pendiente de la recta (m)
        """
        self.line = self.ax.axline((0, intercept_y), slope=slope, color=color, linewidth=1.5)
        self.fig.canvas.draw_idle()

    def unset_line(self):
        """Removes regression line if exists."""
        if self.line:
            self.line.remove()
            self.line = None

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
    plt.show()
