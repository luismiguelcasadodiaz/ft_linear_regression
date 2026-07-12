import canvas
from files_functions import ft_load_csv

data_file = "data.csv"
data = ft_load_csv(data_file)
nube = canvas.NubeDePuntos(xlim=(0, 250000), ylim=(0, 10000))
for row in range(1, len(data)):
    nube.add_point(data[row][0], data[row][1])
nube.plot()
