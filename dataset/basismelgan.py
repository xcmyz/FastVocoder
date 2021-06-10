import os

data_path = os.path.join("..", "Basis-MelGAN-dataset", "generated")
with open("basismelgan.txt", "w", encoding="utf-8") as f:
    for filename in os.listdir(data_path):
        f.write(os.path.abspath(os.path.join(data_path, filename))+"\n")
