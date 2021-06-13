import os

data_path = os.path.join("..", "BZNSYP", "Wave")
with open("BZNSYP.txt", "w", encoding="utf-8") as f:
    for filename in os.listdir(data_path):
        if filename[0] != ".":
            f.write(os.path.abspath(os.path.join(data_path, filename))+"\n")
