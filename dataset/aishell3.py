import os

data_path = os.path.join("..", "aishell3", "train", "wav")
with open("aishell3.txt", "w", encoding="utf-8") as f:
    for foldername in os.listdir(data_path):
        for filename in os.listdir(os.path.join(data_path, foldername)):
            f.write(os.path.abspath(os.path.join(data_path, foldername, filename))+"\n")
