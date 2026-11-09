import polars as pl 
import numpy as np
from models import models as md
from models import network as ntwk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import subprocess
import os.path as path


data_path = "data/mnist_train.csv.gz"
if path.exists(data_path):
	# decompress data
	subprocess.run(["gzip", "-d", data_path])

"""Begin training phase"""
mnist = pl.read_csv("data/mnist_train.csv", has_header = False)
mnist = mnist.drop_nulls()

labels = mnist[:, 0].to_numpy().flatten()
images = mnist[:, 1:].to_numpy() / 255

network = ntwk.NeuralNetwork(activation_func = md.sigmoid)
# convert labels to output of columns
y = np.eye(10)[labels]

X_train, X_test, y_train, y_test = train_test_split(images, y, test_size = 0.15)

network.fit(X_train, 
y_train,
epochs = 3000, 
learning_rate = 0.5,
batch_size = 10,
)

score = r2_score(y_test, network.predict(X_test))

print(f"Score = {score}")