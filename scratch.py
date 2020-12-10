from genes import *
import numpy as np
import time

f = open("atari_pacman6/sample_gene_446.json", "r")
genes = Genes.load(f, Genes.Metaparameters())
f.close()

genes_json = genes.as_json()

input = [_ for _ in range(128)]

t1 = time.time()
neurons = None
for _ in range(20000):
    neurons = genes.feed_sensor_values(input, neurons)
t2 = time.time()
print(f"Time = {t2 - t1}")
pass


