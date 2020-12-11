from genes import *
import numpy as np
import time

meta = Genes.Metaparameters()
mj = meta.as_json()
g = Genes(5, 5, meta)
neurons = None
input = [0, 1, 2, 3, 4, 5]
neurons = g.feed_sensor_values(input, neurons)
for _ in range(20):
    g.mutate()
out = g.as_json()
g2 = g.clone()
meta.reset_tracking()
for _ in range(20):
    g.mutate()
out2 = g2.as_json()
out3 = g.as_json()
pass
"""
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
"""
pass

