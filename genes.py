class Genes:
    def __init__(self, num_sensors, num_outputs):
        self.inputs = []
        self.innovation = 0
    
    def feed_sensor_values(self, values):
        return []
        # return output values
        pass
    
    def num_nodes(self):
        return 1
        pass

    def add_connection(self, in, out, weight, enabled = True):
        pass

    def add_node(self, in, out):
        pass

    def mutate(self):
        pass

    def breed(self, other):
        return Genes(0, 0)
        pass

    def distance(self, other, c_1, c_2, c_3):
        return 0

    def clone(self):
        return self

    def save(self, out_stream):
        pass

    def load(in_stream):
        return Genes(0, 0)
