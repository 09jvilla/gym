import pickle
import time 

def export_weights_sparse(weight_dict):
	filename = "sparse_weight_" + time.strftime("%m%d_%H%M") + ".pkl"
	output = open(filename, 'wb')
	pickle.dump(weight_dict, output)
	output.close()

