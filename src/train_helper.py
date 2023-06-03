import argparse
import copy
import shutil
import pandas as pd
from os.path import exists

import util
from optuna_nn_trainer import *
from data_wrapper import *


def main():

	torch.set_printoptions(precision = 5)

	parser = argparse.ArgumentParser(description = 'Train/test VNN')
	parser.add_argument('-onto', help = 'Ontology file used to guide the neural network', type = str)
	parser.add_argument('-train', help = 'Training dataset', type = str)
	parser.add_argument('-tasks', help = 'Task list file (drug smiles)', type = str)
	parser.add_argument('-epoch', help = 'Training epochs for training', type = int, default = 300)
	parser.add_argument('-lr', help = 'Learning rate', type = float, default = 0.001)
	parser.add_argument('-wd', help = 'Weight decay', type = float, default = 0.001)
	parser.add_argument('-alpha', help = 'Loss parameter alpha', type = float, default = 0.3)
	parser.add_argument('-batchsize', help = 'Batchsize', type = int, default = 64)
	parser.add_argument('-modeldir', help = 'Folder for trained models', type = str, default = 'model/')
	parser.add_argument('-cuda', help = 'Specify GPU', type = int, default = 0)
	parser.add_argument('-gene2id', help = 'Gene to ID mapping file', type = str)
	parser.add_argument('-cell2id', help = 'Cell to ID mapping file', type = str)
	parser.add_argument('-genotype_hiddens', help = 'Mapping for the number of neurons in each term in genotype parts', type = int, default = 4)
	parser.add_argument('-mutations', help = 'Mutation information for cell lines', type = str)
	parser.add_argument('-cn_deletions', help = 'Copy number deletions for cell lines', type = str)
	parser.add_argument('-cn_amplifications', help = 'Copy number amplifications for cell lines', type = str)
	parser.add_argument('-optimize', help = 'Training option (0=optimize)', type = int, default = 1)
	parser.add_argument('-tuned_hyperparams', help = 'Tuned hyperparameter file', type = str, default = '')
	parser.add_argument('-patience', help = 'Early stopping epoch limit', type = int, default = 30)
	parser.add_argument('-delta', help = 'Minimum change in loss to be considered an improvement', type = float, default = 0.001)
	parser.add_argument('-min_dropout_layer', help = 'Start dropout from this Layer number', type = int, default = 2)
	parser.add_argument('-dropout_fraction', help = 'Dropout Fraction', type = float, default = 0.3)

	opt = parser.parse_args()
	data_wrapper = DataWrapper(opt)

	train_data = pd.read_csv(data_wrapper.train, sep='\t')

	train_features, val_features, train_labels, val_labels = util.prepare_train_data(train_data, data_wrapper.cell_id_mapping)

	vnn_trainer = OptunaNNTrainer(data_wrapper, train_features, val_features, train_labels, val_labels)
	if opt.optimize == 0:
		trial_params = vnn_trainer.exec_study(opt.tuned_hyperparams)
		for key, value in trial_params.items():
			vnn_trainer.update_data_wrapper(key, value)
	else:
		if exists(opt.tuned_hyperparams):
			with open(opt.tuned_hyperparams, 'r') as f:
				key, value = f.readline().strip().split()
				vnn_trainer.update_data_wrapper(key, value)
	vnn_trainer.train_model()
	

if __name__ == "__main__":
	main()
