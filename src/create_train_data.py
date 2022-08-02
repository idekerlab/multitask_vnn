import argparse
import copy
import shutil
import pandas as pd

import util

def main():

	parser = argparse.ArgumentParser(description = 'Create CV train/test data')
	parser.add_argument('-modeldir', help = 'Folder for trained models', type = str, default = 'model/')
	parser.add_argument('-folds', help = 'Number of folds for CV', type = int, default = 5)
	parser.add_argument('-train', help = 'Training dataset', type = str)
	parser.add_argument('-tasks', help = 'Task list file (drug smiles)', type = str)

	opt = parser.parse_args()

	tasks = util.read_file(opt.tasks)
	all_df = util.convert_train_file(opt.train, tasks)
	data_list = util.create_cv_data(all_df, opt.folds)

	for f in range(opt.folds):
		modeldir = opt.modeldir + '_' + str(f+1)
		util.create_dir(modeldir)
		train_data = data_list[f][0]
		train_data.to_csv(modeldir + '/train.txt', sep='\t', header=True, index=False)
		test_data = data_list[f][1]
		test_data.to_csv(modeldir + '/test.txt', sep='\t', header=True, index=False)


if __name__ == "__main__":
	main()

