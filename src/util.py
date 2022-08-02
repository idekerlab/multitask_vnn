import os
import torch
from torch._six import inf
import math
import shutil
import numpy as np
import pandas as pd
import random as rd
from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import scale
from scipy import stats
from sklearn.metrics import r2_score


def pearson_corr(x, y):
	xx = x - torch.mean(x)
	yy = y - torch.mean(y)
	return torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy,2))


def get_pearson_corr(labels, predict, cuda):
	task_len = labels.size(dim=1)
	corr_list = torch.zeros(task_len).cuda(cuda)
	for i in range(task_len):
		xi = labels[:,i]
		yi = predict[:,i]
		mask = torch.isnan(xi)
		xi = xi[~mask]
		yi = yi[~mask]
		corr_list[i] = pearson_corr(xi, yi)
	return corr_list


def convert_train_file(train_file, tasks):
	all_df = pd.read_csv(train_file, sep='\t', header=None, names=['cell_line', 'smiles', 'auc'])
	all_df = all_df.query('smiles in @tasks')
	new_df = all_df.pivot(index='cell_line', columns='smiles', values='auc')
	new_df.reset_index(inplace=True)
	final_df = new_df.rename(columns = {'index':'cell_line'})
	return final_df


def prepare_train_data(train_df, cell_id_mapping):

	train_cell_lines = list(train_df['cell_line'].copy())
	val_cell_lines = []
	val_size = int(len(train_cell_lines)/5)

	for _ in range(val_size):
		r = rd.randint(0, len(train_cell_lines) - 1)
		val_cell_lines.append(train_cell_lines.pop(r))

	val_df = train_df.query('cell_line in @val_cell_lines').reset_index(drop=True)
	train_df = train_df.query('cell_line in @train_cell_lines').reset_index(drop=True)

	train_features = []
	train_labels = []
	for row in train_df.values:
		train_features.append(cell_id_mapping[row[0]])
		train_labels.append([float(auc) for auc in row[1:]])

	val_features = []
	val_labels = []
	for row in val_df.values:
		val_features.append(cell_id_mapping[row[0]])
		val_labels.append([float(auc) for auc in row[1:]])

	return (torch.Tensor(train_features), torch.Tensor(val_features), torch.FloatTensor(train_labels), torch.FloatTensor(val_labels))


def prepare_test_data(test_df, cell_id_mapping):
	test_features = []
	test_labels = []
	for row in test_df.values:
		test_features.append(cell_id_mapping[row[0]])
		test_labels.append([float(auc) for auc in row[1:]])
	return (torch.Tensor(test_features), torch.FloatTensor(test_labels))


def load_mapping(mapping_file, mapping_type):
	mapping = {}
	file_handle = open(mapping_file)
	for line in file_handle:
		line = line.rstrip().split()
		mapping[line[1]] = int(line[0])

	file_handle.close()
	print('Total number of {} = {}'.format(mapping_type, len(mapping)))
	return mapping


def build_input_vector(input_data, cell_features):
	genedim = len(cell_features[0, :])
	featdim = len(cell_features[0, 0, :])
	feature = np.zeros((input_data.size()[0], genedim, featdim))

	for i in range(input_data.size()[0]):
		feature[i] = cell_features[int(input_data[i])]

	feature = torch.from_numpy(feature).float()
	return feature


# build mask: matrix (nrows = number of relevant gene set, ncols = number all genes)
# elements of matrix are 1 if the corresponding gene is one of the relevant genes
def create_term_mask(term_direct_gene_map, gene_dim, cuda_id):
	term_mask_map = {}
	for term, gene_set in term_direct_gene_map.items():
		mask = torch.zeros(len(gene_set), gene_dim).cuda(cuda_id)
		for i, gene_id in enumerate(gene_set):
			mask[i, gene_id] = 1
		term_mask_map[term] = mask
	return term_mask_map


def get_grad_norm(model_params, norm_type):
	"""Gets gradient norm of an iterable of model_params.
	The norm is computed over all gradients together, as if they were
	concatenated into a single vector. Gradients are modified in-place.
	Arguments:
		model_params (Iterable[Tensor] or Tensor): an iterable of Tensors or a
			single Tensor that will have gradients normalized
		norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
			infinity norm.
	Returns:Total norm of the model_params (viewed as a single vector).
	"""
	if isinstance(model_params, torch.Tensor): # check if parameters are tensorobject
		model_params = [model_params] # change to list
	model_params = [p for p in model_params if p.grad is not None] # get list of params with grads
	norm_type = float(norm_type) # make sure norm_type is of type float
	if len(model_params) == 0: # if no params provided, return tensor of 0
		return torch.tensor(0.)

	device = model_params[0].grad.device # get device
	if norm_type == inf: # infinity norm
		total_norm = max(p.grad.detach().abs().max().to(device) for p in model_params)
	else: # total norm
		total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in model_params]), norm_type)
	return total_norm


def read_file(filename, delim='\n'):
	rfile = open(filename, 'r')
	item_list = rfile.read().strip().split(delim)
	rfile.close()
	return item_list


def create_dir(dir_path):
	if os.path.isdir(dir_path):
		shutil.rmtree(dir_path)
	os.mkdir(dir_path)


def create_cv_data(all_df, fold_size):

	data_list = []
	cell_lines = list(all_df['cell_line'].copy())
	cell_count = len(cell_lines)
	for k in range(1, fold_size+1):

		cv_size = int(cell_count/fold_size) + k%2
		if len(cell_lines) < cv_size:
			cv_size = len(cell_lines)

		k_cell_lines = []
		for i in range(cv_size):
			r = rd.randint(0, len(cell_lines) - 1)
			k_cell_lines.append(cell_lines.pop(r))

		k_test_data = all_df.query('cell_line in @k_cell_lines')
		k_train_data = all_df.drop(k_test_data.index)
		data_list.append((k_train_data, k_test_data))

	return data_list
