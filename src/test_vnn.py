import argparse
import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import util


def test(predict_data, gene_dim, model_file, hidden_folder, batch_size, result_file, cell_features):

	feature_dim = gene_dim

	model = torch.load(model_file, map_location='cuda:%d' % CUDA_ID)

	predict_feature, predict_label = predict_data

	predict_label_gpu = predict_label.cuda(CUDA_ID)

	model.cuda(CUDA_ID)
	model.eval()

	test_loader = du.DataLoader(du.TensorDataset(predict_feature, predict_label), batch_size=batch_size, shuffle=False)

	#Test
	test_predict = torch.zeros(0,0).cuda(CUDA_ID)
	hidden_embeddings_map = {}

	saved_grads = {}
	def save_grad(element):
		def savegrad_hook(grad):
			saved_grads[element] = grad
		return savegrad_hook

	for i, (inputdata, labels) in enumerate(test_loader):
		# Convert torch tensor to Variable
		features = util.build_input_vector(inputdata, cell_features)

		cuda_features = Variable(features.cuda(CUDA_ID), requires_grad=True)

		# make prediction for test data
		aux_out_map, hidden_embeddings_map = model(cuda_features)

		if test_predict.size()[0] == 0:
			test_predict = aux_out_map['final'].data
		else:
			test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

		for element, hidden_map in hidden_embeddings_map.items():
			hidden_file = hidden_folder + '/' + element + '.hidden'
			with open(hidden_file, 'ab') as f:
				np.savetxt(f, hidden_map.data.detach().cpu().numpy(), '%.4e')

		for element, _ in hidden_embeddings_map.items():
			hidden_embeddings_map[element].register_hook(save_grad(element))

		## Do backprop
		aux_out_map['final'].backward(torch.ones_like(aux_out_map['final']))

		# Save Feature Grads
		feature_grad = torch.zeros(0,0).cuda(CUDA_ID)
		for i in range(len(cuda_features[0, 0, :])):
			feature_grad = cuda_features.grad.data[:, :, i]
			with open(result_file + '_feature_grad_' + str(i) + '.txt', 'ab') as f:
				np.savetxt(f, feature_grad.detach().cpu().numpy(), '%.4e', delimiter='\t')

		# Save Hidden Grads
		for element, hidden_grad in saved_grads.items():
			hidden_file = hidden_folder + '/' + element + '.hidden_grad'
			with open(hidden_file, 'ab') as f:
				np.savetxt(f, hidden_grad.data.detach().cpu().numpy(), '%.4e', delimiter='\t')

	#test_corr = util.get_pearson_corr(predict_label_gpu, test_predict, CUDA_ID)
	#print("Test correlation", test_corr.detach().cpu().numpy())

	np.savetxt(result_file + '.txt', test_predict.detach().cpu().numpy(),'%.4f')


parser = argparse.ArgumentParser(description='Test VNN')
parser.add_argument('-predict', help='Dataset to be predicted', type=str)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=1000)
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str)
parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str)
parser.add_argument('-load', help='Model file', type=str)
parser.add_argument('-hidden', help='Hidden output folder', type=str, default='hidden/')
parser.add_argument('-result', help='Result file prefix', type=str, default='result/predict')
parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
parser.add_argument('-mutations', help = 'Mutation information for cell lines', type = str)
parser.add_argument('-cn_deletions', help = 'Copy number deletions for cell lines', type = str)
parser.add_argument('-cn_amplifications', help = 'Copy number amplifications for cell lines', type = str)

opt = parser.parse_args()
torch.set_printoptions(precision=5)

test_df = pd.read_csv(opt.predict, sep='\t')
cell_id_mapping = util.load_mapping(opt.cell2id, 'cell lines')
predict_data = util.prepare_test_data(test_df, cell_id_mapping)

# load cell/drug features
mutations = np.genfromtxt(opt.mutations, delimiter = ',')
cn_deletions = np.genfromtxt(opt.cn_deletions, delimiter = ',')
cn_amplifications = np.genfromtxt(opt.cn_amplifications, delimiter = ',')
cell_features = np.dstack([mutations, cn_deletions, cn_amplifications])

gene2id_mapping = util.load_mapping(opt.gene2id, "genes")
num_genes = len(gene2id_mapping)

CUDA_ID = opt.cuda

test(predict_data, num_genes, opt.load, opt.hidden, opt.batchsize, opt.result, cell_features)
