import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du
from torch.autograd import Variable

import util
from data_wrapper import *
from vnn import *


class VNNTrainer():

	def __init__(self, data_wrapper, train_feature, val_feature, train_label, val_label):
		self.data_wrapper = data_wrapper
		self.train_feature = train_feature
		self.val_feature = val_feature
		self.train_label = train_label
		self.val_label = val_label


	def train_model(self):

		epoch_start_time = time.time()
		min_loss = None
		early_stopping_counter = 0

		model = VNN(self.data_wrapper)
		model.cuda(self.data_wrapper.cuda)

		term_mask_map = util.create_term_mask(model.term_direct_gene_map, model.gene_dim, self.data_wrapper.cuda)
		for name, param in model.named_parameters():
			term_name = name.split('_')[0]
			if '_direct_gene_layer.weight' in name:
				param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
			else:
				param.data = param.data * 0.1

		train_loader = du.DataLoader(du.TensorDataset(self.train_feature, self.train_label), batch_size=self.data_wrapper.batchsize, shuffle=True, drop_last=True)
		val_loader = du.DataLoader(du.TensorDataset(self.val_feature, self.val_label), batch_size=self.data_wrapper.batchsize, shuffle=True)

		optimizer = torch.optim.AdamW(model.parameters(), lr=self.data_wrapper.lr, betas=(0.9, 0.99), eps=1e-05, weight_decay=self.data_wrapper.lr)
		optimizer.zero_grad()

		print("epoch\ttrain_loss\tval_loss\tgrad_norm\telapsed_time")
		for epoch in range(self.data_wrapper.epochs):
			# Train
			model.train()
			train_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)
			_gradnorms = torch.empty(len(train_loader)).cuda(self.data_wrapper.cuda) # tensor for accumulating grad norms from each batch in this epoch

			for i, (inputdata, labels) in enumerate(train_loader):
				# Convert torch tensor to Variable
				features = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
				cuda_features = Variable(features.cuda(self.data_wrapper.cuda))
				cuda_labels = Variable(labels.cuda(self.data_wrapper.cuda))

				# Forward + Backward + Optimize
				optimizer.zero_grad()  # zero the gradient buffer

				aux_out_map,_ = model(cuda_features)

				if train_predict.size()[0] == 0:
					train_predict = aux_out_map['final'].data
					train_label_gpu = cuda_labels
				else:
					train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)
					train_label_gpu = torch.cat([train_label_gpu, cuda_labels], dim=0)

				total_loss = 0
				for name, output in aux_out_map.items():
					mask = torch.isnan(cuda_labels)
					masked_cuda_labels = cuda_labels[~mask]
					masked_output = output[~mask]
					loss = nn.MSELoss()
					if name == 'final':
						total_loss += loss(masked_output, masked_cuda_labels)
					else:
						total_loss += self.data_wrapper.alpha * loss(masked_output, masked_cuda_labels)
				total_loss.backward()

				for name, param in model.named_parameters():
					if '_direct_gene_layer.weight' not in name:
						continue
					term_name = name.split('_')[0]
					param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

				_gradnorms[i] = util.get_grad_norm(model.parameters(), 2.0).unsqueeze(0) # Save gradnorm for batch
				optimizer.step()

			gradnorms = sum(_gradnorms).unsqueeze(0).cpu().numpy()[0] # Save total gradnorm for epoch
			train_corr = util.get_pearson_corr(train_label_gpu, train_predict, self.data_wrapper.cuda)

			model.eval()

			val_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)

			for i, (inputdata, labels) in enumerate(val_loader):
				# Convert torch tensor to Variable
				features = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
				cuda_features = Variable(features.cuda(self.data_wrapper.cuda))
				cuda_labels = Variable(labels.cuda(self.data_wrapper.cuda))

				aux_out_map, _ = model(cuda_features)

				if val_predict.size()[0] == 0:
					val_predict = aux_out_map['final'].data
					val_label_gpu = cuda_labels
				else:
					val_predict = torch.cat([val_predict, aux_out_map['final'].data], dim=0)
					val_label_gpu = torch.cat([val_label_gpu, cuda_labels], dim=0)

				val_loss = 0
				for name, output in aux_out_map.items():
					mask = torch.isnan(cuda_labels)
					masked_cuda_labels = cuda_labels[~mask]
					masked_output = output[~mask]
					loss = nn.MSELoss()
					if name == 'final':
						val_loss += loss(masked_output, masked_cuda_labels)

			val_corr = util.get_pearson_corr(val_label_gpu, val_predict, self.data_wrapper.cuda)

			epoch_end_time = time.time()
			print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(epoch, total_loss, val_loss, gradnorms, epoch_end_time - epoch_start_time))
			epoch_start_time = epoch_end_time

			if min_loss == None:
				min_loss = val_loss
				early_stopping_counter = 0
				torch.save(model, self.data_wrapper.modeldir + '/model_final.pt')
				print("Model saved at epoch {}".format(epoch))
				print("Train correlations:", train_corr.detach().cpu().numpy())
				print("Val correlations:", val_corr.detach().cpu().numpy())
			elif min_loss - val_loss >= self.data_wrapper.delta:
				min_loss = val_loss
				early_stopping_counter = 0
				torch.save(model, self.data_wrapper.modeldir + '/model_final.pt')
				print("Model saved at epoch {}".format(epoch))
				print("Train correlations:", train_corr.detach().cpu().numpy())
				print("Val correlations:", val_corr.detach().cpu().numpy())
			else:
				early_stopping_counter += 1

			if early_stopping_counter >= self.data_wrapper.patience:
				break


		return min_loss


	def update_data_wrapper(self, key, value):
		if hasattr(self.data_wrapper, key):
			setattr(self.data_wrapper, key, value)