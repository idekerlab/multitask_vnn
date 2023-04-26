import os
import numpy as np
import pandas as pd
import time
from scipy import stats
from multiprocessing import Pool
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)


class RLIPPCalculator():

	def __init__(self, args):
		self.ontology = pd.read_csv(args.ontology, sep='\t', header=None, names=['S', 'T', 'I'], dtype={0:str, 1:str, 2:str})
		self.terms = self.ontology['S'].unique().tolist()
		self.test_df = self.replace_auc(args.test, args.predicted)
		self.genes = pd.read_csv(args.gene2idfile, sep='\t', header=None, names=['I', 'G'])['G']
		self.cell_index = pd.read_csv(args.cell2idfile, sep="\t", header=None, names=['I', 'C'])
		self.rlipp_file = args.sys_output
		self.gene_rho_file = args.gene_output
		self.cpu_count = args.cpu_count

		self.hidden_dir = args.hidden
		if not self.hidden_dir.endswith('/'):
			self.hidden_dir += '/'


	def replace_auc(self, test_file, predict_file):
		test_df = pd.read_csv(test_file, sep='\t')
		predicted_vals = np.loadtxt(predict_file)
		drug_list = list(test_df.columns)
		drug_list.remove('cell_line')
		for i,d in enumerate(drug_list):
			test_df[d] = np.where(test_df[d].notna(), predicted_vals[:,i], np.nan)
		return test_df


	#Load the hidden file for a given element
	def load_feature(self, element):
		file_name = self.hidden_dir + element + '.hidden'
		return np.loadtxt(file_name)


	def load_term_features(self, term):
		return self.load_feature(term)


	def load_gene_features(self, gene):
		return self.load_feature(gene)


	def create_child_feature_map(self, feature_map, term):
		child_features = []
		child_features.append(term)
		children = [row['T'] for _,row in self.ontology.iterrows() if row['S']==term]
		for child in children:
			child_features.append(feature_map[child])
		return child_features


	#Load hidden features for all the terms and genes
	def load_all_features(self):
		feature_map = {}
		with Pool(self.cpu_count) as p:
			results = p.map(self.load_term_features, self.terms)
		for i,t in enumerate(self.terms):
			feature_map[t] = results[i]
		with Pool(self.cpu_count) as p:
			results = p.map(self.load_gene_features, self.genes)
		for i,g in enumerate(self.genes):
			feature_map[g] = results[i]

		child_feature_map = {t:[] for t in self.terms}
		for term in self.terms:
			children = [row['T'] for _,row in self.ontology.iterrows() if row['S']==term]
			for child in children:
				child_feature_map[term].append(feature_map[child])

		return feature_map, child_feature_map


	#Get a hidden feature matrix of a given term's children
	def get_child_features(self, term_child_features):
		child_features = []
		for f in term_child_features:
			child_features.append(f)
		return np.column_stack([f for f in child_features])


	#Executes 5-fold cross validated Ridge regression for a given hidden features matrix
	#and returns the spearman correlation value of the predicted output
	def exec_lm(self, X, y, pca_dim):

		pca = PCA(n_components=pca_dim)
		X_pca = pca.fit_transform(X)

		regr = RidgeCV(cv=5)
		regr.fit(X_pca, y)
		y_pred = regr.predict(X_pca)
		return stats.spearmanr(y_pred, y)


	# Calculates RLIPP for a given term
	#Executes parallely
	def calc_term_rlipp(self, term_features, term_child_features, term, drug):
		pca_dim = np.size(term_features, axis=1)
		y = np.array(self.test_df[drug])
		mask = np.isnan(y)
		y = y[~mask]
		X_parent = term_features[~mask,:]

		X_child = self.get_child_features(term_child_features)[~mask,:]
		p_rho,_ = self.exec_lm(X_parent, y, pca_dim)
		c_rho,_ = self.exec_lm(X_child, y, pca_dim)
		rlipp = p_rho/c_rho
		result = '{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(drug, term, p_rho, c_rho, rlipp)
		return result


	#Calculates Spearman correlation between Gene embeddings and Predicted AUC
	def calc_gene_rho(self, gene_features, gene, drug):
		pred = np.array(self.test_df[drug])
		rho,_ = stats.spearmanr(pred, gene_features)
		result = '{}\t{}\t{:.4f}\n'.format(drug, gene, rho)
		return result


	#Calculates RLIPP scores for top n drugs (n = drug_count), and
	#prints the result in "Drug Term P_rho C_rho RLIPP" format
	def calc_scores(self):
		print('Starting score calculation')

		start = time.time()
		feature_map, child_feature_map = self.load_all_features()
		print('Time taken to load features: {:.4f}'.format(time.time() - start))

		drug_list = list(self.test_df.columns)
		drug_list.remove('cell_line')

		rlipp_file = open(self.rlipp_file, "w")
		rlipp_file.write('Drug\tTerm\tP_rho\tC_rho\tRLIPP\n')
		gene_rho_file = open(self.gene_rho_file, "w")
		gene_rho_file.write('Drug\tGene\tRho\n')

		with Parallel(backend="multiprocessing", n_jobs=self.cpu_count) as parallel:
			for i, drug in enumerate(drug_list):
				start = time.time()

				rlipp_results = parallel(delayed(self.calc_term_rlipp)(feature_map[term], child_feature_map[term], term, drug) for term in self.terms)
				for result in rlipp_results:
					rlipp_file.write(result)

				gene_rho_results = parallel(delayed(self.calc_gene_rho)(feature_map[gene], gene, drug) for gene in self.genes)
				for result in gene_rho_results:
					gene_rho_file.write(result)

				print('Drug {} completed in {:.4f} seconds'.format((i+1), (time.time() - start)))
		gene_rho_file.close()
		rlipp_file.close()
