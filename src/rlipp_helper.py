import argparse
from rlipp_calculator import *


def main():

	parser = argparse.ArgumentParser(description = 'RLIPP score calculation')
	parser.add_argument('-hidden', help = 'Hidden folders path', type = str)
	parser.add_argument('-ontology', help = 'Ontology file', type = str)
	parser.add_argument('-test', help = 'Test file', type = str)
	parser.add_argument('-predicted', help = 'Predicted result file', type = str)
	parser.add_argument('-gene2idfile', help = 'Gene-index file', type = str)
	parser.add_argument('-cell2idfile', help = 'Cell-index file', type = str)
	parser.add_argument('-sys_output', help = 'RLIPP file', type = str)
	parser.add_argument('-gene_output', help = 'Gene rho file', type = str)
	parser.add_argument('-cpu_count', help = 'No of available cores', type = int, default = 1)

	cmd_args = parser.parse_args()

	rlipp_calculator = RLIPPCalculator(cmd_args)
	rlipp_calculator.calc_scores()


if __name__ == "__main__":
	main()
