import argparse
from config import get_parameters
from inference import inference


print("inicio programa")

def main(config):
	inference(config)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Modelo para segmentacion semántica de displasia cortical\
                                                  focal en imagenes de resonancia magnetica')

	parser.add_argument('-i','--input', type=str, required=True, help="""Ruta de estudio NIfTI a procesar""")

	parser.add_argument('-o','--output', type=str, help="""Ruta de salida para almacenamiento de figuras""")

	parser.add_argument('-nums','--numslice', type=int, help="""Número de slice a segmentar""",default=-1)

	parser.add_argument('-useclas','--useclassifier', type=int, help="""Usar clasificador (0 si no se desea usar, en caso contrario 1)""",default=1)

	args = parser.parse_args()

	if not args.output:
		path = '/'.join(args.input.split('/')[:-1])
		args.output = path + '/out/'
	else:
		args.output = args.output+'/out/'

	# if args.output[-1] != '/':
	# 	args.output = args.output+'/'

	config = get_parameters(args)
	main(config)



