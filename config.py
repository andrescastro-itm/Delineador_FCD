import datetime

def get_parameters(args):

    # mode = 'reg' # available modes: 'reg', 'train', 'test'
    print(args.input)
    print(args.output)
    print(args.numslice)
    print(args.useclassifier)
    hyperparams = {'model_dims': (128, 128, 64), # Dimensiones de entrada al modelo
                   'lr'        : 0.00001,         # Taza de aprendizaje
                   'epochs'    : 20,             # Numero de epocas
                   'batch_size': 1,              # Tama;o del batch
                   'new_z'     : [2, 2, 2],      # Nuevo tama;o de zooms
                   'n_heads'   : 23,             # Numero de cabezas
                   'n_train'   : 19,             # "" Entrenamiento
                   'n_val'     : 2,              # "" Validacion
                   'n_test'    : 2,              # "" Prueba
                   'batchnorm' : False           # Normalizacion de batch
    }

    files = {'input': args.input,
             'model': './model/weights-BCEDice-10_eps-25_heads-2023-06-30-_nobn-e7.pth', 
             'output': args.output,
             'clasModel': './model/Deteccion-weights-e15.pth'}

    labels = {'bgnd': 0, # Image background
              'FCD' : 1, # Focal cortical dysplasia
    }

    return {'labels'   : labels,
            'hyperparams' : hyperparams,
            'files'    : files,
            'numslice' : args.numslice,
            'useclas': args.useclassifier}