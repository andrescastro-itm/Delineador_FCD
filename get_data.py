import nibabel as nib
import nibabel.processing
import numpy as np

def preprocess(path, config, norm=False):

    scan = nib.load(path)
    aff  = scan.affine

    vol  = scan.get_fdata()

    # if 'Ras_msk' in path:

    try:
        vol = scan.get_fdata().squeeze(3)
        scan = nib.Nifti1Image(vol, aff)
    except:
        #vol = np.where(vol==0., 0., 1.)
        scan = nib.Nifti1Image(vol, aff)


    # Remuestrea volumen y affine a un nuevo shape

    #new_zooms  = np.array(scan.header.get_zooms()) * config['new_z']
    #new_shape  = np.array(vol.shape) // config['new_z']

    new_affine = nibabel.affines.rescale_affine(aff, 
                                                vol.shape, 
                                                config['hyperparams']['new_z'], 
                                                config['hyperparams']['model_dims']
    )

    scan       = nibabel.processing.conform(scan, 
                                            config['hyperparams']['model_dims'], 
                                            config['hyperparams']['new_z']
    )
     
    ni_img     = nib.Nifti1Image(scan.get_fdata(), new_affine)
    vol        = ni_img.get_fdata() 

    # if 'Ras_msk' in path:
    #     vol = np.where(vol <= 0.1, 0., 1.)

    if norm:
        vol = (vol - np.min(vol))/(np.max(vol) - np.min(vol))

    return vol