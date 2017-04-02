 #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:14:22 2016

calc_acompcor


@author: eaxfjord
"""
import os
import nibabel as nb
import numpy as np
from nipype.interfaces.spm.preprocess import Realign, Coregister, Normalize12
from nipype.interfaces.spm.utils import Analyze2nii
from SPMCustom import NewSegment

from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.utility import Function

from joblib import Parallel, delayed
import nibabel as nib

def convert_to_4d(functional_data):

    for idx in xrange(0,len(functional_data)):
        img = nib.load(functional_data[idx])
        affine = img.affine
        header = img.header
        data = img.get_data()
        
        if idx is 0:
            all_data = data
            all_data = all_data[..., np.newaxis]
        else:
            all_data = np.concatenate((all_data, data[..., np.newaxis]), 
                                      axis = -1)
    functional_4d_data = nib.Nifti1Image(all_data, affine, header)

    return functional_4d_data

def calc_noise_timeseries(masks, functional_images, ventricle_mask):
    from calc_acompcor import convert_to_4d
    import numpy as np
    import nibabel as nib
    from sklearn.decomposition import PCA
    from skimage.morphology import binary_erosion
    from nilearn.image import resample_img
    csf_mask = nib.load(masks[2][0])
    wm_mask = nib.load(masks[1][0])
    
    if len(functional_images) > 1: 
        functional_image = convert_to_4d(functional_images)
    
    csf_rs = resample_img(csf_mask,target_affine=functional_image.affine,
                          target_shape=functional_image.shape[:-1],
                          interpolation='nearest')
    wm_rs = resample_img(wm_mask,target_affine=functional_image.affine,
                         target_shape=functional_image.shape[:-1],
                         interpolation='nearest')

    # reduce masks to voxels with 99% probability or higher
    csf_rs = csf_rs.get_data() >=0.99
    wm_rs = wm_rs.get_data() >=0.99

    
    
    functional_data = np.squeeze(functional_image.get_data())
    ventricle_mask = resample_img(ventricle_mask, 
                                  target_affine=functional_image.affine,
                              target_shape = functional_data.shape[:-1],
                                interpolation='nearest')
    ventdata = ventricle_mask.get_data() > 50

    # intersect csf with ventricles
    csf_rs = csf_rs & ventdata

    # get time series data
    csf_ts = functional_data[csf_rs>0]

    wm_rs = binary_erosion(wm_rs)

    wm_ts = functional_data[wm_rs>0]

    X_csf = csf_ts.T
    X_wm = wm_ts.T

    stdCSF = np.std(X_csf,axis=0)
    stdWM = np.std(X_wm,axis=0)

    X_csf = (X_csf - np.mean(X_csf,axis=0))/(stdCSF + 1e-6)
    X_wm = (X_wm - np.mean(X_wm,axis=0))/(stdWM + 1e-6)

    pca = PCA()

    components_csf = pca.fit(X_csf.T).components_
    csf_explained_var = pca.explained_variance_ratio_
    
    
    components_wm = pca.fit(X_wm.T).components_
    wm_explained_var = pca.explained_variance_ratio_
    
    csf_num_components_50 = np.argmax(np.cumsum(csf_explained_var) > 0.5 )
    wm_num_components_50 = np.argmax(np.cumsum(wm_explained_var) > 0.5 )
    csf_50 = components_csf[:,:csf_num_components_50]
    wm_50 = components_wm[:,:wm_num_components_50] 
    
    wm_mask = nib.Nifti1Image(wm_rs, affine=functional_image.affine,
                             header = functional_image.header)
    csf_mask = nib.Nifti1Image(csf_rs, affine=functional_image.affine,
                              header = functional_image.header)
    
    nib.save(wm_mask, 'wm_mask.nii')
    nib.save(csf_mask, 'csf_mask.nii')

    np.savetxt('all_csf_components.txt', components_csf, fmt='%.10f')
    np.savetxt('all_wm_components.txt', components_wm, fmt='%.10f')
    
    np.savetxt('50_percent_csf_components.txt', csf_50, fmt='%.10f')
    np.savetxt('50_percent_wm_components.txt', wm_50, fmt='%.10f')
    
    out_masks = [wm_mask, csf_mask]

    return components_csf, components_wm, out_masks

def calc_global(masks,functional_images):
    from calc_acompcor import convert_to_4d
    from nilearn.image import resample_img
    import nibabel as nib
    
    if len(functional_images) > 1: 
        functional_image = convert_to_4d(functional_images)
    
    
    gray_matter = nib.load(masks[0][0]).get_data() > 0
    white_matter = nib.load(masks[1][0]).get_data() > 0
    csf = nib.load(masks[2][0]).get_data() > 0
    
    brain = gray_matter + white_matter + csf
    
    
    
    brain_img = nb.load(brain_mask_path)



    brain_rs = resample_img(brain_img,target_affine=data_img.affine,
                          target_shape=data_img.shape[:-1],
                          interpolation='nearest')

    brain_data = brain_rs.get_data()
    brain = brain_data >= 0.5

    data = data_img.get_data()

    brain_ts  = data[brain]
    std_brain = np.std(brain_ts.T,axis=0)
    mean_brain = np.mean(brain_ts.T,axis=0)
    x_global = (brain_ts.T - mean_brain)/(std_brain + 1e-6)

    global_signal = np.mean(x_global,1)

    return global_signal

def short_pipeline(functional_data, anatomical_data):

    spm_path = '/home/egill/matlabtools/spm12/'
    alvin_path = '/home/egill/global_vs_acompcor/alvin_mask/ALVIN_mask_v1.hdr'
    working_dir = '/home/egill/global_vs_acompcor/working_dir'
    
    from nipype import config
    config.set('execution', 'remove_unnecessary_outputs', 'False')  
    
    
    realigner = Node(interface = Realign(register_to_mean=True), 
                     name = 'realigner')
    realigner.inputs.in_files = functional_data
    
    
    coregister = Node(interface = Coregister(), name = 'coregister')
    coregister.inputs.jobtype = 'estimate'
    coregister.inputs.source =  anatomical_data
    
    
    segment = Node(interface = NewSegment(), name = 'segment')
    segment.inputs.channel_info = (0.001, 60, (True, True))
    segment.inputs.write_deformation_fields = [True, True]
    tpm = os.path.join(spm_path,'tpm/TPM.nii')
    tissue1 = ((tpm, 1), 1, (True, False), (False, False))
    tissue2 = ((tpm, 2), 1, (True, False), (False, False))
    tissue3 = ((tpm, 3), 2, (True, False), (False, False))
    tissue4 = ((tpm, 4), 3, (False, False), (False, False))
    tissue5 = ((tpm, 5), 4, (False, False), (False, False))
    tissue6 = ((tpm, 6), 2, (False, False), (False, False))
    segment.inputs.tissues = [tissue1, tissue2, tissue3, tissue4, 
                              tissue5, tissue6]
    segment.inputs.affine_regularization = 'mni'
    
    alvin_to_nifti = Node(interface = Analyze2nii(),
                          name = 'alvin_to_nifti')
    alvin_to_nifti.inputs.analyze_file = alvin_path
    
    alvin_to_native = Node(interface = Normalize12(),
                           name = 'alvin_to_native')
    alvin_to_native.inputs.jobtype = 'write'
    
    calc_acompcor = Node(interface = Function(input_names = \
                            ['masks', 'functional_images', 'ventricle_mask'],
                        output_names = ['components_csf', 'components_wm',
                                   'out_masks'],
                                   function = calc_noise_timeseries),
                        name = 'calc_acompcor')
    
    
    preproc = Workflow(name = 'preproc')
    preproc.base_dir = working_dir
    
    preproc.connect([(realigner, coregister, [('mean_image',
                                               'target')]),
                    (coregister, segment, [('coregistered_source', 
                                            'channel_files')]),
                    (alvin_to_nifti, alvin_to_native, [('nifti_file',
                                                        'apply_to_files')]),
                    (segment, alvin_to_native, [('inverse_deformation_field',
                                                 'deformation_file')]),
                    (segment, calc_acompcor, [('native_class_images',
                                               'masks')]),
                    (realigner, calc_acompcor, [('realigned_files',
                                                 'functional_images')]),
                    (alvin_to_native, calc_acompcor, [('normalized_files',
                                                       'ventricle_mask')])
                    ])
    
        
    preproc.run('MultiProc')


def run(idx,each,data_dir):
    
    
    subject_list = ['con001_T1', 'con002_T1', 'con003_T1']
    subject = subject_list[0]
    data_dir = '/home/egill/Dropbox/test_data/'
    functional_folder = os.path.join(data_dir, subject)
    anatomical_folder = os.path.join(data_dir, subject, 'T1Img')
    
    functional_data = [os.path.join(functional_folder,fn) for fn in os.listdir(functional_folder)
                        if fn.endswith('.nii')]
    functional_data.sort()
    anatomical_data = [os.path.join(anatomical_folder,fn) for fn in
                       os.listdir(anatomical_folder) if fn.endswith('.nii')]
    
    short_pipeline(functional_data, anatomical_data)
    


    comp_csf, comp_wm, masks = calc_noise_timeseries(masks, smooth,
                                              ventricles)

    t1_subj_dir = os.path.join(t1_dir,each)
    brain_mask =  [fn for  fn in os.listdir(t1_subj_dir)
                if fn.startswith('wBrain')]
    brain_mask_path = os.path.join(t1_subj_dir,brain_mask[0])
    global_signal = calc_global(brain_mask_path,smooth)

    corr = np.corrcoef(comp_wm,global_signal)


    save_loc = os.path.join(mask_dir,'FunImg', each, 'GS_test')

    if not os.path.exists(save_loc):
        os.mkdir(save_loc)

    np.savetxt(os.path.join(save_loc,'csf.txt'), comp_csf,fmt='%.10f')
    np.savetxt(os.path.join(save_loc,'wm.txt'), comp_wm, fmt='%.10f')
    np.savetxt(os.path.join(save_loc, 'global.txt'), global_signal,
                            fmt= '%.10f')
    nb.save(masks[0],os.path.join(save_loc, 'wm_mask.nii'))
    nb.save(masks[1],os.path.join(save_loc, 'csf_mask.nii'))

    components = [comp_wm, comp_csf]

    return corr, components,  global_signal

if __name__ == "__main__":
    
    data_dir = '/data/eaxfjord/fmriRSWorkingDir/nipype/output_dir_PreProc_Final/'
    subject_dir = os.path.join(data_dir,'filtered')
    subject_list= sorted(os.listdir(subject_dir)) #all subjects included in the study

    results = Parallel(n_jobs=6)(delayed(run)(idx,each,data_dir)
            for idx,each in enumerate(subject_list))
