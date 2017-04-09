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
    from calc_acompcor import acompcor_nipype
    from calc_acompcor import convert_to_4d
    import numpy as np
    import nibabel as nib
    from sklearn.decomposition import PCA
    from skimage.morphology import binary_erosion
    from nilearn.image import resample_img
    csf_mask = nib.load(masks[2][0])
    wm_mask = nib.load(masks[1][0])
    gm_mask = nib.load(masks[0][0])
    
    
    if len(functional_images) > 1: 
        functional_image = convert_to_4d(functional_images)
    
    csf_rs = resample_img(csf_mask,target_affine=functional_image.affine,
                          target_shape=functional_image.shape[:-1],
                          interpolation='nearest')
    wm_rs = resample_img(wm_mask,target_affine=functional_image.affine,
                         target_shape=functional_image.shape[:-1],
                         interpolation='nearest')
    gm_rs = resample_img(gm_mask,target_affine=functional_image.affine,
                         target_shape=functional_image.shape[:-1],
                         interpolation='nearest')
    

    # reduce masks to voxels with 99% probability or higher
    csf_rs = csf_rs.get_data() >=0.99
    wm_rs = wm_rs.get_data() >=0.99
    gm_rs = gm_rs.get_data() >=0.99
    
    
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
    
    gm_ts = functional_data[gm_rs>0]

    
    components_wm = acompcor_nipype(wm_ts)
    components_csf = acompcor_nipype(csf_ts)
  
    
    wm_mask = nib.Nifti1Image(wm_rs, affine=functional_image.affine,
                             header = functional_image.header)
    csf_mask = nib.Nifti1Image(csf_rs, affine=functional_image.affine,
                              header = functional_image.header)
    
    nib.save(wm_mask, 'wm_mask.nii')
    nib.save(csf_mask, 'csf_mask.nii')

    np.savetxt('csf_components.txt', components_csf, fmt='%.10f')
    np.savetxt('wm_components.txt', components_wm, fmt='%.10f')
        
    np.savetxt('all_wm_timeseries.txt', wm_ts, fmt='%.10f')
    np.savetxt('all_csf_timeseries.txt', csf_ts, fmt='%.10f')
    np.savetxt('all_gm_timeseries.txt', gm_ts, fmt='%.10f')
    
    out_masks = [wm_mask, csf_mask]
    components = np.column_stack((components_csf, components_wm))
    return components, out_masks

def calc_global(masks,functional_images):
    from calc_acompcor import convert_to_4d
    from nilearn.image import resample_img
    import numpy as np
    import nibabel as nib
    
    if len(functional_images) > 1: 
        functional_image = convert_to_4d(functional_images)
    
    
    gray_matter = nib.load(masks[0][0]).get_data() > 0
    white_matter = nib.load(masks[1][0]).get_data() > 0
    csf = nib.load(masks[2][0]).get_data() > 0
    
    brain = gray_matter + white_matter + csf
    
    affine = nib.load(masks[0][0]).affine
    header = nib.load(masks[0][0]).header
    brain_img = nib.Nifti1Image(brain, affine, header)
      

    brain_rs = resample_img(brain_img,target_affine=functional_image.affine,
                          target_shape=functional_image.shape[:-1],
                          interpolation='nearest')

    brain_data = brain_rs.get_data()
    brain = brain_data >= 0.5

    data = functional_image.get_data()

    brain_ts  = data[brain]
    std_brain = np.std(brain_ts.T,axis=0)
    mean_brain = np.mean(brain_ts.T,axis=0)
    x_global = (brain_ts.T - mean_brain)/(std_brain + 1e-6)

    global_signal = np.mean(x_global,1)

    nib.save(brain_rs, 'brain_mask.nii')
    np.savetxt('global_signal.txt', global_signal, fmt = '%.10f')
    
    global_signal = global_signal[...,np.newaxis]
    
    return global_signal , brain_rs


def glm(csf_components, wm_components, global_signal):
    import statsmodels.api as sm
    
    y = global_signal
    X = wm_components
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X)
    results = model.fit()
    
    print results
    
    
def nuisance_regress(regressors, brainmask, functional_images):
    import os
    from calc_acompcor import convert_to_4d
    import nibabel as nib
    import numpy as np
    
    if len(functional_images) > 1: 
        functional_image = convert_to_4d(functional_images) 
    
    functional_data = functional_image.get_data()
    
    mask_data = brainmask.get_data()
    
    ijk = mask_data==1
    
    timeseries = functional_data[ijk].T
    
    x, _, _, _ = np.linalg.lstsq(regressors, timeseries)
    
    timeseries_hat = np.dot(regressors,x)
    
    residuals = timeseries - timeseries_hat
    
    indexes = np.where(mask_data==1)
    
    rebuilt_array = np.zeros(functional_data.shape)
    rebuilt_array[indexes[0], indexes[1], indexes[2]] = residuals.T
    
    residuals_image = nib.Nifti1Image(rebuilt_array, functional_image.affine,
                                      functional_image.header)
    
    out_file = os.path.join(os.getcwd(), 'residuals.nii')
    nib.save(residuals_image, out_file)
    
    return out_file
        
def corr_each_voxel(global_residuals, acompcor_residuals, brainmask):
    import nibabel as nib
    import numpy as np
    import os
    
    global_data = global_residuals.get_data()
    acompcor_data = acompcor_residuals.get_data()
    
    mask_data = brainmask.get_data()
    
    ijk = mask_data==1
    
    timeseries_global = global_data[ijk].T
    timeseries_acompcor = acompcor_data[ijk].T


    corr = np.zeros(timeseries_global.shape[1])
    for i in xrange(timeseries_global.shape[1]):
        corr[i] = np.corrcoef(timeseries_global[:,i], 
            timeseries_acompcor[:,i])[0,1]
    
    
    indexes = np.where(mask_data==1)
    rebuilt_array = np.zeros(global_data.shape[:-1])
    rebuilt_array[indexes[0], indexes[1], indexes[2]] = corr

    corr_image = nib.Nifti1Image(rebuilt_array, global_residuals.affine,
                                      global_residuals.header)
    
    out_file = os.path.join(os.getcwd(), 'correlations.nii')
    nib.save(corr_image, out_file)     
        
        
    

 

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
                        output_names = ['components','out_masks'],
                                   function = calc_noise_timeseries),
                        name = 'calc_acompcor')
    
    calc_global_signal = Node(interface = Function(
            input_names = ['masks', 'functional_images'], 
            output_names = ['global_signal', 'brain_img'], 
            function = calc_global), name = 'calc_global_signal')
    
    regressor = Node(interface = Function(
            input_names = ['regressors', 'brainmask', 'functional_images'],
            output_names = 'out_file', function = nuisance_regress),
        name = 'regressor')
    
    regressor_global = regressor.clone(name='global_regress')
    
    
    
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
                                                       'ventricle_mask')]),
                    (realigner, calc_global_signal, 
                     [('realigned_files', 'functional_images')]),
                     (segment, calc_global_signal, [('native_class_images',
                                                     'masks')]),
                    (calc_acompcor, regressor, [('components', 'regressors')]),
                    (calc_global_signal, regressor, [('brain_img', 'brainmask')]),
                    (realigner, regressor, [('realigned_files', 
                                             'functional_images')]),
                    (calc_global_signal, regressor_global, [('global_signal', 
                                                        'regressors')]),
                    (calc_global_signal, regressor_global, [('brain_img', 
                                                      'brainmask')]),
                    (realigner, regressor_global, [('realigned_files', 
                                             'functional_images')])
                    ])
    
    preproc.write_graph(dotfilename='graph.dot', graph2use='hierarchical',
                        format = 'png')   
    preproc.run('MultiProc')
    
    
def acompcor_nipype(voxel_timecourses):
    from scipy import linalg
    
    
    M = voxel_timecourses.T

    # "Voxel time series from the noise ROI (either anatomical or tSTD) were
    # placed in a matrix M of size Nxm, with time along the row dimension
    # and voxels along the column dimension."
    stdM = np.std(M, axis=0)
        # set bad values to x
    stdM[stdM == 0] = 1
    stdM[np.isnan(stdM)] = 1
    
    M = M / stdM

    # "The covariance matrix C = MMT was constructed and decomposed into its
    # principal components using a singular value decomposition."
    u, _, _ = linalg.svd(M, full_matrices=False)
    components = u[:, :5]
    
    return components


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
    
    
    
    



    

if __name__ == "__main__":
    
    data_dir = '/data/eaxfjord/fmriRSWorkingDir/nipype/output_dir_PreProc_Final/'
    subject_dir = os.path.join(data_dir,'filtered')
    subject_list= sorted(os.listdir(subject_dir)) #all subjects included in the study

    results = Parallel(n_jobs=6)(delayed(run)(idx,each,data_dir)
            for idx,each in enumerate(subject_list))
