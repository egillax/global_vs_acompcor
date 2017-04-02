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
from nipype.interfaces.spm.preprocess import Realign, Coregister
from nipype.interfaces.spm.utils import Analyze2nii, ApplyInverseDeformation
from nipype.utils.filemanip import list_to_filename

from SPMCustom import NewSegment

from nipype.pipeline.engine import Node, Workflow

from nilearn.image import resample_img, threshold_img
from nilearn.masking import apply_mask, intersect_masks
from skimage.morphology import binary_erosion
from sklearn.decomposition import PCA

from joblib import Parallel, delayed


def short_pipeline(functional_data, anatomical_data):

    spm_path = '/home/egill/matlabtools/spm12/'
    alvin_path = '/home/egill/'
    working_dir = '/home/egill/global_vs_acompcor/working_dir'
    
    realigner = Node(interface = Realign(register_to_mean=True), 
                     name = 'realigner')
    realigner.inputs.in_files = functional_data
    
    
    coregister = Node(interface = Coregister(), name = 'coregister')
    coregiser.inputs.jobtype = 'estimate'
    coregister.inputs.source =  anatomical_data
    
    
    segment = Node(interface = NewSegment(), name = 'segment')
    segment.inputs.channel_info = (0.001, 60, (True, True))
    segment.inputs.write_deformation_fields = (True, True)
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
    
    alvin_to_native = Node(interface = ApplyInverseDeformation(),
                           name = 'alvin_to_native')
    
    preproc = Workflow(name = 'preproc')
    preproc.base_dir = working_dir
    
    preproc.connect([(realigner, coregister, [('mean_image',
                                               'target')]),
                    (coregister, segment, [('coregistered_source', 
                                            'channel_files')]),
                    (alvin_to_nifti, alvin_to_native, [('nifti_file',
                                                        list_to_filename,
                                                        'in_files')]),
                    (segment, alvin_to_native, [('inverse_deformation_field',
                                                 'deformation_field')])                  
                    ])
    
    
    
    #load smoothed data from subject
    smoothed_filter = 'sw'


    fnames_smoothed = [fn for  fn in os.listdir(smoothed_dir)
            if fn.startswith(smoothed_filter)]
    fnames_smoothed.sort()

    for idx2 in xrange(0,len(fnames_smoothed)):
        fname_abspath = os.path.join(smoothed_dir,fnames_smoothed[idx2])
        img = nb.load(fname_abspath)
        affine = img.affine
        header = img.header
        data = img.get_data()

        if idx2 is 0:
            smoothed_data=data
            smoothed_data = smoothed_data[...,np.newaxis]
        else:
            smoothed_data = np.concatenate((smoothed_data,
                                            data[...,np.newaxis]),axis=-1)
    smoothed_img = nb.Nifti1Image(smoothed_data,affine, header)
    return smoothed_img

def load_masks(t1_dir,each):

    #load csf and wm mask and reslice to match subject fmri data
    csf_filter = list(['wc3'])
    wm_filter = list(['wc2'])

    #subject specific dir
    subj_dir = os.path.join(t1_dir, each)


    fname_csf = [fn for  fn in os.listdir(subj_dir)
            if fn.startswith(csf_filter[0])]

    fname_wm = [fn for  fn in os.listdir(subj_dir)
            if fn.startswith(wm_filter[0])]

    fname_csf = os.path.join(subj_dir,fname_csf[0])
    fname_wm = os.path.join(subj_dir,fname_wm[0])

    csf_mask = nb.load(fname_csf)
    wm_mask = nb.load(fname_wm)
    masks = [csf_mask, wm_mask]

    return masks

def calc_noise_timeseries(masks, smooth_img, ventricles):
    csf_mask = masks[0]
    wm_mask = masks[1]

    csf_rs = resample_img(csf_mask,target_affine=smooth_img.affine,
                          target_shape=smooth_img.shape[:-1],
                          interpolation='nearest')
    wm_rs = resample_img(wm_mask,target_affine=smooth_img.affine,
                         target_shape=smooth_img.shape[:-1],
                         interpolation='nearest')

    # reduce masks to voxels with 99% probability or higher
    csf_rs = csf_rs.get_data() >=0.99
    wm_rs = wm_rs.get_data() >=0.99


    smoothed_data = np.squeeze(smooth_img.get_data())
    ventricles = resample_img(ventricles,target_affine=smooth_img.affine,
                              target_shape = smoothed_data.shape[:-1],
                                interpolation='nearest')
    ventdata = ventricles.get_data()

    # intersect csf with ventricles
    csf_rs = csf_rs & ventdata

    # get time series data
    csf_ts = smoothed_data[csf_rs>0]

    wm_rs = binary_erosion(wm_rs)

    wm_ts = smoothed_data[wm_rs>0]

    X_csf = csf_ts.T
    X_wm = wm_ts.T

    stdCSF = np.std(X_csf,axis=0)
    stdWM = np.std(X_wm,axis=0)

    X_csf = (X_csf - np.mean(X_csf,axis=0))/(stdCSF + 1e-6)
    X_wm = (X_wm - np.mean(X_wm,axis=0))/(stdWM + 1e-6)

    pca = PCA()

    components_csf = pca.fit(X_csf.T).components_
    components_wm = pca.fit(X_wm.T).components_

    wm_mask = nb.Nifti1Image(wm_rs, affine=smooth_img.affine,
                             header = smooth_img.header)
    csf_mask = nb.Nifti1Image(csf_rs, affine=smooth_img.affine,
                              header = smooth_img.header)
    masks = [wm_mask, csf_mask]

    return components_csf, components_wm, masks

def calc_global(brain_mask_path,data_img):

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




def run(idx,each,data_dir):
    
    
    subject_list = ['con001_T1', 'con002_T1', 'con003_T1']
    subject = subject_list[0]
    data_dir = '/home/egill/Dropbox/test_data/'
    functional_folder = os.path.join(data_dir, subject)
    anatomical_folder = os.path.join(data_dir, subject, 'T1Img')
    
    functional_data = [os.path.abspath(fn) for fn in os.listdir(functional_folder)
                        if fn.endswith('.nii')]
    functional_data.sort()
    anatomical_data = [os.path.abspath(fn) for fn in
                       os.listdir(anatomical_folder) if fn.endswith('.nii')]
    
    short_pipeline(functional_data, anatomical_data)
    
    smoothed_dir = os.path.join(data_dir,'filtered')
    subj_dir = os.path.join(smoothed_dir, each)
    mask_dir = os.path.join(data_dir, 'segment')
    t1_dir = mask_dir

    smooth = load_data(subj_dir)
    masks = load_masks(t1_dir,each)

    vent_fname = '/data/eaxfjord/fmriRSWorkingDir/nipype/data/masks/ventricles/ventricles.nii'
    ventricles = nb.load(vent_fname)

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


data_dir = '/data/eaxfjord/fmriRSWorkingDir/nipype/output_dir_PreProc_Final/'
subject_dir = os.path.join(data_dir,'filtered')
subject_list= sorted(os.listdir(subject_dir)) #all subjects included in the study

results = Parallel(n_jobs=6)(delayed(run)(idx,each,data_dir)
            for idx,each in enumerate(subject_list))
