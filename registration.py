#!/usr/bin/python
# This file is subject to the terms and conditions defined in
# file 'LICENCE', which is part of this source code package.
# Author: Leo Guignard (guignardl...@AT@...janelia.hhmi.org)

from time import time
import os
import json
from multiprocessing import Pool
import numpy as np
from IO import imread, imsave, SpatialImage
from pyklb import readheader
path_to_bin = ''

def read_trsf(path):
    ''' Read a transformation from a text file
        Args:
            path: string, path to a transformation
    '''
    f = open(path)
    if f.read()[0] == '(':
        f.close()
        f = open(path)
        lines = f.readlines()[2:-1]
        f.close()
        return np.array([[float(v) for v in l.split()]  for l in lines])
    f.close()
    return np.loadtxt(path)

def produce_trsf(params):
    ''' Given an output path, an image path format, a reference time, two time points,
        two locks on their respective images, an original voxel size and a voxel size,
        compute the transformation that registers together two consecutive in time images.
        The registration is done by cross correlation of the MIP along the 3 main axes.
        This function is meant to be call by multiprocess.Pool
    '''
    (p_out, p, r, t1, t2, vs,
     trsf_type, make, registration_depth) = params
    if isinstance(t1, int):
        # t_for_string = (t1,)*p.count('%')
        # p_im_1 = p%(t_for_string)
        p_im_1 = p.format(t=t1)
    else:
        p_im_1 = t1
        t1 = int([k for k in ref_TP.split('/') if 'TM' in k][0][2:9])
    if not make:
        print 'trsf tp %d-%d not done'%(t1, t2)
        if t1 < r:
            np.savetxt(p_out + 't%06d-%06d.txt'%(t1, t2), np.identity(4))
        else:
            p_out + 't%06d-%06d.txt'%(t2, t1)
        return
    if (not os.path.exists(p_out + 't%06d-%06d.txt'%(t1, t2))
            or os.path.exists(p_out + 't%06d-%06d.txt'%(t2, t1))):
        # path_format = p
        # t_for_string = (t2,)*p.count('%')
        p_im_2 = p.format(t=t2)#(t_for_string)
        if t1 < r:
            os.system('blockmatching -ref ' + p_im_2 + ' -flo ' + p_im_1 + \
                      ' -reference-voxel %f %f %f'%vs + \
                      ' -floating-voxel %f %f %f'%vs + \
                      ' -trsf-type %s -py-hl 6 -py-ll %d'%(trsf_type, registration_depth) + \
                      ' -res-trsf ' + p_out + 't%06d-%06d.txt'%(t1, t2))
        else:
            os.system('blockmatching -ref ' + p_im_1 + ' -flo ' + p_im_2 + \
                      ' -reference-voxel %f %f %f'%vs + \
                      ' -floating-voxel %f %f %f'%vs + \
                      ' -trsf-type %s -py-hl 6 -py-ll %d'%(trsf_type, registration_depth) + \
                      ' -res-trsf ' + p_out + 't%06d-%06d.txt'%(t2, t1))

def run_produce_trsf(p, r, trsf_p, tp_list, vs=(3., 3., 5.), nb_cpu=1,
                     trsf_type='translation', not_to_do=None, registration_depth=3):
    ''' Parallel processing of the transformations from t to t-1/t-1 to t (depending on t<r)
        The transformation is computed using blockmatching algorithm
        Args:
            p: string, path pattern to the images to register
            r: int, reference time point
            nb_times (not used): int, number of time points on which to apply the transformation
            trsf_p: string, path to the transformation
            tp_list: [int, ], list of time points on which to apply the transformation
            ors: float, original aspect ratio
            first_TP (not used): int, first time point on which to apply the transformation
            vs: float, aspect ratio
            nb_cpy: int, number of cpus to use
    '''
    if not os.path.exists(trsf_p):
        os.mkdir(trsf_p)
    if not_to_do is None:
        not_to_do = []
    mapping = [(trsf_p, p, r, t1, t2, vs, trsf_type,
                0 if (t1 in not_to_do or t2 in not_to_do) else 1,
                registration_depth)
               for t1, t2 in zip(tp_list[:-1], tp_list[1:])]
    tic = time()
    if nb_cpu == 1:
        tmp = []
        for params in mapping:
            tmp += [produce_trsf(params)]
    else:
        pool = Pool(processes=nb_cpu)
        tmp = pool.map(produce_trsf, mapping)
        pool.close()
        pool.terminate()
    tac = time()
    whole_time = tac - tic
    secs = whole_time%60
    whole_time = whole_time//60
    mins = whole_time%60
    hours = whole_time//60
    print '%dh:%dmin:%ds'%(hours, mins, secs)

def compose_trsf(flo_t, ref_t, trsf_p, tp_list):
    ''' Recusrively build the transformation that allows
        to register time `flo_t` onto the frame of time `ref_t`
        assuming that it exists the necessary intermediary transformations
        Args:
            flo_t: int, time of the floating image
            ref_t: int, time of the reference image
            trsf_p: string, path to folder containing the transformations
            tp_list: [int, ], list of time points that have been processed
        Returns:
            out_trsf: string, path to the result composed transformation
    '''
    out_trsf = trsf_p + 't%06d-%06d.txt'%(flo_t, ref_t)
    if not os.path.exists(out_trsf):
        flo_int = tp_list[tp_list.index(flo_t) + np.sign(ref_t - flo_t)]
        # the call is recursive, to build `T_{flo\leftarrow ref}`
        # we need `T_{flo+1\leftarrow ref}` and `T_{flo\leftarrow ref-1}`
        trsf_1 = compose_trsf(flo_int, ref_t, trsf_p, tp_list)
        trsf_2 = compose_trsf(flo_t, flo_int, trsf_p, tp_list)
        os.system(path_to_bin + 'composeTrsf ' + out_trsf + ' -trsfs ' + trsf_2 + ' ' + trsf_1)
    return out_trsf

def read_param_file():
    ''' Asks for, reads and formats the parameter file
    '''
    p_param = raw_input('Please enter the path to the parameter file:\n')
    p_param = p_param.replace('"', '')
    p_param = p_param.replace("'", '')
    p_param = p_param.replace(" ", '')
    if os.path.isdir(p_param):
        f_names = [os.path.join(p_param, f) for f in os.listdir(p_param)
                   if '.json' in f and not '~' in f]
    else:
        f_names = [p_param]
    path_to_datas = []
    path_outs = []
    v_sizes = []
    trsf_types = []
    ref_TPs = []
    suffixs = []
    file_name_patterns = []
    firsts = []
    lasts = []
    p_trsfs = []
    not_to_do = []
    registration_depth = []
    for file_name in f_names:
        with open(file_name) as f:
            param_dict = json.load(f)
            f.close()
        path_to_datas += [param_dict['path_to_data']]
        path_outs += [param_dict['path_out']]
        v_sizes += [tuple(param_dict['voxel_size'])]
        trsf_types += [param_dict['trsf_type']]
        ref_TPs += [param_dict['ref_TP']]
        suffixs += [param_dict['suffix']]
        file_name_patterns += [param_dict['file_name']]
        firsts += [int(param_dict['first'])]
        lasts += [int(param_dict['last'])]
        p_trsfs += [param_dict.get('trsf_folder', None)]
        not_to_do += [param_dict.get('not_to_do', [])]
        registration_depth += [param_dict.get('registration_depth', 3)]
    return (path_to_datas, path_outs, v_sizes, trsf_types, ref_TPs,
            suffixs, file_name_patterns, firsts, lasts,
            p_trsfs, not_to_do, registration_depth)

if __name__ == '__main__':
    params = read_param_file()
    for (p_to_data, p_out,
         v_size, trsf_type, ref_TP,
         suffix, file_name_pattern,
         first, last, p_trsf,
         not_to_do, registration_depth) in zip(*params):
        im_ext = file_name_pattern.split('.')[-1]
        A0 = p_to_data + file_name_pattern
        A0_out = p_to_data + file_name_pattern.replace(im_ext, suffix + '.' + im_ext)
        time_points = np.arange(first, last + 1)
        if p_trsf is None:
            if not os.path.exists(p_out):
                os.makedirs(p_out)
            try:
                to_register = time_points
                max_t = max(time_points)
                run_produce_trsf(A0, ref_TP, p_out, to_register,
                                 vs=v_size, nb_cpu=1, trsf_type=trsf_type,
                                 not_to_do=not_to_do, registration_depth=registration_depth)
                compose_trsf(min(to_register), ref_TP, p_out, list(to_register))
                compose_trsf(max(to_register), ref_TP, p_out, list(to_register))
                np.savetxt('{:s}t{t:06d}-{t:06d}.txt'.format(p_out, t=ref_TP), np.identity(4))
            except Exception as e:
                print p_to_data
                print e

            if isinstance(ref_TP, int):
                if A0.split('.')[-1] == 'klb':
                    im_shape = readheader(A0.format(t=ref_TP))['imagesize_tczyx'][-1:-4:-1]
                else:
                    im_shape = imread(A0.format(t=ref_TP)).shape
                im = SpatialImage(np.ones(im_shape), dtype=np.uint8)
                im.voxelsize = v_size
                imsave(p_out + 'tmp.klb', im)
                os.system('changeMultipleTrsfs -trsf-format ' + p_out + 't%%06d-%06d.txt '%(ref_TP) + \
                                              '-index-reference %d -first %d -last %d '%(ref_TP, min(time_points), max(time_points)) + \
                                              '-template ' + p_out + 'tmp.klb ' + \
                                              '-res ' + p_out + 't%%06d-%06d-padded.txt '%(ref_TP) + \
                                              '-res-t ' + p_out + 'template.klb ' + \
                                              '-trsf-type %s -vs %f %f %f'%((trsf_type,)+v_size))
        else:
            p_out = p_trsf
        if isinstance(ref_TP, int):
            print 'single time series registration'
            X, Y, Z = readheader(p_out + 'template.klb')['imagesize_tczyx'][-1:-4:-1]
            xy_proj = np.zeros((X, Y, len(time_points)), dtype=np.uint16)
            xz_proj = np.zeros((X, Z, len(time_points)), dtype=np.uint16)
            yz_proj = np.zeros((Y, Z, len(time_points)), dtype=np.uint16)
            for i, t in enumerate(sorted(time_points)):
                os.system("applyTrsf '%s' '%s' -trsf "%(A0.format(t=t), A0_out.format(t=t)) + \
                          p_out + 't%06d-%06d-padded.txt '%(t, ref_TP) + \
                          '-template ' + p_out + 'template.klb -floating-voxel %f %f %f -interpolation linear'%v_size)
                im = imread(A0_out.format(t=t))
                xy_proj[:, :, i] = SpatialImage(np.max(im, axis=2))
                xz_proj[:, :, i] = SpatialImage(np.max(im, axis=1))
                yz_proj[:, :, i] = SpatialImage(np.max(im, axis=0))
            if not os.path.exists(p_to_data.format(t=-1)):
                os.makedirs(p_to_data.format(t=-1))
            imsave((p_to_data + file_name_pattern.replace(im_ext, 'xyProjection.klb')).format(t=-1),
                   SpatialImage(xy_proj))
            imsave((p_to_data + file_name_pattern.replace(im_ext, 'xzProjection.klb')).format(t=-1),
                   SpatialImage(xz_proj))
            imsave((p_to_data + file_name_pattern.replace(im_ext, 'yzProjection.klb')).format(t=-1),
                   SpatialImage(yz_proj))
        else:
            print 'two time series registration'
            for t in sorted(time_points):
                os.system('applyTrsf %s %s -trsf '%(A0%(t, t), A0_out%(t, t)) + \
                          p_out + 't%06d-%06d-filtered.txt '%(t, ref_TP) + \
                          '-template %s -floating-voxel %f %f %f -interpolation linear'%((ref_TP,)+v_size))
                im = imread(A0_out%(t, t))
                imsave(A0_out.replace('.klb', '_xyProjection.klb')%(t, t),
                       SpatialImage(np.max(im, axis=2).reshape(im.shape[:2] + (1,))))
