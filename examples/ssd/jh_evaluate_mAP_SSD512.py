from __future__ import print_function
import caffe
from caffe.model_libs import *
from google.protobuf import text_format
import numpy as np

import math, os, shutil, stat, sys

# copy the function from SSD implementation from liuwei89
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

if __name__ == "__main__":

    caffe_root = os.environ['CAFFE_ROOT']
    job_name = "jh_ssd512_test"
    model_name = "VGG_VOC0712_{}".format(job_name)
    weights = os.path.join( caffe_root , "models/VGGNet/VOC0712Plus/SSD_512x512_ft/VGG_VOC0712_SSD_512x512_ft_iter_120000.caffemodel" )
    save_dir = os.path.join( caffe_root , "models/VGGNet/VOC0712Plus/{}".format(job_name) )
    job_dir = os.path.join( caffe_root , "jobs/VGGNet/VOC0712Plus/{}".format(job_name) )
    test_net_file = '/home/jh/working_lib/caffe/models/VGGNet/VOC0712Plus/SSD_512x512_ft/test.prototxt'

    test_batch_size = 8
    num_test_image = 4952

    check_if_exist(weights)
    make_if_not_exist(save_dir)
    make_if_not_exist(job_dir)

    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = caffe.Net( test_net_file , weights , caffe.TEST )

    test_iter = num_test_image / test_batch_size

    num_classes = 21
    #test_iter = 50
    num_gt_cls = np.zeros( num_classes -1 ).astype(int)
    num_tp_det_cls = [ [] for _ in xrange( num_classes - 1 ) ]
    num_fp_det_cls = [ [] for _ in xrange( num_classes - 1 ) ]

    for iiter in xrange( test_iter ):
        output = net.forward()
        print ( "iter = {}".format(iiter) )
        d = output['detection_eval'][0][0]
        for index in xrange(d.shape[0]):

            dd = d[index]
            if ( int(dd[0]) == -1 ):
                num_gt_cls[ int(dd[1]) -1 ] += int(dd[2])
                continue

            cls = int(dd[1]) - 1
            conf = dd[2]
            tp = int(dd[3])
            fp = int(dd[4])

            num_tp_det_cls[cls].append([conf , tp ])
            num_fp_det_cls[cls].append([conf , fp ])

    ap = np.zeros( num_classes - 1 )
    mAP = 0.

    for icls in xrange( len(num_tp_det_cls) ):
        num_tp_det_cls[icls].sort( reverse = True , key = lambda x: x[0] )
        num_fp_det_cls[icls].sort( reverse = True , key = lambda x: x[0] )

        num_pos = num_gt_cls[icls]

        tmp1 = np.array( num_tp_det_cls[icls] )
        tmp2 = np.array( num_fp_det_cls[icls] )

        numpy_tp = np.array( tmp1[:,1] )
        numpy_fp = np.array( tmp2[:,1] )

        numpy_tp = np.cumsum( numpy_tp )
        numpy_fp = np.cumsum( numpy_fp )

        rec = numpy_tp / float( num_pos )
        prec = numpy_tp / np.maximum(numpy_tp + numpy_fp, np.finfo(np.float64).eps)

        ap[icls] = voc_ap( rec , prec , use_07_metric = True )

        print ( "icls = {} , ap = {}".format( icls , ap[icls]) )

    mAP = np.mean(ap)
    print( "mAP = {}".format(mAP) )

