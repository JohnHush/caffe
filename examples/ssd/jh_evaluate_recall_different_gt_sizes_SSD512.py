from __future__ import print_function
import caffe
from caffe.model_libs import *
from google.protobuf import text_format
import numpy as np
import bbox as bbb
import matplotlib.pyplot as plt

import math, os, shutil, stat, sys
import cPickle

def decode_bbox( bbox , h , w ):
    """
    bbox is numpy array or list
    """
    #clip the boxes
    bbox = map( lambda x: max( min( x , 1.), 0. ), bbox )

    #scale it
    bbox[0] = int( bbox[0] * w )
    bbox[1] = int( bbox[1] * h )
    bbox[2] = int( bbox[2] * w )
    bbox[3] = int( bbox[3] * h )

    return bbox

def bbox_overlaps( b , q , normalized ):
    """
    boxes: [N,4] ndarray of float
    query_boxes :[K, 4] ndarray of float

    return :
    overlap :[N, K] ndarray of overlaps 
    """
    N = b.shape[0]
    K = q.shape[0]

    overlaps = np.zeros( ( N , K) , dtype=np.float )
    if normalized:
        for k in xrange(K):
            box_area = (q[k,2] - q[k,0]) * (q[k,3] - q[k,1])
            for n in xrange(N):
                iw = min( b[n,2] , q[k,2] ) - max( b[n,0] , q[k,0])
                if iw > 0:
                    ih = min( b[n,3] , q[k,3] ) - max( b[n,1] , q[k,1] )
                    if ih > 0:
                        ua = (b[n,2]-b[n,0]) * (b[n,3]-b[n,1]) + box_area - iw * ih

                        overlaps[n,k] = iw * ih /ua
    else:
        for k in xrange(K):
            box_area = (q[k,2] - q[k,0]  + 1) * (q[k,3] - q[k,1] + 1 )
            for n in xrange(N):
                iw = min( b[n,2] , q[k,2] ) - max( b[n,0] , q[k,0]) + 1
                if iw > 0:
                    ih = min( b[n,3] , q[k,3] ) - max( b[n,1] , q[k,1] ) + 1
                    if ih > 0:
                        ua = (b[n,2]-b[n,0] + 1) * (b[n,3]-b[n,1] +1 ) + box_area - iw * ih

                        overlaps[n,k] = iw * ih /ua
    return overlaps

def batch_gt_overlap( gt_boxes , det_boxes , num_batch , _sizes , use_diff = False \
        , low_area = None , high_area = None , area = 'all'):
    """
    gt_boxes assumes to be 2d numpy array,
    [num_gt][ batch_id , group_label , instance_id , xmin , ymin , xmax , ymax ,
    diff ]
    det_boxes assumes to be 2d numpy array has the following form:
    [num_det][ batch_id , label , score , xmin , ymin , xmax , ymax ]

    _sizes keep the sizes of the input image batches ..
    to filter the qualified area boxes
    with the form [num_batch][ height , widht ]
    """
    areas = { 'all': 0, 'small': 1, 'medium': 2, 'large': 3,
		'96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
    area_ranges = [ [0**2, 1e5**2],    # all
		    [0**2, 32**2],     # small
		    [32**2, 96**2],    # medium
		    [96**2, 1e5**2],   # large
		    [96**2, 128**2],   # 96-128
		    [128**2, 256**2],  # 128-256
		    [256**2, 512**2],  # 256-512
		    [512**2, 1e5**2],  # 512-inf
		    ]
    assert areas.has_key(area), 'unknown area range: {}'.format(area)
    area_range = area_ranges[areas[area]]

    if low_area != None and high_area != None:
        area_range = [ low_area , high_area ]

    num_pos = 0
    gt_overlaps = np.zeros(0)

    for i in xrange( num_batch ):
        height = _sizes[i][0]
        width  = _sizes[i][1]

        gt_boxes_per_image = np.zeros(0)
        dt_boxes_per_image = np.zeros(0)
        #collect gt boxes from gt_boxes numpy ndarray
        for j in xrange( gt_boxes.shape[0] ):
            box = gt_boxes[j]

            if int(box[0]) == i:
                if not use_diff and int(box[7]) == 1:
                    continue

                bbox = decode_bbox( box[3:7] , height , width )
                bbox_area = ( bbox[2] - bbox[0] + 1 ) * ( bbox[3] - bbox[1] + 1 )
                if bbox_area < area_range[0] or bbox_area >= area_range[1]:
                    continue

                gt_boxes_per_image = np.hstack(( gt_boxes_per_image , bbox ))
        num_gt = gt_boxes_per_image.size / 4
        gt_boxes_per_image = gt_boxes_per_image.reshape( ( num_gt , 4) )

        num_pos += num_gt

        #collect all th det boxes 
        for j in xrange( det_boxes.shape[0] ):
            box = det_boxes[j]

            if int(box[0]) == i:
                #bbox = np.array( ( box[3], box[4] , box[5] , box[6] ) )
                bbox = decode_bbox( box[3:7] , height , width )
                dt_boxes_per_image = np.hstack(( dt_boxes_per_image , bbox ))
        num_dt = dt_boxes_per_image.size /4
        dt_boxes_per_image = dt_boxes_per_image.reshape(( num_dt , 4 ))

        #overlaps = bbox_overlaps( dt_boxes_per_image , gt_boxes_per_image , normalized = False )
        overlaps = bbb.bbox_overlaps( dt_boxes_per_image , gt_boxes_per_image )

        _gt_overlaps = np.zeros( (gt_boxes_per_image.shape[0]) )
        for j in xrange( gt_boxes_per_image.shape[0] ):
            argmax_overlaps = overlaps.argmax(axis=0)
            max_overlaps = overlaps.max(axis=0)
            gt_ind = max_overlaps.argmax()
            gt_ovr = max_overlaps.max()

            box_ind = argmax_overlaps[gt_ind]
            _gt_overlaps[j] = overlaps[box_ind , gt_ind]

            overlaps[box_ind,:] = -1
            overlaps[:,gt_ind] = -1
        gt_overlaps = np.hstack(( gt_overlaps , _gt_overlaps ))

    return gt_overlaps , num_pos

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
    job_name = "jh_ssd512_test_recall_diff_gt_size"
    model_name = "VGG_VOC0712_{}".format(job_name)
    weights = os.path.join( caffe_root , "models/VGGNet/VOC0712Plus/SSD_512x512_ft/VGG_VOC0712_SSD_512x512_ft_iter_120000.caffemodel" )
    save_dir = os.path.join( caffe_root , "models/VGGNet/VOC0712Plus/{}".format(job_name) )
    job_dir = os.path.join( caffe_root , "jobs/VGGNet/VOC0712Plus/{}".format(job_name) )
    test_net_file = '/home/jh/working_lib/caffe/models/VGGNet/VOC0712Plus/SSD_512x512_ft/test.prototxt'

    recall_save_file = "{}/recall_different_gt_size_SSD512.pkl".format(job_dir)

    name_size_file = os.path.join( caffe_root, "data/VOC0712/test_name_size.txt" )
    test_batch_size = 8
    num_test_image = 4952
    num_classes = 21

    check_if_exist(weights)
    make_if_not_exist(save_dir)
    make_if_not_exist(job_dir)

    test_iter = num_test_image / test_batch_size

    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = caffe.Net( test_net_file , weights , caffe.TEST )

    # read name size file
    f = open( name_size_file , "r" )
    lines = f.readlines()

    _sizes = np.zeros(0)
    for line in lines:
        line_list = line.split()

        _sizes = np.hstack( (_sizes , [ int(line_list[1]) , int(line_list[2]) ] ) )
        #print ( "height = %d , width = %d"%( int(line_list[1]) , int(line_list[2]) ))
    _sizes = _sizes.reshape(( _sizes.size/2 , 2 ))

    area_ranges = [ [96,3773], [3773,11368], [11368,28302], [28302,64630],[64630,249001]]
    #area_ranges = [ [96,3773], [3773,11368] ]
    evaluate_list = [ {} for _ in xrange(len( area_ranges )) ]

    for_plt = [ [] for _ in xrange(len( area_ranges )) ]

    for irange , area_range in enumerate( area_ranges ):

        total_num = 0
        size_count = 0
        gt_overlaps = np.zeros(0)
        for iiter in xrange( test_iter ):
            output = net.forward()

            batch_label = net.blobs['label'].data.copy()
            batch_det   = net.blobs['detection_out'].data.copy()

            _overlaps , num_pos = batch_gt_overlap( batch_label[0][0] ,
                    batch_det[0][0] , test_batch_size, _sizes[ size_count:size_count +
                        test_batch_size ,: ]  , low_area = area_range[0] , high_area=area_range[1] )

            gt_overlaps = np.hstack( (gt_overlaps , _overlaps ) )
            total_num += num_pos

            size_count += test_batch_size
            if iiter % 20 == 0:
               print ( "iter = %d , irange# = %d"%(iiter, irange) )

        gt_overlaps = np.sort( gt_overlaps )
        step = 0.05
        thresholds = np.arange( 0.5 , 0.95 + 1e-5 , step )
        recalls = np.zeros_like( thresholds )

        for i , t in enumerate( thresholds ):
            recalls[i] = ( gt_overlaps >= t ).sum() / float( total_num )

        ar = recalls.mean()
        evaluate_list[irange]['ar'] = ar
        evaluate_list[irange]['recalls'] = recalls
        evaluate_list[irange]['thresholds'] = thresholds
        evaluate_list[irange]['gt_overlaps'] = gt_overlaps

        def recall_at(t):
            ind = np.where(thresholds > t-1e-5)[0][0]
            return recalls[ind]

        for_plt[irange] = [ar , recall_at(0.5) , recall_at(0.6), recall_at(0.7), \
                recall_at(0.8) ,recall_at(0.9) ]

        print( " ar = %f"%(ar))
        print( "Recall@0.5:{:.3f}".format(recall_at(0.5)))
        print( "Recall@0.6:{:.3f}".format(recall_at(0.6)))
        print( "Recall@0.7:{:.3f}".format(recall_at(0.7)))
        print( "Recall@0.8:{:.3f}".format(recall_at(0.8)))
        print( "Recall@0.9:{:.3f}".format(recall_at(0.9)))
        print( "total num of pos in fixed size is %d"%(total_num))

    # save the recall evaluation file
    with open( recall_save_file , 'wb' ) as f:
        cPickle.dump( evaluate_list, f, cPickle.HIGHEST_PROTOCOL)

    for_plt = np.array(for_plt)

    xticks_label = [ [] for _ in xrange(len( area_ranges )) ]

    for irange, area_range in enumerate( area_ranges ):
        x_start = int (np.sqrt( area_range[0] ) )
        x_end   = int (np.sqrt( area_range[1] ) )

        xs_str = str( x_start )
        xe_str = str( x_end )

        xticks_label[irange] = xs_str + '*' + xs_str + ' - ' + xe_str + '*' + xe_str

    #ar
    v1 = for_plt[:,0]
    #recall at 0.5
    v2 = for_plt[:,1]
    #recall at 0.6
    v3 = for_plt[:,2]
    #recall at 0.7
    v4 = for_plt[:,3]
    #recall at 0.8
    v5 = for_plt[:,4]
    #recall at 0.9
    v6 = for_plt[:,5]

    plt.figure(1)
    
    area_len = len( area_ranges )

    line1, = plt.plot( np.arange( area_len) , v1 , 'ko-', markerfacecolor='black', markersize=12)
    line2, = plt.plot( np.arange( area_len) , v2 , 'ko-', markerfacecolor='yellow', markersize=10)
    line3, = plt.plot( np.arange( area_len) , v3 , 'ko-', markerfacecolor='red',markersize=10)
    line4, = plt.plot( np.arange( area_len) , v4 , 'ko-', markerfacecolor='green',markersize=10)
    line5, = plt.plot( np.arange( area_len) , v5 , 'ko-', markerfacecolor='blue',markersize=10)
    line6, = plt.plot( np.arange( area_len) , v6 , 'ko-', markerfacecolor='magenta',markersize=10)
    plt.xlabel('size of proposals')
    plt.ylabel('Recall@overlap')
    plt.xticks( np.arange(len( area_ranges )) , xticks_label )
    plt.figlegend( (line1,line2,line3,line4,line5,line6) , ('Average Recall', \
            'Recall@0.5', 'Recall@0.6', 'Recall@0.7','Recall@0.8','Recall@0.9') , 'upper right' )
    plt.show()
