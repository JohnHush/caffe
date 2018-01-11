from __future__ import print_function
import caffe
from caffe.model_libs import *
from google.protobuf import text_format
import numpy as np

import math, os, shutil, stat, sys

def bbox_overlaps( boxes , query_boxes ):
    """
    boxes: [N,4] ndarray of float
    query_boxes :[K, 4] ndarray of float

    return :
    overlap :[N, K] ndarray of overlaps 
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    overlaps = np.zeros( ( N , K) , dtype=np.float )
    for k in xrange(K):
        box_area = (query_boxes[k,2] - query_boxes[k,0]) * (query_boxes[k,3]
                -query_boxes[k,1])
        for n in xrange(N):
            iw = min( boxes[n,2] , query_boxes[k,2] ) - max( boxes[n,0] ,
                    query_boxes[k,0])
            if iw > 0:
                ih = min( boxes[n,3] , query_boxes[k,3] ) - max( boxes[n,1] ,
                        query_boxes[k,1] )
                if ih > 0:
                    ua = (boxes[n,2]-boxes[n,0]) * (boxes[n,3]-boxes[n,1]) + box_area - iw * ih

                    overlaps[n,k] = iw * ih /ua
    return overlaps

def batch_gt_overlap( gt_boxes , det_boxes , num_batch , use_diff = False ):
    """
    gt_boxes assumes to be 2d numpy array,
    [num_gt][ batch_id , group_label , instance_id , xmin , ymin , xmax , ymax ,
    diff ]
    det_boxes assumes to be 2d numpy array has the following form:
    [num_det][ batch_id , label , score , xmin , ymin , xmax , ymax ]
    """
    num_pos = 0
    gt_overlaps = np.zeros(0)
    gt_boxes_per_image = np.zeros(0)
    dt_boxes_per_image = np.zeros(0)

    for i in xrange( num_batch ):
        #collect gt boxes from gt_boxes numpy ndarray
        for j in xrange( gt_boxes.shape[0] ):
            box = gt_boxes[j]
            if int(box[0]) == i:
                if not use_diff and int(box[7]) == 1:
                    continue
                gt_boxes_per_image = np.hstack(( gt_boxes_per_image , [ box[3] ,
                    box[4] , box[5] , box[6] ] ))
        num_gt = gt_boxes_per_image.size / 4
        gt_boxes_per_image = gt_boxes_per_image.reshape( ( num_gt , 4) )

        num_pos += num_gt

        #collect all th det boxes 
        for j in xrange( det_boxes.shape[0] ):
            box = det_boxes[j]
            if int(box[0]) == i:
                dt_boxes_per_image = np.hstack(( dt_boxes_per_image , [ box[3] ,
                    box[4] , box[5] , box[6] ] ))
        num_dt = dt_boxes_per_image.size /4
        dt_boxes_per_image = dt_boxes_per_image.reshape(( num_dt , 4 ))

        overlaps = bbox_overlaps( dt_boxes_per_image , gt_boxes_per_image )

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

# add additional layers to the base net
def AddExtraLayers(net, use_batchnorm=True, lr_mult=1):
    use_relu = True

    # 19 x 19
    from_layer = net.keys()[-1]

    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
        lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2,
        lr_mult=lr_mult)

    # 5 x 5
    from_layer = out_layer
    out_layer = "conv7_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv7_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
      lr_mult=lr_mult)

    # 3 x 3
    from_layer = out_layer
    out_layer = "conv8_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv8_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    # 1 x 1
    from_layer = out_layer
    out_layer = "conv9_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv9_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    return net


if __name__ == "__main__":

    caffe_root = os.environ['CAFFE_ROOT']
    job_name = "jh_ssd_test"
    model_name = "VGG_VOC0712_{}".format(job_name)
    weights = os.path.join( caffe_root , "models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel" )
    save_dir = os.path.join( caffe_root , "models/VGGNet/VOC0712/{}".format(job_name) )
    job_dir = os.path.join( caffe_root , "jobs/VGGNet/VOC0712/{}".format(job_name) )
    test_net_file = "{}/test.prototxt".format(save_dir)
    test_data = os.path.join( caffe_root, "examples/VOC0712/VOC0712_test_lmdb" )
    test_transform_param = {
        'mean_value': [104, 117, 123],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': 300,
                'width': 300,
                'interp_mode': [P.Resize.LINEAR],
                },
        }
    label_map_file = os.path.join( caffe_root, "data/VOC0712/labelmap_voc.prototxt" )
    name_size_file = os.path.join( caffe_root, "data/VOC0712/test_name_size.txt" )
    test_batch_size = 8
    num_test_image = 4952
    use_batchnorm = False
    lr_mult = 1
    mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
    code_type = P.PriorBox.CENTER_SIZE
    background_label_id=0
    num_classes = 21
    min_dim = 300
    min_ratio = 20
    max_ratio = 90
    step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in xrange(min_ratio, max_ratio + 1, step):
        min_sizes.append(min_dim * ratio / 100.)
        max_sizes.append(min_dim * (ratio + step) / 100.)

    min_sizes = [min_dim * 10 / 100.] + min_sizes
    max_sizes = [min_dim * 20 / 100.] + max_sizes
    steps = [8, 16, 32, 64, 100, 300]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    normalizations = [20, -1, -1, -1, -1, -1]
    prior_variance = [0.1, 0.1, 0.2, 0.2]
    flip = True
    clip = False
    share_location = True

    det_out_param = {
    'num_classes': num_classes,
    'share_location': share_location,
    'background_label_id': background_label_id,
    'nms_param': {'nms_threshold': 0.45, 'top_k': 200},
    'keep_top_k': 200,
    'confidence_threshold': 0.0001,
    'code_type': code_type,
    }

    det_eval_param = {
    'num_classes': num_classes,
    'background_label_id': background_label_id,
    'overlap_threshold': 0.5,
    'evaluate_difficult_gt': False,
    'name_size_file': name_size_file,
    }

    check_if_exist(test_data)
    check_if_exist(label_map_file)
    check_if_exist(weights)
    make_if_not_exist(save_dir)
    make_if_not_exist(job_dir)

    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
        train=False, output_label=True, label_map_file=label_map_file,
        transform_param=test_transform_param)

    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True, dropout=False)
    AddExtraLayers(net, use_batchnorm, lr_mult=lr_mult)

    mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
                use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
                aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
                num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
                prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)

    conf_name = "mbox_conf"
    reshape_name = "{}_reshape".format(conf_name)
    net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
    softmax_name = "{}_softmax".format(conf_name)
    net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
    flatten_name = "{}_flatten".format(conf_name)
    net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
    mbox_layers[1] = net[flatten_name]

    net.detection_out = L.DetectionOutput(*mbox_layers,
        detection_output_param=det_out_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
        detection_evaluate_param=det_eval_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))

    with open(test_net_file, 'w') as f:
        print('name: "{}_test"'.format(model_name), file=f)
        print(net.to_proto(), file=f)
    shutil.copy(test_net_file, job_dir)

    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = caffe.Net( test_net_file , weights , caffe.TEST )

    test_iter = num_test_image / test_batch_size
    #test_iter = 50
    num_gt_cls = np.zeros( num_classes -1 ).astype(int)
    num_tp_det_cls = [ [] for _ in xrange( num_classes - 1 ) ]
    num_fp_det_cls = [ [] for _ in xrange( num_classes - 1 ) ]

    for iiter in xrange( test_iter ):
        output = net.forward()

        if iiter == 2:
            label_batch = net.blobs['label'].data.copy()
            do = net.blobs['detection_out'].data.copy()

            for ioutput in xrange( do.shape[2] ):
                dd = do[0,0,ioutput,:]
                print( "item = %d, label= %d , score = %6.3f , xmin=%6.3f, ymin=%6.3f, xmax=%6.3f , ymax = %6.3f"%( int(dd[0]) , int(dd[1]) ,\
                            dd[2] , dd[3] , dd[4] , dd[5]\
                    , dd[6] ))
            print ( do.shape )
#            for k, v in net.blobs.items():
#                print( k , v.data.shape )
            print ( label_batch.shape )
            print ( label_batch )
            sys.exit()
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
