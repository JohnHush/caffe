import numpy as np
import cPickle
import os, sys
import matplotlib.pyplot as plt

def recall_at( t , thresholds , recalls ):
    ind = np.where(thresholds > t-1e-5)[0][0]
    return recalls[ind]

if __name__ == '__main__':

    #load the list of dict

    recall_file = '/home/jh/working_data/EXP_output/pva_recall_different_keepTopK.pkl'
    recall_file2 = '/home/jh/working_data/EXP_output/ssd_recall_different_keepTopK.pkl'
    with open( recall_file , 'rb' ) as fid:
        pvanet_diff_number = cPickle.load(fid)
    with open( recall_file2 , 'rb' ) as fid:
        ssd_diff_number = cPickle.load(fid)

    proposal_num_list = [ 10, 15 , 30 , 50 , 100 , 200 , 300, 500 ]

    Xnumber = len( pvanet_diff_number )
    Ynumber = 6
    pvanet_diff_number_4show = [ [ [] for _ in xrange(Xnumber) ] for _ in xrange(Ynumber) ]
    ssd_diff_number_4show = [ [ [] for _ in xrange(Xnumber) ] for _ in xrange(Ynumber) ]

    for idict , item  in enumerate( pvanet_diff_number ):
        ar          = item['ar']
        recalls     = item['recalls']
        thresholds  = item['thresholds']

        pvanet_diff_number_4show[0][ idict ] = ar
        pvanet_diff_number_4show[1][ idict ] = recall_at( 0.5 , thresholds , recalls )
        pvanet_diff_number_4show[2][ idict ] = recall_at( 0.6 , thresholds , recalls )
        pvanet_diff_number_4show[3][ idict ] = recall_at( 0.7 , thresholds , recalls )
        pvanet_diff_number_4show[4][ idict ] = recall_at( 0.8 , thresholds , recalls )
        pvanet_diff_number_4show[5][ idict ] = recall_at( 0.9 , thresholds , recalls )

    for idict , item  in enumerate( ssd_diff_number ):
        ar          = item['ar']
        recalls     = item['recalls']
        thresholds  = item['thresholds']

        ssd_diff_number_4show[0][ idict ] = ar
        ssd_diff_number_4show[1][ idict ] = recall_at( 0.5 , thresholds , recalls )
        ssd_diff_number_4show[2][ idict ] = recall_at( 0.6 , thresholds , recalls )
        ssd_diff_number_4show[3][ idict ] = recall_at( 0.7 , thresholds , recalls )
        ssd_diff_number_4show[4][ idict ] = recall_at( 0.8 , thresholds , recalls )
        ssd_diff_number_4show[5][ idict ] = recall_at( 0.9 , thresholds , recalls )

    plt.figure(1)

    pp = pvanet_diff_number_4show
    pp2 = ssd_diff_number_4show

    line1, = plt.plot( proposal_num_list , pp[0] , 'ko-', markersize=12)
    line2, = plt.plot( proposal_num_list , pp[1] , 'ys-', markersize=10)
    line3, = plt.plot( proposal_num_list , pp[2] , 'r*-', markersize=10)
    line4, = plt.plot( proposal_num_list , pp[3] , 'gD-', markersize=10)
    line5, = plt.plot( proposal_num_list , pp[4] , 'bp-', markersize=10)
    line6, = plt.plot( proposal_num_list , pp[5] , 'mh-', markersize=10)

    line7, = plt.plot( proposal_num_list , pp2[0] , 'ko--', markersize=12)
    line8, = plt.plot( proposal_num_list , pp2[1] , 'ys--', markersize=10)
    line9, = plt.plot( proposal_num_list , pp2[2] , 'r*--', markersize=10)
    line10, = plt.plot( proposal_num_list , pp2[3] , 'gD--', markersize=10)
    line11, = plt.plot( proposal_num_list , pp2[4] , 'bp--', markersize=10)
    line12, = plt.plot( proposal_num_list , pp2[5] , 'mh--', markersize=10)

    plt.xlabel('# of proposals')
    plt.ylabel('Recall@overlap')
    fl = '300'
    plt.figlegend( (line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11,line12) , \
            ('PVANet_Average Recall', 'PVANet_Recall@0.5', 'PVANet_Recall@0.6', 'PVANet_Recall@0.7', \
            'PVANet_Recall@0.8','PVANet_Recall@0.9' , 'SSD'+fl+'_Average Recall', 'SSD'+fl+'_Recall@0.5', \
            'SSD'+fl+'_Recall@0.6', 'SSD'+fl+'_Recall@0.7', 'SSD'+fl+'_Recall@0.8','SSD'+fl+'_Recall@0.9' ) , 'upper right' )
    plt.show()

    recall_file = '/home/jh/working_data/EXP_output/ssd_recall_different_keepTopK.pkl'
    recall_file2 = '/home/jh/working_data/EXP_output/ssd_recall_different_keepTopK_SSD512.pkl'
    with open( recall_file , 'rb' ) as fid:
        pvanet_diff_number = cPickle.load(fid)
    with open( recall_file2 , 'rb' ) as fid:
        ssd_diff_number = cPickle.load(fid)

    proposal_num_list = [ 10, 15 , 30 , 50 , 100 , 200 , 300, 500 ]

    Xnumber = len( pvanet_diff_number )
    Ynumber = 6
    pvanet_diff_number_4show = [ [ [] for _ in xrange(Xnumber) ] for _ in xrange(Ynumber) ]
    ssd_diff_number_4show = [ [ [] for _ in xrange(Xnumber) ] for _ in xrange(Ynumber) ]

    for idict , item  in enumerate( pvanet_diff_number ):
        ar          = item['ar']
        recalls     = item['recalls']
        thresholds  = item['thresholds']

        pvanet_diff_number_4show[0][ idict ] = ar
        pvanet_diff_number_4show[1][ idict ] = recall_at( 0.5 , thresholds , recalls )
        pvanet_diff_number_4show[2][ idict ] = recall_at( 0.6 , thresholds , recalls )
        pvanet_diff_number_4show[3][ idict ] = recall_at( 0.7 , thresholds , recalls )
        pvanet_diff_number_4show[4][ idict ] = recall_at( 0.8 , thresholds , recalls )
        pvanet_diff_number_4show[5][ idict ] = recall_at( 0.9 , thresholds , recalls )

    for idict , item  in enumerate( ssd_diff_number ):
        ar          = item['ar']
        recalls     = item['recalls']
        thresholds  = item['thresholds']

        ssd_diff_number_4show[0][ idict ] = ar
        ssd_diff_number_4show[1][ idict ] = recall_at( 0.5 , thresholds , recalls )
        ssd_diff_number_4show[2][ idict ] = recall_at( 0.6 , thresholds , recalls )
        ssd_diff_number_4show[3][ idict ] = recall_at( 0.7 , thresholds , recalls )
        ssd_diff_number_4show[4][ idict ] = recall_at( 0.8 , thresholds , recalls )
        ssd_diff_number_4show[5][ idict ] = recall_at( 0.9 , thresholds , recalls )

    plt.figure(1)

    pp = pvanet_diff_number_4show
    pp2 = ssd_diff_number_4show

    line1, = plt.plot( proposal_num_list , pp[0] , 'ko-', markersize=12)
    line2, = plt.plot( proposal_num_list , pp[1] , 'ys-', markersize=10)
    line3, = plt.plot( proposal_num_list , pp[2] , 'r*-', markersize=10)
    line4, = plt.plot( proposal_num_list , pp[3] , 'gD-', markersize=10)
    line5, = plt.plot( proposal_num_list , pp[4] , 'bp-', markersize=10)
    line6, = plt.plot( proposal_num_list , pp[5] , 'mh-', markersize=10)

    line7, = plt.plot( proposal_num_list , pp2[0] , 'ko--', markersize=12)
    line8, = plt.plot( proposal_num_list , pp2[1] , 'ys--', markersize=10)
    line9, = plt.plot( proposal_num_list , pp2[2] , 'r*--', markersize=10)
    line10, = plt.plot( proposal_num_list , pp2[3] , 'gD--', markersize=10)
    line11, = plt.plot( proposal_num_list , pp2[4] , 'bp--', markersize=10)
    line12, = plt.plot( proposal_num_list , pp2[5] , 'mh--', markersize=10)

    plt.xlabel('# of proposals')
    plt.ylabel('Recall@overlap')
    fl = '512'
    plt.figlegend( (line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11,line12) , \
            ('SSD300_Average Recall', 'SSD300_Recall@0.5', 'SSD300_Recall@0.6', 'SSD300_Recall@0.7', \
            'SSD300_Recall@0.8','SSD300_Recall@0.9' , 'SSD'+fl+'_Average Recall', 'SSD'+fl+'_Recall@0.5', \
            'SSD'+fl+'_Recall@0.6', 'SSD'+fl+'_Recall@0.7', 'SSD'+fl+'_Recall@0.8','SSD'+fl+'_Recall@0.9' ) , 'upper right' )
    plt.show()

    recall_file2 = '/home/jh/working_data/EXP_output/ssd_recall_different_gt_size.pkl'
    recall_file = '/home/jh/working_data/EXP_output/pva_recall_different_gt_size.pkl'

    with open( recall_file , 'rb' ) as fid:
        pvanet_diff_number = cPickle.load(fid)
    with open( recall_file2 , 'rb' ) as fid:
        ssd_diff_number = cPickle.load(fid)

    area_ranges = [ [96,3773], [3773,11368], [11368,28302], [28302,64630],[64630,249001]]
    xticks_label = [ [] for _ in xrange(len( area_ranges )) ]
    for irange, area_range in enumerate( area_ranges ):
        xticks_label[irange] = r"$ {}^2 - {}^2 $".format( int(np.sqrt(area_range[0])) , int(np.sqrt(area_range[1])) )

    Xnumber = len( pvanet_diff_number )
    Ynumber = 6
    pvanet_diff_number_4show = [ [ [] for _ in xrange(Xnumber) ] for _ in xrange(Ynumber) ]
    ssd_diff_number_4show = [ [ [] for _ in xrange(Xnumber) ] for _ in xrange(Ynumber) ]

    for idict , item  in enumerate( pvanet_diff_number ):
        ar          = item['ar']
        recalls     = item['recalls']
        thresholds  = item['thresholds']

        pvanet_diff_number_4show[0][ idict ] = ar
        pvanet_diff_number_4show[1][ idict ] = recall_at( 0.5 , thresholds , recalls )
        pvanet_diff_number_4show[2][ idict ] = recall_at( 0.6 , thresholds , recalls )
        pvanet_diff_number_4show[3][ idict ] = recall_at( 0.7 , thresholds , recalls )
        pvanet_diff_number_4show[4][ idict ] = recall_at( 0.8 , thresholds , recalls )
        pvanet_diff_number_4show[5][ idict ] = recall_at( 0.9 , thresholds , recalls )

    for idict , item  in enumerate( ssd_diff_number ):
        ar          = item['ar']
        recalls     = item['recalls']
        thresholds  = item['thresholds']

        ssd_diff_number_4show[0][ idict ] = ar
        ssd_diff_number_4show[1][ idict ] = recall_at( 0.5 , thresholds , recalls )
        ssd_diff_number_4show[2][ idict ] = recall_at( 0.6 , thresholds , recalls )
        ssd_diff_number_4show[3][ idict ] = recall_at( 0.7 , thresholds , recalls )
        ssd_diff_number_4show[4][ idict ] = recall_at( 0.8 , thresholds , recalls )
        ssd_diff_number_4show[5][ idict ] = recall_at( 0.9 , thresholds , recalls )

    plt.figure(1)

    pp = pvanet_diff_number_4show
    pp2 = ssd_diff_number_4show

    xxx = np.arange( len(area_ranges) )
    line1, = plt.plot( xxx , pp[0] , 'ko-', markersize=12)
    line2, = plt.plot( xxx , pp[1] , 'ys-', markersize=10)
    line3, = plt.plot( xxx , pp[2] , 'r*-', markersize=10)
    line4, = plt.plot( xxx , pp[3] , 'gD-', markersize=10)
    line5, = plt.plot( xxx , pp[4] , 'bp-', markersize=10)
    line6, = plt.plot( xxx , pp[5] , 'mh-', markersize=10)

    line7, = plt.plot( xxx , pp2[0] , 'ko--', markersize=12)
    line8, = plt.plot( xxx , pp2[1] , 'ys--', markersize=10)
    line9, = plt.plot( xxx , pp2[2] , 'r*--', markersize=10)
    line10, = plt.plot( xxx , pp2[3] , 'gD--', markersize=10)
    line11, = plt.plot( xxx , pp2[4] , 'bp--', markersize=10)
    line12, = plt.plot( xxx , pp2[5] , 'mh--', markersize=10)

    plt.xlabel('Ground Truth BBox Size')
    plt.ylabel('Recall@overlap')
    plt.xticks( np.arange(len( area_ranges )) , xticks_label )
    fl='300'
    plt.figlegend( (line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11,line12) , \
            ('PVANet_Average Recall', 'PVANet_Recall@0.5', 'PVANet_Recall@0.6', 'PVANet_Recall@0.7', \
            'PVANet_Recall@0.8','PVANet_Recall@0.9' , 'SSD'+fl+'_Average Recall', 'SSD'+fl+'_Recall@0.5', \
            'SSD'+fl+'_Recall@0.6', 'SSD'+fl+'_Recall@0.7', 'SSD'+fl+'_Recall@0.8','SSD'+fl+'_Recall@0.9' ) , 'upper right' )
    plt.show()

    recall_file = '/home/jh/working_data/EXP_output/ssd_recall_different_gt_size.pkl'
    recall_file2 = '/home/jh/working_data/EXP_output/ssd_recall_different_gt_size_SSD512.pkl'

    with open( recall_file , 'rb' ) as fid:
        pvanet_diff_number = cPickle.load(fid)
    with open( recall_file2 , 'rb' ) as fid:
        ssd_diff_number = cPickle.load(fid)

    area_ranges = [ [96,3773], [3773,11368], [11368,28302], [28302,64630],[64630,249001]]
    xticks_label = [ [] for _ in xrange(len( area_ranges )) ]
    for irange, area_range in enumerate( area_ranges ):
        xticks_label[irange] = r"$ {}^2 - {}^2 $".format( int(np.sqrt(area_range[0])) , int(np.sqrt(area_range[1])) )

    Xnumber = len( pvanet_diff_number )
    Ynumber = 6
    pvanet_diff_number_4show = [ [ [] for _ in xrange(Xnumber) ] for _ in xrange(Ynumber) ]
    ssd_diff_number_4show = [ [ [] for _ in xrange(Xnumber) ] for _ in xrange(Ynumber) ]

    for idict , item  in enumerate( pvanet_diff_number ):
        ar          = item['ar']
        recalls     = item['recalls']
        thresholds  = item['thresholds']

        pvanet_diff_number_4show[0][ idict ] = ar
        pvanet_diff_number_4show[1][ idict ] = recall_at( 0.5 , thresholds , recalls )
        pvanet_diff_number_4show[2][ idict ] = recall_at( 0.6 , thresholds , recalls )
        pvanet_diff_number_4show[3][ idict ] = recall_at( 0.7 , thresholds , recalls )
        pvanet_diff_number_4show[4][ idict ] = recall_at( 0.8 , thresholds , recalls )
        pvanet_diff_number_4show[5][ idict ] = recall_at( 0.9 , thresholds , recalls )

    for idict , item  in enumerate( ssd_diff_number ):
        ar          = item['ar']
        recalls     = item['recalls']
        thresholds  = item['thresholds']

        ssd_diff_number_4show[0][ idict ] = ar
        ssd_diff_number_4show[1][ idict ] = recall_at( 0.5 , thresholds , recalls )
        ssd_diff_number_4show[2][ idict ] = recall_at( 0.6 , thresholds , recalls )
        ssd_diff_number_4show[3][ idict ] = recall_at( 0.7 , thresholds , recalls )
        ssd_diff_number_4show[4][ idict ] = recall_at( 0.8 , thresholds , recalls )
        ssd_diff_number_4show[5][ idict ] = recall_at( 0.9 , thresholds , recalls )

    plt.figure(1)

    pp = pvanet_diff_number_4show
    pp2 = ssd_diff_number_4show

    xxx = np.arange( len(area_ranges) )
    line1, = plt.plot( xxx , pp[0] , 'ko-', markersize=12)
    line2, = plt.plot( xxx , pp[1] , 'ys-', markersize=10)
    line3, = plt.plot( xxx , pp[2] , 'r*-', markersize=10)
    line4, = plt.plot( xxx , pp[3] , 'gD-', markersize=10)
    line5, = plt.plot( xxx , pp[4] , 'bp-', markersize=10)
    line6, = plt.plot( xxx , pp[5] , 'mh-', markersize=10)

    line7, = plt.plot( xxx , pp2[0] , 'ko--', markersize=12)
    line8, = plt.plot( xxx , pp2[1] , 'ys--', markersize=10)
    line9, = plt.plot( xxx , pp2[2] , 'r*--', markersize=10)
    line10, = plt.plot( xxx , pp2[3] , 'gD--', markersize=10)
    line11, = plt.plot( xxx , pp2[4] , 'bp--', markersize=10)
    line12, = plt.plot( xxx , pp2[5] , 'mh--', markersize=10)

    plt.xlabel('Ground Truth BBox Size')
    plt.ylabel('Recall@overlap')
    plt.xticks( np.arange(len( area_ranges )) , xticks_label )
    fl='512'
    plt.figlegend( (line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11,line12) , \
            ('SSD300_Average Recall', 'SSD300_Recall@0.5', 'SSD300_Recall@0.6', 'SSD300_Recall@0.7', \
            'SSD300_Recall@0.8','SSD300_Recall@0.9' , 'SSD'+fl+'_Average Recall', 'SSD'+fl+'_Recall@0.5', \
            'SSD'+fl+'_Recall@0.6', 'SSD'+fl+'_Recall@0.7', 'SSD'+fl+'_Recall@0.8','SSD'+fl+'_Recall@0.9' ) , 'upper right' )
    plt.show()
