import sys
import numpy as np
import lmdb
import caffe
import cv2

blob1_path = "/Users/pitaloveu/testtest/PROTO1"
blob2_path = "/Users/pitaloveu/testtest/PROTO2"

BP1= caffe.proto.caffe_pb2.BlobProto()
BP2= caffe.proto.caffe_pb2.BlobProto()

BP1.ParseFromString( open(blob1_path , 'rb' ).read() )
BP2.ParseFromString( open(blob2_path , 'rb' ).read() )

BP1_array = caffe.io.blobproto_to_array(BP1)
BP2_array = caffe.io.blobproto_to_array(BP2)

for id in range( BP1_array.shape[0] ):
    image = BP1_array[id].transpose( 1 , 2 , 0 )
    cv2.imshow( 'cv2.png' , image )
    cv2.waitKey(0)

    txt = ""
    for item in BP2_array[id]:
        txt = txt + str(int(item)) + " "
    print txt

cv2.destroyAllWindows()

