import sys
import numpy as np
import lmdb
import caffe
import cv2

key = 0 
lmdb_path = "/Users/pitaloveu/WORKING_DATA/PA-100K/val_lmdb"
env = lmdb.open( lmdb_path )

count = 0

lmdb_env = lmdb.open( lmdb_path , readonly = True )
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

MLD= caffe.proto.caffe_pb2.MultiLabelDatum()

for key , value in lmdb_cursor:
    MLD.ParseFromString(value)

    data = caffe.io.datum_to_array(MLD.datum)
    print data.shape
    mt_label = MLD.mt_label
    image = data.transpose(1,2,0)
    cv2.imshow( 'cv2.png' , image )
    cv2.waitKey(300)
    count = count +1
    if count > 10:
        break

cv2.destroyAllWindows()
lmdb_env.close()

    
