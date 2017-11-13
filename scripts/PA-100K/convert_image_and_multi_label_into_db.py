import os
import scipy.io as sio
import numpy as np
import subprocess
import shutil

load_data = sio.loadmat( '/Users/pitaloveu/WORKING_DATA/PA-100K/annotation/annotation.mat' )
img_path_prefix = '/Users/pitaloveu/WORKING_DATA/PA-100K/release_data'

txt_names = ['train_images_name' , 'val_images_name' , 'test_images_name' , \
        'train_label' , 'val_label' , 'test_label' , 'attributes']

PARAM1 = "train_images_name.txt" , "train_label.txt" , "train_lmdb"
PARAM2 = "val_images_name.txt" , "val_label.txt" , "val_lmdb"
PARAM3 = "test_images_name.txt" , "test_label.txt" , "test.lmdb"

redo = True
caffe_root = '/Users/pitaloveu/WORKING_LIB/ssd_caffe'
Data_file  = 'train_images_name.txt'
Label_file = 'train_label.txt'
Db_file = 'train_lmdb'
backend = 'lmdb'
gray = False
encode_type = 'jpg'
encoded = False
min_dim = 0
max_dim = 0
resize_height = 227
resize_width = 227
shuffle = False

for name in txt_names:
    data = load_data[name]
    if name == 'train_images_name' or name == 'val_images_name' or name == 'test_images_name':
        if os.path.exists( os.getcwd() + '/' + name + '.txt'):
            os.remove( name + '.txt' )
        data_list = data.tolist()
        with open( name+'.txt' , 'aw' ) as f:
            for item in data_list:
                f.write( img_path_prefix + '/' + item[0][0] + '\n' )

    if name == 'attributes':
        if os.path.exists( os.getcwd() + '/' + name + '.txt'):
            os.remove( name + '.txt' )
#        data_list = data.tolist()
        with open( name+'.txt' , 'aw' ) as f:
            for item in data:
                f.write( item[0][0] + '\n' )

    if name == 'train_label' or name == 'val_label' or name == 'test_label':
        if os.path.exists( os.getcwd() + '/' + name + '.txt'):
            os.remove( name + '.txt' )
        data_list = data.tolist()
        with open( name+'.txt' , 'aw' ) as f:
            for item in data_list:
                for ele in item:
                    f.write( str(ele) + ' ' )
                f.write( '\n')


for Data_file, Label_file, Db_file in ( PARAM1 , PARAM2 , PARAM3 ):
    if redo and os.path.exists( Db_file ):
        shutil.rmtree( Db_file )
    cmd = "{}/build/tools/convert_multi_label_dataset" \
            " --min_dim={}" \
            " --max_dim={}" \
            " --resize_height={}" \
            " --resize_width={}" \
            " --backend={}" \
            " --shuffle={}" \
            " --encode_type={}" \
            " --encoded={}" \
            " --gray={}" \
            " {} {} {}" \
            .format( caffe_root, min_dim, max_dim, resize_height, resize_width, backend, shuffle,
                    encode_type, encoded, gray, Data_file, Label_file, Db_file )
                
    print cmd
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out = process.communicate()[0]
    print out
