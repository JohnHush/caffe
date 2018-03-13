import os
import caffe
import tempfile
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import sys
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

caffe_root = os.environ['CAFFE_ROOT']

def cifar10_AlexNet( train_imdb , test_imdb , mean_file , if_learnable = True , lr_mult = 1):
    """
    accept only  IMDB  format dataset
    force user to input mean proto

    > 30     n.conv1 = L.Convolution( n.data , kernel_size = 5 , num_output = 32 , pad = 2, stride = 1, \
    """
    n = caffe.NetSpec()
    #n.name = 'CIFAR10_quick'

    n.data , n.label = L.Data( batch_size = 100 , backend = P.Data.LMDB , source = train_imdb , \
            transform_param = dict( mean_file = mean_file , scale=1. ) , ntop = 2 , include = {'phase':caffe.TRAIN})

    train_head = str(n.to_proto())

    n.data , n.label = L.Data( batch_size = 100 , backend = P.Data.LMDB , source = test_imdb , \
            transform_param = dict(mean_file = mean_file, scale=1. ) , ntop = 2 , include = {'phase':caffe.TEST} )


    n.conv1 = L.Convolution( n.data , kernel_size = 5 , num_output = 32 , pad = 2, stride = 1, \
            weight_filler = dict( type = 'gaussian' , std=0.0001 ) , bias_filler = dict( type = 'constant'), \
            param = [ dict( lr_mult = 1 ) , dict( lr_mult = 2 ) ])
    n.pool1 = L.Pooling( n.conv1 , kernel_size = 3 , stride = 2 , pool = P.Pooling.MAX )
    
    if if_learnable:
        n.lsig1 = L.LSigmoid( n.pool1  , filler = dict( type = 'constant' , value = 1. ) , param = [dict(lr_mult=lr_mult)] )
    else:
        n.lsig1 = L.Sigmoid( n.pool1 , in_place = False )

    n.conv2 = L.Convolution( n.lsig1 , kernel_size = 5 , num_output = 32 , pad = 2, stride = 1, \
            weight_filler = dict( type = 'gaussian' , std=0.01 ) , bias_filler = dict( type = 'constant'), \
            param = [ dict( lr_mult = 1 ) , dict( lr_mult = 2 ) ])
    #n.relu2 = L.ReLU( n.conv2 , in_place= True )
    if if_learnable:
        n.lsig2 = L.LSigmoid( n.conv2  , filler = dict( type = 'constant' , value = 1. ) , param = [dict(lr_mult=lr_mult)] )
    else:
        n.lsig2 = L.Sigmoid( n.conv2 , in_place = False )
    n.pool2 = L.Pooling( n.lsig2 , kernel_size = 3 , stride = 2 , pool = P.Pooling.AVE )


    n.conv3 = L.Convolution( n.pool2 , kernel_size = 5 , num_output = 64 , pad = 2, stride = 1, \
            weight_filler = dict( type = 'gaussian' , std=0.01 ) , bias_filler = dict( type = 'constant'), \
            param = [ dict( lr_mult = 1 ) , dict( lr_mult = 2 ) ])

    if if_learnable:
        n.lsig3 = L.LSigmoid( n.conv3  , filler = dict( type = 'constant' , value = 1. ) , param = [dict(lr_mult=lr_mult)] )
    else:
        n.lsig3 = L.Sigmoid( n.conv3 , in_place = False )
    n.pool3 = L.Pooling( n.lsig3 , kernel_size = 3 , stride = 2 , pool = P.Pooling.AVE )


    n.ip1 = L.InnerProduct( n.pool3 , num_output = 64 , \
            weight_filler = dict( type = 'gaussian' , std=0.1 ) , bias_filler = dict( type = 'constant'), \
            param = [ dict( lr_mult = 1 ) , dict( lr_mult = 2 ) ] )
    n.ip2 = L.InnerProduct( n.ip1 , num_output = 10 , \
            weight_filler = dict( type = 'gaussian' , std=0.1 ) , bias_filler = dict( type = 'constant'), \
            param = [ dict( lr_mult = 1 ) , dict( lr_mult = 2 ) ] )


    n.loss = L.SoftmaxWithLoss( n.ip2 , n.label )
    n.accuracy = L.Accuracy( n.ip2 , n.label , include={'phase':caffe.TEST} )

    return train_head + str( n.to_proto() )

def cifar10_Solver(train_net_path, test_net_path=None, base_lr=0.001 , max_iter = 10000 , solver_type = 'SGD'):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 500  # Test after every 500 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    #s.iter_size = 1
    s.device_id = 0
    
    s.max_iter = max_iter     # # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = solver_type

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.

    s.lr_policy = 'fixed'
#    s.gamma = 0.1
#    s.stepsize = 20000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.

    s.momentum = 0.9
    s.weight_decay = 0.004
    s.random_seed = 0xCAFFE

    # Display the current training loss and accuracy every 1000 iterations.

    s.display = 100

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 4000
    s.snapshot_prefix = os.path.join( caffe_root , 'examples/cifar10/cifar10_quick' )
    s.snapshot_format = caffe_pb2.SolverParameter.HDF5
    
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    
    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        return f.name

def run_solvers(niter, solvers, disp_interval=10):
    """
    Run solvers for niter iterations,
    returning the loss and accuracy recorded each iteration.
    `solvers` is a list of (name, solver) tuples.
    """
    blobs = ('loss', 'accuracy')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(100)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)     
    # Save the learned weights from both nets.
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights

if __name__ == '__main__':

    caffe.set_mode_gpu()
    caffe.set_device(3)
    #cifar10_solver.solve()

    train_imdb = os.path.join( caffe_root , 'examples/cifar10/cifar10_train_lmdb' )
    test_imdb = os.path.join( caffe_root , 'examples/cifar10/cifar10_test_lmdb' )
    mean_imdb = os.path.join( caffe_root , 'examples/cifar10/mean.binaryproto' )

    net_str = cifar10_AlexNet( train_imdb , test_imdb , mean_imdb , if_learnable = False)
    learnable_net_str = cifar10_AlexNet( train_imdb , test_imdb , mean_imdb , if_learnable = True , lr_mult = 2.)
    with open( "net.prototxt" , 'w') as f:
        f.write( net_str )
    with open( "learnable_net.prototxt" , 'w') as f:
        f.write( learnable_net_str )
    #with tempfile.NamedTemporaryFile(delete=False) as f:
    #    f.write( cifar10_AlexNet( train_imdb , test_imdb , mean_imdb ) )

    s = caffe_pb2.SolverParameter()

    #s.net = f.name
    s.net = 'net.prototxt'

    s.test_iter.append(100)
    s.test_interval = 500

    s.device_id = 0
    s.max_iter = 5000     # # of times to update the net (training iterations)
    s.base_lr = 0.01

    s.lr_policy = 'fixed'
    s.momentum = 0.9
    s.weight_decay = 0.
    s.random_seed = 0xCAFFE

    s.display = 100

    s.snapshot = 40000
    s.snapshot_prefix = os.path.join( caffe_root , 'examples/cifar10/cifar10_quick' )
    s.snapshot_format = caffe_pb2.SolverParameter.HDF5
    
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))

    s.net = 'learnable_net.prototxt'
    with tempfile.NamedTemporaryFile(delete=False) as f2:
        f2.write(str(s))

    solver = None
    solver2 = None
    solver = caffe.SGDSolver( f2.name )
    #solver.solve()

    niter = 1000
    test_interval = 100
    test_iter = 100

    save_interval = 40
    show_width = 40
    save_number = int(np.ceil( niter/save_interval ))

    train_loss = np.zeros( niter )
    kernel1_data = [ [] for _ in xrange( niter ) ]

    test_acc = np.zeros( int(np.ceil(niter/test_interval) ) )
    lsig1 = np.zeros(( niter , 32 ))
    train_pool1 = np.zeros(( int(np.ceil( niter/save_interval )) , 100 , 32 , 16 , 16 ) )
    #train_pool1 = np.zeros(( int(np.ceil( niter/save_interval )) , 64 , 32 , 5 , 5 ) )

    learnable_train_loss = np.zeros( niter )
    learnable_test_acc = np.zeros( int(np.ceil(niter/test_interval) ) )

    # feature blobs
    FBs_name = [ 'lsig3' , 'conv3', 'lsig2', 'conv2', 'lsig1', 'conv1' ]
    FBs_ext = ['diff'] * len( FBs_name )
    FBs_ext[1] = 'diff'
    FBs_ext[3] = 'diff'
    FBs_ext[5] = 'diff'
    FBs = {}
    Params_name = [ 'conv3' , 'conv2', 'conv1']
    Params = {}

    eva_num = 5000
    show_interval = 5
    for blob_name in FBs_name:
        assert( solver.net.blobs.has_key(blob_name) )
        FBs[blob_name] = np.zeros( [save_number] + list( solver.net.blobs[blob_name].data.shape ) )
        print "blob %s has shape:"%(blob_name) + str( solver.net.blobs[blob_name].data.shape )

    for blob_name in Params_name:
        assert( solver.net.params.has_key(blob_name) )
        Params[blob_name] = np.zeros( [save_number] + list(solver.net.params[blob_name][0].data.shape) )
        print "param blob %s has shape:"%(blob_name) + str( solver.net.params[blob_name][0].data.shape )

    for it in xrange( niter ):
        solver.step(1)
        train_loss[it] = solver.net.blobs['loss'].data
        kernel1_data[it] = solver.net.params ['conv3'][0].data.flatten()

        if it % save_interval == 0:
            print it//save_interval
            for i, blob_name in enumerate( FBs_name ):
                if FBs_ext[i] == 'diff':
                    FBs[blob_name][ it // save_interval ] = solver.net.blobs[blob_name].diff
                else:
                    FBs[blob_name][ it // save_interval ] = solver.net.blobs[blob_name].data
            for blob_name in Params_name:
                Params[blob_name][ it // save_interval ] = solver.net.params[blob_name][0].diff
#            train_pool1[ it // save_interval ] = solver.net.blobs['pool1'].diff.copy()
        if it % test_interval == 0:
            acc = 0.
            for test_it in xrange( test_iter ):
                solver.test_nets[0].forward()
                acc += solver.test_nets[0].blobs['accuracy'].data
            test_acc[it // test_interval ] = acc / test_iter

    del( solver )

    if True:
        for k, v in FBs.items():
            v = v.transpose( 2, 0, 1, 3, 4 )
            v = v.reshape( v.shape[0] , v.shape[1] , -1 )
            v = v[:,:,np.random.randint( 0 , v.shape[2] , size = eva_num)]
            print "%s blob has the shape:"%(k) + str( v.shape )
            FBs[k] = v.tolist()

        for k, v in Params.items():
            v = v.transpose( 1, 0, 2, 3, 4)
            v = v.reshape( v.shape[0] , v.shape[1] , -1 )
            print "PARAMS %s blob has the shape:"%(k) + str( v.shape )
            Params[k] = v.tolist()

        fig, axes = plt.subplots( nrows = int(np.ceil( (len(FBs_name) +len(Params_name))/ 2. )), ncols = 2 )
        x = ( np.arange(save_number) * save_interval ).tolist()

        # drawing every blob's fig

        for index , item in enumerate( FBs_name ):
            fig_name = item + '_' + FBs_ext[index]
            row = index / 2
            col = index % 2
            for ifeature in range( 0 , len( FBs[item] ) , show_interval ):
                data = FBs[item][ifeature]
                axes[row, col].violinplot( data , x , points=100, showextrema=False, showmeans=True, widths=show_width )
            axes[row, col].set_title( fig_name , fontsize = 10 )

        for index , item in enumerate( Params_name ):
            fig_name = 'PARAM_' + item
            row = (index + len(FBs_name) ) / 2
            col = (index + len(FBs_name) ) % 2
            for ifeature in range( 0 , len( Params[item] ) , show_interval ):
                data = Params[item][ifeature]
                axes[row, col].violinplot( data , x , points=100, showextrema=False, showmeans=True, widths=show_width )
            axes[row, col].set_title( fig_name , fontsize = 10 )

        fig.suptitle("CIFAR10 blobs data statistics")
        fig.subplots_adjust(hspace=0.4)
        plt.show()

    kernel1_data = np.array( kernel1_data )
    
    kernel1_data = kernel1_data[:1000, :1000]

    X = np.arange( kernel1_data.shape[1] )
    Y = np.arange( kernel1_data.shape[0] )
    X, Y = np.meshgrid( X, Y )

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #surf = ax.plot_surface( X, Y, kernel1_data, cmap=cm.coolwarm, linewidth=0, antialiased=True )
    ax.plot_wireframe(X, Y, kernel1_data, rstride=0, cstride=20)
    plt.show()

    sys.exit()
    train_pool1 = train_pool1.transpose( 2, 0, 1, 3, 4)
    #train_pool1 = train_pool1.transpose( 1, 0, 2, 3, 4)
    train_pool1 = train_pool1.reshape( train_pool1.shape[0] , train_pool1.shape[1] , -1 )

    train_pool1 = train_pool1.tolist()

    _,ax1 = plt.subplots()
    for index in range( 0 , len(train_pool1) , 10 ):
        data = train_pool1[index]
        x = (np.arange( len(data) ) * save_interval).tolist()

        ax1.violinplot( data , x, points = 100 , showmeans=True , showextrema=False , showmedians=True , widths=20)

    plt.show()
    sys.exit()

    solver2 = caffe.SGDSolver( f2.name )

    for it in xrange( niter ):
        solver2.step(1)
        learnable_train_loss[it] = solver2.net.blobs['loss'].data
        lsig1[it] = solver2.net.params['lsig1'][0].data
        
        if it % test_interval == 0:
            acc = 0.
            for test_it in xrange( test_iter ):
                solver2.test_nets[0].forward()
                acc += solver2.test_nets[0].blobs['accuracy'].data
            learnable_test_acc[it // test_interval ] = acc / test_iter

    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot( np.arange( niter ) , train_loss , 'b' )
    ax1.plot( np.arange( niter ) , learnable_train_loss , 'r' )
    ax2.plot( np.arange( len(test_acc) ) * test_interval , test_acc , 'b' )
    ax2.plot( np.arange( len(test_acc) ) * test_interval , learnable_test_acc , 'r' )
    plt.savefig("los_acc.png")
    plt.show()

    lsig1_mean = lsig1.mean( axis = 1 )
    _, ax = plt.subplots()
    ax.plot( np.arange( niter ) , lsig1_mean , 'k')
    for index in xrange( 32 ):
        ax.plot( np.arange( niter ) , lsig1[:,index] , 'g')
    plt.savefig("alpha.png")
    plt.show()

    print lsig1[-1,...]

