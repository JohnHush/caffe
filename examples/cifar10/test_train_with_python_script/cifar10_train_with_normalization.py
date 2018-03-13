import os
import caffe
import tempfile
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import sys
import numpy as np

caffe_root = os.environ['CAFFE_ROOT']
caffe.set_mode_gpu()
caffe.set_device(1)

def cifar10_AlexNet( imdb , batch_size , mean_file , train = True , if_normalize = False ):
    """
    accept only  IMDB  format dataset
    force user to input mean proto

    ^^ happy hacking.shit
    """
    n = caffe.NetSpec()

    n.data , n.label = L.Data( batch_size = batch_size , backend = P.Data.LMDB , source = imdb , \
           transform_param = dict( mean_file = mean_file ) , ntop = 2 )


    n.conv1 = L.Convolution( n.data , kernel_size = 5 , num_output = 32 , pad = 2, stride = 1, \
            weight_filler = dict( type = 'gaussian' , std=0.0001 ) , bias_filler = dict( type = 'constant'), \
            param = [ dict( lr_mult = 1 ) , dict( lr_mult = 2 ) ])
    n.pool1 = L.Pooling( n.conv1 , kernel_size = 3 , stride = 2 , pool = P.Pooling.MAX )
    n.relu1 = L.ReLU( n.pool1 , in_place= True )


    n.conv2 = L.Convolution( n.pool1 , kernel_size = 5 , num_output = 32 , pad = 2, stride = 1, \
            weight_filler = dict( type = 'gaussian' , std=0.01 ) , bias_filler = dict( type = 'constant'), \
            param = [ dict( lr_mult = 1 ) , dict( lr_mult = 2 ) ])
    n.relu2 = L.ReLU( n.conv2 , in_place= True )
    n.pool2 = L.Pooling( n.conv2 , kernel_size = 3 , stride = 2 , pool = P.Pooling.AVE )


    n.conv3 = L.Convolution( n.pool2 , kernel_size = 5 , num_output = 64 , pad = 2, stride = 1, \
            weight_filler = dict( type = 'gaussian' , std=0.01 ) , bias_filler = dict( type = 'constant'), \
            param = [ dict( lr_mult = 1 ) , dict( lr_mult = 2 ) ])
    n.relu3 = L.ReLU( n.conv3 , in_place= True )
    n.pool3 = L.Pooling( n.conv3 , kernel_size = 3 , stride = 2 , pool = P.Pooling.AVE )


    n.ip1 = L.InnerProduct( n.pool3 , num_output = 64 , \
            weight_filler = dict( type = 'gaussian' , std=0.01 ) , bias_filler = dict( type = 'constant'), \
            param = [ dict( lr_mult = 1 ) , dict( lr_mult = 2 ) ] )
    n.ip2 = L.InnerProduct( n.ip1 , num_output = 10 , \
            weight_filler = dict( type = 'gaussian' , std=0.1 ) , bias_filler = dict( type = 'constant'), \
            param = [ dict( lr_mult = 1 ) , dict( lr_mult = 2 ) ] )

    if if_normalize and train:
        n.normalized = L.Normalize( n.ip2 )

    if if_normalize and train:
        n.loss = L.SoftmaxWithLoss( n.normalized , n.label )
    else:
        n.loss = L.SoftmaxWithLoss( n.ip2 , n.label )

    n.accuracy = L.Accuracy( n.ip2 , n.label )

    return n.to_proto()

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
    s.iter_size = 1
    s.device_id = 3
    
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

    train_net = 'cifar_train_using_python_script.prototxt'
    test_net  = 'cifar_test_using_python_script.prototxt'

    with open( train_net , 'w' ) as f:
        f.write( str( cifar10_AlexNet( os.path.join( caffe_root , 'examples/cifar10/cifar10_train_lmdb'), 512 , \
                os.path.join( caffe_root , 'examples/cifar10/mean.binaryproto' ) , train = True, if_normalize= True ) ) )

    with open( test_net , 'w' ) as f:
        f.write( str( cifar10_AlexNet( os.path.join( caffe_root , 'examples/cifar10/cifar10_test_lmdb'), 100 , \
                os.path.join( caffe_root , 'examples/cifar10/mean.binaryproto' ) , train = False , if_normalize = True ) ) )

    #cifar10_solver_filename = cifar10_Solver( train_net , test_net , base_lr=0.001 , max_iter = 4000 , solver_type = 'SGD' )
    cifar10_solver_filename = cifar10_Solver( train_net , test_net , base_lr=0.001 , max_iter = 4000 , solver_type = 'SGD' )

    niter = 4000
    cifar10_solver = caffe.get_solver( cifar10_solver_filename )

    print 'Running solvers for %d iterations...' % niter
#    solvers = [ ('scratch', cifar10_solver ) ,]
#    loss, acc, weights = run_solvers(niter, solvers)
#    print 'Done.'

    blobs = ( 'loss' , 'accuracy' )
    loss , acc = np.zeros(niter) , np.zeros(niter)
    loss2 , acc2 = np.zeros(niter) , np.zeros(niter)

    for it in xrange( niter ):
        cifar10_solver.step(1)
        loss[it] , acc[it] = ( cifar10_solver.net.blobs[b].data.copy() for b in blobs )
        #loss2[it] , acc2[it] = ( cifar10_solver.test_nets[0].blobs[b].data.copy() for b in blobs )

        if it % 10 == 0 or it +1 == niter:

            for iit in xrange( 100 ):
                output = cifar10_solver.test_nets[0].forward()
                loss2[it] += output['loss']
                acc2[it]  += output['accuracy']
            loss2[it] /= 100.
            acc2[it] /= 100.
            loss_disp = 'train loss = %.3f , train accurary = %2d%%' % ( loss[it] , np.round(100*acc[it]) )
            loss_disp2= 'test loss = %.3f , test accurary = %2d%%' % ( loss2[it] , np.round(100*acc2[it]) )
            print '%3d) %s' % (it, loss_disp)
            print '%3d) %s' % (it, loss_disp2)

#    train_loss, scratch_train_loss = loss['pretrained'], loss['scratch']
#    train_acc, scratch_train_acc = acc['pretrained'], acc['scratch']
#    style_weights, scratch_style_weights = weights['pretrained'], weights['scratch']

