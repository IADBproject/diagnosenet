"""
A graph object is a neural network architecture defined to be trained in the target computing platform.
The computational graph applieds more than one operation to the weights of the neural network architecture.
"""

from typing import Sequence, NamedTuple
import tensorflow as tf

from diagnosenet.layers import Layer
from diagnosenet.losses import Loss
from diagnosenet.optimizers import Optimizer
from diagnosenet.monitor import Metrics


class SequentialGraph:
    """
    Implements the back-propagation algorithm...
    Args: A neural network architecture defined by the user.
    Returns: A graph object with trainable parameters;
            that will be assign data in the executors class.
    """
    def __init__(self, input_size: int, output_size: int,
                        layers: Sequence[Layer],
                        loss: Loss,
                        optimizer: Optimizer,
                        dropout: float = 1.0) -> None:

        ## A neural network architecture:
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.dropout = dropout

        self.global_step = tf.Tensor
        self.init_op = tf.Tensor

        ## Graph object trainable parameters:
        self.graph = tf.Graph()
        self.X: tf.placeholder
        self.Y: tf.placeholder
        self.keep_prob: tf.placeholder
        self.projection: tf.Tensor
        self.loss: tf.Tensor
        self.grad_op: tf.Tensor

        ## metrics
        # self.metrics = Metrics()
        self.accuracy: tf.Tensor

        ## MultiGPU
        # self.reuse_vars: bool = False
        self.PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

    def stacked(self, input_holder, keep_prob) -> tf.Tensor:
        for i in range(len(self.layers)):
            ## Prevention to use dropout in the projection layer
            if len(self.layers)-1 == i:
                input_holder = self.layers[i].activation(input_holder)
            else:
                input_holder = self.layers[i].dropout_activation(input_holder, keep_prob)
        return input_holder

    def stacked_valid(self, input_holder) -> tf.Tensor:
        for layer in self.layers:
            print("layer: {}".format(layer.__class__.__name__))
            if layer.__class__.__name__ == "Dropout":
                pass
            else:
                input_holder = layer.activation(input_holder)
                print("layer: {}".format(input_holder))
        return input_holder

    def desktop_graph(self) -> tf.Tensor:
        with tf.Graph().as_default() as self.graph:
            self.X = tf.placeholder(tf.float32, shape=(None, self.input_size), name="Inputs")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.output_size), name="Output")
            self.keep_prob = tf.placeholder(tf.float32)

            self.projection = self.stacked(self.X, self.keep_prob)
            self.loss = self.loss.desktop_loss(self, self.projection, self.Y)
            self.grad_op = self.optimizer.desktop_Grad(self.loss)

            ## Accuracy
            # self.accuracy = Metrics().accuracy(self.Y, self.projection)
            # print("self.accuracy: {}".format(self.accuracy))

            ## # Convert prediction to one hot encoding
            self.soft_projection = tf.nn.softmax(self.projection)
            self.max_projection = tf.argmax(tf.nn.softmax(self.projection), 1)
            self.projection_1hot = tf.one_hot(self.max_projection, depth = int(self.output_size))

            
    def distributed_grpc_graph(self, cluster, task_index) -> tf.Tensor:
        #with tf.Graph().as_default() as self.graph:
        with tf.device(tf.train.replica_device_setter(
                                worker_device="/job:worker/task:%d" % task_index,
                                cluster=cluster)) as self.graph:

            self.X = tf.placeholder(tf.float32, shape=(None, self.input_size), name="Inputs")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.output_size), name="Output")
            self.keep_prob = tf.placeholder(tf.float32)

            self.projection = self.stacked(self.X, self.keep_prob)

            with tf.variable_scope("global_step", reuse=True):
                print("++ Datamaneger+Issue: Pass batch_size ++")
                self.global_step = tf.Variable(500)	#datamanager.batch_size

            self.loss = self.loss.desktop_loss(self, self.projection, self.Y)
            self.grad_op = self.optimizer.desktop_Grad(self.loss)

            ## Accuracy
            ## self.accuracy = Metrics().accuracy(self.Y, self.projection)
            ## print("self.accuracy: {}".format(self.accuracy))

            ## # Convert prediction to one hot encoding
            self.soft_projection = tf.nn.softmax(self.projection)
            self.max_projection = tf.argmax(tf.nn.softmax(self.projection), 1)
            self.projection_1hot = tf.one_hot(self.max_projection, depth = int(self.output_size))
            
            self.init_op = tf.group(tf.global_variables_initializer(),
				                              tf.local_variables_initializer())
            
            
    def distributed_mpi_graph(self) -> tf.Tensor:
        with tf.Graph().as_default() as self.graph:
            self.X = tf.placeholder(tf.float32, shape=(None, self.input_size), name="Inputs")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.output_size), name="Output")
            self.keep_prob = tf.placeholder(tf.float32)

            #for the training part
            self.projection = self.stacked(self.X,  self.keep_prob)
            self.loss = self.loss.desktop_loss(self,self.projection, self.Y)
            self.adam_op = tf.train.AdamOptimizer(self.optimizer.lr)
            self.grad_op = self.adam_op.compute_gradients(self.loss)
            self.sub_grad_op = self.optimizer.desktop_Grad(self.loss)

            self.soft_projection = tf.nn.softmax(self.projection)
            self.max_projection = tf.argmax(self.soft_projection, 1)
            self.projection_1hot = tf.one_hot(self.max_projection, depth = int(self.output_size))

            avg_grads_and_vars = []
            self._grad_placeholders = []
            for grad, var in self.grad_op:
                grad_ph = tf.placeholder(grad.dtype, grad.shape)
                self._grad_placeholders.append(grad_ph)
                avg_grads_and_vars.append((grad_ph, var))
            self._grad_op = [x[0] for x in self.grad_op]
            self._train_op = self.adam_op.apply_gradients(avg_grads_and_vars)
            self._gradients = []



    ###################################
    ## Warning: Developing MultiGPU grpah
    def stacked_multigpu(self, input_holder, keep_prob, reuse) -> tf.Tensor:
        """
        """
        # with tf.variable_scope("BackPropagation", reuse=reuse):
        w1 = tf.Variable(tf.random_normal([14637, 2048], stddev=0.1), dtype=tf.float32)
        b1 = tf.Variable(tf.random_normal([2048]), dtype=tf.float32)
        l1= tf.nn.relu(tf.matmul(input_holder, w1) + b1)

        w2 = tf.Variable(tf.random_normal([2048, 14], stddev=0.1), dtype=tf.float32)
        b2 = tf.Variable(tf.random_normal([14]), dtype=tf.float32)
        l2 = tf.matmul(l1, w2 + b2)
        return l2

    def stacked_multigpu__API__(self, input_holder, keep_prob, reuse) -> tf.Tensor:
        """
        """
        # with tf.variable_scope("BackPropagation", reuse=reuse):
        for i in range(len(self.layers)):
                ## Prevention to use dropout in the projection layer
                if len(self.layers)-1 == i:
                    input_holder = self.layers[i].activation(input_holder)
                else:
                    input_holder = self.layers[i].dropout_activation(input_holder, keep_prob)
        return input_holder

    def multiGPU_loss(self, y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        """
        """
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        cross_entropy_reduce = tf.reduce_mean(cross_entropy)

        return cross_entropy_reduce

    def tower_loss(self, y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        """
        """
        # loss = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
        loss = self.multiGPU_loss(y_pred, y_true)
        # ## Assemble all the losses for the current
        losses = tf.get_collection('losses', loss)

        ## Calculate the total loss for the current tower
        # total_loss = tf.add_n(losses(losses, name='total_loss'))
        # total_loss = tf.add_n(losses)
        #
        # ##
        # loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        # loss_averages_op = loss_averages.apply(losses + [total_loss])
        #
        # with tf.control_dependencies([loss_averages_op]):
        #     total_loss = tf.identity(total_loss)
        # return total_loss
        return loss

    def average_gradients(self, tower_grads):
        """
        Merge the grads computations done by each GPU tower
        """
        ### First Print
        print("\n \n")
        # print("tower_grads: {}".format(tower_grads))
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            ## Second print
            # print("grad_and_vars: {}".format(grad_and_vars))
            grads = []
            for g, _ in grad_and_vars:
                ## Third Print
                print("+ Grad by Tower: {}".format(g))
                if g is None:
                    pass
                else:
                    # Add 0 dimension to the gradients to represent the tower.
                    expanded_g = tf.expand_dims(g, 0)

                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)

        #### JAGH DEbug
        #         grads.append(g)
        # return grads

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    def assign_to_device(self, device, ps_device='/cpu:0'):
        def _assign(op):
            node_def = op if isinstance(op, tf.NodeDef) else op.node_def
            if node_def.op in self.PS_OPS:
                return "/" + ps_device
            else:
                return device

        return _assign

    def multiGPU_graph(self, batch_size, num_gpus) -> tf.Tensor:
        """
        """
        self.num_gpus=num_gpus
        self.gpu_batch_size=int((batch_size/self.num_gpus))


        with tf.Graph().as_default() as self.mlp_graph:

            with tf.device('/cpu:0'):
                ###########################
                self.total_projection = []
                self.total_losses = []
                self.total_grads = []

                self.X = tf.placeholder(tf.float32, shape=(None, self.input_size), name="Inputs")
                self.Y = tf.placeholder(tf.float32, shape=(None, self.output_size), name="Targets")
                self.keep_prob = tf.placeholder(tf.float32)

                self.adam_op = tf.train.AdamOptimizer(learning_rate=0.001)
                reuse_vars = False

                ################
                for gpu in range(self.num_gpus):
                    # with tf.device('/gpu:%d' % gpu):
                    with tf.device(self.assign_to_device('/gpu:{}'.format(gpu), ps_device='/cpu:0')):

                        with self.mlp_graph.name_scope("Tower") as scope:
                            # tf.variable_scope.reuse_variables()
                            # Split data between GPUs
                            self._X = self.X[(gpu * self.gpu_batch_size):
                                    (gpu * self.gpu_batch_size) + (self.gpu_batch_size)]
                            self._Y = self.Y[(gpu * self.gpu_batch_size):
                                    (gpu * self.gpu_batch_size) + (self.gpu_batch_size)]

                            ## Projection by Tower Model operations
                            if gpu == 0:
                                with tf.variable_scope("BackPropagation", reuse=False):
                                        self.projection = self.stacked_multigpu(self._X, self.keep_prob, reuse_vars)
                            else:
                                with tf.variable_scope("BackPropagation", reuse=True):
                                    self.projection = self.stacked_multigpu(self._X, self.keep_prob, reuse_vars)
                            self.total_projection.append(self.projection)


                            ## Loss by Tower Model operations
                            self.loss = self.multiGPU_loss(self.projection, self._Y)
                            self.total_losses.append(self.loss)

                            ## Grads by Tower Model operations
                            self.grads_computation = self.adam_op.compute_gradients(self.loss)
                                                            # gate_gradients=2,
                                                            # colocate_gradients_with_ops=True,)
                            # reuse_vars = True
                            self.total_grads.append(self.grads_computation)


                            print("{}".format("+"*20))
                            print("+ GPU: {}".format(gpu))
                            print("+ Split_X: {}, {}".format((gpu * self.gpu_batch_size),
                                (gpu * self.gpu_batch_size) + (self.gpu_batch_size)))
                            print("+ Tower_Projection: {}".format(self.projection.name))
                            print("+ Gradient_T: {}".format(self.grads_computation))
                            print("{}".format("+"*20))


                with tf.device('/cpu:0'):
                    self.output1 = tf.concat(self.total_projection, axis=0)
                    self.output2 = self.total_losses
                    self.output3 = self.average_gradients(self.total_grads)
                    self.train_op = tf.group(self.adam_op.apply_gradients(self.output3))
                    # self.output3 = tf.concat(self.total_grads, axis=0)
        ## End Graph


class CustomGraph:

    def __init__(self, input_size_1: int,input_size_2: int, output_size: int,loss: Loss,
                        optimizer: Optimizer,layers:Sequence[Layer],
                        dropout: float = 1.0) -> None:

        ## A neural network architecture:
        self.input_size_1 = input_size_1
        self.input_size_2 = input_size_2
        self.output_size = output_size
        self.loss = loss
        self.optimizer = optimizer
        self.dropout = dropout
        self.layers = layers
        self.projection: tf.Tensor

    def r_block(self,in_layer,k,keep_prob,is_training):
        x = tf.layers.batch_normalization(in_layer)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=keep_prob, training=is_training)
        x = tf.layers.conv1d(x,64*k,16,padding='same',kernel_initializer=tf.glorot_uniform_initializer())
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=keep_prob, training=is_training)
        x = tf.layers.conv1d(x,64*k,16,padding='same',kernel_initializer=tf.glorot_uniform_initializer())
        x = tf.add(x,in_layer)
        return x

    def subsampling_r_block(self,in_layer,k,keep_prob,is_training):
        x = tf.layers.batch_normalization(in_layer)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=keep_prob, training=is_training)
        x = tf.layers.conv1d(x,64*k,16,kernel_initializer=tf.glorot_uniform_initializer(),padding='same')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=keep_prob, training=is_training)
        x = tf.layers.conv1d(x, 64*k, 1, strides=2,kernel_initializer=tf.glorot_uniform_initializer())
        pool = tf.layers.max_pooling1d(in_layer,1,strides=2)
        x = tf.add(x,pool)
        return x

    def stacked(self,x,keep_prob):
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet'):
            is_training =tf.cond( keep_prob<1.0,lambda:True,lambda: False)

            act1 = tf.layers.conv1d(x, 64, 16, padding='same',kernel_initializer=tf.glorot_uniform_initializer())
            x = tf.layers.batch_normalization(act1)
            x = tf.nn.relu(x)


            x = tf.layers.conv1d(x, 64, 16, padding='same',kernel_initializer=tf.glorot_uniform_initializer())
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)

            x = tf.layers.dropout(x, rate=keep_prob, training=is_training)
            x1 = tf.layers.conv1d(x, 64, 1, strides=2,kernel_initializer=tf.glorot_uniform_initializer())

            x2 = tf.layers.max_pooling1d(act1,2,strides=2)
            x = tf.add(x1,x2)

            k=1
            for i in range(1,3,1):
                if i%2 ==0:
                    k+=1
                x=tf.layers.conv1d(x,64*k,16,padding='same',kernel_initializer=tf.glorot_uniform_initializer())
                x=self.r_block(x,k,keep_prob,is_training)
                x=self.subsampling_r_block(x,k,keep_prob,is_training)

            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = tf.contrib.layers.flatten(x)
            out = tf.layers.dense(x, 4,kernel_initializer=tf.glorot_uniform_initializer())
        return out

    def desktop_graph(self) -> tf.Tensor:
        with tf.Graph().as_default() as self.graph:
            self.X = tf.placeholder(tf.float32, shape=(None, self.input_size_1,self.input_size_2), name="Inputs")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.output_size), name="Output")
            self.keep_prob = tf.placeholder(tf.float32)
            #for the training part
            self.projection = self.stacked(self.X,  self.keep_prob)
            self.loss = self.loss.desktop_loss(self, self.projection, self.Y)
            self.grad_op = self.optimizer.desktop_Grad(self.loss)

            self.soft_projection = tf.nn.softmax(self.projection)
            self.max_projection = tf.argmax(self.soft_projection, 1)
            self.projection_1hot = tf.one_hot(self.max_projection, depth = int(self.output_size))


    def distributed_grpc_graph(self, cluster, task_index) -> tf.Tensor:
        #with tf.Graph().as_default() as self.graph:
        with tf.device(tf.train.replica_device_setter(
                                worker_device="/job:worker/task:%d" % task_index,
                                cluster=cluster)) as self.graph:

            self.X = tf.placeholder(tf.float32, shape=(None, self.input_size_1,self.input_size_2), name="Inputs")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.output_size), name="Output")
            self.keep_prob = tf.placeholder(tf.float32)
            self.projection = self.stacked(self.X,  self.keep_prob)

            with tf.variable_scope("global_step", reuse=True):
                print("++ Datamaneger+Issue: Pass batch_size ++")
                self.global_step = tf.Variable(500)     #datamanager.batch_size

            self.loss = self.loss.desktop_loss(self, self.projection, self.Y)
            self.grad_op = self.optimizer.desktop_Grad(self.loss)

            ## Accuracy
            ## self.accuracy = Metrics().accuracy(self.Y, self.projection)
            ## print("self.accuracy: {}".format(self.accuracy))

            ## # Convert prediction to one hot encoding
            self.soft_projection = tf.nn.softmax(self.projection)
            self.max_projection = tf.argmax(tf.nn.softmax(self.projection), 1)
            self.projection_1hot = tf.one_hot(self.max_projection, depth = int(self.output_size))

            self.init_op = tf.group(tf.global_variables_initializer(),
                                                              tf.local_variables_initializer())




    def distributed_mpi_graph(self) -> tf.Tensor:
        with tf.Graph().as_default() as self.graph:
            self.X = tf.placeholder(tf.float32, shape=(None, self.input_size_1,self.input_size_2), name="Inputs")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.output_size), name="Output")
            self.keep_prob = tf.placeholder(tf.float32)

            #for the training part
            self.projection = self.stacked(self.X,  self.keep_prob)
            self.loss = self.loss.desktop_loss(self,self.projection, self.Y)
            self.adam_op = tf.train.AdamOptimizer(self.optimizer.lr)
            self.sub_grad_op = self.optimizer.desktop_Grad(self.loss)
            
            self.grad_op = self.adam_op.compute_gradients(self.loss)
            self.soft_projection = tf.nn.softmax(self.projection)
            self.max_projection = tf.argmax(self.soft_projection, 1)
            self.projection_1hot = tf.one_hot(self.max_projection, depth = int(self.output_size))

            avg_grads_and_vars = []
            self._grad_placeholders = []
            for grad, var in self.grad_op:
                grad_ph = tf.placeholder(grad.dtype, grad.shape)
                self._grad_placeholders.append(grad_ph)
                avg_grads_and_vars.append((grad_ph, var))
            self._grad_op = [x[0] for x in self.grad_op]
            self._train_op = self.adam_op.apply_gradients(avg_grads_and_vars)
            self._gradients = []



##########################################
## Warning: Adding CNNs on SequentialGraph
import numpy as np
class ConvNetworks:
    """
    Implements a fully-connected algorithm for convolutional networks.
    Args: A convolution architecture defined by the user.
    Returns: A graph object with trainable parameters;
            that will be assign data in the executors class.
    """
    def __init__(self,  input_size: int,
                        input_length: int,
                        dimension: int,
                        layers: Sequence[Layer]) -> None:
                        # input_size: int, output_size: int,
                        # layers: Sequence[Layer],
                        # loss: Loss,
                        # optimizer: Optimizer,
                        # dropout: float = 1.0) -> None:

        ## A neural network architecture:
        self.input_size = input_size
        self.input_length = input_length
        self.dimension = dimension
        self.layers = layers

    def stacked(self, input_holder) -> tf.Tensor:
        for i in range(len(self.layers)):
            ## Prevention to use dropout in the projection layer
            if len(self.layers)-1 == i:
                input_holder = self.layers[i].activation(input_holder)
        return input_holder


    def desktop_graph(self) -> tf.Tensor:
        with tf.Graph().as_default() as self.conv1d_graph:
            self.X = tf.placeholder(tf.float32, shape=(None, self.input_length, self.dimension), name="Inputs")
            output = self.stacked(self.X)

            # filter=tf.zeros([1300, 1, 1])
            # output = tf.nn.conv1d(self.X, filter, stride=2, padding='VALID')

            init_op = tf.global_variables_initializer()
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.7

            with tf.Session(config=config) as sess:
                sess.run(init_op)
                matrix = np.load('/home/jagarcia/Documents/05_dIAgnoseNET/04-stage-2019/Version1/input/xdata.npy')
                output_l = sess.run(output, feed_dict={self.X: matrix})
                print(output_l.shape)
