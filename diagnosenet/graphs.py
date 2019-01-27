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

class FullyConnected:
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

        ## Graph object trainable parameters:
        self.mlp_graph = tf.Graph()
        self.X: tf.placeholder
        self.Y: tf.placeholder
        self.keep_prob: tf.placeholder
        self.projection: tf.Tensor
        self.mlp_loss: tf.Tensor
        self.mlp_grad_op: tf.Tensor

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

    # def stacked_multigpu(self, input_holder, keep_prob, reuse) -> tf.Tensor:
    #     """
    #     """
    #     # with tf.variable_scope("BackPropagation", reuse=reuse):
    #     for i in range(len(self.layers)):
    #             ## Prevention to use dropout in the projection layer
    #             if len(self.layers)-1 == i:
    #                 input_holder = self.layers[i].activation(input_holder)
    #             else:
    #                 input_holder = self.layers[i].dropout_activation(input_holder, keep_prob)
    #     return input_holder


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
        with tf.Graph().as_default() as self.mlp_graph:
            self.X = tf.placeholder(tf.float32, shape=(None, self.input_size), name="Inputs")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.output_size), name="Output")
            self.keep_prob = tf.placeholder(tf.float32)

            self.projection = self.stacked(self.X, self.keep_prob)
            self.mlp_loss = self.loss.desktop_loss(self, self.projection, self.Y)
            self.mlp_grad_op = self.optimizer.desktop_Grad(self.mlp_loss)

            ## Accuracy
            # self.accuracy = Metrics().accuracy(self.Y, self.projection)
            # print("self.accuracy: {}".format(self.accuracy))

            ## # Convert prediction to one hot encoding
            self.soft_projection = tf.nn.softmax(self.projection)
            self.max_projection = tf.argmax(tf.nn.softmax(self.projection), 1)
            self.projection_1hot = tf.one_hot(self.max_projection, depth = int(self.output_size))



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




    # def multiGPU_graph_OLD(self, batch_size) -> tf.Tensor:
    #
    #     with tf.Graph().as_default() as self.mlp_graph:
    #
    #         self.num_gpus=1
    #         self.gpu_batch_size=int((batch_size/self.num_gpus))
    #         ###########################
    #         self.tower_grads = []
    #         self.mlp_losses = []
    #
    #         self.X = tf.placeholder(tf.float32, shape=(None, self.input_size), name="Inputs")
    #         self.Y = tf.placeholder(tf.float32, shape=(None, self.output_size), name="Targets")
    #         self.keep_prob = tf.placeholder(tf.float32)
    #
    #         for gpu in range(self.num_gpus):
    #             print("gpu: {}".format(gpu))
    #
    #             with tf.device('/gpu:%d' % gpu):
    #
    #                 # Split data between GPUs
    #                 _X = self.X[(gpu * self.gpu_batch_size):
    #                             (gpu * self.gpu_batch_size) + (self.gpu_batch_size)]
    #                 _Y = self.Y[(gpu * self.gpu_batch_size):
    #                             (gpu * self.gpu_batch_size) + (self.gpu_batch_size)]
    #
    #                 print("_X: {}, {}".format((gpu * self.gpu_batch_size),
    #                                 (gpu * self.gpu_batch_size) + (self.gpu_batch_size)))
    #
    #
    #                 # self.projection = self.stacked(_X)
    #                 # self.projection = self.stacked(_X, self.keep_prob)
    #                 self.projection = self.stacked_multigpu(_X, self.keep_prob)
    #
    #                 print("{}".format("+"*20))
    #                 print("self.projection: {}".format(self.projection))
    #
    #                 ### Loss ###
    #                 ## Desktop function
    #                 # self.mlp_loss = self.loss.multiGPU_loss(self.projection, _Y)
    #
    #                 self.mlp_loss = self.multiGPU_loss(self.projection, _Y)
    #
    #                 # self.mlp_losses.append(self.mlp_loss)
    #
    #                 ### optimizer ###
    #                 ## Desktop function
    #                 self.grad_from_optimizer = self.optimizer.desktop_Grad(self.mlp_loss)
    #
    #                 ### New grads
    #                 self.adam_op = tf.train.AdamOptimizer(learning_rate=0.001)
    #                 self.grads_computation = self.adam_op.compute_gradients(self.mlp_loss)
    #                 self.tower_grads.append(self.grads_computation)
    #
    #         self.mlp_tower_grads = self.average_gradients(self.tower_grads)
    #         self.train_op = self.adam_op.apply_gradients(self.mlp_tower_grads)
    #         # self.train_op = optimizer.apply_gradients(tower_grads)
    #
    #         #self.mlp_tower_grads = self.average_gradients(self.tower_grads)
    #         # self.train_op = mgpu_optimizer.apply_gradients(tower_grads)
