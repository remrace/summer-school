import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import SegEval as ev
import SynGraph as syn

GRAPH_WIDTH = 10
GRAPH_HEIGHT = GRAPH_WIDTH

def KerasTest():
    model = keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    #model.add(keras.layers.Dense(64, activation='relu'))
    # Add another:
    #model.add(keras.layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    #model.add(keras.layers.Dense(10, activation='softmax'))

    #model.compile(optimizer=tf.train.AdamOptimizer(0.001),
    #          loss='rand_loss_function',
    #          metrics=['accuracy'])
    

def SynTestData(numFeatures, numSamples, plotme = None):

    X_train, Y_train = make_blobs(n_features=numFeatures, n_samples=numSamples, centers=2, random_state=500)
    #Y_train = OneHotEncoder().fit_transform(y_flat.reshape(-1, 1)).todense()
    #Y_train = np.array(Y_train)
    # Optional line: Sets a default figure size to be a bit larger.
    if plotme is not None:
        plt.rcParams['figure.figsize'] = (24, 10)
        plt.scatter(X_train[:,0], X_train[:,1], c=Y_train, alpha=0.4, s=150)
        plt.show() 

    Y_train = Y_train * 2 - 1    
    
    return(X_train, Y_train)

def rand_loss_function(YT, YP):
    # Need to get from YP to A
    #G = syn.InitWithAffinities(GRAPH_WIDTH, GRAPH_HEIGHT, A)

    # Needs the GT on nodes (labeling) 
    #[posCounts, negCounts, mstEdges] = ev.FindRandCounts(G, GT)
    
    # Need to get counts into a TF matrix of weights
    #randW = numpy array (posCounts - negCounts) / totalWeight

    # YT need to be ground truth for edges...
    #edge_loss = tf.square(tf.subtract(YT, YP))
    
    # I think if all the pieces are tensorflow, gradient could be automatic..
    #return tf.reduce_sum(tf.multiply(randW, edge_loss))

    return tf.squared_difference(YT, YP)

def test_loss(YT, YP):
    return tf.squared_difference(YT, YP)
        

def Test1():
    numFeatures = 2
    numSamples = 200 
    X_train, Y_train = SynTestData(numFeatures, numSamples, plotme = False)
        
    weights_shape = (numFeatures, 1) 
    bias_shape = (1, 1)    

    W = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(weights_shape))  # Weights of the model
    b = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(bias_shape))

    X = tf.placeholder(tf.float32, name="X")
    Y = tf.placeholder(tf.float32, name="Y")
    
    YP = X * W + b
    
    loss_function = test_loss(Y, YP)

    learner = tf.train.GradientDescentOptimizer(0.1).minimize(loss_function)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50):
            result = sess.run(learner, {X: X_train, Y: Y_train})
            #if i % 100 == 0:
            #print("Iteration {}:\tLoss={:.6f}".format(i, sess.run(loss_function, {X: X_test, Y_true: y_test})))
        y_pred = sess.run(Y_pred, {X: X_train})
        W_final, b_final = sess.run([W, b])

    #predicted_y_values = np.argmax(y_pred, axis=1)
    #predicted_y_values
    print(W_final)



        
if __name__ == '__main__':
    print(tf.__version__)
    Test1()

'''
# Data file provided by the Stanford course CS 20SI: TensorFlow for Deep Learning Research.
# https://github.com/chiphuyen/tf-stanford-tutorials
DATA_FILE = "data/fire_theft.xls"
# read the data from the .xls file.
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
num_samples = sheet.nrows - 1
#######################
## Defining flags #####
#######################
tf.app.flags.DEFINE_integer(
    'num_epochs', 50, 'The number of epochs for training the model. Default=50')
# Store all elements in FLAG structure!
FLAGS = tf.app.flags.FLAGS

# creating the weight and bias.
# The defined variables will be initialized to zero.
W = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")

def inputs():
    """
    Defining the place_holders.
    :return:
            Returning the data and label lace holders.
    """
    X = tf.placeholder(tf.float32, name="X")
    Y = tf.placeholder(tf.float32, name="Y")
    return X,Y


def inference():
    """
    Forward passing the X.
    :param X: Input.
    :return: X*W + b.
    """
    return X * W + b

def loss(X, Y):
    """
    compute the loss by comparing the predicted value to the actual label.
    :param X: The input.
    :param Y: The label.
    :return: The loss over the samples.
    """
    # Making the prediction.
    Y_predicted = inference(X)
    return tf.squared_difference(Y, Y_predicted)

def train(loss):
    learning_rate = 0.0001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            
with tf.Session() as sess:
    # Initialize the variables[w and b].
    sess.run(tf.global_variables_initializer())
    # Get the input tensors
    X, Y = inputs()
    # Return the train loss and create the train_op.
    train_loss = loss(X, Y)
    train_op = train(train_loss)
    # Step 8: train the model
    for epoch_num in range(FLAGS.num_epochs): # run 100 epochs
        for x, y in data:
          train_op = train(train_loss)
          # Session runs train_op to minimize loss
          loss_value,_ = sess.run([train_loss,train_op], feed_dict={X: x, Y: y})
        # Displaying the loss per epoch.
        print('epoch %d, loss=%f' %(epoch_num+1, loss_value))
        # save the values of weight and bias
        wcoeff, bias = sess.run([W, b])


    X = tf.placeholder(dtype=tf.float32)
    Y = tf.placeholder(dtype=tf.float32)
    
    weights_shape = (numFeatures, 1) 
    W = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(weights_shape))  # Weights of the model
    bias_shape = (1, 1)    
    b = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(bias_shape))
    # creating the weight and bias.
    # The defined variables will be initialized to zero.
    #W = tf.Variable(0.0, name="weights")
    #b = tf.Variable(0.0, name="bias")

    #define network
    Y = X * W + b
    #tf.matmul(X, W)

    #loss_function = tf.losses.softmax_cross_entropy(Y_true, Y_pred)
    loss_function = rand_loss_function(Y_true, Y_pred)

    learner = tf.train.GradientDescentOptimizer(0.1).minimize(loss_function)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(500):
            result = sess.run(learner, {X: X_train, Y_true: y_train})
            #if i % 100 == 0:
            #print("Iteration {}:\tLoss={:.6f}".format(i, sess.run(loss_function, {X: X_test, Y_true: y_test})))
        y_pred = sess.run(Y_pred, {X: X_test})
        W_final, b_final = sess.run([W, b])

    predicted_y_values = np.argmax(y_pred, axis=1)
    predicted_y_values        

    #potential other way to get loss in
    ########### Defining place holders ############
    ###############################################
    #image_place = tf.placeholder(tf.float32, shape=([None, num_features]), name='image')
    #label_place = tf.placeholder(tf.int32, shape=([None,]), name='gt')
    #label_one_hot = tf.one_hot(label_place, depth=FLAGS.num_classes, axis=-1)
    #dropout_param = tf.placeholder(tf.float32)
    ##################################################
    ########### Model + Loss + Accuracy ##############
    ##################################################
    # A simple fully connected with two class and a softmax is equivalent to Logistic Regression.
    #logits = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs = FLAGS.num_classes, scope='fc')    
    #with tf.name_scope('loss'):
    #    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot))
    # Accuracy
    #with tf.name_scope('accuracy'):
    #    # Evaluate the model
    #    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))
    #    # Accuracy calculation
    #    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        '''


'''
def TFRandLoss(
        affs,
        gt_affs,
        gt_seg,
        neighborhood,
        gt_aff_mask=None,
        gt_seg_unlabelled=None,
        name=None):

    Returns a tensorflow op to compute the constrained MALIS loss, using the
    squared distance to the target values for each edge as base loss.

    In the simplest case, you need to provide predicted affinities (``affs``),
    ground-truth affinities (``gt_affs``), a ground-truth segmentation
    (``gt_seg``), and the neighborhood that corresponds to the affinities.

    This loss also supports masks indicating unknown ground-truth. We
    distinguish two types of unknowns:

        1. Out of ground-truth. This is the case at the boundary of your
           labelled area. It is unknown whether objects continue or stop at the
           transition of the labelled area. This mask is given on edges as
           argument ``gt_aff_mask``.

        2. Unlabelled objects. It is known that there exists a boundary between
           the labelled area and unlabelled objects. Withing the unlabelled
           objects area, it is unknown where boundaries are. This mask is also
           given on edges as argument ``gt_aff_mask``, and with an additional
           argument ``gt_seg_unlabelled`` to indicate where unlabelled objects
           are in the ground-truth segmentation.

    Both types of unknowns require masking edges to exclude them from the loss:
    For "out of ground-truth", these are all edges that have at least one node
    inside the "out of ground-truth" area. For "unlabelled objects", these are
    all edges that have both nodes inside the "unlabelled objects" area.

    Args:

        affs (Tensor): The predicted affinities.

        gt_affs (Tensor): The ground-truth affinities.

        gt_seg (Tensor): The corresponding segmentation to the ground-truth
            affinities. Label 0 denotes background.

        neighborhood (Tensor): A list of spatial offsets, defining the
            neighborhood for each voxel.

        gt_aff_mask (Tensor): A binary mask indicating where ground-truth
            affinities are known (known = 1, unknown = 0). This is to be used
            for sparsely labelled ground-truth and at the borders of labelled
            areas. Edges with unknown affinities will not be constrained in the
            two malis passes, and will not contribute to the loss.

        gt_seg_unlabelled (Tensor): A binary mask indicating where the
            ground-truth contains unlabelled objects (labelled = 1, unlabelled
            = 0). This is to be used for ground-truth where only some objects
            have been labelled. Note that this mask is a complement to
            ``gt_aff_mask``: It is assumed that no objects cross from labelled
            to unlabelled, i.e., the boundary is a real object boundary.
            Ground-truth affinities within the unlabelled areas should be
            masked out in ``gt_aff_mask``. Ground-truth affinities between
            labelled and unlabelled areas should be zero in ``gt_affs``.

        name (string, optional): A name to use for the operators created.

    Returns:

        A tensor with one element, the MALIS loss.

    weights = malis_weights_op(
        affs,
        gt_affs,
        gt_seg,
        neighborhood,
        gt_aff_mask,
        gt_seg_unlabelled,
        name)
    edge_loss = tf.square(tf.subtract(gt_affs, affs))

    return tf.reduce_sum(tf.multiply(weights, edge_loss))



    '''
