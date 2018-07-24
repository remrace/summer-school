import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import SegEval as ev
import SynGraph as syn


NUM_OUTPUTS = 2
PATCH_SIZE = 11
TRUTH_SIZE = 10
EDGE_SIZE = 5
N = TRUTH_SIZE * TRUTH_SIZE
D = PATCH_SIZE * PATCH_SIZE * 3

    
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
    ind = YT == 0.0
    
    sloss = tf.maximum(0.0, tf.subtract(1.0, tf.multiply(YP, YT)))
    #sloss = tf.maximum(0.0, tf.multiply(-1.0, tf.multiply(YP, YT)))
    #sloss = tf.square(tf.subtract(YT, YP))
    #print(sloss)
    #return tf.reduce_sum(sloss) 
    return tf.reduce_mean(sloss)         

def ShowPatch(img, seg):
    
    fig=plt.figure(figsize=(2, 4))

    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    fig.add_subplot(2, 2, 2)
    plt.imshow(seg)
    plt.show()    
    return    
        
def UnrollData(img, seg, nlabels, elabels):

    X_train = np.zeros( (N, D), np.float32)
    Y_train = np.zeros( (N, 2), np.float32)
    upto = 0
    for j in range(PATCH_SIZE):
        for i in range(PATCH_SIZE):
            x = img[i:(i+PATCH_SIZE), j:(j+PATCH_SIZE), :]
            X_train[upto,:] = np.reshape(x, (1, D))
            if j == (PATCH_SIZE-1) or i == (PATCH_SIZE-1):
                if i == (PATCH_SIZE-1):
                    Y_train[upto, 0] = 0.0
                else:
                    Y_train[upto, 1] = 0.0                    
            else:
                if abs(seg[i,j] - seg[i+1,j]) < 1.0:
                    Y_train[upto, 0] = 1.0
                else:
                    Y_train[upto, 0] = -1.0
                if abs(seg[i,j] - seg[i,j+1]) < 1.0:    
                    Y_train[upto, 1] = 1.0
                else:
                    Y_train[upto, 1] = -1.0
            upto = upto + 1

    return (X_train, Y_train)

def Train(img, seg):
    
    imgn = (img / img.max()) * 2.0 - 1.0
    
    (patchImg, patchSeg, nlabels, elabels) = ev.SamplePatch(imgn, seg)
    #ShowPatch(patchImg, patchSeg)
    (X_train, Y_train) = UnrollData(patchImg, patchSeg, nlabels, elabels)
       
    #print(X_train.shape)
    #print(Y_train.shape)
    
    #pca = PCA(n_components=D, whiten=True)    
    #X_train = pca.fit_transform(X_train)
    #print(X_train.shape)

    weights_shape = (D, NUM_OUTPUTS) 
    bias_shape = (1, NUM_OUTPUTS)    

    #W = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(weights_shape))  # Weights of the model
    #b = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(bias_shape))
    W = tf.Variable(dtype=tf.float32, initial_value=tf.zeros(weights_shape))  # Weights of the model
    b = tf.Variable(dtype=tf.float32, initial_value=tf.ones(bias_shape))

    X = tf.placeholder(tf.float32,name="X")
    Y = tf.placeholder(tf.float32, name="Y")
    #X = tf.placeholder(tf.float32, [batchSize, numFeatures], name="X")
    #Y = tf.placeholder(tf.float32, [batchSize, numOutputs], name="Y")
        
    YP = tf.subtract(tf.matmul(X, W), b)
    
    loss_function = test_loss(Y, YP)    

    learner = tf.train.GradientDescentOptimizer(0.1).minimize(loss_function)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            result = sess.run(learner, {X: X_train, Y: Y_train})
            if i % 10 == 0:
                loss = sess.run(loss_function, {X: X_train, Y: Y_train})
                #print("Iteration {}:\tLoss={:.6f}".format(i, sess.run(error_function, {X: X_train, Y: Y_train})))
                print("Iteration " + str(i) + ": " + str(loss)) 
        
        y_pred = sess.run(YP, {X: xmap})
        W_final, b_final = sess.run([W, b])

    

def TrainConv(img, seg):

    
    (patchImg, patchSeg, nlabels, elabels) = ev.SamplePatch(img, seg)
    #ShowPatch(patchImg, patchSeg)
       
    X_train = tf.reshape(patchImg, [1, patchImg.shape[0], patchImg.shape[1], patchImg.shape[2]], name='image')
    Y_train = np.ones((121, 1), np.float32) 
        
    #X_train, Y_train = SynTestData(numFeatures, numSamples)    
    #Y_train = Y_train.reshape(Y_train.shape[0], 1)
    #print(Y_train.shape)

    #pca = PCA(n_components=numFeatures, whiten=True)
    #X_train = pca.fit_transform(X_train)
            
    weights_shape = (kernelSize, kernelSize, numChannels, numOutputs) 
    bias_shape = (1, numOutputs)    

    W = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(weights_shape))  # Weights of the model
    b = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(bias_shape))
    #W = tf.Variable(dtype=tf.float32, initial_value=tf.zeros(weights_shape))  # Weights of the model
    #b = tf.Variable(dtype=tf.float32, initial_value=tf.ones(bias_shape))

    X = tf.placeholder(tf.float32,name="X")
    Y = tf.placeholder(tf.float32, name="Y")
    #X = tf.placeholder(tf.float32, [batchSize, numFeatures], name="X")
    #Y = tf.placeholder(tf.float32, [batchSize, numOutputs], name="Y")

    YP = tf.subtract(tf.squeeze(tf.nn.conv2d(X, W, [1, 1, 1, 1], "VALID")), b)          
    
    loss_function = test_loss(Y, YP)    

    learner = tf.train.GradientDescentOptimizer(0.1).minimize(loss_function)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            result = sess.run(learner, {X: X_train, Y: Y_train})
            if i % 10 == 0:
                mistakes = sess.run(loss_function, {X: X_train, Y: Y_train})
                #print("Iteration {}:\tLoss={:.6f}".format(i, sess.run(error_function, {X: X_train, Y: Y_train})))
                print("Iteration " + str(i) + ": " + str(mistakes)) 
        
        #y_pred = sess.run(YP, {X: xmap})
        W_final, b_final = sess.run([W, b])
    
    print(W_final)
    print(b_final)    

        
if __name__ == '__main__':
    print(tf.__version__)

