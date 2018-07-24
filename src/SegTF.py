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
import VizGraph as viz
import matplotlib.pyplot as plt
import SegGraph as seglib

NUM_OUTPUTS = 1
PATCH_SIZE = 21
KERNEL_SIZE = 11
N = (PATCH_SIZE-1) * PATCH_SIZE * 2
D = KERNEL_SIZE * KERNEL_SIZE * 3
    

def ShowPatch(img, seg):
    
    fig=plt.figure(figsize=(4, 2))

    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    fig.add_subplot(1, 2, 2)
    plt.imshow(seg)
    plt.show()    
    return    

def ShowResult(img, seg, pred):
    
    fig=plt.figure(figsize=(6, 2))

    fig.add_subplot(1, 3, 1)
    plt.imshow(img)
    fig.add_subplot(1, 3, 2)
    plt.imshow(seg)
    fig.add_subplot(1, 3, 3)
    plt.imshow(pred)    
    plt.show()    
    return    
        
def UnrollData(img, seg, G, nlabels, elabels):

    X_train = np.zeros( (N, D), np.float32)
    Y_train = np.zeros( (N, 1), np.float32)
    upto = 0
    for (u,v) in G.edges():
        uimg = img[u[0]:(u[0]+KERNEL_SIZE), u[1]:(u[1]+KERNEL_SIZE), :]
        vimg = img[v[0]:(v[0]+KERNEL_SIZE), v[1]:(v[1]+KERNEL_SIZE), :]
        aimg = (uimg + vimg) / 2.0
        X_train[upto,:] = np.reshape(aimg, (1, D))
        Y_train[upto, 0] = elabels[(u,v)]
        upto = upto + 1

    return (X_train, Y_train)

def Train(img, seg):
    
    imgn = (img / img.max()) * 2.0 - 1.0
    
    mySeed = 37
    np.random.seed(mySeed)

    (patchImg, patchSeg, G, nlabels, elabels) = ev.SamplePatch(imgn, seg, PATCH_SIZE, KERNEL_SIZE)
    
    ShowPatch(patchImg, patchSeg)    

    (X_train, Y_train) = UnrollData(patchImg, patchSeg, G, nlabels, elabels)
    
    #return
    pca = PCA(n_components=D, whiten=True)    
    X_train = pca.fit_transform(X_train)
    print(X_train.shape)
    print(Y_train.shape)

    def rand_loss_function(YT, YP):

        def GetRandWeights(A):
            upto = 0
            for u, v, d in G.edges(data = True):
                d['weight'] = A[upto]
                upto = upto + 1

            [posCounts, negCounts, mstEdges, edgeInd] = ev.FindRandCounts(G, nlabels)
            #randW = numpy array (posCounts - negCounts) / totalWeight            
            mstW = posCounts - negCounts
            
            posMst = mstW > 0
            posEdges = edgeInd[ posMst ]
            posWeight = mstW[ posMst ]

            negMst = mstW < 0
            negEdges = edgeInd[ negMst ]
            negWeight = -mstW[ negMst ]
            
            W = np.zeros( (N, 1), np.float32)

            totalPos = 0.0
            for ei in range(len(posMst)):                
                loss = 1 - A[ posEdges[ei] ]
                if loss > 0.0:
                    wloss = posWeight[ei] * loss
                    W[ posEdges[ei] ] = wloss
                    totalPos = totalPos + wloss

            totalNeg = 0.0
            for ei in range(len(negMst)):                
                loss = 1 + A[ negEdges[ei] ]
                if loss > 0.0:
                    wloss = negWeight[ei] * loss
                    W[ negEdges[ei] ] = wloss
                    totalNeg = totalNeg + wloss

            totalW = totalPos + totalNeg
            if totalW > 1.0e-8: 
                W = W / totalW
            
            return W
                
        RW = tf.py_func(GetRandWeights, [YP], [tf.float32]) 
                
        #weightedLoss = tf.multiply(RW, YP)

        return tf.reduce_sum(RW) 
    
    def test_loss(YT, YP):           
        sloss = tf.maximum(0.0, tf.subtract(1.0, tf.multiply(YP, YT)))
        #sloss = tf.maximum(0.0, tf.multiply(-1.0, tf.multiply(YP, YT)))
        #sloss = tf.square(tf.subtract(YT, YP))
        #print(sloss)
        #return tf.reduce_sum(sloss) 
        return tf.reduce_mean(sloss)         
    

    weights_shape = (D, NUM_OUTPUTS) 
    bias_shape = (1, NUM_OUTPUTS)    

    #W = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(weights_shape))  # Weights of the model
    #b = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(bias_shape))
    W = tf.Variable(dtype=tf.float32, initial_value=tf.zeros(weights_shape))  # Weights of the model
    b = tf.Variable(dtype=tf.float32, initial_value=tf.ones(bias_shape))

    X = tf.placeholder(tf.float32,name="X")
    Y = tf.placeholder(tf.float32, name="Y")
        
    YP = tf.subtract(tf.matmul(X, W), b)
    
    #loss_function = test_loss(Y, YP)    
    loss_function = rand_loss_function(Y, YP)    

    learner = tf.train.GradientDescentOptimizer(0.1).minimize(loss_function)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            result = sess.run(learner, {X: X_train, Y: Y_train})
            if i % 10 == 0:
                loss = sess.run(loss_function, {X: X_train, Y: Y_train})
                #print("Iteration {}:\tLoss={:.6f}".format(i, sess.run(error_function, {X: X_train, Y: Y_train})))
                print("Iteration " + str(i) + ": " + str(loss)) 
        
        y_pred = sess.run(YP, {X: X_train})
        W_final, b_final = sess.run([W, b])
    upto = 0

    for u, v, d in G.edges(data = True):
        d['weight'] = float(y_pred[upto])
        upto = upto + 1
    
    predMin = y_pred.min()
    predMax = y_pred.max()
    print("Prediction has range " + str(predMin) + " -> " + str(predMax))

    labels = seglib.GetLabelsAtThreshold(G, 1.0)

    patchPred = np.zeros( (patchSeg.shape[0], patchSeg.shape[1]), np.float32)
    for j in range(patchSeg.shape[1]):
        for i in range(patchSeg.shape[0]):
            patchPred[i,j] = labels[(i,j)]

    ShowResult(patchImg, patchSeg, patchPred)
    #plt.show()

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

