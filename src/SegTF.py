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

GRAPH_WIDTH = 10
GRAPH_HEIGHT = GRAPH_WIDTH
numOutputs = 1
numFeatures = 2
numSamples = 200 
batchSize = 200

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
    

def SynTestData(numFeatures, numSamples):

    X_train, Y_train = make_blobs(n_features=numFeatures, n_samples=numSamples, centers=2, random_state=500)
    Y_train = Y_train * 2.0 - 1.0    
    
    return(X_train, Y_train)

def VizTest(X_values):
    h = 1
    x_min, x_max = X_values[:, 0].min() - 2 * h, X_values[:, 0].max() + 2 * h
    y_min, y_max = X_values[:, 1].min() - 2 * h, X_values[:, 1].max() + 2 * h
    x_0, x_1 = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    decision_points = np.c_[x_0.ravel(), x_1.ravel()]
    return(decision_points, x_0, x_1)

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
    sloss = tf.maximum(0.0, tf.subtract(1.0, tf.multiply(YP, YT)))
    #sloss = tf.maximum(0.0, tf.multiply(-1.0, tf.multiply(YP, YT)))
    #sloss = tf.square(tf.subtract(YT, YP))
    #print(sloss)
    #return tf.reduce_sum(sloss) 
    return tf.reduce_mean(sloss) 

def test_error(YT, YP):       
    #YP = tf.Print(YP, [YP.shape])
    #YT = tf.Print(YT, [YT.shape])
     
    YPB = tf.greater(YP, 0.0)
    #num = tf.reduce_sum(tf.cast( YPB, tf.float32))
    YTB = tf.greater(YT, 0.0)
    #num = tf.reduce_sum(tf.cast( YTB, tf.float32))
    
    YERR = tf.logical_xor(YTB, YPB)
    num = tf.reduce_sum(tf.cast( YERR, tf.float32))
    
    numSamples = tf.cast( tf.shape(YT)[0], tf.float32)

    return tf.divide(num, numSamples)
        

def Test1():

    X_train, Y_train = SynTestData(numFeatures, numSamples)
    #print(Y_train.shape)
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    print(Y_train.shape)

    pca = PCA(n_components=numFeatures, whiten=True)
    
    X_train = pca.fit_transform(X_train)
    
    xmap, x_0, x_1 = VizTest(X_train)        
    
    #Y2 = OneHotEncoder().fit_transform(Y_train.reshape(-1, 1)).todense()
    #Y2 = np.array(Y2)
        
    weights_shape = (numFeatures, numOutputs) 
    bias_shape = (1, numOutputs)    

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
    error_function = test_error(Y, YP)

    learner = tf.train.GradientDescentOptimizer(0.1).minimize(loss_function)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            result = sess.run(learner, {X: X_train, Y: Y_train})
            if i % 10 == 0:
                mistakes = sess.run(error_function, {X: X_train, Y: Y_train})
                #print("Iteration {}:\tLoss={:.6f}".format(i, sess.run(error_function, {X: X_train, Y: Y_train})))
                print("Iteration " + str(i) + ": " + str(mistakes)) 
        
        y_pred = sess.run(YP, {X: xmap})
        W_final, b_final = sess.run([W, b])

    
    print("Scatter 1")
    print(Y_train.shape)
    Y_train = Y_train.reshape(Y_train.shape[0])
    plt.rcParams['figure.figsize'] = (24, 10)
    plt.scatter(X_train[:,0], X_train[:,1], c=Y_train>0, alpha=0.4, s=150)

    Z = np.array(y_pred)
    Z = Z.reshape(Z.shape[0])
    Z = Z > 0
    #Z = Z.reshape(xx.shape)
    
    print("Scatter 2")
    print(Z.shape)
    plt.scatter(xmap[:,0], xmap[:,1], c=Z, marker='x') #alpha=0.3)
    #plt.scatter(X_test[:,0], X_test[:,1], c=predicted_y_values, marker='x', s=200)
    #plt.contourf(x_0, x_1, Z, alpha=0.1)
    #plt.xlim(x_0.min(), x_0.max())
    #plt.ylim(x_1.min(), x_1.max())

    #plt.scatter(X_train[:,0], X_train[:,1], c=y_train_flat, alpha=0.3)
    #plt.scatter(X_test[:,0], X_test[:,1], c=predicted_y_values, marker='x', s=200)

    #print(y_pred)
    #predicted_y_values = np.argmax(y_pred, axis=1)
    #print(predicted_y_values)
    print(W_final)
    print(b_final)

    plt.show() 




        
if __name__ == '__main__':
    print(tf.__version__)
    Test1()

