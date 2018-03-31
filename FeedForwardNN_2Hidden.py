import tensorflow as tf
import numpy as np
import pandas as pd
import cPickle as pickle

########################
### Load Training Set ##
########################

train_filename = 'train.csv'
DT = pd.read_csv(train_filename,sep=',',dtype=np.float64)
str_output = 'labels'

#Column names for Predictors
cols = DT.columns[1:-1]

#Create Train & Test Sets
X_train = DT[cols]
y_train = pd.DataFrame(DT[str_output])

########################
### Load Testing Set ###
########################

test_filename = 'test.csv'
DT = pd.read_csv(test_filename,sep=',',dtype=np.float64)
str_output = 'labels'

#Column names for Predictors
cols = DT.columns[1:-1]

#Create Train & Test Sets
X_test = DT[cols]
y_test = pd.DataFrame(DT[str_output])

##################################################
### Hidden & Output Layer Generation Function  ###
##################################################

# Definition Requires Specific Keys for Weights and Biases Dictionaries
def multilayer_perceptron(x, weights, biases,keep_var):
    # Hidden layer 1 with RELU Activation & Dropout (Post-ReLU)
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1_drop = tf.nn.dropout(layer_1, keep_var,seed=435234)
    
    # Hidden layer with RELU Activation & Dropout
    layer_2 = tf.add(tf.matmul(layer_1_drop, weights['w2']), biases['b2'])    
    layer_2 = tf.nn.relu(layer_2)
    layer_2_drop=tf.nn.dropout(layer_2, keep_var,seed=435234)
    
    # Output layer with Sigmoid/Logistic Activation 
    # Consider Modifying to Softmax for Multi-Class Classification (2+ Independent Classes)
    out_layer = tf.add(tf.matmul(layer_2_drop, weights['w_out']) , biases['b_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

###########################################################
# Define Tuning Parameters, Loss Function, and Optimizer ##
###########################################################

# Tuning Parameters [USER INPUT]
out_filename= 'output'

hidden1_width = [40,50]
hidden2_width = [40,50]

dropout = [0.5]
learning_rate = [1.0,0.1,0.01,0.001]

# Optimization Parameters
max_training_iter = 1000
min_training_iter = 20
rel_tol = 1E-6
abs_tol = 1E-4

#Random Seeds
graph_seed=1234567
ops_seed=4563

# Data Descriptors
batch_size = X_train.shape[0] # Used to Define Predictor Tensor Size 
n_classes = 1                 # Number of Outputs(Probability of Match)

###################
## Launch Graph ###
###################

train_config=[]
train_LL = []
test_LL =[]

current_cost=[]
last_cost=[]
temp=[]

for width1 in hidden1_width:
    for width2 in hidden2_width:
        for rate in learning_rate:
            for drop in dropout:
                print('')
                print('H1 Width: %s , H2 Width: %s ,Learning Rate: %s , Dropout: %s'  % (width1,width2,rate,drop))

                # Network Parameters (Modifiable)
                n_hidden_1 = width1   # 1st layer number of features
                n_hidden_2 = width2   # 2nd layer number of features
                n_features = X_train.shape[1]   # Number of Input Features

                ##Graph Cleanup
                tf.reset_default_graph() 
                tf.set_random_seed(graph_seed)

                # tf Graph inputs
                x = tf.placeholder(tf.float32, shape=[batch_size,n_features],name='x')
                y = tf.placeholder(tf.float32, shape=[batch_size,1],name='y')
                keep_prob=tf.placeholder(tf.float32)

                # Weight & Bias Tensors
                weights = {
                    'w1': tf.Variable(tf.random_normal([n_features, n_hidden_1],seed=ops_seed+1.), name='w1'),
                    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],seed=ops_seed-2.), name='w2'),
                    'w_out': tf.Variable(tf.random_normal([n_hidden_2, n_classes],seed=ops_seed+3.), name='w_out')
                }

                biases = {
                    'b1': tf.Variable(tf.random_normal([n_hidden_1],seed=ops_seed-4.),  name='b1'),
                    'b2': tf.Variable(tf.random_normal([n_hidden_2],seed=ops_seed+5.), name='b2'),
                    'b_out': tf.Variable(tf.random_normal([n_classes],seed=ops_seed-.6),   name='b_out')
                }

                ### Create model, loss, and optimizer
                pred_train = multilayer_perceptron(x=x, weights=weights, biases=biases,keep_var=keep_prob)
                cost = tf.losses.log_loss(labels=y,predictions=pred_train)
                optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)
                #optimizer = tf.train.GradientDescentOptimizer(learning_rate = rate).minimize(cost)
                #optimizer = tf.train.AdagradOptimizer(learning_rate = rate).minimize(cost)

                ###############

                # Initialize the variables
                init = tf.global_variables_initializer()

                # Launch the graph(Session)
                with tf.Session() as sess:
                    sess.run(init)

                    # Training cycle
                    for iter in range(max_training_epochs):
                        # Run Optimizer and Cost Operations
                        _, current_cost =sess.run([optimizer, cost], feed_dict={x: X_train,y: y_train, keep_prob: 1.0-drop})

                        # Display Cost (LogLoss) Per Iteration
                        print("Iteration:", '%04d' % (iter+1), "cost=", "{:.9f}".format(current_cost))

                        if epoch>min_training_iter:
                            if ((np.abs(current_cost-last_cost)/last_cost <rel_tol) and \
                                 np.abs(current_cost-last_cost)<abs_tol):
                                break

                        last_cost=current_cost

                    # Cross Validation 
                    train_config.append((width1,width2,rate,drop))

                    #Final Training LL
                    train_LL.append(current_cost)

                    #Test Set LL
                    test_preds=sess.run(pred_train, feed_dict={x: X_test, y: y_test,keep_prob: 1.0})
                    temp = tf.losses.log_loss(labels=y,predictions=test_preds)
                    test_LL.append(sess.run(temp,feed_dict={y:y_test}))