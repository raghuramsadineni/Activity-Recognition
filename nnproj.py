import numpy as np
import pandas as pd
import tensorflow as tf
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.preprocessing import LabelEncoder
#from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split

def read_dataset():
    #input data
    d1=pd.read_csv('C:/Users/Raghuram/Desktop/Python/intern/Fdata1.csv')# change the path name
    d1['label'] = d1['label'].map({'Standing': 1, 'Walking': 0})
    msk = np.random.rand(len(d1)) < 0.8
    X=d1[msk]
    Y=d1[~msk]
    print(X)
    print(Y)
    return(X,Y)

X,Y=read_dataset()
#X,Y=shuffle(X,Y,random_state=1)

x_train=X.iloc[:,0:3]
y_train=X.pop('label')
y_train = pd.DataFrame(y_train)

x_test=Y.iloc[:,0:3]
y_test=Y.pop('label')
y_test = pd.DataFrame(y_test)


x_train=(x_train-x_train.mean())/x_train.std()
x_test=(x_test-x_test.mean())/x_test.std()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)


def multilayer_perceptron(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


n_hidden_1 = 30
n_hidden_2 = 30
n_input = x_train.shape[1]
n_classes = y_train.shape[1]

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

keep_prob = tf.placeholder("float")
training_epochs = 1000
display_step = 100
batch_size = 30


x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

predictions = multilayer_perceptron(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.3).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(len(x_train) / batch_size)
        x_batches = np.array_split(x_train, total_batch)
        y_batches = np.array_split(y_train, total_batch)
        for i in range(total_batch):
            batch_x, batch_y = x_batches[i], y_batches[i]
            _, c = sess.run([optimizer, cost], 
                            feed_dict={
                                x: batch_x, 
                                y: batch_y, 
                                keep_prob: 0.8
                            })
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test, keep_prob: 1.0}))
    print(correct_prediction)

