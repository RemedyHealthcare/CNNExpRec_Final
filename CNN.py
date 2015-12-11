import tensorflow as tf
import numpy as np
import jaffe_parser
import fer_parser
import pickle

# Train parameters
num_iterations = 50 # number of epochs to run
minibatch_size = 64 # train minibatch size
test_batch_size = 128 # test batch size


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def filters_to_images(filters, num_filters, name):
    filter_images = tf.squeeze(tf.pack(tf.split(3, num_filters, filters)),[4])
       
    tf.image_summary(name, filter_images)

def model(X, filter1, filter2, filter3, weights4, weights_out,bias1, bias2, bias3, bias4, p_keep_conv, p_keep_hidden):
    layer1conv = tf.nn.relu(tf.nn.conv2d(X, filter1, [1, 1, 1, 1], 'SAME') + bias1)
    layer1pool = tf.nn.max_pool(layer1conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer1pool = tf.nn.dropout(layer1pool, p_keep_conv)

    layer2conv = tf.nn.relu(tf.nn.conv2d(layer1pool, filter2, [1, 1, 1, 1], 'SAME') + bias2)
    layer2pool = tf.nn.max_pool(layer2conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer2pool = tf.nn.dropout(layer2pool, p_keep_conv)

    layer3conv = tf.nn.relu(tf.nn.conv2d(layer2pool, filter3, [1, 1, 1, 1], 'SAME') + bias3)
    layer3pool = tf.nn.max_pool(layer3conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer3pool = tf.nn.dropout(layer3pool, p_keep_conv)

    layer3pool = tf.reshape(layer3pool, [-1, weights4.get_shape().as_list()[0]])
    layer3pool = tf.nn.dropout(layer3pool, p_keep_conv)

    layer4 = tf.nn.relu(tf.matmul(layer3pool, weights4)+ bias4)
    layer4 = tf.nn.dropout(layer4, p_keep_hidden)

    probs = tf.matmul(layer4, weights_out)
    return probs

def fprint(output):
   print output
   with open("outfile_train.txt", "a") as f:
       f.write("{}\n".format(output))

parser = fer_parser.Fer_Parser()
X_tr, Y_tr, X_te, Y_te = parser.parse_all()
flipped_X_tr = X_tr[:,:,::-1]
X_tr = np.vstack([X_tr, flipped_X_tr])
Y_tr = np.vstack([Y_tr,Y_tr])

image_dim = 48

X = tf.placeholder("float", [None, image_dim, image_dim, 1])
Y = tf.placeholder("float", [None, 7])

filter1 = init_weights([6, 6, 1, image_dim])
filter2 = init_weights([6, 6, image_dim,2*image_dim])
filter3 = init_weights([6, 6, 2*image_dim, 4*image_dim])
weights4 = init_weights([image_dim*144, 1250])
weights_out = init_weights([1250, 7])

bias1 = init_weights([image_dim])
bias2 = init_weights([2*image_dim])
bias3 = init_weights([4*image_dim])
bias4 = init_weights([1250])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
probs = model(X, filter1, filter2, filter3, weights4, weights_out, bias1, bias2, bias3, bias4, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(probs, Y)) + .02*tf.nn.l2_loss(weights4) + .02*tf.nn.l2_loss(weights_out)
train_op = tf.train.AdagradOptimizer(0.01).minimize(cost)
predict_op = tf.argmax(probs,1)

filter1_images = tf.pack(tf.split(2, image_dim, tf.reduce_mean(filter1,2)))
tf.image_summary('filter1', filter1_images, image_dim)
filter2_images =  tf.pack(tf.split(2, 2*image_dim, tf.reduce_mean(filter2,2)))
tf.image_summary('filter2', filter2_images, 2*image_dim)
filter3_images =  tf.pack(tf.split(2, 4*image_dim, tf.reduce_mean(filter3,2)))
tf.image_summary('filter3', filter3_images, 4*image_dim)
summary_op = tf.merge_all_summaries()


sess = tf.Session()
summary_writer = tf.train.SummaryWriter('summaries/', sess.graph_def)
init = tf.initialize_all_variables()
sess.run(init)



train_correctness = []
test_correctness = []
#fig = plt.figure()
#ax = fig.add_subplot(111)
#Ln, = ax.plot(train_correctness)
#Ln2, = ax.plot(test_correctness)
#ax.autoscale(enable = 'True', axis = 'both', tight = None)

#plt.ion()
#plt.show()

saver = tf.train.Saver(max_to_keep = 1)
fprint('Training model...')
fprint('')
for i in range(num_iterations):
    subbatch_count = 1
    shuffle = np.random.permutation(len(X_tr))
    X_tr = X_tr[shuffle]
    Y_tr = Y_tr[shuffle]

    for start, end in zip(range(0, len(X_tr), minibatch_size), range(64, len(X_tr),minibatch_size)):

        #p_keep_conv originally .8
        subbatch_count += 1
        sess.run(train_op, feed_dict={X:X_tr[start:end], Y:Y_tr[start:end], p_keep_conv: 0.5, p_keep_hidden: 0.5})

    saver.save(sess, 'Trained_CNN', global_step = i)

    test_indices = np.arange(len(X_te)) 
    np.random.shuffle(test_indices)
    test_indices = test_indices[0:test_batch_size]


    train_eval_indices = np.arange(len(X_tr))
    np.random.shuffle(train_eval_indices)
    train_eval_indices = train_eval_indices[0:test_batch_size]


    fprint('Iteration: ' + str(i))
    predictions = sess.run(predict_op, feed_dict={X: X_tr[train_eval_indices],
                                                     Y: Y_tr[train_eval_indices],
                                                     p_keep_conv: 1.0,
                                                     p_keep_hidden: 1.0})

    train_correctness_iter= np.mean(np.argmax(Y_tr[train_eval_indices], axis=1) == predictions)
    
    fprint('Train correctness:')
    fprint(train_correctness_iter)

    predictions = sess.run(predict_op, feed_dict={X: X_te[test_indices],
                                                     Y: Y_te[test_indices],
                                                     p_keep_conv: 1.0,
                                                     p_keep_hidden: 1.0})

    test_correctness_iter = np.mean(np.argmax(Y_te[test_indices], axis=1) == predictions)
    fprint('Test correctness:')
    fprint(test_correctness_iter)

    train_correctness.append(train_correctness_iter)
    test_correctness.append(test_correctness_iter)
    summary_op_string = sess.run(summary_op)
    summary_writer.add_summary(summary_op_string, i)
    summary_writer.flush()


    fprint('')
fprint(test_correctness)
fprint(train_correctness)
