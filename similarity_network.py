
import tensorflow as tf

from tensorflow_vgg import vgg16


def similarity_net(images , targets = None , lr = None):
    
    vgg = vgg16.Vgg16(vgg16_npy_path = 'models/vgg16.npy')
    with tf.name_scope("vgg_body"):
        vgg.build(images)

    feat_layers =  vgg.fc7
    feat_dim = feat_layers.get_shape().as_list()


    base_feat , comparisions_feat = tf.split(feat_layers,[1,-1],0)
    

    simi_feat = tf.square(comparisions_feat - base_feat)
    out = tf.layers.dense(inputs=simi_feat, units=4096, activation=tf.nn.sigmoid)
    out = tf.layers.dense(inputs=simi_feat, units=1024, activation=tf.nn.sigmoid)
    out = tf.layers.dense(inputs=out, units=1, activation=tf.nn.sigmoid)
    
    if lr :
        with tf.name_scope("train_ops"):
            loss = tf.reduce_sum(tf.square(out - targets))
            # loss = -tf.reduce_sum(targets*tf.log(tf.clip_by_value(out,1e-10,1.0)))
            # loss = -tf.reduce_sum(targets*tf.log(out))

            train_op = tf.train.AdamOptimizer(lr).minimize(loss)
            # train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

        with tf.name_scope("evaluation"):
            prediction = tf.round(out)
            predictions_correct = tf.cast(tf.equal(prediction, targets), tf.float32)
            accuracy = tf.reduce_mean(predictions_correct)


        return out , train_op , loss , accuracy 
    else :
        return out