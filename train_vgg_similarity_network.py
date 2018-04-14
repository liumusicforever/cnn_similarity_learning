import numpy as np
import tensorflow as tf

from data import data_sampler
from similarity_network import similarity_net

def train():
    # configuration
    network_name = 'similarity_net'
    train_root = '/home/share/data/dessert/classification/training'
    eval_root = '/home/share/data/dessert/classification/validation'
    pretrained =  None
    batch_size = 24
    num_iteration = 200
    learning_rate = 0.00001
    start_epoch = 0
    end_epoch = 1000

    images = tf.placeholder("float", [None, 224, 224, 3])
    targets = tf.placeholder("float", [None,1])
    train_ops = similarity_net(images , targets , learning_rate)

    # load model
    sess = tf.Session()
    saver = tf.train.Saver()  

    init = tf.global_variables_initializer()

    # load pretrain model or initial
    if pretrained:
        saver.restore(sess , pretrained)
    else:
        init = tf.global_variables_initializer()
        sess.run(init) 

    for epoch in range(start_epoch,end_epoch,1):
        # training 
        train_iter = data_sampler(train_root,batch_size,num_iteration)
        for iteration , batch in enumerate(train_iter):
            out , train_op , loss , accuracy  = sess.run(train_ops,feed_dict = {images : batch[0] , targets : batch[1][1:]})
            if iteration % 50 == 0:
                print iteration ,  loss , accuracy
            
        # evaluation
        eval_iter = data_sampler(eval_root,batch_size,50)
        acc_collect= []
        for iteration , batch in enumerate(eval_iter):
            out , train_op , loss , accuracy  = sess.run(train_ops,feed_dict = {images : batch[0] , targets : batch[1][1:]})
            acc_collect.append(accuracy)
        acc = sum(acc_collect)/len(acc_collect)
        print ('epoch : {} , samples : {} , accuracy : {}'.format(epoch , iteration * batch_size,acc))
        # save_path = saver.save(sess, "models/{}_{}.ckpt".format(network_name , epoch))
        print ("Model saved in file: ", save_path)
    


        

train()