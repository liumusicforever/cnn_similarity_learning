import tensorflow as tf

from data import comparsions_iter , load_image
from similarity_network import similarity_net


def test():
    base_img_path = '/home/share/data/dessert/classification/training/2-2/3287eaeb-9d61-433e-849b-1bb20332f053.jpg'
    pretrained =  'models/similarity_net_51.ckpt'
    comparsion_dir = '/home/share/data/dessert/classification/comparision'
    batch_size = 5

    images = tf.placeholder("float", [None, 224, 224, 3])
    out = similarity_net(images)

    # load model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()  

    saver.restore(sess , pretrained)


    base_img = load_image(base_img_path)
    data_iter = comparsions_iter(comparsion_dir ,batch_size -1 )

    voting_dict = dict()

    for sample_no , (comparisions_batch  , clssname_batch) in enumerate(data_iter):
        batch = [base_img] + comparisions_batch
        res =  sess.run(out , feed_dict = {images : batch}).flatten().tolist()
        
        for i in range(len(clssname_batch)):
            clss = clssname_batch[i]
            acc = res[i]
            if clss  in  voting_dict:
                voting_dict[clss].append(acc)
            else:
                voting_dict.update({clss:[acc]}) 
    
    result = dict()
    for clss, accs in voting_dict.items():
        result.update({clss:sum(accs) / float(len(accs))}) 
    
    print ('Total comparsions images : {}'.format(data_iter.num_samples))
    print result
        
    
test()
    
    






