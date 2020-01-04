import os
import sys
from shutil import rmtree
from random import random,randint
from glob import glob
import tensorflow as tf
import numpy as np
from tiffio.baseio import open_tif,save_tif

MODEL_DIR = '/home/csdl/deli/neutu/light/fcn_r_model'
LOG_DIR = '/home/csdl/deli/neutu/light/fcn_r_log'
DATA_DIR = '/home/csdl/deli/neutu/light/refiner_example/data_refined/patch'
TEST_FILE = '/home/csdl/deli/neutu/data/gold/test/test.tif'
MODE = 'test'
SHAPE = (8,64,64)
MAX_STEPS = 50000
BATCH_SIZE = 10
STEP_EVALUATE = 10
SEG_THRESHOLD = 0.5


def label_name(name):
    forder,file_name = os.path.split(name)
    id_,postfix = file_name.split('_')
    return  os.path.join(forder,id_+'_label.tif')


def batch_train():
    all_names = glob(os.path.join(DATA_DIR,'*.tif'))
    train_file_names = [ name for name in all_names if name.find('img')!=-1 ]
    num_train_file_names = len(train_file_names)
    while True:
        imgs = np.zeros(dtype = np.float32, shape = (BATCH_SIZE,SHAPE[0],SHAPE[1],SHAPE[2],1))
        labels = np.zeros(dtype = np.float32, shape = (BATCH_SIZE,SHAPE[0],SHAPE[1],SHAPE[2],1))
        for i in range(BATCH_SIZE):
            name = train_file_names[randint(0,num_train_file_names)-1]
            imgs[i] = open_tif(name).reshape((SHAPE[0],SHAPE[1],SHAPE[2],1)).astype(np.float32)
            labels[i] = open_tif(label_name(name)).reshape((SHAPE[0],SHAPE[1],SHAPE[2],1)).astype(np.float32)
        yield imgs,labels


def define_fcn(x):
    with tf.variable_scope('fcn',reuse = False) as sc:
        def block(x, filters=64):
            layer = tf.layers.conv3d(x,filters,3,1,'SAME')
            layer = tf.nn.relu(layer)
            layer = tf.layers.conv3d(layer,filters,3,1,'SAME')
            return tf.nn.relu(tf.add(layer,x))
        
        layer = tf.layers.conv3d(x,30,5,1,'SAME')
        
        for i in range(3):
            layer = block(layer,30)
        
        layer = tf.layers.conv3d(layer,60,3,1,'SAME')
        for i in range(3):
            layer = block(layer,60)
        
        layer = tf.layers.conv3d(layer,120,3,1,'SAME')
        for i in range(3):
            layer = block(layer,120)
        
        logits = tf.layers.conv3d(layer,1,1,1,'SAME')

        return logits
    

def define_graph():
    rv = {}
    
    rv['imgs'] = tf.placeholder(tf.float32,[BATCH_SIZE,SHAPE[0],SHAPE[1],SHAPE[2],1])
    rv['labels'] = tf.placeholder(tf.float32,[BATCH_SIZE,SHAPE[0],SHAPE[1],SHAPE[2],1])
    
    rv['logits'] = define_fcn(rv['imgs'])
    
    #rv['loss']  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = rv['logits'], labels = rv['labels']))
    rv['loss'] = tf.reduce_mean(tf.square(rv['logits']-rv['labels']))
    tf.summary.scalar('loss', rv['loss'])
    
    rv['train_op'] = tf.train.AdamOptimizer(1e-4).minimize(rv['loss'])#,var_list=r_variables
    
    img = tf.cast(rv['imgs'],tf.uint8)
    segmentation = tf.cast(rv['logits'],tf.uint8)
    segmentation = 255*segmentation
    
    for i in range(BATCH_SIZE):
        for j in range(SHAPE[0]):
            tf.summary.image('img'+str(i)+'depth_'+str(j),tf.stack([img[i,j,:,:,:],segmentation[i,j,:,:,:]]))
        
    rv['summary'] = tf.summary.merge_all()
    
    return rv


def train():
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if os.path.exists(LOG_DIR):
        rmtree(LOG_DIR)
        os.mkdir(LOG_DIR)

    g = define_graph()
    batch_it_train = batch_train()
    #batch_it_test = batch_test()

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(LOG_DIR,s.graph)
        saver = tf.train.Saver(max_to_keep=5)

        for step in range(MAX_STEPS):
            print('----------------',step)
            
            train_imgs, train_labels = next(batch_it_train)         
            s.run(g['train_op'], feed_dict = {g['imgs']:train_imgs, g['labels']:train_labels})
              
            if step % STEP_EVALUATE == 0:
                #test_imgs = next(batch_it_test)
                print('loss:', s.run(g['loss'], feed_dict = {g['imgs']:train_imgs, g['labels']:train_labels}))
                writer.add_summary(s.run(g['summary'],feed_dict = {g['imgs']:train_imgs, g['labels']:train_labels}),step)
                saver.save(s,os.path.join(MODEL_DIR,'model.ckpt'), global_step=step+1)
                    
            print('----------------')
    

def test():
    g = define_graph()
    with tf.Session() as s:
        saver = tf.train.Saver()
        saver.restore(s,tf.train.latest_checkpoint(MODEL_DIR))

        stack = open_tif(TEST_FILE)
        
        d,h,w = stack.shape
        
        segmentation = np.zeros_like(stack,dtype=np.uint8)

        for k in range(0,d,SHAPE[0]):
            for j in range(0,h,SHAPE[1]):
                for i in range(0,w,SHAPE[2]):
                    print(k,j,i)
                    batch = np.zeros(dtype=np.float32,shape=[BATCH_SIZE,SHAPE[0],SHAPE[1],SHAPE[2],1])
                    patch = stack[k:k+SHAPE[0],j:j+SHAPE[1],i:i+SHAPE[2]]
                    r_d,r_h,r_w = patch.shape
                    for b in range(BATCH_SIZE):
                        batch[b,:r_d,:r_h,:r_w] = patch.reshape(r_d,r_h,r_w,1)

                    seg = s.run(g['logits'],feed_dict = {g['imgs']:batch})[0].reshape(SHAPE)
                    seg[seg>=SEG_THRESHOLD] = 1.0
                    seg[seg<SEG_THRESHOLD] = 0.0
                    seg.astype(np.uint8)
                    segmentation[k:k+r_d,j:j+r_h,i:i+r_w] = seg[:r_d,:r_h,:r_w]
        save_tif(segmentation,TEST_FILE+".seg.tif")


if __name__ == '__main__':
    if MODE == 'train':
        train()
    elif MODE == 'test':
        test()
