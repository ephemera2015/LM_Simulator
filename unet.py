import os
import sys
from shutil import rmtree
from random import random,randint
from glob import glob
import tensorflow as tf
import numpy as np
from tiffio.baseio import open_tif,save_tif

REFINED = True
DATA_DIR = '/home/csdl/deli/neutu/light/data/4'
#MODE = 'train'
MODE = 'test'

if REFINED:
    MODEL_DIR = os.path.join(DATA_DIR,'unet_refined_model')
    LOG_DIR = os.path.join(DATA_DIR,'unet_refined_log')
else:
    MODEL_DIR = os.path.join(DATA_DIR,'unet_model')
    LOG_DIR = os.path.join(DATA_DIR,'unet_log')
 
TEST_FILES = glob(os.path.join(DATA_DIR,'real/*.tif'))

SHAPE = (8,64,64)
MAX_STEPS = 50000
BATCH_SIZE = 50
STEP_EVALUATE = 10
SEG_THRESHOLD = 0.5


def label_name(name):
    forder,file_name = os.path.split(name)
    id_,postfix = file_name.split('.')
    return  os.path.join(forder,id_+'.gt.tif')


def batch_train():
    if REFINED:
        all_names = glob(os.path.join(DATA_DIR,'refined/*.tif'))
        all_names.extend(glob(os.path.join(DATA_DIR,'seg/*.tif')))
    else:
        all_names = glob(os.path.join(DATA_DIR,'fake/fake_patches/*.tif'))
    train_file_names = [ name for name in all_names if name.find('gt')==-1 ]
    num_train_file_names = len(train_file_names)
    while True:
        imgs = np.zeros(dtype = np.float32, shape = (BATCH_SIZE,SHAPE[0],SHAPE[1],SHAPE[2],1))
        labels = np.zeros(dtype = np.float32, shape = (BATCH_SIZE,SHAPE[0],SHAPE[1],SHAPE[2],1))
        for i in range(BATCH_SIZE):
            name = train_file_names[randint(0,num_train_file_names)-1]
            imgs[i] = open_tif(name).reshape((SHAPE[0],SHAPE[1],SHAPE[2],1)).astype(np.float32)
            labels[i] = open_tif(label_name(name)).reshape((SHAPE[0],SHAPE[1],SHAPE[2],1)).astype(np.float32)
            labels[i][labels[i]==0] = 0.05
            labels[i][labels[i]==1] = 0.95
        yield imgs,labels


def define_unet(x):
    with tf.variable_scope('unet',reuse = False) as sc:
        
        layer = tf.layers.conv3d(x,32,3,1,'SAME')
        layer = tf.nn.relu(layer)
        layer = tf.layers.conv3d(layer,32,3,1,'SAME')
        layer_1 = tf.nn.relu(layer)
        layer = tf.nn.max_pool3d(layer_1,(1,2,2,2,1),(1,2,2,2,1),'SAME')

        #4*32*32
        layer = tf.layers.conv3d(layer,64,3,1,'SAME')
        layer = tf.nn.relu(layer)
        layer = tf.layers.conv3d(layer,64,3,1,'SAME')
        layer_2 = tf.nn.relu(layer)
        layer = tf.nn.max_pool3d(layer_2,(1,2,2,2,1),(1,2,2,2,1),'SAME')

        #2*16*16
        layer = tf.layers.conv3d(layer,128,3,1,'SAME')
        layer = tf.nn.relu(layer)
        layer = tf.layers.conv3d(layer,128,3,1,'SAME')
        layer_3 = tf.nn.relu(layer)
        layer = tf.nn.max_pool3d(layer_3,(1,2,2,2,1),(1,2,2,2,1),'SAME')
        
        #1*8*8
        layer = tf.layers.conv3d_transpose(layer,128,2,2,padding='SAME')
        layer = tf.concat([layer,layer_3],axis=4)
        
        layer = tf.layers.conv3d(layer,128,3,1,'SAME')
        layer = tf.nn.relu(layer)
        layer = tf.layers.conv3d(layer,128,3,1,'SAME')
        layer = tf.nn.relu(layer)

        layer = tf.layers.conv3d_transpose(layer,64,2,2,padding='SAME')
        layer = tf.concat([layer,layer_2],axis=4)

        layer = tf.layers.conv3d(layer,64,3,1,'SAME')
        layer = tf.nn.relu(layer)
        layer = tf.layers.conv3d(layer,64,3,1,'SAME')
        layer = tf.nn.relu(layer)

        layer = tf.layers.conv3d_transpose(layer,32,2,2,padding='SAME')
        layer = tf.concat([layer,layer_1],axis=4)

        layer = tf.layers.conv3d(layer,32,3,1,'SAME')
        layer = tf.nn.relu(layer)
        layer = tf.layers.conv3d(layer,32,3,1,'SAME')
        layer = tf.nn.relu(layer)

        logits = tf.nn.sigmoid(tf.layers.conv3d(layer,1,1,1,'SAME'))
        
        return logits
    

def define_graph():
    rv = {}
    
    rv['imgs'] = tf.placeholder(tf.float32,[BATCH_SIZE,*SHAPE,1])
    rv['labels'] = tf.placeholder(tf.float32,[BATCH_SIZE,*SHAPE,1])
    #weights = (rv['labels']*10+1)

    rv['logits'] = define_unet(rv['imgs'])
    
    #rv['loss'] = tf.reduce_sum(tf.square(rv['logits']-rv['labels']))
    rv['loss'] = tf.reduce_mean((-rv['labels']*tf.log(1e-4+rv['logits'])-(1-rv['labels'])*tf.log(1e-4+1-rv['logits'])))
    tf.summary.scalar('loss', rv['loss'])
    
    rv['train_op'] = tf.train.AdamOptimizer(1e-4).minimize(rv['loss'])
    
    img = tf.cast(rv['imgs'],tf.uint8)
    rv['segmentation'] = segmentation = 255*tf.cast(tf.to_int32(rv['logits']>=SEG_THRESHOLD),tf.uint8)
    
    for j in range(SHAPE[0]):
        tf.summary.image('depth_'+str(j),tf.concat([img[:,j,:,:,:],segmentation[:,j,:,:,:]],axis=2))
        
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

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(LOG_DIR,s.graph)
        saver = tf.train.Saver(max_to_keep=5)

        for step in range(MAX_STEPS):
            print('----------------',step)
            
            train_imgs, train_labels = next(batch_it_train)         
            s.run(g['train_op'], feed_dict = {g['imgs']:train_imgs, g['labels']:train_labels})
              
            if step % STEP_EVALUATE == 0:
                print('loss:', s.run(g['loss'], feed_dict = {g['imgs']:train_imgs, g['labels']:train_labels}))
                writer.add_summary(s.run(g['summary'],feed_dict = {g['imgs']:train_imgs, g['labels']:train_labels}),step)
                saver.save(s,os.path.join(MODEL_DIR,'model.ckpt'), global_step=step+1)
                    
            print('----------------')
    

def test():
    g = define_graph()
    with tf.Session() as s:
        saver = tf.train.Saver()
        saver.restore(s,tf.train.latest_checkpoint(MODEL_DIR))
        
        for file_name in TEST_FILES:
            stack = open_tif(file_name)
            print(file_name) 
            d,h,w = stack.shape
            
            segmentation = np.zeros_like(stack,dtype=np.float32)
            seg_list = []
            
            batch = np.zeros(dtype=np.float32,shape=[BATCH_SIZE,*SHAPE,1])
            cnt = 0

            for k in range(0,d,SHAPE[0]):
                for j in range(0,h,SHAPE[1]):
                    for i in range(0,w,SHAPE[2]):
                        patch = stack[k:k+SHAPE[0],j:j+SHAPE[1],i:i+SHAPE[2]]
                        r_d,r_h,r_w = patch.shape
                        batch[cnt,:r_d,:r_h,:r_w] = patch.reshape(r_d,r_h,r_w,1).astype(np.float32)
                        cnt += 1
                        if cnt == BATCH_SIZE:
                            seg = s.run(g['logits'],feed_dict = {g['imgs']:batch})
                            for ss in seg:
                                seg_list.append(ss.reshape(SHAPE))
                            cnt = 0
            if cnt > 0:
                seg = s.run(g['logits'],feed_dict = {g['imgs']:batch})
                for ss in seg:
                    seg_list.append(ss.reshape(SHAPE))

            index = 0
            for k in range(0,d,SHAPE[0]):
                for j in range(0,h,SHAPE[1]):
                    for i in range(0,w,SHAPE[2]):
                        shape = segmentation[k:k+SHAPE[0],j:j+SHAPE[1],i:i+SHAPE[2]].shape 
                        seg = seg_list[index]
                        segmentation[k:k+SHAPE[0],j:j+SHAPE[1],i:i+SHAPE[2]] = seg[:shape[0],:shape[1],:shape[2]]
                        index += 1
            
            for t in (0.3,0.5,0.9):
                seg = np.zeros(shape = segmentation.shape,dtype=np.uint8)
                seg[segmentation>=t] = 1
                save_tif(seg,file_name+'.'+str(t)+".seg.tif")


if __name__ == '__main__':
    if MODE == 'train':
        train()
    elif MODE == 'test':
        test()
