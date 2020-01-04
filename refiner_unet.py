import os
import sys
from shutil import rmtree
from random import random,randint
from glob import glob
import tensorflow as tf
import numpy as np
from skimage.transform import resize
from tiffio.baseio import open_tif,save_tif
from tiffio.simulator import simulator, gaussianNoisyAddGray3D


MODEL_DIR = '/home/csdl/deli/neutu/light/refiner_unet_model'
EXAMPLE_DIR = '/home/csdl/deli/neutu/light/refiner_unet_example'
LOG_DIR = '/home/csdl/deli/neutu/light/refiner_unet_log'
DATA_DIR = '/home/csdl/deli/neutu/light/data/light_dataset_single'
MODE = 'test'
EXAMPLE_NUM = 10
SHAPE = (8,64,64)
MAX_STEPS = 20000
BATCH_SIZE = 5
STEP_REFINER = 5
STEP_DISCRIMINATOR = 1
STEP_EVALUATE = 50
LAMBDA = 0.5



def batch_test():
    all_names = glob(os.path.join(DATA_DIR,'test/*.tif'))
    test_file_names = [name for name in all_names if name.find('label')==-1 ]
    test_file_names = [name for name in test_file_names if name.find('refined')==-1 ]
    num_test_file_names = len(test_file_names)
    i = 0
    while i + BATCH_SIZE < num_test_file_names:
        imgs = np.zeros(dtype = np.float32, shape = (BATCH_SIZE,SHAPE[0],SHAPE[1],SHAPE[2],1))
        names = []
        for j in range(i,i+BATCH_SIZE):
            name = test_file_names[j]
            names.append(name)
            imgs[j-i] = open_tif(name).reshape((SHAPE[0],SHAPE[1],SHAPE[2],1)).astype(np.float32)
        yield imgs,names
        i += BATCH_SIZE


def batch_real():
    real_file_names = glob(os.path.join(DATA_DIR,'real/train/*.tif'))
    num_real_file_names = len(real_file_names)
    while True:
        imgs = np.zeros(dtype = np.float32, shape = (BATCH_SIZE,SHAPE[0],SHAPE[1],SHAPE[2],1))
        for i in range(BATCH_SIZE):
            img = open_tif(real_file_names[randint(0,num_real_file_names)-1])
            imgs[i] = img.reshape((SHAPE[0],SHAPE[1],SHAPE[2],1)).astype(np.float32)
        yield imgs
    

def batch_fake():
    fake_file_names = glob(os.path.join(DATA_DIR,'fake/train/*.tif'))
    num_fake_file_names = len(fake_file_names)
    while True:
        imgs = np.zeros(dtype = np.float32, shape = (BATCH_SIZE,SHAPE[0],SHAPE[1],SHAPE[2],1))
        for i in range(BATCH_SIZE):
            img = open_tif(fake_file_names[randint(0,num_fake_file_names)-1])
            imgs[i] = img.reshape((SHAPE[0],SHAPE[1],SHAPE[2],1)).astype(np.float32)
        yield imgs
    

def define_discriminator(x,reuse=False):
    with tf.variable_scope('discriminator',reuse = reuse) as sc:
        def block(x, filters=64):
            layer = tf.layers.conv3d(x,filters,3,1,'SAME')
            layer = tf.nn.relu(layer)
            layer = tf.layers.conv3d(layer,filters,3,1,'SAME')
            return tf.nn.relu(tf.add(layer,x))
    
        layer = tf.layers.conv3d(x,40,5,2,'SAME')
        layer = tf.nn.max_pool3d(layer,(1,2,2,2,1),(1,2,2,2,1),'SAME')

        for i in range(3):
            layer = block(layer,40)

        layer = tf.layers.conv3d(layer,80,3,1,'SAME')
        for i in range(3):
            layer = block(layer,80)

        layer = tf.layers.conv3d(layer,160,3,1,'SAME')
        for i in range(3):
            layer = block(layer,160)

        layer = tf.nn.avg_pool3d(layer,(1,2,2,2,1),(1,2,2,2,1),'SAME')
        layer = tf.contrib.layers.flatten(layer)
        layer = tf.layers.dense(layer,1024)
        logits = tf.layers.dense(layer,2)
        
        return logits
    
    
def define_refiner(x):
    with tf.variable_scope('refiner',reuse = False) as sc:
        layer = tf.layers.conv3d(x,32,3,1,'SAME')
        layer = tf.nn.relu(layer)
        layer = tf.layers.conv3d(layer,32,3,1,'SAME')
        layer_1 = tf.nn.relu(layer)
        layer = tf.nn.max_pool3d(layer_1,(1,2,2,2,1),(1,2,2,2,1),'SAME')
        
        layer = tf.layers.conv3d(layer,64,3,1,'SAME')
        layer = tf.nn.relu(layer)
        layer = tf.layers.conv3d(layer,64,3,1,'SAME')
        layer_2 = tf.nn.relu(layer)
        layer = tf.nn.max_pool3d(layer_2,(1,2,2,2,1),(1,2,2,2,1),'SAME')
        
        layer = tf.layers.conv3d(layer,128,3,1,'SAME')
        layer = tf.nn.relu(layer)
        layer = tf.layers.conv3d(layer,128,3,1,'SAME')
        layer_3 = tf.nn.relu(layer)
        layer = tf.nn.max_pool3d(layer_3,(1,2,2,2,1),(1,2,2,2,1),'SAME')
        
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
        layer = tf.concat([layer,layer_1,x],axis=4)
        layer = tf.layers.conv3d(layer,32,3,1,'SAME')
        layer = tf.nn.relu(layer)
        layer = tf.layers.conv3d(layer,32,3,1,'SAME')
        layer = tf.nn.relu(layer)

        logits = tf.layers.conv3d(layer,1,1,1,'SAME')

        return logits



def define_graph():
    rv = {}
    rv['real_img'] = tf.placeholder(tf.float32,[BATCH_SIZE,SHAPE[0],SHAPE[1],SHAPE[2],1])
    rv['fake_img'] = tf.placeholder(tf.float32,[BATCH_SIZE,SHAPE[0],SHAPE[1],SHAPE[2],1])
    
    rv['r_logits'] = define_refiner(rv['fake_img'])
    
    rv['d_real_logits'] = define_discriminator(rv['real_img'])
    rv['d_refine_logits'] = define_discriminator(rv['r_logits'],reuse=True)
    
    d_real_labels = tf.one_hot(tf.zeros(BATCH_SIZE,dtype=tf.int32), 2, dtype = tf.float32)
    d_refine_labels = tf.one_hot(tf.ones(BATCH_SIZE,dtype=tf.int32), 2, dtype = tf.float32)
    
    d_loss_real  = tf.nn.softmax_cross_entropy_with_logits(logits = rv['d_real_logits'], labels = d_real_labels)
    d_loss_refine = tf.nn.softmax_cross_entropy_with_logits(logits = rv['d_refine_logits'], labels = d_refine_labels)
    
    rv['d_loss_real']  = tf.reduce_mean(d_loss_real)
    rv['d_loss_refine'] = tf.reduce_mean(d_loss_refine)
    
    rv['d_loss'] = tf.add(rv['d_loss_real'], rv['d_loss_refine'])
    
    r_loss_realism = tf.nn.softmax_cross_entropy_with_logits(logits = rv['d_refine_logits'], labels = d_real_labels)
    rv['r_loss_realism'] = tf.reduce_mean(r_loss_realism)
    rv['r_loss_reg'] = LAMBDA*tf.reduce_mean(tf.square(rv['fake_img']-rv['r_logits']))
    rv['r_loss'] = tf.add(rv['r_loss_realism'], rv['r_loss_reg'])
    
    all_variables = tf.trainable_variables()
    r_variables = [var for var in all_variables if var.name.startswith('refiner')]
    d_variables = [var for var in all_variables if var.name.startswith('discriminator')]
    
    rv['r_train_op'] = tf.train.AdamOptimizer(1e-4).minimize(rv['r_loss'],var_list=r_variables)
    rv['d_train_op'] = tf.train.AdamOptimizer(1e-4).minimize(rv['d_loss'],var_list=d_variables)
    
    tf.summary.scalar('r_loss', rv['r_loss'])
    tf.summary.scalar('d_loss', rv['d_loss'])
    tf.summary.scalar('d_loss_real', rv['d_loss_real'])
    tf.summary.scalar('d_loss_refine', rv['d_loss_refine'])
    tf.summary.scalar('r_loss_realism', rv['r_loss_realism'])
    tf.summary.scalar('r_loss_reg', rv['r_loss_reg'])
    
    original = tf.cast(rv['fake_img'],tf.uint8)
    refined = tf.cast(rv['r_logits'],tf.uint8)
    for i in range(SHAPE[0]):
        tf.summary.image('depth_'+str(i),tf.stack([original[0,i,:,:,:],refined[0,i,:,:,:]]))
    
    rv['summary'] = tf.summary.merge_all()
    return rv


def train():
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if os.path.exists(LOG_DIR):
        rmtree(LOG_DIR)
        os.mkdir(LOG_DIR)
    
    if not os.path.exists(EXAMPLE_DIR):
        os.mkdir(EXAMPLE_DIR)

    g = define_graph()
    batch_it_real = batch_real()
    batch_it_fake = batch_fake()
    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(LOG_DIR,s.graph)
        saver = tf.train.Saver(max_to_keep=5)

        for step in range(MAX_STEPS):
            print('----------------',step)
            
            real_img = next(batch_it_real)
            fake_img = next(batch_it_fake)
            
            for step_d in range(STEP_DISCRIMINATOR):
                s.run(g['d_train_op'], feed_dict = {g['real_img']:real_img,g['fake_img']:fake_img})
                          
            for step_r in range(STEP_REFINER):
                s.run(g['r_train_op'], feed_dict = {g['fake_img']:fake_img})
              
            if step % STEP_EVALUATE == 0:
                print('d_loss:',s.run(g['d_loss'], feed_dict = {g['real_img']:real_img,g['fake_img']:fake_img}))
                print('r_loss:',s.run(g['r_loss'], feed_dict = {g['fake_img']:fake_img}))
                writer.add_summary(s.run(g['summary'],feed_dict = {g['real_img']:real_img,g['fake_img']:fake_img}),step)
                saver.save(s,os.path.join(MODEL_DIR,'model.ckpt'), global_step=step+1)        
            print('----------------')
    
    
def create_examples():
    with tf.Session() as s:
        g = define_graph()
        saver = tf.train.Saver()
        saver.restore(s,tf.train.latest_checkpoint(MODEL_DIR))
        
        for num_example in range(EXAMPLE_NUM):
            stack = simulator(250,5,50,0.5,0,0)
            stack = resize(stack,(80,640,640))
            label = np.zeros_like(stack)
            label[stack>0] = 1
            stack = gaussianNoisyAddGray3D(stack,15,5)

            d,h,w = stack.shape
            refined = np.zeros_like(stack,dtype=np.uint8)
            for k in range(0,d,SHAPE[0]):
                for j in range(0,h,SHAPE[1]):
                    for i in range(0,w,BATCH_SIZE*SHAPE[2]):
                        fake = np.zeros(dtype = np.float32, shape = (BATCH_SIZE,SHAPE[0],SHAPE[1],SHAPE[2],1))
                        for b in range(BATCH_SIZE):
                            patch = stack[k:k+SHAPE[0],j:j+SHAPE[1],(i+b*SHAPE[2]):(i+(b+1)*SHAPE[2])].astype(np.float32)
                            fake[b] = patch.reshape((SHAPE[0],SHAPE[1],SHAPE[2],1))
                        refined_patch = s.run(g['r_logits'],feed_dict = {g['fake_img']:fake}).astype(np.uint8)
                        for b in range(BATCH_SIZE):
                            refined[k:k+SHAPE[0],j:j+SHAPE[1],(i+b*SHAPE[2]):(i+(b+1)*SHAPE[2])] = refined_patch[b].reshape(SHAPE)
            refined[refined>240] = 0
            save_tif(stack,os.path.join(EXAMPLE_DIR,str(num_example)+'.tif'))
            save_tif(label,os.path.join(EXAMPLE_DIR,str(num_example)+'_label.tif'))
            save_tif(refined,os.path.join(EXAMPLE_DIR,str(num_example)+'_refined.tif'))

if __name__ == '__main__':
    if MODE == 'train':
        train()
    elif MODE == 'test':
        create_examples()
