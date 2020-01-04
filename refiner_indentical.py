import os
import sys
sys.path.append('../')
from shutil import rmtree
from random import random,randint
from glob import glob
import tensorflow as tf
import numpy as np
from lib.baseio import open_tif,save_tif

SHAPE = (16,128,128)
DATA_DIR = '/home/csdl/deli/neutu/light/data/light_dataset'
MODEL_DIR = '/home/csdl/deli/neutu/light/refiner_model'
EXAMPLE_DIR = '/home/csdl/deli/neutu/light/refiner_example'
LOG_DIR = '/home/csdl/deli/neutu/light/refiner_log'
BATCH_SIZE = 5
MAX_STEPS = 10000


def batch_it_real():
    real_file_names = glob(os.path.join(DATA_DIR,'real/train/*.tif'))
    num_real_file_names = len(real_file_names)
    while True:
        imgs = np.zeros(dtype = np.float32, shape = (BATCH_SIZE,SHAPE[0],SHAPE[1],SHAPE[2],1))
        for i in range(BATCH_SIZE):
            img = open_tif(real_file_names[randint(0,num_real_file_names)-1])
            imgs[i] = img.reshape((SHAPE[0],SHAPE[1],SHAPE[2],1)).astype(np.float32)
        yield imgs


def batch_it_fake():
    fake_file_names = glob(os.path.join(DATA_DIR,'fake/train/*.tif'))
    num_fake_file_names = len(fake_file_names)
    while True:
        imgs = np.zeros(dtype = np.float32, shape = (BATCH_SIZE,SHAPE[0],SHAPE[1],SHAPE[2],1))
        for i in range(BATCH_SIZE):
            img = open_tif(fake_file_names[randint(0,num_fake_file_names)-1])
            imgs[i] = img.reshape((SHAPE[0],SHAPE[1],SHAPE[2],1)).astype(np.float32)
        yield imgs


def graph():
    with tf.variable_scope('refiner',reuse = False) as sc:
        def block(x, filters=64):
            layer = tf.layers.conv3d(x,filters,3,1,'SAME')
            layer = tf.nn.relu(layer)
            layer = tf.layers.conv3d(layer,filters,3,1,'SAME')
            return tf.nn.relu(tf.add(layer,x))
        
        x = tf.placeholder(tf.float32,[None,SHAPE[0],SHAPE[1],SHAPE[2],1])
        layer = tf.layers.conv3d(x,16,5,1,'SAME')
        for i in range(3):
            layer = block(layer,16)
        layer = tf.layers.conv3d(layer,32,3,1,'SAME')
        for i in range(3):
            layer = block(layer,32)
        layer = tf.layers.conv3d(layer,64,3,1,'SAME')
        for i in range(3):
            layer = block(layer,64)
        logits = tf.layers.conv3d(layer,1,1,1,'SAME')

        loss = tf.reduce_sum(tf.square(x-logits))#identical
        tf.summary.scalar('loss',loss)
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
        
        refined = tf.cast(logits,tf.uint8)
        original = tf.cast(x,tf.uint8)

        for i in range(SHAPE[0]):
            tf.summary.image('depth_'+str(i),tf.stack([original[0,i,:,:,:],refined[0,i,:,:,:]]))

        g = {}
        g['logits'] = logits
        g['x'] = x
        g['train_op'] = train_op
        g['loss'] = loss
        g['summary'] = tf.summary.merge_all()
        g['refined'] = refined
        return g
        

def train():
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if os.path.exists(LOG_DIR):
        rmtree(LOG_DIR)
        os.mkdir(LOG_DIR)
    
    if not os.path.exists(EXAMPLE_DIR):
        os.mkdir(EXAMPLE_DIR)
    #batch_real = batch_it_real()
    batch_fake = batch_it_fake()

    with tf.Session() as s:
        g = graph()
        saver = tf.train.Saver(max_to_keep=3)
        s.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(LOG_DIR,s.graph)

        for step in range(MAX_STEPS):
            print(step)
            #real_imgs = batch_real()
            fake_imgs = next(batch_fake)
            s.run(g['train_op'],feed_dict = {g['x']:fake_imgs})
            if step % 10 == 0:
                writer.add_summary(s.run(g['summary'],feed_dict={g['x']:fake_imgs}),step)
                refined = s.run(g['refined'],feed_dict = {g['x']:fake_imgs})
                i = 0
                for img in refined:
                    save_tif(fake_imgs[i],os.path.join(EXAMPLE_DIR,str(step)+'_original_'+str(i)+'.tif'))
                    save_tif(img,os.path.join(EXAMPLE_DIR,str(step)+'_refined_'+str(i)+'.tif'))
                    i += 1
                print(s.run(g['loss'],feed_dict = {g['x']:fake_imgs}))


def test():
    pass


if __name__ == '__main__':
    train()
