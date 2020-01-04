import os
import sys
sys.path.append('../')
from shutil import rmtree
from random import random,randint
from glob import glob
import tensorflow as tf
import numpy as np
from lib.baseio import open_tif

SHAPE = (16,128,128)
DATA_DIR = '/home/csdl/deli/neutu/light/data/light_dataset'
MODEL_DIR = '/home/csdl/deli/neutu/light/dis_f_r_model'
LOG_DIR = '/home/csdl/deli/neutu/light/dis_f_r_log'
BATCH_SIZE = 10
MAX_STEPS = 20000


def batch_it_test(num=50):
    real_test_file_names = glob(os.path.join(DATA_DIR,'real/test/*.tif'))
    fake_test_file_names = glob(os.path.join(DATA_DIR,'fake/test/*.tif'))
    num_real_test_file_names = len(real_test_file_names)
    num_fake_test_file_names = len(fake_test_file_names)
    while True:
        imgs = np.zeros(dtype = np.float32, shape = (num,SHAPE[0],SHAPE[1],SHAPE[2],1))
        labels = np.zeros(dtype = np.float32, shape= (num,2))

        for i in range(num):
            if random() < 0.5:
                img = open_tif(real_test_file_names[randint(0,num_real_test_file_names)-1])
                imgs[i] = img.reshape((SHAPE[0],SHAPE[1],SHAPE[2],1)).astype(np.float32)
                labels[i] = (1,0)
            else:
                img = open_tif(fake_test_file_names[randint(0,num_fake_test_file_names)-1])
                imgs[i] = img.reshape((SHAPE[0],SHAPE[1],SHAPE[2],1)).astype(np.float32)
                labels[i] = (0,1)

        yield imgs,labels


def batch_it_train():
    real_train_file_names = glob(os.path.join(DATA_DIR,'real/train/*.tif'))
    fake_train_file_names = glob(os.path.join(DATA_DIR,'fake/train/*.tif'))
    num_real_train_file_names = len(real_train_file_names)
    num_fake_train_file_names = len(fake_train_file_names)
    while True:
        imgs = np.zeros(dtype = np.float32, shape = (BATCH_SIZE,SHAPE[0],SHAPE[1],SHAPE[2],1))
        labels = np.zeros(dtype = np.float32, shape= (BATCH_SIZE,2))

        for i in range(BATCH_SIZE):
            if random() < 0.5:
                img = open_tif(real_train_file_names[randint(0,num_real_train_file_names)-1])
                imgs[i] = img.reshape((SHAPE[0],SHAPE[1],SHAPE[2],1)).astype(np.float32)
                labels[i] = (1,0)
            else:
                img = open_tif(fake_train_file_names[randint(0,num_fake_train_file_names)-1])
                imgs[i] = img.reshape((SHAPE[0],SHAPE[1],SHAPE[2],1)).astype(np.float32)
                labels[i] = (0,1)

        yield imgs,labels


def graph():
    with tf.variable_scope('dis_f_r',reuse = False) as sc:
        def block(x, filters=64):
            layer = tf.layers.conv3d(x,filters,3,1,'SAME')
            layer = tf.nn.relu(layer)
            layer = tf.layers.conv3d(layer,filters,3,1,'SAME')
            return tf.nn.relu(tf.add(layer,x))

        x = tf.placeholder(tf.float32,[None,SHAPE[0],SHAPE[1],SHAPE[2],1])
        y = tf.placeholder(tf.float32,[None,2]) 
    
        layer = tf.layers.conv3d(x,64,5,2,'SAME')
        layer = tf.nn.max_pool3d(layer,(1,2,2,2,1),(1,2,2,2,1),'SAME')

        for i in range(3):
            layer = block(layer,64)

        layer = tf.layers.conv3d(layer,128,3,1,'SAME')
        for i in range(3):
            layer = block(layer,128)

        layer = tf.layers.conv3d(layer,256,3,1,'SAME')
        for i in range(3):
            layer = block(layer,256)

        '''layer = tf.layers.conv3d(layer,512,3,1,'SAME')
        for i in range(3):
            layer = block(layer,512)'''

        layer = tf.nn.avg_pool3d(layer,(1,2,2,2,1),(1,2,2,2,1),'SAME')
        layer = tf.contrib.layers.flatten(layer)
        layer = tf.layers.dense(layer,1024)
        logits = tf.layers.dense(layer,2)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss',loss)
    
        predict = tf.argmax(logits,1)
        gt = tf.argmax(y,1)

        acc = tf.reduce_mean(tf.cast(tf.equal(predict,gt),tf.float32))
        tf.summary.scalar('acc',acc)

        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

        summary = tf.summary.merge_all()
        
        rv = {}
        rv['x'] = x
        rv['y'] = y
        rv['predict'] = predict
        rv['acc'] = acc
        rv['loss'] = loss
        rv['train_op'] = train_op
        rv['summary'] = summary
        return rv


def train():
    batch_train = batch_it_train()
    batch_test = batch_it_test(100)

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if os.path.exists(LOG_DIR):
        rmtree(LOG_DIR)
        os.mkdir(LOG_DIR)

    last_acc = 0.0

    with tf.Session() as s:
        g = graph()
        saver = tf.train.Saver(max_to_keep=3)
        s.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(LOG_DIR,s.graph)

        for step in range(MAX_STEPS):
            print(step)
            imgs, labels = next(batch_train)
            s.run(g['train_op'],feed_dict = {g['x']:imgs, g['y']:labels})

            if(step %50 == 0):
                imgs, labels = next(batch_test)
                writer.add_summary(s.run(g['summary'],feed_dict = {g['x']:imgs,g['y']:labels}),step)
                acc = s.run(g['acc'],feed_dict={g['x']:imgs,g['y']:labels})
                if (acc > 0.98 and last_acc >0.98) or (step == MAX_STEPS-1):
                    saver.save(s,os.path.join(MODEL_DIR,'model.ckpt'), global_step = step+1)
                    print('final acc:',acc)
                    break
                last_acc = acc
            

def inference(x):
    with tf.Session() as s:
        g = graph()
        saver = tf.train.Saver()
        saver.restore(s,os.path.join(MODEL_DIR,'model.ckpt-1251'))
        return s.run(g['predict'],feed_dict={g['x']:x})


def test():
    it = batch_it_test(200)
    imgs,labels = next(it)
    predict = inference(imgs)
    gt = np.argmax(labels,1)
    print(np.sum(predict==gt))

if __name__ == '__main__':
    #train()
    test()
