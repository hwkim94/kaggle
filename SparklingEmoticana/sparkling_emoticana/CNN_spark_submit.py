
# coding: utf-8

# In[1]:


import pyspark.sql
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import scipy.io
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sys
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession

# In[2]:


class CNN() :
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        
    def convolution(self, X_input, filters, kernel_size, strides, name, padding="SAME") :
        with tf.variable_scope(name) :
            bn = tf.layers.batch_normalization(X_input)
            conv = tf.layers.conv2d(bn, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu = tf.nn.leaky_relu(conv)
            
            return relu
            
    def build(self) :
        with tf.variable_scope(self.name) :
            ### Input
            #input : 128x126x1
            #output : 8
            self.X = tf.placeholder(tf.float32, [None, 128, 126, 1])
            self.Y = tf.placeholder(tf.float32, [None, 8])
            self.training = tf.placeholder(tf.bool)
            self.learning_rate = tf.placeholder(tf.float32)
            print(self.X.shape)
            
        ### Input Layer
        #input : 128x126x1
        #output : 32x31x8
        conv1 = self.convolution(self.X, 8, [3,3], 2, "conv1")
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2, name="pool1")
        print(conv1.shape)
        print(pool1.shape)

        ### Hidden Layer1
        #input : 32x31x8
        #output : 32x31x16
        conv2 = self.convolution(conv1, 16, [3,3], 1, "conv2")
        print(conv2.shape)
            
        ### Hidden Layer2
        #input : 32x31x16
        #output : 32x31x32
        conv3 = self.convolution(conv2, 32, [3,3], 1, "conv3")
        print(conv3.shape)
            
        ### Pooling Layer2
        #input : 32x31x32
        #output : 16x15x32
        pool2 = tf.layers.max_pooling2d(conv3, pool_size=[2,2], strides=2, name="pool2")
        print(pool2.shape)
            
        ### Hidden Layer3
        #input : 16x15x32
        #output : 16x15x64
        conv4 = self.convolution(pool2, 64, [3,3], 1, "conv4")
        print(conv4.shape)
        
        ### Hidden Layer4
        #input : 16x15x64
        #output : 16x15x128
        conv5 = self.convolution(conv4, 128, [3,3], 1, "conv5")
        print(conv5.shape)
        
        ### Pooling Layer3
        #input : 16x15x128
        #output : 8x7x128
        pool3 = tf.layers.max_pooling2d(conv5, pool_size=[2,2], strides=2, name="pool3")
        print(pool3.shape)
        
        ### Hidden Layer5
        #input : 8x7x128
        #output : 8x7x32
        conv6 = self.convolution(pool3, 32, [1,1], 1, "conv6")
        print(conv6.shape)
        
        with tf.variable_scope("global_avg_pooling") :
            ### global avg pooling
            #input : 8x7x32
            #output : 1x1x32
            global_avg_pooling = tf.reduce_mean(conv6, [1, 2], keep_dims=True)
            print(global_avg_pooling.shape)
        
        with tf.variable_scope("fully_connected") :
            ###Output Layer
            #input : 1x1x32
            #ouput : 8
            shape = global_avg_pooling.get_shape().as_list()
            dimension = shape[1] * shape[2] * shape[3]
            flat = tf.reshape(global_avg_pooling, shape=[-1, dimension])

            fc = tf.layers.dense(inputs=flat, units=8, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.logits = fc

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))     
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        feed_dict={self.X: x_test, self.training: training}
        
        return self.sess.run(self.logits, feed_dict=feed_dict)

    def get_accuracy(self, x_test, y_test, training=False):
        feed_dict={self.X: x_test,self.Y: y_test, self.training: training}
        
        return self.sess.run(self.accuracy, feed_dict=feed_dict)

    def train(self, x_data, y_data, learning_rate, training=True):
        feed_dict={self.X: x_data, self.Y: y_data, self.learning_rate: learning_rate, self.training: training}
        
        return self.sess.run([self.cost, self.optimizer], feed_dict=feed_dict)
    
    def evaluate(self, X_input, Y_input, batch_size=None, training=False):
        N = X_input.shape[0]
            
        total_loss = 0
        total_acc = 0
            
        for i in range(0, N, batch_size):
            X_batch = X_input[i:i + batch_size]
            Y_batch = Y_input[i:i + batch_size]
                
            feed_dict = {self.X: X_batch, self.Y: Y_batch, self.training: training}
                
            loss = self.cost
            accuracy = self.accuracy
                
            step_loss, step_acc = self.sess.run([loss, accuracy], feed_dict=feed_dict)
                
            total_loss += step_loss * X_batch.shape[0]
            total_acc += step_acc * X_batch.shape[0]
            
        total_loss /= N
        total_acc /= N
            
        return total_loss, total_acc
    
    def save(self, ver) :
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "CNN_" + str(ver) + ".ckpt")
        
        print("Model saved in path: %s" % save_path)
                 


# In[28]:


def load_wav_data(path, num =150) :
    file_lst = os.listdir(path)
    random.shuffle(file_lst)
    
    file_lst = file_lst[:num]
    
    train = []
    valid = []
    test = []
    all_data = []
    
    for file in file_lst :
        try : 
            y, sr = librosa.load(path+file)
            emotion = int(file.split("-")[2])
            actor = int(file.split("-")[6].split(".")[0])
        
            melspectrogram = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        
            if actor in [1,2] :
                valid.append((melspectrogram, emotion))
            elif actor in [3,4] :
                test.append((melspectrogram, emotion))
            else :
                train.append((melspectrogram, emotion))
            
            all_data.append((melspectrogram, emotion))
        except :
            pass
    
    return file_lst, train, valid, test, all_data


# In[4]:


def cutting(train, valid, test, all_data, size=1025, num=276) :
    result = []
    half = int(num/2)
    
    for dataset in [train, valid, test, all_data] :
        zero = np.zeros([len(dataset), size, num])
        emotion_lst = []

        idx = 0
        for spectrogram, emotion in dataset:
            mid = int(spectrogram.shape[1]/2)
            zero[idx, :, 0:len(spectrogram[0])] = spectrogram[:, mid-half:mid+half]
            emotion_lst.append(emotion-1)
            idx += 1
            
        result.append((zero, emotion_lst))
        
    return result


# In[5]:


def onehot_encoding(data, num=8) :
    return np.eye(num)[data]


# In[8]:


def making_pair(name, x) :
    return pyspark.sql.Row(file = name,
                           neutral = round(float(x[0]),2),
                           calm = round(float(x[1]),2),
                           happy = round(float(x[2]),2),
                           sad = round(float(x[3]),2),
                           angry = round(float(x[4]),2),
                           fearful = round(float(x[5]),2),
                           disgust = round(float(x[6]),2),
                           surprised = round(float(x[7]),2))
                        


#  

# In[ ]:


directory = sys.argv[1]


# In[ ]:


# melspectrogram from wav
file_lst, train, valid, test, all_data = load_wav_data(directory)
cut_train, cut_valid, cut_test, cut_all = cutting(train, valid, test, all_data, size =128 , num=126)

all_data = cut_all[0].reshape([-1, 128, 126, 1])
all_label = onehot_encoding(cut_all[1])

train = []
valid = []
test = []
cut_train = []
cut_valid = []
cut_test = []
cut_all = []


#  

# In[9]:


sess = tf.Session()

model = CNN(sess, "CNN")
model.build()


# In[10]:


ver = 2

saver = tf.train.Saver()
saver.restore(sess, "./CNN/CNN_model/ver_2/CNN_2.ckpt")


# In[30]:


print(model.get_accuracy(all_data, all_label))


# In[32]:


result = sess.run([tf.nn.softmax(model.logits)], feed_dict={model.X : all_data, model.Y : all_label, model.learning_rate:0.01, model.training:False})
result_array = np.array(result).reshape([-1, 8])


# In[33]:
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.appName("CNN").getOrCreate()
file_rdd = sc.parallelize(file_lst)
result_rdd = sc.parallelize(result_array)
rdd = file_rdd.zip(result_rdd)


# In[34]:


rdd2 = rdd.map(lambda x: making_pair(x[0], list(np.round(x[1]*100, 2))))
result_lst = rdd2.collect()
result_df = spark.createDataFrame(result_lst)


# In[35]:


result_df2 = result_df.select(["file", "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"])
result_df2.show(10)


# In[36]:


result_df2.createOrReplaceTempView("result")


# In[37]:


while True :
    print("Table's name is 'result'")
    query = inpuy("query >>> ")

    if query == "exit" :
        break
    
    try :
        spark.sql(query).show(30)
    except :
        print("Invalid SQL syntax")
        print("\n\n")
    
    

