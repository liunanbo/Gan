import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow.contrib.layers as layers
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

df=pd.read_csv('.\\MSEUMAP_3.csv')
X=df.filter(like='pixel').values
Y=df.filter(like='GT_Label').values
Y_OH=[]
for i in Y.flatten():
    temp=[0]*10
    temp[int(i)]=1
    Y_OH.append(temp)
Y_OH=np.array(Y_OH)
random_dim=100
batch_size=64
epoch=10000
def MiniBatcher(X,Y, batch_size, shuffle=False):
    n_samples = X.shape[0]
    if shuffle:
        idx = np.random.permutation(n_samples)
    else:
        idx = list(range(n_samples))
    for k in range(int(np.ceil(n_samples / batch_size))):
        from_idx = k * batch_size
        to_idx = (k + 1) * batch_size
        yield X[idx[from_idx:to_idx]],Y[idx[from_idx:to_idx]]


def plot(samples):
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(10, 7)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)



def generator(random_input,y_Label,is_train,reuse=False):
    with tf.variable_scope('gen',reuse=reuse) as scope:
        random_input=tf.concat(values=[random_input,y_Label],axis=1)
        flat_conv1=layers.fully_connected(random_input,7*7*4*28,activation_fn=None)
        #1st convolution layer
        conv1=tf.reshape(flat_conv1,shape=[-1,7,7,4*28])
        #conv1=layers.batch_norm(conv1,is_training=is_train)
        conv1=tf.nn.relu(conv1)

        #2nd convolution layer
        conv2=layers.conv2d_transpose(conv1,2*28,5,2,activation_fn=None)
        #conv2 = layers.batch_norm(conv2, is_training=is_train)
        act2 = tf.nn.relu(conv2)

        # 3nd convolution layer
        conv3 = layers.conv2d_transpose(act2, 28, 5, 2, activation_fn=None)
        #conv3 = layers.batch_norm(conv3, is_training=is_train)
        conv3 = tf.nn.relu(conv3)

        # 4rd convolution layer
        conv4 = layers.conv2d_transpose(conv3, 1, 5, 1,activation_fn=None)
        output = tf.nn.sigmoid(conv4)
    return tf.reshape(output ,[-1,784])

def discriminator(img_input,y_Label,is_train,reuse=False):
    with tf.variable_scope('dis',reuse=reuse) as scope:
        y_Label=tf.reshape(y_Label,[-1,1,1,y_Label.get_shape()[1]])
        y_Label=tf.tile(y_Label,[1,28,28,1])
        img_input = tf.reshape(img_input, [-1, 28, 28, 1])
        input = tf.concat(values=[img_input, y_Label], axis=3)


        # 1st convolution layer
        conv1 = layers.conv2d(input,28,5,2,activation_fn=None)
        #conv1 = layers.batch_norm(conv1, is_training=is_train)
        conv1 = lrelu(conv1)

        # 2nd convolution layer
        conv2 = layers.conv2d(conv1, 28*2, 5, 2,activation_fn=None)
        #conv2 = layers.batch_norm(conv2, is_training=is_train)
        conv2 = lrelu(conv2)

        # 3nd convolution layer
        conv3 = layers.conv2d(conv2, 28 * 4, 5, 2, activation_fn=None)
        #conv3 = layers.batch_norm(conv3, is_training=is_train)
        conv3 = lrelu(conv3)
        #flatten and fully connected layer
        conv3=layers.flatten(conv3)
        logits=layers.fully_connected(conv3,1,activation_fn=None)
        return logits

random_input = tf.placeholder(tf.float32,[None,random_dim])
img_input = tf.placeholder(tf.float32,[None,784])
y_Label= tf.placeholder(tf.float32,[None,10])
is_train= tf.placeholder(tf.bool)

#Wgan-GP
fake_img=generator(random_input,y_Label,is_train)

real_result=discriminator(img_input,y_Label,is_train)
fake_result=discriminator(fake_img,y_Label,is_train,reuse=True)

#Gradient penality
LAMBDA=10
epsilon=tf.random_uniform([batch_size, 1] ,minval=0,maxval=1)
interpolates = epsilon*(img_input)+(1.-epsilon)*fake_img
grad = tf.gradients(discriminator(interpolates,y_Label,is_train,reuse=True), [interpolates])[0]
grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), reduction_indices=[1]))
grad_pen = LAMBDA * tf.reduce_mean((grad_norm - 1.)**2)



d_loss = tf.reduce_mean(fake_result) -tf.reduce_mean(real_result)+ grad_pen# This optimizes the discriminator.
g_loss = - tf.reduce_mean(fake_result)  # This optimizes the generator.

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'dis' in var.name]
g_vars = [var for var in t_vars if 'gen' in var.name]
trainer_d = tf.train.AdamOptimizer(learning_rate=2e-4,
                                beta1=0.5,beta2=0.9).minimize(d_loss, var_list=d_vars)
trainer_g = tf.train.AdamOptimizer(learning_rate=1e-4,
                                beta1=0.5,beta2=0.9).minimize(g_loss, var_list=g_vars)


sess=tf.Session(config=tf.ConfigProto(log_device_placement=False))
Saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())

print('batch size: %d , epoch num: %d' % (batch_size,epoch))
print('start training...')
noise = np.random.uniform(-1.0, 1.0, size=[70, random_dim]).astype(np.float32)
noise_Label=[]
for i in range(10):
    for _  in range(7):
        noise_Label.append(np.eye(10)[i])
noise_Label=np.array(noise_Label)
for i in range(epoch):
    d=[]
    g=[]
    MB=MiniBatcher(X,Y_OH,batch_size,shuffle=True)
    for _ in range(500):
        mbX,mbY = next(MB)

        train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
        # Update the discriminator
        _, dLoss = sess.run([trainer_d, d_loss],
                            feed_dict={random_input: train_noise,
                                       img_input: mbX, y_Label: mbY, is_train: True})
        d.append(dLoss)
        # Update the generator
        _, gLoss = sess.run([trainer_g, g_loss],
                            feed_dict={random_input: train_noise,
                                       y_Label: mbY,is_train: True})
        g.append(gLoss)


    # save check point every epoch
    if not os.path.exists('.\\model\\' ):
        os.makedirs('.\\model\\' )
    if not os.path.exists('.\\out\\' ):
        os.makedirs('.\\out\\' )
    Saver.save(sess, '.\\model\\saved_model' )
    print('Epoch %d, D_loss is:%f G_loss is:%f ' % (i, np.mean(d), np.mean(g)))

    samples = sess.run(fake_img,{random_input: noise,y_Label: noise_Label,is_train :False})
    fig = plot(samples)
    plt.savefig('out/{}.png'
                .format(str(i).zfill(3)), bbox_inches='tight')
    plt.close(fig)












