import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.pooling import MaxPooling2D 
from keras.layers.core import Flatten
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import numpy as np
from os.path import exists, join, basename
from os import mkdir, makedirs
import time
from tqdm import tqdm

from utils import get_images, combine_images, resize_data

class DCGAN(object):
    """DCGAN model for generation images

    Available functions:
    - build_model: build model with keras and tensorflow at __init__
    - train: optimize established model
    """

    def __init__(self, config, sess=None):
        """
        config should have the following attributes

        Args:
            device: the name of the device (e.g.'/gpu:0')
            image_size(int): the size of pciture is (image_size, image_size, c_dim)
            sample_size(int): sample size of the generator
            k_h, k_w(int): the kernel size of convolutions
            d_h, d_w(int): the scale of down samnplings
            z_dim(int): the dimention of generator's input
            c_dim(int): color dimention
            dataset_name(str): e.g. 'sun'
            learning_rate(float)
            batcsh_size(int): The size of batch. that should be specified before training
            n_epoch(int): the number of training epochs
            checkpoint_dir: '/path/to/your/checkpoint'
            save_img_dir: '/path/to/your/save/directory'
        """
        if sess is None:
            sess =  tf.Session()
        self.sess = sess
        self.device = config.device
        self.image_size = config.image_size
        self.sample_size = config.sample_size
        self.image_size = config.image_size
        self.batch_size = config.batch_size
        self.n_epoch = config.n_epoch
        self.learning_rate = config.learning_rate
        self.z_dim = config.z_dim
        
        # color configuration
        self.is_color = config.is_color
        if self.is_color:
            self.c_dim = config.c_dim
        else:
            self.c_dim = 1
 
        # shape of convolution
        self.k_h, self.k_w, self.d_h, self.d_w =\
            config.k_h, config.k_w, config.d_h, config.d_w

        self.dataset_name = config.dataset_name
        self.checkpoint_dir = config.checkpoint_dir
        self.save_img_dir = config.save_img_dir
        # operation will be set in the train mode
        K.set_learning_phase(1)
        # all operators will be controlled under self.sess
        K.set_session(self.sess)
        with self.sess.as_default():
            self.build_model()

        
    def train(self, data):
        """train model
        Args:
            data: image data, whose elements are integers [0, 255]
                  its shape: (number of iamges, height, width, colordimention)
        """
        # make optimization graph
        with tf.device(self.device):
            d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
                          .minimize(self.d_loss, var_list=self.D_logit.trainable_weights)
            g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
                          .minimize(self.g_loss, var_list=self.G.trainable_weights)
        tf.initialize_all_variables().run(session=self.sess)
        
        # reshape data for training
        X_train = resize_data(data, self.image_size, self.image_size, self.c_dim, self.is_color)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        sample_images = X_train[:self.sample_size]
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            
        # make directory for saving generated images
        if not (exists(self.save_img_dir)):
            try:
                mkdir(self.save_img_dir)
            except:
                makedirs(self.save_img_dir)
        
        # train model
        count = 0
        for epoch in tqdm(range(self.n_epoch)):
            batch_indices = len(X_train) // self.batch_size
            shuffle_idx = np.arange(batch_indices)
            np.random.shuffle(shuffle_idx)
            batch_count = 0
            for idx in iter(shuffle_idx):
                batch_count += 1
                count += 1
                batch = X_train[idx*self.batch_size:(idx + 1)*self.batch_size]
                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
                            .astype(np.float32)
                    
                for _ in range(1):
                    d_optim.run(session=self.sess, feed_dict={self.images: batch_images, self.z: batch_z})
                g_optim.run(session=self.sess, feed_dict={self.z: batch_z})
                
                errD_fake = self.d_loss_fake.eval({self.z: batch_z}, session=self.sess)
                errD_real = self.d_loss_real.eval({self.images: batch_images}, session=self.sess)
                errG = self.g_loss.eval({self.z: batch_z}, session=self.sess)
                
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, batch_count, batch_indices,
                        time.time() - start_time, errD_fake+errD_real, errG))
            
            if np.mod(epoch + 1, 1) == 0:
                errD_fake = self.d_loss_fake.eval({self.z: batch_z}, session=self.sess)
                errD_real = self.d_loss_real.eval({self.images: batch_images}, session=self.sess)
                errG = self.g_loss.eval({self.z: batch_z}, session=self.sess)
                
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, batch_count, batch_indices,
                        time.time() - start_time, errD_fake+errD_real, errG))

            if np.mod(epoch + 1, 1) == 0:
                samples = self.sess.run(self.g, feed_dict={self.z: sample_z})
                
                image = combine_images(samples)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(join(self.save_img_dir, str(epoch)+".png"))

            if np.mod(epoch + 1, 5) == 0:
                self.save(self.checkpoint_dir, epoch)
    
    def build_model(self):
        print ('build model ...')
        # we will use (self.output_size, self.output_size) picture as images
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim],  name='real_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        
        with tf.device(self.device):
            self.G = self.get_generator_model()
            self.g = self.G(self.z)
            self.D_logit = self.get_discriminator_logit_model()
            self.d_logit = self.D_logit(self.images)
            self.d_fake_logit = self.D_logit(self.g)
        
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.d_logit, tf.ones_like(self.d_logit)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.d_fake_logit, tf.zeros_like(self.d_fake_logit)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.d_fake_logit, tf.ones_like(self.d_fake_logit)))
        
        self.saver = tf.train.Saver()


    def get_discriminator_logit_model(self):
        """return disriminator keras model

        to stabilize learning, we will not applay batch normalization to 
        the output layer of generator and the input layer of discriminator
        """
        leak = 0.2
        s = self.image_size
        c = 1024
        model = Sequential()
        # conv1
        model.add(Convolution2D(nb_filter=int(c/8), nb_row=self.k_w, nb_col=self.k_h,
                                subsample=(self.d_w, self.d_h), border_mode='same',
                                input_shape=(s, s, self.c_dim)))
        model.add(LeakyReLU(leak))
        # conv2
        model.add(Convolution2D(nb_filter=int(c/4), nb_row=self.k_w, nb_col=self.k_h,
                                subsample=(self.d_w, self.d_h), border_mode='same'))
        model.add(BatchNormalization(mode=2, axis=-1))
        model.add(LeakyReLU(leak))
        # conv3
        model.add(Convolution2D(nb_filter=int(c/2), nb_row=self.k_w, nb_col=self.k_h,
                                subsample=(self.d_w, self.d_h), border_mode='same'))
        model.add(BatchNormalization(mode=2, axis=-1))
        model.add(LeakyReLU(leak))
        # conv4
        # we will use the tanh for the activation function of the last layer
        model.add(Convolution2D(nb_filter=c, nb_row=self.k_w, nb_col=self.k_h,
                                subsample=(self.d_w, self.d_h), border_mode='same'))
        model.add(BatchNormalization(mode=2, axis=-1))
        model.add(LeakyReLU(leak))
        # output layer
        model.add(Flatten())
        model.add(Dense(1))
        
        return model

    def get_generator_model(self):
        """ return generator keras model

        to stabilize learning, we will not applay batch normalization to 
        the output layer of generator and the input layer of discriminator
        use transpose convolution for generator
        """
        s = int(self.image_size / 16)
        c = 1024
        model = Sequential()
        # reshape and project
        model.add(Dense(output_dim=c*s*s, input_dim=self.z_dim))
        model.add(BatchNormalization(mode=1))
        model.add(Activation('relu'))
        model.add(Reshape([s, s, c]))
        # conv1
        model.add(UpSampling2D(size=(self.d_w, self.d_h)))
        model.add(Convolution2D(nb_filter=int(c/2), nb_row=self.k_w, nb_col=self.k_h, border_mode='same'))
        model.add(BatchNormalization(mode=2, axis=-1))
        model.add(Activation('relu'))
        # conv2
        model.add(UpSampling2D(size=(self.d_w, self.d_h)))
        model.add(Convolution2D(nb_filter=int(c/4), nb_row=self.k_w, nb_col=self.k_h, border_mode='same'))
        model.add(BatchNormalization(mode=2, axis=-1))
        model.add(Activation('relu'))
        # conv3
        model.add(UpSampling2D(size=(self.d_w, self.d_h)))
        model.add(Convolution2D(nb_filter=int(c/8), nb_row=self.k_w, nb_col=self.k_h, border_mode='same'))
        model.add(BatchNormalization(mode=2, axis=-1))
        model.add(Activation('relu'))
        # conv4
        # we will use the tanh for the activation function of the last layer
        model.add(UpSampling2D(size=(self.d_w, self.d_h)))
        model.add(Convolution2D(nb_filter=self.c_dim, nb_row=self.k_w, nb_col=self.k_h, border_mode='same'))
        model.add(Activation('tanh'))
        return model

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.image_size)
        checkpoint_dir = join(checkpoint_dir, model_dir)
        try:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, join(checkpoint_dir, ckpt_name))
                return True
            else:
                return False
        except:
            return False
