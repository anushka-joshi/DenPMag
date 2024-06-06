import tensorflow as tf

class DendriteLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            n_units=100,
            act=tf.identity,
            W_init=tf.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='uniform'),
            b_init=tf.constant_initializer(value=0.0),
            name='my_dense_layer',
            branch=2,
            is_scale=False,
            is_train=False
    ):
        super(DendriteLayer, self).__init__(name=name)
        self.n_units = n_units
        self.act = act
        self.W_init = W_init
        self.b_init = b_init
        self.branch = branch
        self.is_scale = is_scale
        self.is_train = is_train

    def build(self, input_shape):
        n_in = input_shape[-1]
        self.W = self.add_weight(shape=(self.n_units * self.branch, n_in),
                                  initializer=self.W_init,
                                  trainable=True,
                                  name='W')
        self.b = self.add_weight(shape=(self.n_units,),
                                  initializer=self.b_init,
                                  trainable=True,
                                  name='b')
        self.mask = self.add_weight(shape=(self.branch * self.n_units, n_in),
                                     initializer='ones',
                                     trainable=False,
                                     name='mask')
        super(DendriteLayer, self).build(input_shape)

    def call(self, inputs):
        mask_weight = tf.multiply(self.mask, self.W)
        out = tf.tensordot(inputs, mask_weight, axes=[[1], [1]])
        h1 = out
        h1 = tf.nn.max_pool(tf.reshape(h1, shape=[-1, self.branch * self.n_units, 1, 1]),
                            ksize=[1, self.branch, 1, 1], strides=[1, self.branch, 1, 1],
                            padding='VALID', data_format="NHWC")
        outputs = tf.reshape(h1, shape=[-1, self.n_units])
        outputs = self.act(outputs + self.b)
        if self.is_scale:
            outputs = outputs * self.branch
        return outputs

    def get_config(self):
        config = super(DendriteLayer, self).get_config()
        config.update({
            "n_units": self.n_units,
            "act": self.act,
            "W_init": self.W_init,
            "b_init": self.b_init,
            "branch": self.branch,
            "is_scale": self.is_scale,
            "is_train": self.is_train
        })
        return config
