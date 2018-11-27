# FCN图像语义分割


### 项目简介
采用Pascal2 VOC2012的数据中的语义分割数据，应用TensorFlow框架和FCN-8s模型，训练一个可以对20类物体进行语义分割的模型。

代码地址：https://github.com/koala0qoo/quiz-w10

数据集地址：https://www.tinymind.com/yh001/datasets/data-quiz-w10

运行地址：https://www.tinymind.com/executions/cfncevbn

### 文件说明
convert_fcn_dataset.py 用于生成tfrecord格式数据

train.py 用于训练及验证，包含FCN-8s实现过程

results 文件夹包含部分验证结果

### 运行结果
运行log输出：

![输入图片说明](https://github.com/koala0qoo/img/blob/master/log_w10.png?raw=true "在这里输入图片标题")

运行2800个step后的验证图片：

原图：

![输入图片说明](https://github.com/koala0qoo/img/blob/master/val_2800_img.jpg?raw=true)

标签：

![输入图片说明](https://github.com/koala0qoo/img/blob/master/val_2800_annotation.jpg?raw=true "在这里输入图片标题")

预测：

![输入图片说明](https://github.com/koala0qoo/img/blob/master/val_2800_prediction.jpg?raw=true "在这里输入图片标题")

CRF后的预测：

![输入图片说明](https://github.com/koala0qoo/img/blob/master/val_2800_prediction_crfed.jpg?raw=true "在这里输入图片标题")

### FCN-8s实现过程
1. 进行一些初始设置。由于这里需要实现FCN-8s，upsample_factor设置为8；分类数为21。
```
upsample_factor = 8
number_of_classes = 21
```

2. 定义要使用的分类网络（VGG），用于对特征进行提取。这里的VGG网络已经将后面的全连接层改为卷积层，设置时注意要采用same padding。
```
# Define the model that we want to use
with slim.arg_scope(vgg.vgg_arg_scope()):
    logits, end_points = vgg.vgg_16(image_tensor,
                                    num_classes=number_of_classes,
                                    is_training=is_training_placeholder,
                                    spatial_squeeze=False,
                                    fc_conv_padding='SAME')
```
3. 计算最终上采样后输出logits的尺寸。
```
downsampled_logits_shape = tf.shape(logits)

img_shape = tf.shape(image_tensor)

# Calculate the ouput size of the upsampled tensor
# The shape should be batch_size X width X height X num_classes
upsampled_logits_shape = tf.stack([
                                  downsampled_logits_shape[0],
                                  img_shape[1],
                                  img_shape[2],
                                  downsampled_logits_shape[3]
                                  ])
```
4. 这里取出原网络中第三次和第四次池化后的feature map，并分别通过1x1的卷积核对其中每个像素进行分类操作，用于辅助上采样过程。
```
pool4_feature = end_points['vgg_16/pool4']
pool3_feature = end_points['vgg_16/pool3']

with tf.variable_scope('vgg_16/fc8'):
    aux_logits_16s = slim.conv2d(pool4_feature, number_of_classes, [1, 1],
                                 activation_fn=None,
                                 weights_initializer=tf.zeros_initializer,
                                 scope='conv_pool4')
    aux_logits_8s = slim.conv2d(pool3_feature, number_of_classes, [1, 1],
                                 activation_fn=None,
                                 weights_initializer=tf.zeros_initializer,
                                 scope='conv_pool3')
```
5. 计算两倍上采样的反卷积核的大小，并使用双线性插值初始化反卷积核，对输出层的logits进行两倍上采样，并与第四次池化后分类输出的aux_logits相加（elementwise addition）。
```
# Perform the upsampling of logits
upsample_filter_np_x2 = bilinear_upsample_weights(2,  # upsample_factor,
                                                  number_of_classes)

upsample_filter_tensor_x2_1 = tf.Variable(upsample_filter_np_x2, name='vgg_16/fc8/t_conv_x2_1')

upsampled_logits = tf.nn.conv2d_transpose(logits, upsample_filter_tensor_x2_1,
                                          output_shape=tf.shape(aux_logits_16s),
                                          strides=[1, 2, 2, 1],
                                          padding='SAME')

upsampled_logits = upsampled_logits + aux_logits_16s
```
6. 同样使用双线性插值初始化反卷积核，对上一步的upsampled_logits进行两倍上采样，并与第三次池化后分类输出的aux_logits相加（elementwise addition）。
```
# Perform the upsampling of upsampled_logits(16s to 8s)
upsample_filter_tensor_x2_2 = tf.Variable(upsample_filter_np_x2, name='vgg_16/fc8/t_conv_x2_2')

upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits, upsample_filter_tensor_x2_2,
                                          output_shape=tf.shape(aux_logits_8s),
                                          strides=[1, 2, 2, 1],
                                          padding='SAME')

upsampled_logits = upsampled_logits + aux_logits_8s
```
7. 计算8倍上采样的反卷积核的大小，再次使用双线性插值初始化反卷积核，对上一步的upsampled_logits进行8倍上采样。
```
# Perform the final upsampling
upsample_filter_np_x8 = bilinear_upsample_weights(upsample_factor,
                                                   number_of_classes)

upsample_filter_tensor_x8 = tf.Variable(upsample_filter_np_x8, name='vgg_16/fc8/t_conv_x8')
upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits, upsample_filter_tensor_x8,
                                          output_shape=upsampled_logits_shape,
                                          strides=[1, upsample_factor, upsample_factor, 1],
                                          padding='SAME')
```
8. 完成8倍的上采样后，输出的logits大小已经和原图一致。此时对ground truth进行onehot编码，与激活后的upsampled_logits逐像素比对并计算交叉熵损失。与基本的分类网络相同。
```
lbl_onehot = tf.one_hot(annotation_tensor, number_of_classes)
cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=upsampled_logits,
                                                          labels=lbl_onehot)

cross_entropy_loss = tf.reduce_mean(tf.reduce_sum(cross_entropies, axis=-1))
```


### 对FCN的理解
图像语义分割本质上也可以认为是足够稠密的目标识别，这个“稠密”一般而言需要达到像素级别，也即预测每个像素点的类别。FCN实现的这种“稠密预测”，主要通过全卷积和上采样完成。

#### 对全卷积的理解：

FCN保留了传统VGG网络的前五层，并将后面的全连接层换成了卷积层，这样做有三个意义：
- 传统VGG网络经过5次卷积+池化后，生成的是7x7的特征图，这时7x7的卷积操作实际上等价于全连接，因而可以继续使用已训练好的权重进行finetune。
- 由于没有了全连接层，网络中间输出的feature map不需要为固定大小，因此可以处理任意大小的图片。
- 通过采用same padding，无论输入图尺寸如何，最终输入分类层的都不再是特征向量而是特征图；而分类层采用1x1的卷积，网络输出不再是类别而是 heatmap，heatmap上的每一个像素对应原图的一个感受野及在该区域的分类预测，即得到了更稠密的分类结果。

#### 对Upsampling的理解：

这里的upsampling是通过双线性插值的方式对图片进行扩大。
语义分割的目的是得到像素级分类，只有像素级的分类结果才能与ground truth进行逐像素比对并计算损失。而由于在特征提取过程中经过了5次卷积+池化，图像尺寸依次缩小了2、4、8、16、32倍，上述全卷积生成的“稠密结果”也只是原图像素尺寸的1/32，需要进行32倍上采样，才能得到与原图一样的大小。这就是FCN-32s采用的方式。

#### 对Skip Architecture的理解：

仅对第5层做32倍反卷积，得到的结果是很不精确的。我们知道，较浅的卷积层感知域较小，可以学习到一些局部的特征，同时也保留了更多的位置信息；而较深的卷积层具有较大的感知域，能够学习到更加整体或抽象的特征（也是更有利于分类的特征），但却丢失了较精确的位置信息。

对FCN网络优化的目标是提高预测的密度，并同原图有更准确的位置对应（即一个预测结果尽量对应原图上较小的区域）。同时，还应要保留较深卷积层学到的更有利于分类的特征。

Skip Architecture通过利用网络不同深度输出的特征图，在不同的密度和特征层面上分别进行分类，并将深层网络和浅层网络的预测结果结合起来，或者说，是利用浅层网络保留的位置信息辅助进行逐步升采样，同时保留了较准确的定位和较好的分类能力。
