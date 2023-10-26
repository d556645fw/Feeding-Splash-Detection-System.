import tensorflow as tf
from tensorflow.keras import backend
from transformers import TFSegformerForSemanticSegmentation


def build_segformer(backbone,name): # b0,b1,b2,b3,b4,b5
    model_checkpoint = "nvidia/mit-"+backbone
    id2label = {0: "outer", 1: "fish"}
    label2id = {label: id for id, label in id2label.items()}
    num_labels = len(id2label)

    # mean   = tf.constant([0.485, 0.456, 0.406])
    # std    = tf.constant([0.229, 0.224, 0.225])
    # inputs = tf.keras.layers.Input(shape=(img_h,img_w,3))
    # x      = inputs[:,:,:,::-1]
    # x      = (x - mean)/tf.maximum(std, backend.epsilon())
    model  = TFSegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        name = name)
    # x = tf.transpose(x,[0,3,1,2])
    # x = model(x)
    # x = tf.transpose(x,[0,2,3,1])
    # outputs = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='bilinear')(x)
    # segmodel = tf.keras.Model(inputs=inputs,outputs=outputs,name='SegFormer')
    return model

