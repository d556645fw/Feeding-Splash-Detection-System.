import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.WARNING)
from io import StringIO
import sys
from typing import Any, Callable, Dict, List, Optional, Union
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import logging
import matplotlib.patches as patches
from matplotlib.cm import get_cmap
import numpy as np
import time
import cv2
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import tensorflow as tf
import random
import glob
import tqdm
import build_segformer
from sklearn.metrics import confusion_matrix
import argparse

tf.compat.v1.logging.set_verbosity(50)

def TensorFlowFLOPs(image_model,inputs=None):
  def try_count_flops(model,inputs_kwargs = None,output_path= None):
      if hasattr(model, 'inputs'):
          try:
              # Get input shape and set batch size to 1.
              if model.inputs:
                  inputs = [
                      tf.TensorSpec([1] + input.shape[1:], input.dtype)
                      for input in model.inputs
                  ]
                  concrete_func = tf.function(model).get_concrete_function(inputs)
              # If model.inputs is invalid, try to use the input to get concrete
              # function for model.call (subclass model).
              else:
                  concrete_func = tf.function(model.call).get_concrete_function(
                      **inputs_kwargs)
              frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)

              # Calculate FLOPs.
              run_meta = tf.compat.v1.RunMetadata()
              opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
              if output_path is not None:
                  opts['output'] = f'file:outfile={output_path}'
              else:
                  opts['output'] = 'none'
              flops = tf.compat.v1.profiler.profile(
                  graph=frozen_func.graph, run_meta=run_meta, options=opts)
              return flops.total_float_ops
          except Exception as e:  # pylint: disable=broad-except
              logging.info(
                  'Failed to count model FLOPs with error %s, because the build() '
                  'methods in keras layers were not called. This is probably because '
                  'the model was not feed any input, e.g., the max train step already '
                  'reached before this run.', e)
              return None
      return None
  model_s = 0
  def model_size(s):
    nonlocal model_s
    if s.find('Total params:') >= 0:
      model_s=int(''.join(s.split(':')[-1].split(',')))

  if inputs is not None:
    inp = tf.keras.layers.Input(shape=inputs)
    model= tf.keras.Model(inputs=inp,outputs=image_model(inp))
    flops= try_count_flops(model)
    model.summary(print_fn=model_size)
  else:
    flops= try_count_flops(image_model)
    image_model.summary(print_fn=model_size)
  #print(image_model.name,flops/1e9/2,"GFlops",model_s,' bytes')
  return flops/1e9/2, model_s/1e6  # GFLOPS, MPARAM
#


# for k, v in models.items():
#     model = v()
#     print(model.name,*TensorFlowFLOPs(model,(3,540,960) if model.name.startswith('SegFormer') else (540,960,3)))



def read_annotation_file(filename, draw=True,img_h=480,img_w=640,gray=False):
    image            = cv2.imread(filename+'.jpg')
    label_map        = np.zeros(image.shape[:2],dtype=np.uint8)

    with open(filename+'.json','r') as fp:
        data = json.load(fp)
        label_id         = {'ripple':1,'positive':1,'bg':0} 

    if draw:
        plt.figure(figsize=(12,4))
        ax = plt.gca()
        plt.subplot(1,2,1)
        plt.imshow(image[:,:,::-1])
        color = {'cage':'b','ripple':'r','positive':'r','bg':'darkgrey'}

    for s in data['shapes']:
        pt = np.array(s['points'],dtype=np.int32)
        if s['label'] in {'ripple','positive'}:
            cv2.fillPoly(label_map,[pt],(label_id[s['label']],label_id[s['label']],label_id[s['label']]))
            if draw:
                plt.fill(pt[:,0],pt[:,1],color[s['label']])             

    if draw:
        plt.axis('off')
        plt.title(filename)

    if draw:
        plt.subplot(1,2,2)
        plt.imshow(label_map.astype(np.float32),cmap='gray')
        plt.axis('off')
        plt.title('label image')

        plt.tight_layout()
        plt.show()
        
        
    image = cv2.resize(image,(img_w,img_h))
    label_map = cv2.resize(label_map,(img_w,img_h),cv2.INTER_NEAREST)
    truth = label_map.copy()
#     print(np.max(truth))
    label_map   = tf.one_hot(label_map,2).numpy()
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.concatenate([image[...,np.newaxis].copy(),image[...,np.newaxis].copy(),image[...,np.newaxis].copy()],axis=-1)
    return  image[:,:,::-1], label_map,truth
        
def data_shuffle(x_train,y_train,x_image,truthes):
    new_x=[]
    new_y=[]
    new_image=[]
    new_truthes=[]
    indexes = list(range(len(x_train)))
    random.shuffle(indexes)
    for i in indexes:
        new_x.append(x_train[i])
        new_y.append(y_train[i])
        new_image.append(x_image[i])
        new_truthes.append(truthes[i])
    return new_x,new_y,new_image,new_truthes

def SegFormerPreprocessing(img):
    mean   = np.array([0.485, 0.456, 0.406])
    std    = np.array([0.229, 0.224, 0.225])
    img    = (img - mean)/std
    return img

class WaterRipple_data(tf.keras.utils.Sequence):
    def __init__(self, x_set, batch_size, SegFormer=False):
        self.x = np.array(x_set,dtype=object)
        self.batch_size = batch_size
        skf = StratifiedKFold((len(x_set)+self.batch_size-1)//self.batch_size)
        idx = []
        self.idx   = np.array(list(range(len(x_set))))
        self.SegFormer = SegFormer
    def __len__(self):
        return (len(self.idx)+self.batch_size-1)//self.batch_size
    
    def __getitem__(self,idx):
        lst = self.idx[[i%len(self.idx) for i in range(idx*self.batch_size,idx*self.batch_size+self.batch_size)]]
        X   = []
        segy= []
        if self.SegFormer:
            for f in self.x[lst]:
                image, _, label_map = read_annotation_file(f.replace(".json",""),False,img_h,img_w)
                label_map = np.expand_dims(label_map,axis=-1)
                image = SegFormerPreprocessing(image)
                X.append(np.expand_dims(image,axis=0))
                segy.append(np.expand_dims(np.squeeze(label_map),axis=0))
            return np.transpose(np.concatenate(X,axis=0),(0,3,1,2)), np.concatenate(segy,axis=0).astype(np.float32)
        else:
            for f in self.x[lst]:
                image, label_map, _ = read_annotation_file(f.replace(".json",""),False,img_h,img_w)
                X.append(np.expand_dims(image,axis=0))
                segy.append(np.expand_dims(label_map,axis=0))

            return tf.keras.applications.vgg16.preprocess_input(np.concatenate(X,axis=0)), np.concatenate(segy,axis=0)

models={"SegFormer-b0":lambda : build_segformer.build_segformer('b0','SegFormer-b0'),
        "SegFormer-b1":lambda : build_segformer.build_segformer('b1','SegFormer-b1'),
        "SegFormer-b2":lambda : build_segformer.build_segformer('b2','SegFormer-b2'),
        "SegFormer-b3":lambda : build_segformer.build_segformer('b3','SegFormer-b3'),
        "SegFormer-b4":lambda : build_segformer.build_segformer('b4','SegFormer-b4'),
        "SegFormer-b5":lambda : build_segformer.build_segformer('b5','SegFormer-b5')}




def training(trainpath,valpath):

    x_val = []
    y_val = []
    
    train_f = glob.glob(os.path.join(trainpath,'*.json'))

    np.random.shuffle(train_f)

    train_data = WaterRipple_data(train_f,3,SegFormer = model_name.startswith('SegFormer'))
    val_f      = glob.glob(os.path.join(valpath,'*.json'))
    np.random.shuffle(val_f)
    val_data = WaterRipple_data(val_f,1,SegFormer = model_name.startswith('SegFormer'))
    
    print(len(x_val),len(y_val))


    model=models[model_name]()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00006))    
    model.summary()
        

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join('.','{}.h5'.format(model.name)),
                                                        monitor='val_loss',
                                                        save_weights_only=True,
                                                        save_best_only=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=1)

    print(tf.config.list_physical_devices('GPU'))

    t0 = time.time()

    history = model.fit(train_data,validation_data=val_data,callbacks=[model_checkpoint,early_stopping],epochs=100)

    t1 = time.time()

    print('training time:',t1-t0)


def testing(model_name,testpath,draw=False):

    model=models[model_name]()

    model.load_weights(os.path.join('.','{}.h5'.format(model.name)))

    x_test = []
    y_test = []
    test_image = []
    test_truthes=[]
    for p in tqdm.tqdm(glob.iglob(os.path.join(testpath,'*.json'))):
        image, label_map, truth= read_annotation_file(p.replace(".json",""),False,img_h,img_w)
        test_image.append(image[np.newaxis,...].copy())
        x_test.append(np.transpose(SegFormerPreprocessing(image[np.newaxis,...]).copy(),(0,3,1,2)))
        y_test.append(label_map[np.newaxis,...].copy()) 
        test_truthes.append(truth[np.newaxis,...].copy()) 

    x_test       = np.concatenate(x_test,axis=0)    
    y_test       = np.concatenate(y_test,axis=0)  
    test_image   = np.concatenate(test_image,axis=0) 
    test_truthes = np.concatenate(test_truthes,axis=0)
    print('test:',x_test.shape,y_test.shape,test_truthes.shape)

    t0 = time.time()
    for i in range(x_test.shape[0]):

        x_pred = model.predict(x_test[i:i+1]).logits
        x_pred = tf.transpose(x_pred,(0,2,3,1))
        x_pred = tf.image.resize(x_pred,(img_h,img_w))
        x_pred = tf.nn.softmax(x_pred)
        x_pred = np.where(x_pred>0.5,x_pred,0)

        image = test_image[i].copy()
        image[:,:,0] = np.clip(test_image[i][:,:,0]+255*x_pred[0,:,:,1],0,255)
    #     image[:,:,1] = np.clip(x_image[i][:,:,1]+255*x_pred[0,:,:,2],0,255)    
    #     image[:,:,2] = np.clip(x_image[i][:,:,2]+255*x_pred[0,:,:,3],0,255)
        if draw:
            plt.figure(figsize=(16,6)) 
            plt.subplot(1,2,1)  
            plt.imshow(test_image[i])
            plt.title('original pic')
            plt.axis('off')  
            plt.subplot(1,2,2)  
            plt.imshow(image)
            plt.axis('off') 
            plt.show() 


model_name     ='SegFormer-b2'

