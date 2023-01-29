import numpy as np
import cv2
import os
from skimage.io import imread_collection
import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM
from tensorflow.keras import backend as K
from tensorflow.keras import Model
import argparse

class ScaleLayer(KL.Layer):                 #custom classes for importing tensorflow model layers
    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        if K.image_data_format() == 'channels_last':
            ndim = int(input_shape[-1])
        else:
            ndim = int(input_shape[1])

        self.gamma = self.add_weight(name='gamma', shape=(ndim, ))
        self.beta = self.add_weight(name='beta', shape=(ndim, ))

        super(ScaleLayer, self).build(input_shape)

    def call(self, x):
        input_shape = K.int_shape(x)

        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[bn_axis] = input_shape[bn_axis]

        broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
        broadcast_beta = K.reshape(self.beta, broadcast_shape)

        output = tf.math.multiply(x, broadcast_gamma)
        output = tf.math.add(output, broadcast_beta)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape
        
class ReshapeLayer(KL.Layer):
    def __init__(self, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        super(ReshapeLayer, self).build(input_shape)

    def call(self, x):
        s = K.shape(x)
        zeros_w = tf.zeros((s[0], 1, s[2], s[3]), tf.float32)
        r = K.concatenate([x, zeros_w], 1)

        s = K.shape(r)
        zeros_h = tf.zeros((s[0], s[1], 1, s[3]), tf.float32)
        r = K.concatenate([r, zeros_h], 2)
        return r    

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        if K.image_data_format() == 'channels_last':
            shape[1] = shape[1] + 1
            shape[2] = shape[2] + 1
        else:
            shape[2] = shape[2] + 1
            shape[3] = shape[3] + 1
        return tf.TensorShape(shape)


model_from_file = tf.keras.models.load_model('dlib_face_recognition_resnet_model_v1.h5',    #importing tensorflow model
    custom_objects={'ScaleLayer': ScaleLayer, 'ReshapeLayer': ReshapeLayer})

def run(
        faces_dir,
        face_distance_tolerance=0.18,  # Tolerance value used to compare two faces
        min_faces_cluster=10,  # Minimum faces to form cluster
        percentage_for_non_identified=0.3,  # Percentage value for comparing non identified faces to already made clusters
):
    faces_dir=os.path.join(faces_dir,'*jpg')
    col_dir = faces_dir
    col = imread_collection(col_dir)
    np_vars=[np.asarray(i, dtype='float32') for i in col]
    normalized_np_vars=[normalize_image(i) for i in np_vars]
    final_vars = list(map(lambda x: np.reshape(np.resize(x,(150,150,3)),(1,150,150,3)), normalized_np_vars))
    vars=[get_embedding_tf(i) for i in final_vars]
    vars=[(vars[i],i) for i in range(len(vars))]
    vars_dup=vars
    di={}
    non_identified=[]
    while vars_dup:             # running cluster algo step 1
        for i in vars_dup:
            #cluster_hits=[i for i, x in enumerate(distance_based_compare([i[0] for i in vars_dup],i[0])) if x == True]
            cluster_hits_bool=distance_based_compare([i[0] for i in vars_dup],i[0],face_distance_tolerance)
            cluster_hits=[(cluster_hits_bool[j],vars_dup[j][1]) for j in range(len(vars_dup)) if cluster_hits_bool[j]==True]     
            if len(cluster_hits)>min_faces_cluster:
                di[i[1]]=[]
                di[i[1]].extend(cluster_hits)
                vars_dup = [j for j in vars_dup if j[1] not in [x[1] for x in di[i[1]]]]
                break
            else:
                non_identified.append(i)
                vars_dup=[j for j in vars_dup if j[1]!=i[1]]
    print('Total',str(len(di)), 'clusters found in dataset')
    for i in non_identified:            # running cluster algo step 2
        non_arr=[]
        for j in di.items():
                checking=[vars[k[1]][0] for k in j[1]]
                #print(checking)
                gg=[1 for i in distance_based_compare(checking,i[0],face_distance_tolerance) if i==True]
                non_arr.append(sum(gg)/len(checking))
        highest_match=max(non_arr)
        if highest_match>percentage_for_non_identified:
                index=non_arr.index(highest_match)
                match_index=list(di.keys())[index]
                di[match_index].append((True,i[1]))
        else:
                non_identified=[x for x in non_identified if i[1]!=x[1]]
    path,person='Results',0                     #saving clusters
    if not os.path.exists(path):
            os.mkdir(path)
    for i in di.items():
        person+=1
        dirr=os.path.join(path, str(person))
        if not os.path.exists(dirr):
            os.mkdir(dirr)
        ctr=0
        for j in i[1]:
            img=col[j[1]]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(dirr+'/'+str(ctr)+'.jpg', img)
            ctr+=1
    print("Results saved to",path)

def normalize_image(img):    
    [R,G,B] = np.dsplit(img,img.shape[-1])
    Rx = (R - 122.782) / 256.
    Gx = (G - 117.001) / 256.
    Bx = (B - 104.298) / 256.
    return np.dstack((Rx,Gx,Bx))
def get_embedding_tf(img):
    try:
        embedding=model_from_file.predict(img, batch_size=1)[0]
        return embedding
    except:
        return None
def distance_based_compare(to_check_from,checker,face_distance_tolerance):
    if len(to_check_from)==0:
        return np.empty((0))
    result=np.linalg.norm(to_check_from - checker, axis=1)
    return list(result<=face_distance_tolerance)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--faces_dir', type=str, help='Path to directory consisting faces to cluster')
    parser.add_argument('--face_distance_tolerance', type=float, default=0.18, help='Tolerance value used to compare two faces')
    parser.add_argument('--min_faces_cluster', type=int, default=10, help='Minimum faces to form cluster')
    parser.add_argument('--percentage_for_non_identified', type=float, default=0.3, help='Percentage value for comparing non identified faces to already made clusters')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)