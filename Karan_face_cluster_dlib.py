import dlib
import numpy as np
import cv2
import os
from skimage.io import imread_collection
import argparse
import logging

#set_logging("Face cluster by Karan Singh")  # run before defining LOGGER
LOGGER = logging.getLogger("Face cluster by Karan Singh")

pose_predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')       # importing dlib models
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def run(
        faces_dir,
        face_distance_tolerance=0.5,  # Tolerance value used to compare two faces
        min_faces_cluster=15,  # Minimum faces to form cluster
        percentage_for_non_identified=0.35,  # Percentage value for comparing non identified faces to already made clusters
):
    faces_dir=os.path.join(faces_dir,'*jpg')
    col_dir = faces_dir
    col = imread_collection(col_dir)
    vars=[get_embedding(i) for i in col]
    col=[col[i] for i in range(len(vars)) if str(type(vars[i]))!="<class 'NoneType'>"]  # extracting embeddings
    vars=[i for i in vars if str(type(i))!="<class 'NoneType'>"]
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

def encodings(img,pose_predictor,face_encoder):
    face_locations=[dlib.rectangle(left=0, top=0, right=img.shape[1], bottom=img.shape[0])]
    predictors = [pose_predictor(img, face_location) for face_location in face_locations]
    return [np.array(face_encoder.compute_face_descriptor(img, predictor, 1)) for predictor in predictors]
def get_embedding(img):
    try:
        embedding = encodings(img,pose_predictor,face_encoder)[0]
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
    parser.add_argument('--face_distance_tolerance', type=float, default=0.5, help='Tolerance value used to compare two faces')
    parser.add_argument('--min_faces_cluster', type=int, default=15, help='Minimum faces to form cluster')
    parser.add_argument('--percentage_for_non_identified', type=float, default=0.35, help='Percentage value for comparing non identified faces to already made clusters')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)