
# Face Clustering by Karan Singh

A custom made face clustering algorithm which uses dlib's face recognition model to get 
face embeddings and then perform clustering on it efficiently and accurately with changable parameters 


## Installing Libraries

To successfully run the code first install the required libraries using 

```bash
  pip install -r requirements.txt
```

## Inference
To run the inference use the Karan_face_cluster.py file with changable parameters:
- faces_dir *path to directory consisting faces*
- face_distance_tolerance (default 0.5)  *Tolerance value used to compare two faces*
- min_faces_cluster (default 15),  *Minimum faces to form cluster*
- percentage_for_non_identified (default 0.35)  *Percentage value for comparing non identified faces to already made clusters*

**Note: This algorithm works only on datasets comparing already cropped faces, not full images**
```bash
  python Karan_face_cluster.py --faces_dir 'test' --min_faces_cluster 20
```
## Authors

- [@Karan Singh](https://github.com/K-prog)

