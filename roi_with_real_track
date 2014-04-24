import numpy as np
import nibabel as nib
from dipy.tracking.distances import track_roi_intersection_check
from sklearn.neighbors import KDTree
from dipy.tracking.metrics import inside_sphere
import time
import pickle

def my_inside_sphere(xyz,center,radius):
	tmp = xyz-center
	return (np.sum((tmp * tmp), axis=1)<=(radius * radius)).any()==True

subject = '213'

""" 
 Make a dictionary with the subject ids and cordinate of ROI for Left CST
"""

subject_voxels = {'101': (np.array([61.5,66.5,20.5], dtype=np.float32), np.array([55.5,66.5,40.5], dtype=np.float32)),
	          '103': (np.array([61.5,62.5,20.5], dtype=np.float32), np.array([54.5,61.5,39.5], dtype=np.float32)),
                  '105': (np.array([62.5,66.5,12.5], dtype=np.float32), np.array([54.5,66.5,35.5], dtype=np.float32)), 
                  '104': (np.array([59.5,64.5,19.5], dtype=np.float32), np.array([54.5,63.5,36.5], dtype=np.float32)), 
                  '107': (np.array([61.5,67.5,15.5], dtype=np.float32), np.array([54.5,69.5,35.5], dtype=np.float32)), 
                  '109': (np.array([61.5,64.5,12.5], dtype=np.float32), np.array([53.5,62.5,37.5], dtype=np.float32)),  
                  '111': (np.array([60.5,65.5,15.5], dtype=np.float32), np.array([54.5,70.5,38.5], dtype=np.float32)),   
                  '112': (np.array([59.5,65.5,22.5], dtype=np.float32), np.array([54.5,64.5,34.5], dtype=np.float32)), 
                  '113': (np.array([61.5,56.5,23.5], dtype=np.float32), np.array([53.5,58.5,41.5], dtype=np.float32)),
                  '201': (np.array([59.5,66.5,20.5], dtype=np.float32), np.array([53.5,66.5,38.5], dtype=np.float32)), 
                  '202': (np.array([61.5,64.5,22.5], dtype=np.float32), np.array([53.5,66.5,38.5], dtype=np.float32)), 
                  '203': (np.array([59.5,65.5,19.5], dtype=np.float32), np.array([53.5,64.5,37.5], dtype=np.float32)), 
                  '204': (np.array([62.5,68.5,13.5], dtype=np.float32), np.array([54.5,68.5,35.5], dtype=np.float32)), 
                  '205': (np.array([63.5,61.5,16.5], dtype=np.float32), np.array([54.5,60.5,38.5], dtype=np.float32)), 
                  '206': (np.array([60.5,67.5,20.5], dtype=np.float32), np.array([54.5,68.5,39.5], dtype=np.float32)), 
                  '207': (np.array([61.5,68.5,15.5], dtype=np.float32), np.array([53.5,67.5,34.5], dtype=np.float32)), 
                  '208': (np.array([60.5,65.5,21.5], dtype=np.float32), np.array([53.5,66.5,39.5], dtype=np.float32)), 
                  '209': (np.array([61.5,67.5,18.5], dtype=np.float32), np.array([53.5,70.5,38.5], dtype=np.float32)), 
                  '210': (np.array([63.5,64.5,20.5], dtype=np.float32), np.array([54.5,67.5,39.5], dtype=np.float32)), 
                  '212': (np.array([61.5,65.5,16.5], dtype=np.float32), np.array([54.5,65.5,38.5], dtype=np.float32)), 
                  '213': (np.array([59.5,70.5,18.5], dtype=np.float32), np.array([53.5,70.5,38.5], dtype=np.float32)),   
		 }

"""
for reading the tracks
"""
streams1,hdr1=nib.trackvis.read('/home/nusrat/dataset_trackvis/'+subject+'.trk',points_space='voxel')


tracks = np.array([s[0] for s in streams1], dtype=np.object)


si = np.concatenate([i*np.ones(len(s)) for i,s in enumerate(tracks)]).astype(np.int)

coords = np.vstack(tracks).T



def compute_intersecting(voxel, R, coords, si):
	x_idx = np.where((coords[0] >= voxel[0] - R) & (coords[0] <= (voxel[0] + R)))[0]
        y_idx = x_idx[np.where((coords[1][x_idx] >=  voxel[1] - R) & (coords[1][x_idx] <= (voxel[1] + R)))[0]]
        z_idx = y_idx[np.where((coords[2][y_idx] >=  voxel[2] - R) & (coords[2][y_idx] <= (voxel[2] + R)))[0]]

        s_idx = np.unique(si[z_idx]).astype(np.int)
        print len(s_idx)
	
	return s_idx[np.array([my_inside_sphere(si, voxel, R) for si in tracks[s_idx]])]


print 'computing intersecting streamlines.'

R = 2

""" 
   finding the streamlines which are going through both of the two ROI
""" 
intersecting = []
for voxel in subject_voxels[subject]:
	print 'voxel:', voxel
	print 'R:', R
        t0=time.time()
	intersecting.append(compute_intersecting(voxel, R, coords, si))
	print len(intersecting[-1]), 'streamlines'
        t_min=time.time()-t0
        print t_min 


common = set(intersecting[0].tolist()).intersection(set(intersecting[1].tolist()))
pickle.dump(common, open('CST'+subject+'Right','w'), protocol=pickle.HIGHEST_PROTOCOL)
print
print "Common streamlines:", len(common)


