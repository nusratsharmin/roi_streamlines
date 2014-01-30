import numpy as np
import nibabel as nib
from dipy.tracking.distances import track_roi_intersection_check
from sklearn.neighbors import KDTree

subject = '103'

""" 
 Make a dictionary with the subject ids and cordinate of ROI for Left CST
"""

subject_voxels = {'101': (np.array([[68,68,20]], dtype=np.float32), np.array([[76,68,40]], dtype=np.float32)),
		  '201': (np.array([[70,67,21]], dtype=np.float32), np.array([[75,67,39]], dtype=np.float32)),
		  '103': (np.array([[69,63,20]], dtype=np.float32), np.array([[76,62,38]], dtype=np.float32)),
		 }

"""
for reading the tracks
"""
filename = '/home/nusrat/Desktop/dataset/Nifti/'+subject+'/DIFF2DEPI_EKJ_64dirs_14/DTI/tracks_dti_3M.trk'
print "Loading", filename
streams,hdr=nib.trackvis.read(filename, points_space='voxel')
tracks = np.array([s[0] for s in streams], dtype=np.object)

"""
put all the points of streamlines from full tracktography together  
"""
coords = np.vstack(tracks)
"""
make a kdt of all the points
"""
kdt=KDTree(coords)
"""
Find the maximum segment of streamlines from the whole tracktography
"""
si = np.concatenate([i*np.ones(len(s)) for i,s in enumerate(tracks)])
max_segment = np.sqrt(np.max([((s[1:] - s[:-1])**2).sum(1).max() for s in tracks]))

def compute_intersecting(voxel, R, kdt, max_segment):
   """ 
   function for finding the streamlines which are inside the specific redius for each ROI
   """     
	subset = np.unique(si[kdt.query_radius(voxel, r=R+max_segment)[0]]).astype(np.int)
	return subset[np.array([track_roi_intersection_check(s, voxel, sq_dist_thr=R**2) for s in tracks[subset]])]

print 'computing intersecting streamlines.'
R = 2.0

""" 
   finding the streamlines which are going through both of the two ROI
""" 
intersecting = []
for voxel in subject_voxels[subject]:
	print 'voxel:', voxel
	print 'R:', R
	intersecting.append(compute_intersecting(voxel, R, kdt, max_segment))
	print len(intersecting[-1]), 'streamlines'

common = set(intersecting[0].tolist()).intersection(set(intersecting[1].tolist()))
print
print "Common streamlines:", len(common)
