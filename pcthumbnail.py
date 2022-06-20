#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 18:12:09 2020

@author: Peter Szutor
"""
# Thumbnail generator for point cloud
# Original article:
# Szutor Péter: Pontfelhő karcsúsítás – simplification,
# Az elmélet és a gyakorlat találkozása a térinformatikában XI.
# Theory meets practice in GIS
import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import hdbscan
from sklearn.cluster import DBSCAN
import time

#feloldogzando XYZ file
bepcfilenev="pipes_chunk_subsam.ply"
pcd = o3d.io.read_point_cloud(bepcfilenev,format='auto')
kezd=time.time()
voxelsampleszorzo=1
vss=47
vsth=30
vsh=105
vsalfa=1.2
mincdarab=45
pcpdarab=np.asarray(pcd.points).shape[0]
if pcpdarab<500000:
    vss=47
    vsth=32
    vsh=110
    vsalfa=1.12
    mincdarab=35

if pcpdarab<100000:
    vss=47
    vsth=34
    vsh=120
    vsalfa=1.1
    mincdarab=15
if pcpdarab<50000:
    vss=49
    vsth=35
    vsh=120
    vsalfa=1.01
    mincdarab=8



print('Beolvasva:',pcpdarab)

#0 ba tarnszformalas a nezegeteshez
eminbound=(pcd.get_min_bound())
emaxbound=(pcd.get_max_bound())
trmap=[[1,0,0,-eminbound[0]],[0,1,0,-eminbound[1]],[0,0,1,0-eminbound[2]],[0,0,0,1]]
pcd.transform(trmap)
minbound=(pcd.get_min_bound())
maxbound=(pcd.get_max_bound())
pckoor=np.asarray(pcd.points)
pcid=np.arange(0,len(pckoor))

meretx=maxbound[0]-minbound[0]
merety=maxbound[1]-minbound[1]
meretz=maxbound[2]-minbound[2]
voxelhezmeret=np.amax(np.array([meretx,merety,meretz]))


#voxel jobb felbontasban
#kitoltesnagy=pcd.voxel_down_sample(voxel_size=(voxelhezmeret)/vss)
#o3d.io.write_point_cloud(bepcfilenev.split('.')[0]+'voxelsuru.xyz', kitoltesnagy)


#voxel alaulmitavetelezes
kitoltes=pcd.voxel_down_sample(voxel_size=(voxelhezmeret)/vsth)
kitolt_hossz=np.asarray(kitoltes.points).shape[0]
voxelpontok=np.asarray(kitoltes.points)

#atlagos suruseg
pcsuruseg=pckoor.shape[0]/(maxbound[0]-minbound[0])*(maxbound[1]-minbound[1])

clusterer = hdbscan.HDBSCAN(min_cluster_size=mincdarab,alpha=vsalfa).fit(np.asarray(pcd.points))
csopok=clusterer.labels_
                
#csopok=np.asarray(pcd.cluster_dbscan(2,90,True))
klaszterek=np.unique(csopok)
print('klaszterek szama:',klaszterek.shape)
csopok2=np.zeros((len(csopok),),dtype='int32')
hatarok=[]
for klasz in klaszterek:
    klaszkoor=pckoor[np.where(csopok==klasz)]
    if len(klaszkoor)>5:
        hatar = ConvexHull(klaszkoor)
        if len(hatarok)==0:
            hatarok=klaszkoor[hatar.vertices]
        else:
            ujhatar=klaszkoor[hatar.vertices]
            hatarok=np.concatenate((hatarok,ujhatar))
hatarokdarab=hatarok.shape[0]
eredetipcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pckoor+maxbound*1.3))     

hatarok=np.concatenate((hatarok,voxelpontok))
szurtpcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hatarok))     
#megvannak a hatarok,de tul suruek, ezert ezket is voxel szerint alaulmintavetelezem
szurtpcd=szurtpcd.voxel_down_sample(voxel_size=(voxelhezmeret)/vsh)
voxelpontok=np.asarray(kitoltes.points)
trmap=[[1,0,0,voxelhezmeret],[0,1,0,voxelhezmeret],[0,0,1,0+voxelhezmeret],[0,0,0,1]]
trszurtkiir=szurtpcd.transform(trmap)
eredetipcd.colors=o3d.utility.Vector3dVector(np.tile([0,1,0],(pckoor.shape[0],1)))
szurtpcd.colors=o3d.utility.Vector3dVector(np.tile([0,0,1],(hatarok.shape[0],1)))
print('Idő:',time.time()-kezd)
#o3d.io.write_point_cloud(bepcfilenev.split('.')[0]+'csakvoxel.xyz', kitoltes)
o3d.io.write_point_cloud(bepcfilenev.split('.')[0]+'thumbnail.xyz', trszurtkiir)
o3d.visualization.RenderOption.point_size=1
o3d.visualization.draw_geometries([szurtpcd,eredetipcd],window_name='DBSCAN hatarok:'+str(pckoor.shape[0])+'->'+str(hatarokdarab)+' voxel:'+str(kitolt_hossz))   
