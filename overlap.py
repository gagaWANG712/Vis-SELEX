import open3d as o3d
import numpy as np
import copy
import os
from multiprocessing import Process
import argparse

def txt2o3d(file_path):
    data = np.genfromtxt(file_path, delimiter=' ') 
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data)
    return point_cloud

def batch_calc_overlap(input_data, dist_val):
    target = input_data[0]
    idx_list = []
    for idx, source in enumerate(input_data[1:]):
        dists = target.compute_point_cloud_distance(source)
        dists = np.asarray(dists)
        idx = np.where(dists < dist_val)[0]
        idx_list.append(idx)
    set_list = [set(idx) for idx in idx_list]
    common_indices = set_list[0].intersection(*set_list[1:])
    common_indices = np.asarray(list(common_indices))
    overlap_pointcloud = target.select_by_index(common_indices)
    return overlap_pointcloud

def calc_overlap(input_data1, input_data2, dist_val):
    dists = input_data1.compute_point_cloud_distance(input_data2)
    dists = np.asarray(dists)
    idx = np.where(dists < dist_val)[0]
    overlap_pointcloud = input_data1.select_by_index(idx)
    return overlap_pointcloud

def pca_compute(data):
    [center, covariance] = data.compute_mean_and_covariance()
    eigenvectors, _, _ = np.linalg.svd(covariance)
    return eigenvectors, center

def pca_icp_registration(P, X, threshold):

    error = [] 
    matrax = [] 
    Up, Cp = pca_compute(P)
    Ux, Cx = pca_compute(X)
    Upcopy = copy.deepcopy(Up)
    sign1 = [1, -1, 1, 1, -1, -1, 1, -1]
    sign2 = [1, 1, -1, 1, -1, 1, -1, -1]
    sign3 = [1, 1, 1, -1, 1, -1, -1, -1]
    for nn in range(len(sign3)):
        Up[:,0] = sign1[nn]*Upcopy[:,0]
        Up[:,1] = sign2[nn]*Upcopy[:,1]
        Up[:,2] = sign3[nn]*Upcopy[:,2]
        R0 = np.dot(Ux, np.linalg.inv(Up))
        T0 = Cx-np.dot(R0, Cp)
        T = np.eye(4)
        T[:3, :3] = R0
        T[:3, 3] = T0
        T[3, 3] = 1
        trans = copy.deepcopy(P).transform(T)
        dists = trans.compute_point_cloud_distance(X)
        dists = np.asarray(dists)  
        mse = np.average(dists)
        error.append(mse)
        matrax.append(T)
    ind = error.index(min(error))  
    final_T = matrax[ind]  
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(P, X, threshold, final_T)
    print(evaluation) 
    print("Apply point-to-point ICP")
    icp_p2p = o3d.pipelines.registration.registration_icp(
        P, X, threshold, final_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000)) 
    print(icp_p2p)  
    print("Transformation is:")
    print(icp_p2p.transformation)  
    pcaregisted = copy.deepcopy(P).transform(icp_p2p.transformation)
    correspoindence_set = np.asarray(icp_p2p.correspondence_set)
    return pcaregisted, correspoindence_set 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ovlap_file", nargs='*', required=True, help=" ")
    parser.add_argument("--PDB_file", nargs='*', required=True, help=" ")
    parser.add_argument("--dist_val", type=float, default=6, help=" ")
    parser.add_argument("--PDB_dist_val", type=float, default=6, help=" ")
    args = parser.parse_args()
    input_data = []
    for i in args.ovlap_file:
        input_data.append(txt2o3d(i))
    PDB_data = []
    for j in args.PDB_file:
        PDB_data.append(txt2o3d(j))
    dist_val = args.dist_val
    PDB_val = args.PDB_dist_val

    overlap_pointcloud = batch_calc_overlap(input_data, dist_val)
    overlap_pointcloud_np = np.asarray(overlap_pointcloud.points)
    np.savetxt('overlap.txt', overlap_pointcloud_np, fmt='%.10f', delimiter=' ')
    overlap_pointcloud.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([overlap_pointcloud],
                                      window_name=f" ",
                                      width=800, height=600,
                                      left=850, top=50)
    for idx, PDB_i in enumerate(PDB_data):
        PDB_overlap = calc_overlap(PDB_i, overlap_pointcloud, PDB_val)
        cur_file_path = args.PDB_file[idx]
        parent_dir = os.path.dirname(cur_file_path)
        filename, file_extension = os.path.splitext(os.path.basename(cur_file_path))
        PDB_overlap_np = np.asarray(PDB_overlap.points)
        pcaregisted_path = os.path.join(parent_dir, filename + '-overlap' + file_extension)
        np.savetxt(pcaregisted_path, PDB_overlap_np, fmt='%.2f', delimiter=' ')
        PDB_overlap.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw_geometries([PDB_overlap, overlap_pointcloud],
                                          window_name=f"{filename}",
                                          width=800, height=600,
                                          left=850, top=50)
        overlap_pointcloud.paint_uniform_color([1, 0.65, 0])
        PDB_i.paint_uniform_color([0, 0.65, 1])
        o3d.visualization.draw_geometries([PDB_i, overlap_pointcloud], width=800, height=600)



