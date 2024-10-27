import open3d as o3d
import numpy as np
import copy
import os
from multiprocessing import Process

def txt2o3d(file_path):
    data = np.genfromtxt(file_path, delimiter=' ') 
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data)
    return point_cloud

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
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, final_T)
    print(evaluation) 
    print("Apply point-to-point ICP")
    icp_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, final_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000)) 
    print(icp_p2p)
    print("Transformation is:")
    print(icp_p2p.transformation)
    correspoindence_set = np.asarray(icp_p2p.correspondence_set)
    return correspoindence_set, icp_p2p.transformation 

def draw_point(information):
    if (len(information) == 4):
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(information[0])
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(information[1])
        source_temp = copy.deepcopy(source)
        source_temp.transform(information[2])
        target.paint_uniform_color([0, 0.651, 0.929])     
        source.paint_uniform_color([1, 0.706, 0])
        source_temp.paint_uniform_color([0.906, 0.125, 0.296])
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"{information[3]+1}", width=1000, height=800, left=0, top=50)
        vis.add_geometry(source_temp)
        vis.add_geometry(source)
        vis.add_geometry(target)
        vis.run()
        vis.destroy_window()
    else:
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(information[0])
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(information[1])
        target.paint_uniform_color([0, 0.651, 0.929])
        source.paint_uniform_color([1, 0.706, 0])
        line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(source, target, information[3])
        line_set.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([source, target, line_set],
                                          window_name=f"{information[4]+1}",
                                          width=1000, height=800,
                                          left=900, top=50)
        o3d.visualization.draw_geometries([source],
                                          window_name=f"{information[4]+1}",
                                          width=1000, height=800,
                                          left=900, top=50)
        o3d.visualization.draw_geometries([target],
                                          window_name=f"{information[4]+1}",
                                          width=1000, height=800,
                                          left=900, top=50)

def draw_registration_result(informations):
    processes = []
    for i, information in enumerate(informations):
        source = information['source']
        target = information['target']
        point_set = information['point_set']
        transformation = information['transformation']

        draw1_list = [source, target, transformation, i]
        draw2_list = [source, target, transformation, point_set, i]

        process = Process(target=draw_point,
                          args=(draw1_list,))
        processes.append(process)
        process.start()
        process = Process(target=draw_point,
                          args=(draw2_list,))
        processes.append(process)
        process.start()
        for process in processes:
            process.join()
if __name__ == "__main__":
    import sys
    if len(sys.argv)<3:
        raise Exception("At least two parameters are required")
    input_parameters = sys.argv[1:]
    input_data = []
    for i in input_parameters:
        input_data.append(txt2o3d(i))
    output_data = [None]
    output_list = []
    target = input_data[0]
    for idx, source in enumerate(input_data[1:]):
        correspoindence_set, transformation = pca_icp_registration(source, target, 1)
        output_dict = {'source': np.asarray(source.points) , 'target': np.asarray(target.points),
                       'point_set': correspoindence_set, 'transformation': transformation}
        output_list.append(output_dict)
        source_trans = copy.deepcopy(source).transform(transformation)
        source_trans_np = np.asarray(source_trans.points)
        cur_file_path = input_parameters[idx+1]
        parent_dir = os.path.dirname(cur_file_path)
        filename, file_extension = os.path.splitext(os.path.basename(cur_file_path))
        output_path = os.path.join(parent_dir,filename + '-register' + file_extension)
        np.savetxt(output_path, source_trans_np, fmt='%.10f', delimiter=' ')
    draw_registration_result(output_list)
