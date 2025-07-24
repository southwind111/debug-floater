#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from shlex import join
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
import torch
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
import open3d as o3d
import cv2
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    mask_path:str
    mask_name:str
    mask: np.array
    width: int
    height: int
    is_test: bool
    focal:np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, masks_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        #! 提取key对应的相机内外参
        extr = cam_extrinsics[key]
        #! 提取key对应的相机内参
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        #! 从世界坐标系到相机坐标系的旋转矩阵R
        R = np.transpose(qvec2rotmat(extr.qvec))
        #! 从世界坐标系到相机坐标系的平移矩阵T
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            #! 通过焦距和图像宽高计算视场角Field of View
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""
        if masks_folder is not None:
            if not os.path.exists(masks_folder):
                assert False, f"Masks folder {masks_folder} does not exist"
            mask_path  = os.path.join(masks_folder, extr.name)
            mask_name = extr.name
            mask = Image.open(mask_path) #! 在cam对象构造的时候计算包围框
        else:
            mask_path = None
            mask_name = None
            mask = None
        
        #! gt_image和mask的读取推迟到cam对象的组装中
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              mask_path = mask_path, mask_name = mask_name, mask=mask, focal=intr.params,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def mask_pointcloud_with_voxelgrid(voxelgrid, pointcloud):
    #! 检查点云中有哪些点在体素内
    mask = np.array(voxelgrid.check_if_included(pointcloud.points))

    #! 保留在体素内的点云
    pointcloud.points = o3d.utility.Vector3dVector(np.asarray(pointcloud.points)[mask])
    pointcloud.colors = o3d.utility.Vector3dVector(np.asarray(pointcloud.colors)[mask])
    pointcloud.normals = o3d.utility.Vector3dVector(np.asarray(pointcloud.normals)[mask])


def getCamO3dParam(camInfo):
    W = camInfo.width
    H = camInfo.height
    focal = camInfo.focal
    
    extrinsic = torch.tensor(getWorld2View2(camInfo.R, camInfo.T)).numpy()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, focal[0], focal[1], W/2-0.5, H/2-0.5)
    param = o3d.camera.PinholeCameraParameters()
    param.extrinsic = extrinsic
    param.intrinsic = intrinsic
    
    return param

def extractPointCloud(point_cloud, train_cam_info, path):
    points = point_cloud.points
    num_points = points.shape[0]
    final_mask = np.ones(num_points, dtype=bool)
    for idx, caminfo in enumerate(train_cam_info):
        R = caminfo.R
        T = caminfo.T.reshape((3, 1))
        mask = caminfo.mask
        mask_array = np.array(mask)
        height, width = caminfo.height, caminfo.width

        fx = (width / 2) / np.tan(caminfo.FovX / 2)
        fy = (height / 2) / np.tan(caminfo.FovY / 2)
        cx = width / 2
        cy = height / 2

        points_cam = (R.T @ points.T + T).T

        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]
        z[z == 0] = 1e-6  #! 避免除0错误

        u = (fx * x / z + cx).astype(np.int32)
        v = (fy * y / z + cy).astype(np.int32)

        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        current_mask = np.zeros(num_points, dtype=bool)

        valid_indices = np.where(valid)[0]  #! 先筛选在图像范围内的点
        mask_nonzero = np.where(mask_array > 0)

        point_image = np.zeros((height, width), dtype=np.uint8)

        for i in valid_indices:
            #! 将有效点云投影像素绘制到图像上
            point_image[v[i], u[i]] = 1
            if v[i] in mask_nonzero[0] and u[i] in mask_nonzero[1]:
                current_mask[i] = True

        #! 检测如果mask有效区域触及边界了，直接将current_mask置为True，即不进行裁剪
        min_v, max_v = np.min(mask_nonzero[0]), np.max(mask_nonzero[0])
        min_u, max_u = np.min(mask_nonzero[1]), np.max(mask_nonzero[1])
        if min_v == 0 or max_v == height - 1 or min_u == 0 or max_u == width - 1:
            current_mask = np.ones(num_points, dtype=bool)

        img = Image.fromarray(point_image * 255)
        os.makedirs(os.path.join(path, "point_img"), exist_ok=True)
        img.save(os.path.join(path, "point_img", "{}".format(caminfo.image_name)))
        former_point_num = np.count_nonzero(final_mask)
        final_mask &= current_mask  #! 取交集
        print(
            f"{former_point_num} -> {np.count_nonzero(final_mask)} cropped by mask: {caminfo.image_name}"
        )

    filtered_points = points[final_mask]
    filtered_colors = point_cloud.colors[final_mask]
    filtered_normals = point_cloud.normals[final_mask]

    return BasicPointCloud(points=filtered_points, colors=filtered_colors, normals=filtered_normals)

def calcuRGBMask(target_mask_path, source_path, output_path):
    print(f"Saving RGB masks to {output_path}, cropped by {target_mask_path}, source from {source_path}")
    for filename in os.listdir(source_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {filename}")
            continue
        #! 读取原图像
        mask_path = os.path.join(target_mask_path, filename)
        image_path = os.path.join(source_path, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image.shape[2] == 4:
            image = image[:,:,:3]
        
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]    
        # cv2.imwrite(os.path.join(output_path, "mask_" + filename), mask)
        
        #! 创建白色背景图像
        white_bg = np.full_like(image, 255)
        masked = cv2.bitwise_and(image, image, mask=mask)
        inverse_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(white_bg, white_bg, mask=inverse_mask)
        final_image = cv2.add(masked, background)
        cv2.imwrite(os.path.join(output_path, filename), final_image)
        print(f"RGB mask {filename} saved...")

def extractPointCloudUsingVoxel(point_cloud_path, train_cam_info, path):
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    #todo voxel_size待确定
    origin_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=0.005)

    for idx, caminfo in enumerate(train_cam_info):
        mask = caminfo.mask #! 需要用到完整掩码
        mask_array = np.array(mask)
        if not np.any(mask_array > 0):
            print(f"Skip mask {caminfo.image_name}, no valid pixels")
            continue
        mask_nonzero = np.where(mask_array > 0)
        min_v, max_v = np.min(mask_nonzero[0]), np.max(mask_nonzero[0])
        min_u, max_u = np.min(mask_nonzero[1]), np.max(mask_nonzero[1])
        if min_v == 0 or max_v == caminfo.height - 1 or min_u == 0 or max_u == caminfo.width - 1:
            print(f"Skip mask {caminfo.image_name}")
            continue
        print(f"Carving voxels using mask {caminfo.image_name}, voxels num : {len(origin_voxel_grid.get_voxels())}")
        mask_array = np.expand_dims(mask_array, axis=-1)
        mask_array_contiguous = np.ascontiguousarray(mask_array.astype(np.float32))
        mask_o3d = o3d.geometry.Image(mask_array_contiguous)
        o3dParam = getCamO3dParam(caminfo)
        origin_voxel_grid.carve_silhouette(mask_o3d, o3dParam)

    mask_pointcloud_with_voxelgrid(origin_voxel_grid, point_cloud)
    return BasicPointCloud(points=np.asarray(point_cloud.points),
                           colors=np.asarray(point_cloud.colors),
                           normals=np.asarray(point_cloud.normals))

def readColmapSceneInfo(path, images, depths, mask_path, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    #! 组装cam_infos，包括R、T、FovY、FovX、image_path、image_name、mask、mask_path等
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, 
        cam_intrinsics=cam_intrinsics, 
        depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        masks_folder=mask_path,
        depths_folder=os.path.join(path, depths) if depths != "" else "", 
        test_cam_names_list=test_cam_names_list
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
        #! 如果有mask_path,使用mask对场景点云进行裁剪
        if mask_path is not None and os.path.exists(mask_path):
            print(f"Extracting point cloud using masks from {mask_path}")
            # pcd = extractPointCloud(pcd, train_cam_infos, path)
            pcd = extractPointCloudUsingVoxel(ply_path, train_cam_infos, path)
            #! 裁剪后将点云存储回数据集下
            #! 修改ply_path
            cropped_point_cloud_path = os.path.join(path, "sparse/0/cropped_points3D.ply")
            storePly(cropped_point_cloud_path, pcd.points, pcd.colors)
            ply_path = cropped_point_cloud_path
        else:
            print("Using original point cloud")
    except Exception as e:
        print(f"Error fetching PLY file: {e}")
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
