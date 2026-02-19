"""Camera calibration and pose estimation using ArUco markers.

Pipeline:
1. Detect ArUco markers in calibration images → estimate intrinsics (K)
2. Detect markers in scene images → solve PnP → estimate camera poses (c2w)
3. Undistort images using estimated distortion coefficients
"""

import cv2
import numpy as np
import glob
import os
from typing import Tuple, List, Dict, Optional


def create_aruco_detector():
    """create an ArUco marker detector using the 4x4_50 dictionary."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def get_tag_world_coords(tag_size_m: float) -> np.ndarray:
    """get 3D world coordinates for the 4 corners of an ArUco tag.  
    assumes the tag lies in z=0 plane with origin at top-left

    Args:
        tag_size_m: Physical size of the tag in meters.

    Returns:
        Corner coordinates (4, 3) in meters.
    """
    return np.array(
        [[0, 0, 0], [tag_size_m, 0, 0], [tag_size_m, tag_size_m, 0], [0, tag_size_m, 0]],
        dtype=np.float32,
    )


def calibrate_camera(
    image_folder: str,
    tag_size_m: float = 0.06,
    pattern: str = "*.jpg",
) -> Tuple[np.ndarray, np.ndarray]:
    """Calibrate camera intrinsics from images of ArUco markers. 
    Detects ArUco markers in all images, extracts 2D-3D correspondences,
    and runs OpenCV's camera calibration.

    Args:
        image_folder: Directory containing calibration images.
        tag_size_m: Physical tag size in meters.
        pattern: Glob pattern for image files.

    Returns:
        camera_matrix: Intrinsic matrix K (3, 3).
        dist_coeffs: Distortion coefficients (1, 5).
    """
    detector = create_aruco_detector()
    obj_points_single = get_tag_world_coords(tag_size_m)

    all_world_points = []
    all_img_points = []
    image_size = None

    image_paths = glob.glob(os.path.join(image_folder, pattern))
    print(f"Found {len(image_paths)} calibration images")

    for img_path in image_paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            for i in range(len(ids)):
                all_img_points.append(corners[i].reshape(4, 2))
                all_world_points.append(obj_points_single)

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        all_world_points, all_img_points, image_size, None, None
    )

    print(f"Calibration reprojection error: {ret:.4f} px")
    print(f"Focal length: ({camera_matrix[0,0]:.1f}, {camera_matrix[1,1]:.1f}) px")
    return camera_matrix, dist_coeffs


def estimate_poses(
    image_folder: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    tag_size_m: float = 0.06,
    pattern: str = "*.jpg",
) -> List[Dict]:
    """Estimate camera-to-world poses for each image using PnP.

    Args:
        image_folder: Directory with scene images containing ArUco markers.
        camera_matrix: Intrinsic matrix K.
        dist_coeffs: Distortion coefficients.
        tag_size_m: Physical tag size in meters.
        pattern: Glob pattern for image files.

    Returns:
        List of pose dicts with keys: image_path, image, c2w, tag_id.
    """
    detector = create_aruco_detector()
    object_points = get_tag_world_coords(tag_size_m)
    image_paths = sorted(glob.glob(os.path.join(image_folder, pattern)))

    poses = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            continue

        image_points = corners[0].reshape(4, 2)
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            continue

        R, _ = cv2.Rodrigues(rvec)
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = (-R.T @ tvec).flatten()

        poses.append({
            "image_path": img_path,
            "image": img,
            "c2w": c2w,
            "tag_id": ids[0][0],
        })

    print(f"Estimated poses for {len(poses)}/{len(image_paths)} images")
    return poses


def safe_undistort(
    img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    alpha: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Undistort an image and crop black borders.

    Uses getOptimalNewCameraMatrix to compute the valid ROI,
    undistorts, crops, and adjusts the intrinsic matrix accordingly.

    Args:
        img: Input BGR image.
        camera_matrix: Original intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        alpha: Free scaling parameter (0 = no black borders).

    Returns:
        undistorted_cropped: Cropped undistorted image.
        new_K: Adjusted intrinsic matrix for the cropped image.
    """
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
    )
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_K)

    x, y, w_roi, h_roi = roi
    undistorted_cropped = undistorted[y : y + h_roi, x : x + w_roi]

    new_K[0, 2] -= x
    new_K[1, 2] -= y

    return undistorted_cropped, new_K


def prepare_dataset(
    poses: List[Dict],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    output_file: str = "dataset.npz",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    target_size: Tuple[int, int] = (200, 200),
) -> Dict:
    """Undistort, resize, and split images into train/val/test.

    Args:
        poses: List of pose dicts from estimate_poses.
        camera_matrix: Original intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        output_file: Path to save the .npz dataset.
        train_ratio: Fraction of images for training.
        val_ratio: Fraction for validation (rest goes to test).
        target_size: (width, height) to resize images to.

    Returns:
        Dict with images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal.
    """
    target_w, target_h = target_size
    processed_images = []
    c2ws = []
    focals = []

    for pose in poses:
        undistorted, new_K = safe_undistort(pose["image"], camera_matrix, dist_coeffs)
        h_crop, w_crop = undistorted.shape[:2]
        resized = cv2.resize(undistorted, target_size, interpolation=cv2.INTER_AREA)

        # Scale intrinsics for resize
        scaled_K = new_K.copy()
        scaled_K[0, :] *= target_w / w_crop
        scaled_K[1, :] *= target_h / h_crop

        processed_images.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        c2ws.append(pose["c2w"])
        focals.append(scaled_K[0, 0])

    images = np.array(processed_images, dtype=np.uint8)
    c2ws = np.array(c2ws, dtype=np.float32)
    focal = float(np.mean(focals))

    # Train/val/test split
    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx = np.random.permutation(n)

    result = {
        "images_train": images[idx[:n_train]],
        "c2ws_train": c2ws[idx[:n_train]],
        "images_val": images[idx[n_train : n_train + n_val]],
        "c2ws_val": c2ws[idx[n_train : n_train + n_val]],
        "c2ws_test": c2ws[idx[n_train + n_val :]],
        "focal": focal,
    }

    np.savez(output_file, **result)
    print(f"Saved dataset to {output_file} ({n_train} train / {n_val} val / {n - n_train - n_val} test)")
    return result
