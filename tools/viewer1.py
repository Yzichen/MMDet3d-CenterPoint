import os
import os.path as osp
import numpy as np
from open3d_vis import Visualizer
import yaml
import argparse
import cv2


# https://www.rapidtables.com/web/color/RGB_Color.html
PALETTE = [[30, 144, 255],  # dodger blue
           [0, 255, 255],   # 青色
           [255, 215, 0],   # 金黄色
           [160, 32, 240],  # 紫色
           [3, 168, 158],   # 锰蓝
           [255, 0, 0]]    # 红色


def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(
        valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid


def depth2color(depth):
    gray = max(0, min((depth + 4) / 8.0, 1.0))
    max_lumi = 200
    colors = np.array(
        [[max_lumi, 0, max_lumi], [max_lumi, 0, 0], [max_lumi, max_lumi, 0],
         [0, max_lumi, 0], [0, max_lumi, max_lumi], [0, 0, max_lumi]],
        dtype=np.float32)
    if gray == 1:
        return tuple(colors[-1].tolist())
    num_rank = len(colors) - 1
    rank = np.floor(gray * num_rank).astype(np.int32)
    diff = (gray - rank / num_rank) * num_rank
    return tuple(
        (colors[rank] + (colors[rank + 1] - colors[rank]) * diff).tolist())


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    template = np.array((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2      # (8, 3)

    # (N, 1, 3) * (1, 8, 3) --> (N, 8, 3)
    corners3d = boxes3d[:, np.newaxis, 3:6] * template[np.newaxis, :, :]
    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3), boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, np.newaxis, 0:3]

    return corners3d


def show_result(data,
                result,
                score_thr=0.0,
                ):
    """Show 3D detection result by meshlab."""
    canva_size = np.array([1000, 800])
    show_range = np.array([102.4, 40.96])
    draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]

    lidar_points = data
    pred_bboxes = result[:, :7]
    pred_scores = result[:, 7]
    pred_labels = result[:, 8]

    # filter out low score bboxes for visualization
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]
        pred_labels = pred_labels[inds]
    print('num_objects: {} after score_thr: {}'.format(pred_bboxes.shape[0], 
                                                       score_thr))

    sort_ids = np.argsort(pred_scores)

    # bird-eye-view
    bev_canvas = np.zeros((1000, 800, 3), dtype=np.uint8)

    lidar_points[:, 1] = -lidar_points[:, 1]
    lidar_points[:, 0] = canva_size[0] - lidar_points[:, 0] / show_range[0] * canva_size[0]
    lidar_points[:, 1] = (lidar_points[:, 1] + show_range[1]) / show_range[1] / 2.0 * canva_size[1]

    for p in lidar_points:
        p = p[:3]
        if check_point_in_img(
                p.reshape(1, 3), bev_canvas.shape[1], bev_canvas.shape[0])[0]:
            color = depth2color(p[2])
            cv2.circle(
                bev_canvas, (int(p[1]), int(p[0])),
                radius=0,
                color=color,
                thickness=1
            )

    # box
    corners_3d = boxes_to_corners_3d(pred_bboxes)      # (N, 8, 3)
    corners_3d[:, :, 1] = -corners_3d[:, :, 1]
    bottom_corners_bev = corners_3d[:, [0, 1, 2, 3], :2]        # (N, 4, 2)
    bottom_corners_bev[..., 0] = canva_size[0] - bottom_corners_bev[..., 0] / show_range[0] * canva_size[0]
    bottom_corners_bev[..., 1] = (bottom_corners_bev[..., 1] + show_range[1]) / show_range[1] / 2.0 * canva_size[1]
    bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)   # (N, 4, 2)

    center_bev = corners_3d[:, [0, 1, 2, 3], :2].mean(axis=1)   # (N, 2)
    head_bev = corners_3d[:, [0, 1], :2].mean(axis=1)   # (N, 2)
    center_bev[:, 0] = canva_size[0] - center_bev[:, 0] / show_range[0] * canva_size[0]
    center_bev[:, 1] = (center_bev[:, 1] + show_range[1]) / show_range[1] / 2.0 * canva_size[1]
    head_bev[:, 0] = canva_size[0] - head_bev[:, 0] / show_range[0] * canva_size[0]
    head_bev[:, 1] = (head_bev[:, 1] + show_range[1]) / show_range[1] / 2.0 * canva_size[1]

    bottom_corners_bev = bottom_corners_bev.astype(np.int32)
    center_bev = center_bev.astype(np.int32)
    head_bev = head_bev.astype(np.int32)

    color = (255, 255, 0)
    for rid in sort_ids:
        score = pred_scores[rid]
        for index in draw_boxes_indexes_bev:
            cv2.line(
                bev_canvas,
                tuple(bottom_corners_bev[rid, index[0], [1, 0]]),
                tuple(bottom_corners_bev[rid, index[1], [1, 0]]),
                (color[0], color[1], color[2]),
                thickness=2,
                lineType=cv2.LINE_AA
            )
        cv2.line(
            bev_canvas,
            tuple(center_bev[rid, [1, 0]]),
            tuple(head_bev[rid, [1, 0]]),
            (color[0], color[1], color[2]),
            2,
            lineType=cv2.LINE_AA)

    # cv2.imwrite("img.png", bev_canvas)
    print(bev_canvas.shape)
    cv2.namedWindow("result")
    cv2.imshow("result", bev_canvas)
    cv2.waitKey(0)


def dataloader(cloud_path, boxes_path, load_dim):
    data = np.fromfile(cloud_path, dtype=np.float32, count=-1).reshape([-1, load_dim])
    result = np.loadtxt(boxes_path).reshape(-1, 9)
    return result, data


parser = argparse.ArgumentParser()
parser.add_argument("--score_thr", type=float, default=0.1)
args = parser.parse_args()


def main():
    with open("../bootstrap.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    result, data = dataloader(config['InputFile'], config['OutputFile'], config['LoadDim'])

    # init visualizer
    gt_bboxes = None
    # show the results
    show_result(
        data,
        result,
        score_thr=args.score_thr,
    )


if __name__ == "__main__":
    main()