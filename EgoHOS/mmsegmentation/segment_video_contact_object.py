from mmseg.apis import inference_segmentor, init_segmentor
from tqdm import tqdm
import argparse
import numpy as np
import os
import glob
import av
import cv2
import imageio
from skimage.io import imsave
import torch
from visualize import visualize_twohands_obj1, visualize_twohands_obj2
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--video", default='../testvideos/testvideo1_short.mp4', type=str)
    parser.add_argument("--output_file", default='../testvideos/testvideo1_short_result.mp4', type=str)
    parser.add_argument("--twohands_config_file", default='./work_dirs/seg_twohands_ccda/seg_twohands_ccda.py', type=str)
    parser.add_argument("--twohands_checkpoint_file", default='./work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth', type=str)
    parser.add_argument("--cb_config_file", default='./work_dirs/twohands_to_cb_ccda/twohands_to_cb_ccda.py', type=str)
    parser.add_argument("--cb_checkpoint_file", default='./work_dirs/twohands_to_cb_ccda/best_mIoU_iter_76000.pth', type=str)
    parser.add_argument("--obj1_config_file", default='./work_dirs/twohands_cb_to_obj1_ccda/twohands_cb_to_obj1_ccda.py', type=str)
    parser.add_argument("--obj1_checkpoint_file", default='./work_dirs/twohands_cb_to_obj1_ccda/best_mIoU_iter_34000.pth', type=str)
    args = parser.parse_args()
    
    return args

def decode_video(path):
    try: 
        reader = av.open(path)
    except:
        print("Issue with opening the video, path:", path)
        assert(False)

    return [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]

def segment_video(video_path, twohands_model, cb_model, obj1_model, catchBadMasks=False):
    # extract video frames and save them into a directory
    print('Reading and extracting video frames......')
    reader = decode_video(video_path)

    height, width = reader[0].shape[0], reader[0].shape[1]
    downsample_resolution_factor = max(width // 400 + 1, height // 400 + 1)
    resolution = width // downsample_resolution_factor, height // downsample_resolution_factor
    
    video_dir = video_path[:-4]
    video_image_dir = os.path.join(video_dir, 'images')
    os.makedirs(video_image_dir, exist_ok = True)
    frames = []
    for num, image in tqdm(enumerate(reader), total=len(reader)):
        save_img_file = os.path.join(video_image_dir, str(num).zfill(8)+'.png')
        image = cv2.resize(image, dsize=resolution, interpolation=cv2.INTER_CUBIC).astype(np.uint8)
        frames.append(image)
        imsave(save_img_file, image)

    print('Predicting hands......')
    video_twohands_dir = os.path.join(video_dir, 'pred_twohands')
    os.makedirs(video_twohands_dir, exist_ok = True)

    for file in tqdm(sorted(glob.glob(video_image_dir + '/*'))):
        fname = os.path.basename(file).split('.')[0]
        seg_result = inference_segmentor(twohands_model, file)[0]
        imsave(os.path.join(video_twohands_dir, fname + '.png'), seg_result.astype(np.uint8))

    print('Predicting countact boundaries......')
    video_cb_dir = os.path.join(video_dir, 'pred_cb')
    os.makedirs(video_cb_dir, exist_ok = True)
    for file in tqdm(sorted(glob.glob(video_image_dir + '/*'))):
        fname = os.path.basename(file).split('.')[0]
        seg_result = inference_segmentor(cb_model, file)[0]
        imsave(os.path.join(video_cb_dir, fname + '.png'), seg_result.astype(np.uint8))
        
    print('Predicting objects......')
    video_obj1_dir = os.path.join(video_dir, 'pred_obj1')
    os.makedirs(video_obj1_dir, exist_ok = True)
    masks = []
    for file in tqdm(sorted(glob.glob(video_image_dir + '/*'))):
        seg_result = inference_segmentor(obj1_model, file)[0]
        masks.append(seg_result.astype(np.uint8))

    masks = [(mask > 0).astype(np.uint8) for mask in masks]
    dilate = lambda m : cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)), iterations=3)
    masks = [dilate(mask) for mask in masks]

    os.system('rm -rf ' + video_dir)

    # check criterion for if segmentation is good
    if catchBadMasks:
        zeroOrOne = lambda x : 1 if x > 0 else 0
        mask_exists = [zeroOrOne(np.sum(mask)) for mask in masks]
        if sum(mask_exists) / len(mask_exists) < 0.5:
            print('Bad segmentation for video:', video_path)
            return None, None

    return frames, masks

if __name__ == "__main__":
    args = get_args()

    # predict twohands
    twohands_model = init_segmentor(args.twohands_config_file, args.twohands_checkpoint_file, device=device)
    cb_model = init_segmentor(args.cb_config_file, args.cb_checkpoint_file, device=device)
    obj_model = init_segmentor(args.obj1_config_file, args.obj1_checkpoint_file, device=device)
    video, obj_masks = segment_video(args.video, twohands_model, cb_model, obj_model)

    # stitch prediction into a video
    print('stitch prediction into a video......')
    writer = imageio.get_writer(args.output_file, fps=20)
    for img, mask in zip(video, obj_masks):
        twohands_obj1_vis = visualize_twohands_obj1(img, mask)
        writer.append_data(twohands_obj1_vis.astype(np.uint8))

    writer.close()