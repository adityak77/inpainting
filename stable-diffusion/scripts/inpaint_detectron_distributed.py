import argparse, os, sys, glob
from syslog import setlogmask
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.multiprocessing as mp
import cv2
from itertools import repeat

from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

sys.path.append('~/inpainting/detectron2/')
from segment_video import get_segmented_frames
from detectron2.utils.logger import setup_logger

def make_batch(image, mask):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(device)

    mask = mask.float()
    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch

def inpaint_frame(frame, mask, model, sampler):
    # get input frame in correct format
    batch = make_batch(frame, mask)

    # encode masked image and concat downsampled mask
    c = model.cond_stage_model.encode(batch["masked_image"])
    cc = torch.nn.functional.interpolate(batch["mask"],
                                            size=c.shape[-2:])
    c = torch.cat((c, cc), dim=1)

    shape = (c.shape[1]-1,)+c.shape[2:]
    samples_ddim, _ = sampler.sample(S=opt.steps,
                                     conditioning=c,
                                     batch_size=c.shape[0],
                                     shape=shape,
                                     verbose=False)
    x_samples_ddim = model.decode_first_stage(samples_ddim)

    frame = torch.clamp((batch["image"]+1.0)/2.0,
                        min=0.0, max=1.0)
    mask = torch.clamp((batch["mask"]+1.0)/2.0,
                        min=0.0, max=1.0)
    predicted_frame = torch.clamp((x_samples_ddim+1.0)/2.0,
                                    min=0.0, max=1.0)

    inpainted = (1-mask)*frame+mask*predicted_frame
    inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255

    # return inpainted frame in correct format (RGB uint8 numpy)
    return inpainted

def f(frame, masks, model, sampler):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    masks = masks.to(device)
    if masks.shape[0] > 0:
        for mask in masks:
            mask = mask[None, None]

            # get input frame in correct format
            batch = make_batch(frame, mask)

            # encode masked image and concat downsampled mask
            c = model.cond_stage_model.encode(batch["masked_image"])
            cc = torch.nn.functional.interpolate(batch["mask"],
                                                    size=c.shape[-2:])
            c = torch.cat((c, cc), dim=1)

            shape = (c.shape[1]-1,)+c.shape[2:]
            samples_ddim, _ = sampler.sample(S=opt.steps,
                                            conditioning=c,
                                            batch_size=c.shape[0],
                                            shape=shape,
                                            verbose=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)

            frame = torch.clamp((batch["image"]+1.0)/2.0,
                                min=0.0, max=1.0)
            mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                min=0.0, max=1.0)
            predicted_frame = torch.clamp((x_samples_ddim+1.0)/2.0,
                                            min=0.0, max=1.0)

            inpainted = (1-mask)*frame+mask*predicted_frame
            inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255

            # return inpainted frame in correct format (RGB uint8 numpy)
            frame = inpainted
        processed_frame = frame
    else:
        processed_frame = frame
    return processed_frame.round().astype(np.uint8)


def inpaint_video(opt, frame_list, mask_list, save_dir):
    config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"],
                          strict=False)

    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(save_dir, exist_ok=True)
    inpainted_frames = []
    with torch.no_grad():
        with model.ema_scope():
            pool = mp.Pool(processes=3)
            nframes = len(frame_list)
            inpainted_frames = pool.starmap(f, zip(frame_list, mask_list, repeat(model, nframes), repeat(sampler, nframes)))


            """
            for frame, masks in tqdm(zip(frame_list, mask_list)):
                masks = masks.to(device)
                if masks.shape[0] > 0:
                    for mask in masks:
                        mask_expand = mask[None, None]
                        
                        frame = inpaint_frame(frame, mask_expand, model, sampler)
                    processed_frame = frame
                else:
                    processed_frame = frame
                inpainted_frames.append(processed_frame.round().astype(np.uint8))
            """

    # save video to save_dir
    video = cv2.VideoCapture(opt.video_input)
    width = inpainted_frames[0].shape[1]
    height = inpainted_frames[0].shape[0]
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    basename = os.path.basename(opt.video_input)
    codec, file_ext = "mp4v", ".mp4"
    if os.path.isdir(save_dir):
        output_fname = os.path.join(save_dir, basename)
        output_fname = os.path.splitext(output_fname)[0] + file_ext
    else:
        output_fname = save_dir
    assert not os.path.isfile(output_fname), output_fname
    output_file = cv2.VideoWriter(
        filename=output_fname,
        fourcc=cv2.VideoWriter_fourcc(*codec),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )

    for frame in inpainted_frames:
        output_file.write(frame)

    output_file.release()

    

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    # Detectron2 args
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--video-input", help="Path to video file.")

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    opt = parser.parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(opt))


    # generate segmented frames in video
    frames, masks = get_segmented_frames(opt, opt.video_input, resolution=(512, 512), human_filter=True)
    inpaint_video(opt, frames[:100], masks[:100], opt.outdir)