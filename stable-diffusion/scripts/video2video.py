import argparse
import av
from contextlib import nullcontext
import cv2
from einops import repeat, rearrange
import imageio
import os
from PIL import Image
import numpy as np
import time
import torch
from torch import autocast
from torchvision.utils import make_grid
from tqdm import trange, tqdm

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from scripts.img2img import chunk, load_model_from_config
from ldm.models.diffusion.ddim import DDIMSampler

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def decode_video(path):
    try: 
        reader = av.open(path)
    except:
        print("Issue with opening the video, path:", path)
        assert(False)

    return [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]

def get_conditioning(opt, model):
    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
        
    assert opt.prompt is not None
    prompts = [opt.prompt]

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)

    return sampler, c

def get_image_encoding(model, img):
    # need half precision
    init_image = img.to(device).half()
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    return init_latent

def decode(opt, model, sampler, c, init_latent):
    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning([""])

                # encode (scaled latent)
                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc], device=device))
                                
                # decode it
                samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                            unconditional_conditioning=uc,)

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                all_samples.append(x_samples)

    return all_samples

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--video_path",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        nargs="?",
        help="path to write results to",
        default="outputs/video2video-samples/sample_results.gif"
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")

    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.to(device).half()

    frames_numpy = decode_video(opt.video_path)
    height, width = frames_numpy[0].shape[0], frames_numpy[0].shape[1]
    frames_numpy = [cv2.resize(frame, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)[None] for frame in frames_numpy]
    frames_numpy = rearrange(frames_numpy, 'f b h w c -> f b c h w')
    frames = [2. * (torch.from_numpy(frame) / 255.) - 1. for frame in frames_numpy]

    output_frames = []
    for img in tqdm(frames):
        sampler, c = get_conditioning(opt, model)
        init_latent = get_image_encoding(model, img)
        all_samples = decode(opt, model, sampler, c, init_latent)
        output_frames.append(all_samples[0][0])

    os.makedirs(os.path.dirname(opt.save_path), exist_ok=True)
    outpath = opt.save_path

    output_frames = [(frame.cpu().detach().numpy() * 255).astype(np.uint8) for frame in output_frames]
    output_frames = rearrange(output_frames, 'f c h w -> f h w c')
    output_frames = [cv2.resize(frame, dsize=(width, height), interpolation=cv2.INTER_CUBIC) for frame in output_frames]
    imageio.mimsave(outpath, output_frames)

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()