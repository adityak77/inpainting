import argparse
import av
from contextlib import nullcontext
import cv2
from einops import repeat, rearrange
import imageio
import os
import json
from PIL import Image
import numpy as np
import random
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

def get_conditioning(opt, model, prompt):
    sampler = DDIMSampler(model)

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                c = model.get_learned_conditioning([prompt])

    return sampler, c

def get_image_encoding(opt, model, sampler, img, strength):
    # need half precision
    init_image = img.to(device).half()
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                # encode (scaled latent)
                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc], device=device))
                
    return z_enc, t_enc

def decode(opt, model, sampler, c, z_enc, t_enc):
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning([""])
                                
                # decode it
                samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                            unconditional_conditioning=uc,)

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                all_samples.append(x_samples)

    return all_samples

# Code from Karpathy implementation: https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355
def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """ helper function to spherically interpolate two arrays v0 v1 """

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

def generate_video(opt, model, frames_numpy, prompt, strength, interpolation_steps=3):
    height, width = frames_numpy[0].shape[0], frames_numpy[0].shape[1]
    frames_numpy = [cv2.resize(frame, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)[None] for frame in frames_numpy]
    frames_numpy = rearrange(frames_numpy, 'f b h w c -> f b c h w')
    frames = [2. * (torch.from_numpy(frame) / 255.) - 1. for frame in frames_numpy]
    if opt.downsample_video:
        frames = frames[::interpolation_steps]

    downsampled_latents = []
    sampler, c = get_conditioning(opt, model, prompt)
    for img in tqdm(frames):
        z_enc, t_enc = get_image_encoding(opt, model, sampler, img, strength)
        downsampled_latents.append(z_enc)
    
    final_frames = []
    for i in range(len(downsampled_latents)-1):
        # do not decode last frame unless it is the last frame
        steps = interpolation_steps - 1 if i < len(downsampled_latents) - 2 else interpolation_steps
        interpolated_latents = []
        for j in range(steps):
            if opt.interpolation_method == 'linear':
                latent = downsampled_latents[i] + j * (downsampled_latents[i+1] - downsampled_latents[i]) / (interpolation_steps - 1)
            else:
                latent = slerp(j / (interpolation_steps - 1), downsampled_latents[i], downsampled_latents[i+1])
            interpolated_latents.append(latent)

        for z in interpolated_latents:
            all_samples = decode(opt, model, sampler, c, z, t_enc)
            final_frames.append(all_samples[0][0])

    output_frames = [(frame.cpu().detach().numpy() * 255).astype(np.uint8) for frame in final_frames]
    output_frames = rearrange(output_frames, 'f c h w -> f h w c')
    output_frames = [cv2.resize(frame, dsize=(width, height), interpolation=cv2.INTER_CUBIC) for frame in output_frames]

    return output_frames
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir_path",
        type=str,
        help='path to Something-Something directory containing inpainted videos and json files for video labels',
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
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
    parser.add_argument(
        '--interpolation_method',
        type=str,
        help="Method to walk latent space",
        choices=["slerp", "linear"],
        default="slerp"
    )
    parser.add_argument(
        '--downsample_video',
        action='store_true',
        help="downsample video frames and interpolate between frames"
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    interpolation_steps = 3

    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.to(device).half()

    inpainted_dir = os.path.join(opt.dir_path, '20bn-something-something-v2-inpainted-egohos-e2fgvi_hq')
    video_names = os.listdir(inpainted_dir)
    random.shuffle(video_names)

    save_dir = os.path.join(opt.dir_path, '20bn-something-something-v2-inpainted-egohos-e2fgvi_hq', 'style_augmented')
    os.makedirs(save_dir, exist_ok=True)
    
    json_file = os.path.join(opt.dir_path, 'annotations', 'something-something-v2-train.json')
    with open(json_file) as f:
        data = json.load(f)
    video_objects = {item['id']: item['placeholders'] for item in data}

    style_modifiers = ['in style of <sim-style>', 'in style of cartoon animation', 'in style of unity engine']
    prob_no_style = 0.25

    logging_file = os.path.join(save_dir, 'augmentation_log.json')
    for video_name in video_names:
        print('Processing video: ', video_name)
        video_path = os.path.join(inpainted_dir, video_name)
        video_id = video_name.split('.')[0]
        save_video_path = os.path.join(save_dir, f'{video_id}_augmented.webm')
        if os.path.exists(save_video_path):
            continue
        if video_id not in video_objects:
            continue

        # need to reseed in order to get random style augmentation prompts and strengths
        seed = random.randint(0, 100000)
        seed_everything(seed)

        frames_numpy = decode_video(video_path)

        if random.random() > prob_no_style: # generate style augmented video
            prompt = ', '.join([f'a {elem}' for elem in video_objects[video_id]])
            style = random.choice(style_modifiers)
            prompt = f'{prompt}, {style}'

            strength = round(random.uniform(0.35, 0.50), 2)
            results_dict = {'prompt': prompt, 'seed': seed, 'strength': strength}

            out_video = generate_video(opt, model, frames_numpy, prompt, strength, interpolation_steps)
        else: # do not generate style augmented video
            results_dict = {'prompt': None, 'seed': None, 'strength': None}
            out_video = frames_numpy

        if os.path.exists(save_video_path):
            continue
        else:
            h, w = out_video[0].shape[:2]
            writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*"VP90"), 24, (w, h))
            for frame in out_video:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

            if os.path.exists(logging_file):
                with open(logging_file) as f:
                    results_info = json.load(f)
            else:
                results_info = {}

            results_info[video_id] = results_dict
            with open(logging_file, 'w') as f:
                json.dump(results_info, f)


if __name__ == "__main__":
    main()
