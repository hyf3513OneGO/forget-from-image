import torch
from PIL import Image
import argparse
import sys

sys.path.append('img2prompt/src/blip')
sys.path.append('img2prompt/clip-interrogator')
from clip_interrogator import Config, Interrogator

def inference(image_path, mode, best_max_flavors=4):
    config = Config()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.blip_offload = False if torch.cuda.is_available() else True
    config.chunk_size = 2048
    config.flavor_intermediate_count = 512
    config.blip_num_beams = 32
    ci = Interrogator(config)

    image = Image.open(image_path).convert('RGB')
    
    if mode == 'best':
        prompt_result = ci.interrogate(image, max_flavors=int(best_max_flavors))
    elif mode == 'classic':
        prompt_result = ci.interrogate_classic(image)
    else:  # 'fast'
        prompt_result = ci.interrogate_fast(image)
    
    print(f"Mode {mode}: {prompt_result}")
    return prompt_result

def main():
    parser = argparse.ArgumentParser(description="Run inference with CLIP Interrogator")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--mode", type=str, choices=['best', 'classic', 'fast'], default='fast', help="Inference mode")
    parser.add_argument("--best_max_flavors", type=int, default=4, help="Maximum flavors for 'best' mode")
    args = parser.parse_args()

    inference(args.image_path, args.mode, args.best_max_flavors)

if __name__ == "__main__":
    main()
