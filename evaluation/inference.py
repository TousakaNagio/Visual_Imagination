import re
import ast
import random
import argparse
import os
import base64
from io import BytesIO
from PIL import Image
import json

# import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook, tqdm

import phyre
from prompts import *
# from schemas import *
from utils import *

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel

random.seed(0)

# def extract_answer_list(text):
#     """
#     Extracts the answer list from within <Answer>...</Answer> tags in the text.
#     Returns the list as a Python list of floats.
#     """
#     match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
#     if match:
#         answer_str = match.group(1).strip()
#         try:
#             answer_list = ast.literal_eval(answer_str)
#             if isinstance(answer_list, list):
#                 return answer_list
#         except Exception as e:
#             print(f"Parsing error: {e}")
#     return None

def extract_answer_list(text):
    """
    Extracts float values from within <answer>...</answer> tags.
    Returns the list as Python floats.
    """
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        inner_text = match.group(1)
        floats = re.findall(r"[-+]?\d*\.\d+|\d+", inner_text)
        return [float(x) for x in floats]
    return None

class QwenVLClient:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        local_load_path=None,
        lora_path=None,
        device_map="auto",
        torch_dtype="auto",
        use_flash_attention=False,
        min_pixels=None,
        max_pixels=None
    ):
        model_kwargs = {"device_map": device_map, "trust_remote_code": True}
        if use_flash_attention:
            model_kwargs.update({
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2"
            })
        else:
            model_kwargs.update({"torch_dtype": torch_dtype})
        
        if local_load_path is not None:
            model_path = local_load_path
        else:
            model_path = model_name

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        if lora_path is not None and os.path.isdir(lora_path):
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print(f"Loaded LoRA weights from: {lora_path}")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=(min_pixels if min_pixels is not None else 4 * 28 * 28),
            max_pixels=(max_pixels if max_pixels is not None else 16384 * 28 * 28),
        )
        self.processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
        self.processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
        self.processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)

    def encode_image(self, image_object):
        """
        Encode an image and get its base64 representation.
        Accepts: local file path or numpy array.
        """
        if isinstance(image_object, str) and os.path.isfile(image_object):
            with open(image_object, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        elif isinstance(image_object, np.ndarray):
            # Normalize and convert to uint8
            if image_object.max() <= 1.0:
                image_object = (image_object * 255).clip(0, 255).astype(np.uint8)
            else:
                image_object = image_object.astype(np.uint8)

            img_pil = Image.fromarray(image_object)
            buffer = BytesIO()
            img_pil.save(buffer, format="PNG")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        else:
            raise ValueError("Unsupported image object type. Use a file path or NumPy array.")

        return f"data:image;base64,{image_base64}"
    
    def place_input_image(self, text, image_pad="<|vision_start|><|image_pad|><|vision_end|>", image_placeholder="<image>") -> str:
        text = text.replace(image_pad, '')
        text = text.replace(image_placeholder, image_pad)
        return text

    def inference_image(self, image, prompt="Describe this image.", max_new_tokens=1024):
        """
        Run inference on a single image (path, numpy, or base64) with a text prompt.

        Args:
            image: File path, NumPy array, or base64 string.
            prompt: Text prompt for the image.
            max_new_tokens: Max number of tokens to generate.

        Returns:
            str: The generated output text.
        """
        if isinstance(image, np.ndarray):
            image = self.encode_image(image)
        elif isinstance(image, str) and not image.startswith("data:image;base64,"):
            image = self.encode_image(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Prepare input
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_input = self.place_input_image(text_input)
        # breakpoint()
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate response
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # breakpoint()
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            # skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text[0]

def main(args):
    eval_setup = args.eval_setups
    fold_id = args.fold_id
    output_root = args.output_root
    model_name = args.model_name
    local_load_path = args.local_load_path
    
    exp_name = f'{eval_setup}_fold_{fold_id}_eval'
    print('Experiment name:', exp_name)
    output_dir = os.path.join(output_root, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    # exp_run = f'run_{len(os.listdir(output_dir))}' # run_001
    exp_run = f'run_{len(os.listdir(output_dir)):03d}' # run_001
    output_dir = os.path.join(output_dir, exp_run)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'inputs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visuals'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'responses'), exist_ok=True)
    
    train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
    print('Size of resulting splits:\n train:', len(train_tasks), '\n dev:', len(dev_tasks), '\n test:', len(test_tasks))
    
    action_tier = phyre.eval_setup_to_action_tier(eval_setup)
    print('Action tier for', eval_setup, 'is', action_tier)
    
    tasks = test_tasks
    task_indices = list(range(len(tasks)))
    
    qwen_client = QwenVLClient(model_name=model_name, local_load_path=local_load_path, lora_path=args.lora_path)

    success, failed, num_invalid, format_error = 0, 0, 0, 0
    eval_num = len(task_indices)

    for idx, task_index in enumerate(tqdm(task_indices[:eval_num], desc="Processing tasks")):
        # Create the simulator from the tasks and tier.
        simulator = phyre.initialize_simulator(tasks, action_tier)
        task_id = simulator.task_ids[task_index]
        print('Task ID:', task_id)
        initial_scene = simulator.initial_scenes[task_index]
        print('Initial scene shape=%s dtype=%s' % (initial_scene.shape, initial_scene.dtype))
        init_img = phyre.observations_to_float_rgb(initial_scene)

        ### Initial Prediction
        prompt = text_input_cot

        # Save the initial image
        save_image(init_img, path=os.path.join(output_dir, 'inputs', f'{task_id}.png'))

        result = qwen_client.inference_image(
            image=init_img,
            prompt=prompt
        )
        print(result)
        action = extract_answer_list(result)
        if (action is None) or (len(action) != 3):
            print("Format error in response.")
            format_error += 1
            output_text = {"response": result, "action": None}  # Wrap in a dict to match expected format
            save_json(output_text, os.path.join(output_dir, 'responses', f'{task_id}.json'))
            continue

        pred_action = np.array(action, dtype=np.float32)
        simulation = simulator.simulate_action(task_index, pred_action, need_images=True, need_featurized_objects=True)

        solved, invalid = log_simulation_results(pred_action, task_index, tasks, simulation)

        if invalid:
            print("Invalid action detected.")
            num_invalid += 1
            output_text = {"response": result, "action": None}  # Wrap in a dict to match expected format
            save_image(init_img, path=os.path.join(output_dir, 'visuals', f'{task_id}_invalid.png'))
            save_json(output_text, os.path.join(output_dir, 'responses', f'{task_id}.json'))
            continue

        img = simulation.images[0]
        first_img = phyre.observations_to_float_rgb(img)
        if solved:
            print("The agent has solved the task.")
            success += 1
        else:
            print("The agent failed to solve the task.")
            failed += 1

        output_text = {"response": result, "action": str(action)}  # Wrap in a dict to match expected format
        save_image(first_img, path=os.path.join(output_dir, 'visuals', f'{task_id}_init.png'))
        save_json(output_text, os.path.join(output_dir, 'responses', f'{task_id}.json'))
        print("Current results: success=%d, failed=%d, invalid=%d, format_error=%d" % (success, failed, num_invalid, format_error))

    # Write results to output directory
    result_summary = {
        'success': success,
        'failed': failed,
        'invalid': num_invalid,
        'format_error': format_error,
        'total_tasks': eval_num,
        'accuracy': success / eval_num if eval_num > 0 else 0,
    }
    with open(os.path.join(output_dir, f'result_summary_{model_name.split("/")[-1]}.json'), 'w') as f:
        json.dump(result_summary, f, indent=4)
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze motion in videos using Gemini-2.5-Pro.")
    parser.add_argument('--eval_setups', type=str, default='ball_within_template') # ball_cross_template ball_within_template two_balls_cross_template two_balls_within_template ball_phyre_to_tools
    parser.add_argument('--fold_id', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--local_load_path', type=str, default=None)
    parser.add_argument('--lora_path', type=str, default=None)
    parser.add_argument('--output_root', type=str, default='./output')

    args = parser.parse_args()
    main(args)