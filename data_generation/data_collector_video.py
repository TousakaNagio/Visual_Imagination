import os, json
import random
import argparse

from gemini_api import GeminiClient
from openai_api import OpenAIClient
from prompts import *
from schemas import *

API_KEY = 'YOUR_KEY'

def map_id_to_filename(file_path):
    id_to_filename = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            id_to_filename[data['id']] = data['file_name']
    return id_to_filename

def get_ids_from_jsonl(file_path):
    ids = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            ids.append(data['id'])
    return ids

def map_id_to_action(file_path):
    id_to_action = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            id_to_action[data['id']] = list(data['action'])
    return id_to_action

def main(args):
    
    ### Configure the Gemini client
    # agent = GeminiClient(upload_file=True, fps=3.0) # model='gemini-2.0-flash', model='gemini-2.5-pro-preview-05-06'
    agent = OpenAIClient(api_key=API_KEY, fps=3.0)  # Using OpenAIClient for Gemini-2.5
    
    data_root = args.data_root
    splits = ['train']
    
    for split in splits:
        split_data_dir = os.path.join(data_root, split)
        input_imgs_dir = os.path.join(split_data_dir, 'input_images')
        helper_imgs_dir = os.path.join(split_data_dir, 'helper_images')
        videos_dir = os.path.join(split_data_dir, 'videos')

        output_dir = os.path.join(split_data_dir, f'prompts_video')
        os.makedirs(output_dir, exist_ok=True)
        
        sample_ids = get_ids_from_jsonl(os.path.join(split_data_dir, 'metadata.jsonl'))
        input_imgs_dict = map_id_to_filename(os.path.join(input_imgs_dir, 'metadata.jsonl'))
        helper_imgs_dict = map_id_to_filename(os.path.join(helper_imgs_dir, 'metadata.jsonl'))
        videos_dict = map_id_to_filename(os.path.join(videos_dir, 'metadata.jsonl'))
        
        for idx, id in enumerate(sample_ids):
            img_id = input_imgs_dict[id].split('.')[0]  # Assuming the ID is the filename without extension
            input_img_path = os.path.join(input_imgs_dir, input_imgs_dict[id])
            helper_img_path = os.path.join(helper_imgs_dir, helper_imgs_dict[id])
            video_path = os.path.join(videos_dir, videos_dict[id])
            
            output_file = os.path.join(output_dir, f'{id}.json')
            if os.path.exists(output_file):
                print(f"Output file {output_file} already exists, skipping ID {id}.")
                continue

            # Describe the first image
            response1 = agent.inference_image(
                input_img_path,
                prompt=data_collection_1,
                history=True
            )
            print(response1)

            # Describe the helper image
            response2 = agent.inference_image(
                helper_img_path,
                prompt=data_collection_2,
                history=True
            )
            print(response2)
            
            response2_5 = agent.inference_image(
                video_path,
                prompt=data_collection_2_5,
            )
            print(response2_5)

            # Construct data
            replace_dict = {
                '<DESCRIPTION>': response1,
                '<PLACEMENT>': response2,
                '<VIDEO_DESCRIPTION>': response2_5
            }
            response3 = agent.inference_text(
                prompt=data_collection_3_5,
                history=False,
                replace_dict=replace_dict
            )
            print(response3)

            # Save the constructed data
            output = {
                'id': id,
                'img_id': img_id,
                'description': response1,
                'placement': response2,
                'video_description': response2_5,
                'reasoning': response3
            }
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=4)
            print(f"Saved constructed data for ID {id} to {output_dir}")
            agent.clean_history()
            
            if idx == 1600:
                break
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate intruction tuning data using Gemini-2.5.")
    parser.add_argument('--data_root', type=str, default='/home/shinji106/Data/phyre/ball_within_template_fold_0_20250704-180146')
    parser.add_argument('--eval_setups', type=str, default='ball_within_template') # ball_cross_template ball_within_template two_balls_cross_template two_balls_within_template ball_phyre_to_tools
    parser.add_argument('--output_root', type=str, default='/home/shinji106/ntu/Mirage/data/images')

    args = parser.parse_args()
    main(args)