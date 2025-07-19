import os
import json
import argparse
import re
from dotenv import load_dotenv 

import time
import cv2
import base64
from PIL import Image
from io import BytesIO

import numpy as np
import openai
from openai import OpenAI

class OpenAIClient:
    def __init__(
        self, api_key='', upload_file=False, model="gpt-4o", fps=3.0
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.fps = fps
        self.upload_file = upload_file
        self.conversation_history = []
        
    def clean_history(self):
        """
        Reset the conversation history and output directories.
        """
        self.conversation_history = []
    
    def _safe_json_load(self, json_str):
        """
        Extract and parse a valid JSON object from a raw string that may contain extra text
        (e.g., markdown formatting or log prefixes).
        Raises ValueError if parsing fails.
        """
        match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {e}")
        else:
            raise ValueError("No valid JSON object found in the input string.")
    
    def _video_to_numpy_list(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / video_fps

        # Include last second explicitly
        sample_times = np.arange(0, duration_sec, step=1 / self.fps)
        if duration_sec not in sample_times:
            sample_times = np.append(sample_times, duration_sec)

        sample_indices = np.clip((sample_times * video_fps).astype(int), 0, total_frames - 1)

        sampled_frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            sampled_frames.append(frame)

        cap.release()
        return sampled_frames

        
    def encode_image(self, image_object):
        """
        Encode an image and get its base64 representation.
        """
        if isinstance(image_object, str) and os.path.isfile(image_object):
            image_path = image_object
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        elif isinstance(image_object, np.ndarray):
            
            # Check if the image is in the range [0, 1] and normalize it
            if image_object.max() <= 1.0:
                # Ensure the values are in [0, 255] range and convert to uint8
                img_uint8 = (image_object * 255).clip(0, 255).astype(np.uint8)
            else:
                img_uint8 = image_object.astype(np.uint8)

            # Convert NumPy array to PIL image
            img_pil = Image.fromarray(img_uint8)

            # Save to a BytesIO buffer
            buffer = BytesIO()
            img_pil.save(buffer, format="PNG")
            buffer.seek(0)

            # Encode the image in base64
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        
        else:
            raise ValueError("Unsupported image object type. Provide a file path or a NumPy array.")
        return image_base64

    def encode_images(self, image_objects):
        """
        Encode a list of images and get their base64 representations.
        """
        image_base64_list = []
        for image_object in image_objects:
            # Save each frame for debug
            
            
            image_base64 = self.encode_image(image_object)
            image_base64_list.append(image_base64)
        return image_base64_list

    def request(self, messages, schema=None, max_retries=3):
        """
        Make a request to OpenAI API with the given messages.
        """
        print("Requesting with user inputs...")
        
        # Prepare the request parameters
        request_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.1
        }
        
        # Add response format if schema is provided
        if schema is not None:
            request_params["response_format"] = {"type": "json_object"}
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**request_params)
                return response.choices[0].message.content
            except Exception as e:
                print(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def inference_text(self, prompt, replace_dict=None, schema=None, history=False, max_retries=5):
        if replace_dict is not None:
            for key, value in replace_dict.items():
                prompt = prompt.replace(key, value)
        
        # Create message for OpenAI format
        message = {"role": "user", "content": prompt}
        
        if history:
            messages = self.conversation_history + [message]
        else:
            messages = [message]
        
        for attempt in range(max_retries):
            try:
                response = self.request(messages, schema=schema)
                if schema is not None:
                    pred_result = self._safe_json_load(response)
                else:
                    pred_result = response
                
                # Add to conversation history
                self.conversation_history += [message, {"role": "assistant", "content": str(pred_result)}]
                return pred_result

            except Exception as e:
                print(f"Error processing (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise e
                print("Retrying...")
                time.sleep(1)

    def inference_image(self, image_object, prompt, replace_dict=None, schema=None, history=False, max_retries=5):
        if replace_dict is not None:
            for key, value in replace_dict.items():
                prompt = prompt.replace(key, value)
        
        # Prepare image content
        if self.upload_file:
            print("OpenAI doesn't support image upload, please set upload_file=False")
            return None
        else:
            # Handle both single image and list of images
            if isinstance(image_object, list):
                # Multiple images
                image_base64_list = self.encode_images(image_object)
                image_contents = []
                for image_base64 in image_base64_list:
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    })
            # If the image_object is a video file path
            elif isinstance(image_object, str) and image_object.endswith('.mp4'):
                # Uniformly sample frames and encode as images
                frames = self._video_to_numpy_list(image_object)
                image_base64_list = self.encode_images(frames)
                image_contents = []
                for image_base64 in image_base64_list:
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    })
                
            else:
                # Single image
                image_base64 = self.encode_image(image_object)
                image_contents = [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }]
        
        # Create message for OpenAI format
        message = {
            "role": "user",
            "content": image_contents + [{"type": "text", "text": prompt}]
        }
        
        if history:
            messages = self.conversation_history + [message]
        else:
            messages = [message]
        
        for attempt in range(max_retries):
            try:
                response = self.request(messages, schema=schema)
                if schema is not None:
                    pred_result = self._safe_json_load(response)
                else:
                    pred_result = response
                
                # Add to conversation history
                self.conversation_history += [message, {"role": "assistant", "content": str(pred_result)}]
                return pred_result

            except Exception as e:
                print(f"Error processing (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise e
                print("Retrying...")
                time.sleep(1)