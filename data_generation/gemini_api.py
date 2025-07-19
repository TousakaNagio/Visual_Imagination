import os
import json
import argparse
import re

import time
import cv2
import base64
from PIL import Image
from io import BytesIO

import numpy as np
from google import genai
from google.genai import types

GOOGLE_API_KEY = "YOUR_KEY"

class GeminiClient:
    def __init__(
        self, api_key=GOOGLE_API_KEY, upload_file=False, model="models/gemini-2.5-flash-preview-05-20", fps=30.0
    ):

        self.client = genai.Client(api_key=api_key)
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
        
    def encode_image(self, image_object):
        """
        Encode an image and get its base64 representation.
        """
        if isinstance(image_object, str) and os.path.isfile(image_object):
            image_path = image_object
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        elif isinstance(image_object, np.ndarray):
            # Ensure the values are in [0, 255] range and convert to uint8
            img_uint8 = (image_object * 255).clip(0, 255).astype(np.uint8)

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
            image_base64 = self.encode_image(image_object)
            image_base64_list.append(image_base64)
        return image_base64_list
    
    def upload_image(self, image_object):
        """
        Upload an image to Gemini and get its file object.
        """
        if isinstance(image_object, str) and os.path.isfile(image_object):
            image_path = image_object
            my_file = self.client.files.upload(file=image_path)

        elif isinstance(image_object, np.ndarray):
            # Ensure the values are in [0, 255] range and convert to uint8
            img_uint8 = (image_object * 255).clip(0, 255).astype(np.uint8)

            # Convert NumPy array to PIL image
            img_pil = Image.fromarray(img_uint8)

            # Save to a BytesIO buffer
            tmp_path = "/home/shinji106/ntu/phyre/videollm/temps/temp_image.png"
            img_pil.save(tmp_path, format="PNG")

            # Upload the image to Gemini
            my_file = self.client.files.upload(file=tmp_path)

        else:
            raise ValueError("Unsupported image object type. Provide a file path or a NumPy array.")
        time.sleep(1)  # Wait for the upload to complete
        return my_file

    def upload_video(self, video_object, fps=None):
        """
        Upload a video file to Gemini and return the file object.
        """
        if fps is None:
            fps = self.fps

        if isinstance(video_object, str) and os.path.isfile(video_object):
            video_path = video_object
            video_file = self.client.files.upload(file=video_path)
        
        elif isinstance(video_object, np.ndarray):
            # Ensure the video is in the correct format (e.g., a list of frames)
            assert video_object.ndim == 4, "Video object must be a 4D NumPy array (frames, height, width, channels)."

            # Convert each frame to an image and save to a temporary video file
            temp_video_path = "/home/shinji106/ntu/phyre/videollm/temps/temp_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width, _ = video_object[0].shape
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

            for frame in video_object:
                out.write(frame)

            out.release()
            video_file = self.client.files.upload(file=temp_video_path)
            os.remove(temp_video_path)
        
        else:
            raise ValueError("Unsupported video object type. Provide a file path or a list of frames (NumPy arrays).")
        time.sleep(1)  # Wait for the upload to complete
        return video_file

    def encode_video(self, video_object, fps=None):
        """
        Encode a video and get its base64 representation.
        """
        if fps is None:
            fps = self.fps

        if isinstance(video_object, str) and os.path.isfile(video_object):
            video_path = video_object
            video_bytes = open(video_path, 'rb').read()

        elif isinstance(video_object, np.ndarray):
            # Ensure the video is in the correct format (e.g., a list of frames)
            if not isinstance(video_object, list):
                raise ValueError("Video object must be a list of frames (NumPy arrays).")
            
            # Convert each frame to an image and save to a temporary video file
            temp_video_path = "/home/shinji106/ntu/phyre/videollm/temps/temp_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width, _ = video_object[0].shape
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

            for frame in video_object:
                out.write(frame)

            out.release()
            with open(temp_video_path, 'rb') as video_file:
                video_bytes = video_file.read()
            os.remove(temp_video_path)

        else:
            raise ValueError("Unsupported video object type. Provide a file path or a list of frames (NumPy arrays).")
        return video_bytes

        
    def request(self, user_inputs, schema=None):
        if schema is not None:
            config={
                'response_mime_type': 'application/json',
                'response_schema': schema,
            }
        else:
            config=None
        print("Requesting with user inputs...")
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_inputs,
            config=config,
        )
        return response.text
    
    def inference_text(self, prompt, replace_dict=None, schema=None, history=False):
        if replace_dict is not None:
            for key, value in replace_dict.items():
                prompt = prompt.replace(key, value)
        request_prompt = types.Part(text=prompt)

        
        if history:
            user_inputs = self.conversation_history + [request_prompt]
        else:
            user_inputs = [request_prompt]
        
        success_flag = False
        while not success_flag:
            try:
                response = self.request(user_inputs, schema=schema)
                if schema is not None:
                    pred_result = self._safe_json_load(response)
                    success_flag = True
                else:
                    pred_result = response
                    success_flag = True

            except Exception as e:
                print(f"Error processing: {e}")
                print("Retrying...")

        self.conversation_history += [request_prompt, types.Part(text=str(pred_result))]
        return pred_result

    def inference_image(self, image_object, prompt, replace_dict=None, schema=None, history=False):
        if replace_dict is not None:
            for key, value in replace_dict.items():
                prompt = prompt.replace(key, value)
        request_prompt = types.Part(text=prompt)
        
        if self.upload_file:
            image_file = self.upload_image(image_object)
        else:
            image_bytes = self.encode_image(image_object)
            image_file = types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            )
        
        if history:
            user_inputs = self.conversation_history + [image_file, request_prompt]
        else:
            user_inputs = [image_file, request_prompt]
        
        success_flag = False
        while not success_flag:
            try:
                response = self.request(user_inputs, schema=schema)
                if schema is not None:
                    pred_result = self._safe_json_load(response)
                    success_flag = True
                else:
                    pred_result = response
                    success_flag = True

            except Exception as e:
                print(f"Error processing: {e}")
                print("Retrying...")

        self.conversation_history += [image_file, request_prompt, types.Part(text=str(pred_result))]
        return pred_result
    
    def inference_multi_images(self, image_objects, prompts, format_instruction, replace_dicts=None, schema=None, history=False):
        """
        image_objects: list of image objects
        prompts: list of prompt strings
        format_instruction: list of strings like ['prompt0', 'image0', 'image1', 'prompt1']
        replace_dicts: list of dicts, one per prompt (optional)
        schema: response schema (optional)
        history: whether to include conversation history
        """
        # Prepare prompts with optional replacements
        prompt_parts = []
        for idx, prompt in enumerate(prompts):
            if replace_dicts and idx < len(replace_dicts) and replace_dicts[idx] is not None:
                for key, value in replace_dicts[idx].items():
                    prompt = prompt.replace(key, value)
            prompt_parts.append(types.Part(text=prompt))

        # Prepare image parts
        image_parts = []
        for image_object in image_objects:
            if self.upload_file:
                image_file = self.upload_image(image_object)
            else:
                image_bytes = self.encode_image(image_object)
                image_file = types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                )
            image_parts.append(image_file)

        # Construct user_inputs from format_instruction
        user_inputs = []
        if history:
            user_inputs.extend(self.conversation_history)

        for item in format_instruction:
            if item.startswith("prompt"):
                idx = int(item.replace("prompt", ""))
                user_inputs.append(prompt_parts[idx])
            elif item.startswith("image"):
                idx = int(item.replace("image", ""))
                user_inputs.append(image_parts[idx])
            else:
                raise ValueError(f"Invalid format instruction: {item}")

        # Retry logic
        success_flag = False
        while not success_flag:
            try:
                response = self.request(user_inputs, schema=schema)
                pred_result = self._safe_json_load(response) if schema else response
                success_flag = True
            except Exception as e:
                print(f"Error processing: {e}")
                print("Retrying...")

        self.conversation_history += user_inputs + [types.Part(text=str(pred_result))]
        return pred_result
    
    def inference_video(self, video_object, prompt, replace_dict=None, schema=None, history=False, fps=None):
        if fps is None:
            fps = self.fps

        if replace_dict is not None:
            for key, value in replace_dict.items():
                prompt = prompt.replace(key, value)
        request_prompt = types.Part(text=prompt)
        
        if self.upload_file:
            video_file = self.upload_video(video_object)
        else:
            video_bytes = self.encode_video(video_object)
            video_file = types.Part(
                inline_data=types.Blob(data=video_bytes, mime_type='video/mp4'),
                video_metadata=types.VideoMetadata(fps=self.fps)
            )

        if history:
            user_inputs = self.conversation_history + [video_file, request_prompt]
        else:
            user_inputs = [video_file, request_prompt]
        
        success_flag = False
        while not success_flag:
            try:
                response = self.request(user_inputs, schema=schema)
                if schema is not None:
                    pred_result = self._safe_json_load(response)
                    success_flag = True
                else:
                    pred_result = response
                    success_flag = True

            except Exception as e:
                print(f"Error processing: {e}")
                print("Retrying...")

        self.conversation_history += [video_file, request_prompt, types.Part(text=str(pred_result))]
        return pred_result
