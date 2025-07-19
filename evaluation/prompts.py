prompt_init = f"""
You are a Phyre expert. You will be given an initial scene of a Phyre task.
Your task is to analyze the scene (256*256 pixels) and provide a solution to the task.
In this environment, all the objects fall under the influence of gravity, and the scene is a 2D representation of a physics simulation.
The goal of the task is to make the green ball and the blue ball touch each other.
You can reach the goal by placing a red ball in the scene.
Your action should be a list of 3 floats, each in the range [0, 1], representing the action to take.
The action should be in the format: [x, y, r], where:
x: pixel of center of mass divided by SCENE_WIDTH (0 = left, 1 = right)
y: pixel of center of mass divided by SCENE_HEIGHT (0 = bottom, 1 = top)
r: radius of the red ball (0 = smallest allowed (2 pixels), 1 = largest allowed (32 pixels))

You should first provide your analysis of the task, and then provide your action in a list.
"""

prompt_check = f"""
Let's draw the red ball in the scene.
I will provide the image of the scene with the red ball placed at the position you predicted. Predicted action: <PREDICTED_ACTION>
You should first focus on the new image and describe the position of the red ball in the scene.
Then, does this placement of the red ball meet your expectations?
You can summarize the strategy you used in the previous stage and evaluate that if the red ball was placed at where you predicted, would it achieve the goal of making the green ball and the blue ball touch each other.
If yes, return the action in the format: [x, y, r] as you predicted in the previous round.
If not, please adjust your action based on the analysis of the scene and provide a new action.
"""

prompt_invalid = f"""
It seems that the predicted action <PREDICTED_ACTION> is invalid due the red ball being placed outside the scene or colliding with other objects.
Please analyze the scene more carefully and provide a new action.
"""

prompt_video = f"""
Now, let's watch the video of the previous round.
You will be given a recorded video of a Phyre task.
In the video, the gravity is applied to the objects, and the scene is a 2D representation of a physics simulation.
It seems that the red ball was placed at <PREDICTED_ACTION> in the previous round, but it did not achieve the goal.
Your task is to analyze the previous video and provide a new solution to the task.
Based the previous video, you should provide the reason for the failure and adjust your action accordingly.
"""

video_prompt = f"""
You are a Phyre expert. You will be given a video of a Phyre task.
Your task is to analyze the previous video and provide a new solution to the task.
In this environment, all the objects fall under the influence of gravity, and the scene is a 2D representation of a physics simulation.
The goal of the task is to make the green ball and the blue ball touch each other.
You can reach the goal by placing a red ball in the scene.

In the previous prediction, the red ball was placed at <PREDICTED_ACTION>.

Your action should be a list of 3 floats, each in the range [0, 1], representing the action to take.
The action should be in the format: [x, y, r], where:
x: pixel of center of mass divided by SCENE_WIDTH (0 = left, 1 = right)
y: pixel of center of mass divided by SCENE_HEIGHT (0 = bottom, 1 = top)
r: radius of the red ball (0 = smallest allowed (2 pixels), 1 = largest allowed (32 pixels))

You should first provide your analysis of the task, and then adjust your action based on the previous video to achieve the goal.
"""

data_construction_prompt = f"""
You are a physics reasoning agent in the Phyre environment.

In this environment:
- Objects fall under the influence of gravity in 2D except for the black objects.
- Goal of the task is to make the green object (such as ball, stick) touch the blue or purple object (such as ball, wall, cup or ground).
- You can reach the goal by placing a red ball in the scene.
- A valid action must be a red ball defined by `[x, y, r]`, where:
  - `x` = x-coordinate of center / scene width (in [0,1], 0.5 indicates the center of the scene, increases to the right)
  - `y` = y-coordinate of center / scene height (in [0,1], 0.5 indicates the center of the scene, increases to the top)
  - `r` = ball radius scaled to [0,1] (0 = 2px, 1 = 32px)

Please use the following format:

You will be given a pair of images:
- The **first image** shows the initial scene of a task.
- The **second image** shows a **helper action** where a red ball is placed in the scene. This red ball causes the green object to make contact with the blue object.

Your goal is to provide **step-by-step reasoning** (visual chain-of-thought) that explains:
1. How to approach solving the task by placing a red ball.
2. Why the red ball placement shown in the second image works to solve the task.

I want to build the intruction tuning data to train a MLLM.
You need to help me to construct the data by providing a step-by-step reasoning of how to solve the task.

I will first provide an example of how the data should be constructed.

Example:
<PROMPT>
<INPUT_IMAGE_1>

<Think>
<Prompt 1> Based on the first image, I need to place a red ball in the scene to make the green object touch the blue object. The red ball should be placed at a position where ... as imagined in <INPUT_IMAGE_2>... So that the green object can roll down and collide with the blue object...
<\Think>

<Answer> The red ball should be placed at <ACTIONS> in the scene. <\Answer>

Based on the exact images provided, please finish the reasoning in <Think> and construct the data in the same format as above.
"""

data_collection_1 = f"""
You are a physics reasoning agent in the Phyre environment.

In this environment:
- Objects fall under the influence of gravity in 2D except for the black objects.
- Goal of the task is to make the green object (such as ball, stick) touch the blue or purple object (such as ball, wall, cup or ground).
- You can reach the goal by placing a red ball in the scene.
- A valid action must be a red ball defined by `[x, y, r]`, where:
  - `x` = x-coordinate of center / scene width (in [0,1], 0.5 indicates the center of the scene, increases to the right)
  - `y` = y-coordinate of center / scene height (in [0,1], 0.5 indicates the center of the scene, increases to the top)
  - `r` = ball radius scaled to [0,1] (0 = 2px, 1 = 32px)

Given the initial scene of a Phyre task, your first mission is to describe the scene.
"""

data_collection_2 = f"""
Now, I will provide you a second image of the scene with a red ball placed in it.
This is a helper action that shows how to solve the task.

Your task is to analyze the second image and provide a rationale for the placement of the red ball.
Explain why the red ball is placed at that position and how it helps to achieve the goal of making the green object touch the blue object.
"""

data_collection_2_5 = f"""
You are a physics reasoning agent in the Phyre environment.

In this environment:
- Objects fall under the influence of gravity in 2D except for the black objects.
- Goal of the task is to make the green object (such as ball, stick) touch the blue or purple object (such as ball, wall, cup or ground).
- You can reach the goal by placing a red ball in the scene.
- A valid action must be a red ball defined by `[x, y, r]`, where:
  - `x` = x-coordinate of center / scene width (in [0,1], 0.5 indicates the center of the scene, increases to the right)
  - `y` = y-coordinate of center / scene height (in [0,1], 0.5 indicates the center of the scene, increases to the top)
  - `r` = ball radius scaled to [0,1] (0 = 2px, 1 = 32px)
  
Now, I will provide you a video of the motion with a red ball placed in it.
This is a expert demonstration which shows how to solve the task.

Your task is to analyze the objects' position and motion in the video and explain why the red ball is placed at that position and how it helps to achieve the goal of making the green object touch the blue or purple object from the whole video.
"""

data_collection_3 = f"""
You are a data construction agent for reasoning and instruction tuning.

You will get the following information:
1. The description of the initial scene of a Phyre task.
2. The action placement in the scene that is a helper action to solve the task.

Your task is to construct reasoning following the process:
1. Analyze the initial scene and describe the task.
2. Provide a possible action to take in the scene.
3. Analyze the helper action and explain how it helps to achieve the goal.

Example:
In the first image, I see a green object and a blue object. The goal is to make them touch each other...
To make them touch, I need to place a red ball at the right position... as imagined in the <IMAGEINE>...
By placing the red ball at that position, the green object will roll down and collide with the blue object, achieving the goal...

Information:
<DESCRIPTION>
<PLACEMENT>

You should must include special tokens <IMAGEINE> once and only once in your response.
"""

data_collection_3_5 = f"""
You are a data construction agent for reasoning and instruction tuning.

You will get the following information as reference:
1. The description of the initial scene of a Phyre task.
2. The action placement in the scene that is a helper action to solve the task.
3. The exact description of the objects' motion in the video.

Your task is to construct reasoning following the process:
1. Analyze the initial scene and describe the task.
2. Provide a possible action to take in the scene.
3. Analyze the helper action and explain how it helps to achieve the goal.

Example:
In the first image, I see a green object and a blue object. The goal is to make them touch each other...
To make them touch, I need to place a red ball at the right side of... as imagined in the <IMAGEINE>...
By placing the red ball at that position, the green object will roll down and collide with the blue object, achieving the goal...

Information:
<DESCRIPTION>
<PLACEMENT>
<VIDEO_DESCRIPTION>

You should must include special tokens <IMAGEINE> once and only once in your response.
"""

text_input = f"""You are a physics reasoning agent in the Phyre environment.
In this environment:
- Objects fall under the influence of gravity in 2D except for the black objects.
- Goal of the task is to make the green object (such as ball, stick) touch the blue or purple object (such as ball, wall, cup or ground).
- You can reach the goal by placing a red ball in the scene.
- A valid action must be a red ball defined by `[x, y, r]`, where:
  - `x` = x-coordinate of center / scene width (in [0,1], 0.5 indicates the center of the scene, increases to the right)
  - `y` = y-coordinate of center / scene height (in [0,1], 0.5 indicates the center of the scene, increases to the top)
  - `r` = ball radius scaled to [0,1] (0 = 2px, 1 = 32px)

You FIRST think about the reasoning process as an internal monologue and then provide the action in [x, y, r] as final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags.
"""

text_output = f"""<think> <REASONING1> </think><output_image><think> <REASONING2> </think> <answer> I will put the red ball at: <ACTION> </answer>"""

qwen_prompt = f"""You are a physics reasoning agent in the Phyre environment.
In this environment:
- Objects fall under the influence of gravity in 2D except for the black objects.
- Goal of the task is to make the green object (such as ball, stick) touch the blue or purple object (such as ball, wall, cup or ground).
- You can reach the goal by placing a red ball in the scene.
- A valid action must be a red ball defined by `[x, y, r]`, where:
  - `x` = x-coordinate of center / scene width (in [0,1], 0.5 indicates the center of the scene, increases to the right)
  - `y` = y-coordinate of center / scene height (in [0,1], 0.5 indicates the center of the scene, increases to the top)
  - `r` = ball radius scaled to [0,1] (0 = 2px, 1 = 32px)

You FIRST think about the reasoning process as an internal monologue and then provide the action in [x, y, r] as final answer. The final answer MUST BE put in <Answer> </Answer> tags such as <Answer> [x, y, r] </Answer>.
"""

text_input_direct = f"""You are a physics reasoning agent in the Phyre environment.
In this environment:
- Objects fall under the influence of gravity in 2D except for the black objects.
- Goal of the task is to make the green object (such as ball, stick) touch the blue or purple object (such as ball, wall, cup or ground).
- You can reach the goal by placing a red ball in the scene.
- A valid action must be a red ball defined by `[x, y, r]`, where:
  - `x` = x-coordinate of center / scene width (in [0,1], 0.5 indicates the center of the scene, increases to the right)
  - `y` = y-coordinate of center / scene height (in [0,1], 0.5 indicates the center of the scene, increases to the top)
  - `r` = ball radius scaled to [0,1] (0 = 2px, 1 = 32px)

This is the input scene: <image>
Provide the action in [x, y, r] as final answer directly. The final answer MUST BE put in <answer> </answer> tags.
"""

text_output_direct = f"""<answer> I will put the red ball at: <ACTION> </answer>"""

text_input_cot = f"""You are a physics reasoning agent in the Phyre environment.
In this environment:
- Objects fall under the influence of gravity in 2D except for the black objects.
- Goal of the task is to make the green object (such as ball, stick) touch the blue or purple object (such as ball, wall, cup or ground).
- You can reach the goal by placing a red ball in the scene.
- A valid action must be a red ball defined by `[x, y, r]`, where:
  - `x` = x-coordinate of center / scene width (in [0,1], 0.5 indicates the center of the scene, increases to the right)
  - `y` = y-coordinate of center / scene height (in [0,1], 0.5 indicates the center of the scene, increases to the top)
  - `r` = ball radius scaled to [0,1] (0 = 2px, 1 = 32px)

This is the input scene: <image>
You FIRST think about the reasoning process as an internal monologue and then provide the action in [x, y, r] as final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags.
"""

text_output_cot = f"""<think> <REASONING> </think> <answer> I will put the red ball at: <ACTION> </answer>"""

text_input_vcot = f"""You are a physics reasoning agent in the Phyre environment.
In this environment:
- Objects fall under the influence of gravity in 2D except for the black objects.
- Goal of the task is to make the green object (such as ball, stick) touch the blue or purple object (such as ball, wall, cup or ground).
- You can reach the goal by placing a red ball in the scene.
- A valid action must be a red ball defined by `[x, y, r]`, where:
  - `x` = x-coordinate of center / scene width (in [0,1], 0.5 indicates the center of the scene, increases to the right)
  - `y` = y-coordinate of center / scene height (in [0,1], 0.5 indicates the center of the scene, increases to the top)
  - `r` = ball radius scaled to [0,1] (0 = 2px, 1 = 32px)

This is the input scene: <image>
You FIRST think about the reasoning process as an internal monologue and then provide the action in [x, y, r] as final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags.
"""

text_output_vcot = f"""<think> <REASONING1> </think><output_image><think> <REASONING2> </think> <answer> I will put the red ball at: <ACTION> </answer>"""