# Define dataset
dataset = dict(
    type="VariableVideoTextDataset",
    data_path=None,
    num_frames=None,
    frame_interval=2,
    image_size=(None, None),
    transform_name="resize_crop",
)
bucket_config = {  # 13s/it
    "360p": {1: (1.0, 128), 16: (1.0, 8), 32: (1.0, 4)},
    # "144p": {1: (1.0, 200), 16: (1.0, 36), 32: (1.0, 18), 64: (1.0, 9), 128: (1.0, 4)},
    # "256": {1: (0.8, 200), 16: (0.5, 22), 32: (0.5, 11), 64: (0.5, 6), 128: (0.8, 4)},
    # "240p": {1: (0.8, 200), 16: (0.5, 22), 32: (0.5, 10), 64: (0.5, 6), 128: (0.5, 3)},
    # "360p": {1: (0.5, 120), 16: (0.5, 9), 32: (0.5, 4), 64: (0.5, 2), 128: (0.5, 1)},
    #"512": {1: (0.5, 120), 16: (0.5, 9), 32: (0.5, 4), 64: (0.5, 2), 128: (0.8, 1)},
    # "512": {16: (1.0, 8), 32: (1.0, 4)},
    # "480p": {1: (0.4, 80), 16: (0.6, 6), 32: (0.6, 3), 64: (0.6, 1), 128: (0.0, None)},
    # "720p": {1: (0.4, 40), 16: (0.6, 3), 32: (0.6, 1), 96: (0.0, None)},
    # "720p": {16: (1.0, 3), 32: (1.0, 1)},
    # "1024": {1: (0.3, 40)},
}
mask_ratios = {
    "mask_no": 0.75,
    "mask_quarter_random": 0.025,
    "mask_quarter_head": 0.025,
    "mask_quarter_tail": 0.025,
    "mask_quarter_head_tail": 0.05,
    "mask_image_random": 0.025,
    "mask_image_head": 0.025,
    "mask_image_tail": 0.025,
    "mask_image_head_tail": 0.05,
}

# Define acceleration
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT2-XL/2",
    from_pretrained=None,
    input_sq_size=512,  # pretrained model is trained on 512x512
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=4,
    local_files_only=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=200,
    shardformer=True,
    local_files_only=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)
scheduler_inference = dict(
    type="iddpm",
    num_sampling_steps=100,
    cfg_scale=7.0,
    cfg_channel=3,  # or None
)

# Others
seed = 42
outputs = "outputs"
wandb = True

epochs = 1000
log_every = 10
ckpt_every = 500
load = None

batch_size = None
lr = 1e-5
grad_clip = 1.0

eval_prompts = [
        "A scuba diver on a coral reef with schools of fish swimming, and a sea turtle and an octopus.",
        "Two pirate ships battling each other with canons as they sail inside a cup filled with coffee. The two pirate ships are fully in view, and the camera view is an aeriel view looking down at a 45 degree angle at the ships.",
        "People eating ice cream and drinkin espresso outside of a cafe on a narrow street in Rome. There are stores along the street selling a variety of wares. One shop sells fruits. Another shop sells vegetables. A third shop sells christmas ornaments. Many people walk along the street.",
        "An astronaut walking on the moon, with the effects of gravity making the walk appear very bouncy.",
        "A person walks down a garden path. The path is surrounded by gorgeous and colorful flowers, lush bushes, and grand trees. Butterflies and bees zip around the scene in the background. The person is walking directly towards the camera.",
        "A shot of the sunset from a beautiful beach with white sand and crashing waves. No people or animals in sight. Few clouds that are lit up in orange and red from the sunset. The Sun is halfway set.",
        "A majestic lion walking confidently through the savannah, its mane flowing in the wind with each step. The golden sunset casts a warm glow on the scene, highlighting the lion's powerful stride. Tall grasses sway gently around it as it moves, and distant acacia trees are silhouetted against the horizon.",
        "A curious owl turning its head almost 180 degrees and blinking its large, expressive eyes. It is perched on a tree branch in a moonlit forest, with shadows of leaves and branches creating a mystical atmosphere. The owl occasionally fluffs its feathers and tilts its head inquisitively.",
        "A ninja wearing a red outfit jumps from one roof of a building to a second building's roof. The full moon is in sight directly behind the ninja.",
        "A cowgirl rides a horse across a wide grass field.",
        "A rock band performs a high energy song on a stage in front of a huge crowd. The camera views the stage from the audience, where part of the view is obstructed by the backs of people's heads. The band members are styled similar to the Kiss rock band.",
        "A newly married couple do their first dance at a wedding. Both partners are wearing white dresses, and are slow dancing in the center of a beautifully decorated wedding hall.",
        "Two knights battle each other on foot with swords and shields. The knights both swing swords at each other and the swords meet in the middle as both knights battle for victory.",
        "A bustling city street at night, filled with the glow of car headlights and the ambient light of streetlights. The scene is a blur of motion, with cars speeding by and pedestrians navigating the crosswalks. The cityscape is a mix of towering buildings and illuminated signs, creating a vibrant and dynamic atmosphere. The perspective of the video is from a high angle, providing a bird's eye view of the street and its surroundings. The overall style of the video is dynamic and energetic, capturing the essence of urban life at night.",
        "A fat rabbit wearing a purple robe walking through a fantasy landscape",
        "A young man walks alone by the seaside",            
]

eval_image_size = (360, 540)
eval_num_frames = 32
eval_fps = 8
eval_batch_size = 8
eval_steps = 1000

wandb_project_name = "lego"
wandb_project_entity = "lambdalabs"