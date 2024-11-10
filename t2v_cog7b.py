import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import time


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def main():
    start_time = time.time()
    
    print("Starting CogVideo generation pipeline...")
    import_custom_nodes()
    
    model_load_start = time.time()
    with torch.inference_mode():
        # Load CogVideo model
        print("\nLoading CogVideo model...")
        downloadandloadcogvideomodel = NODE_CLASS_MAPPINGS["DownloadAndLoadCogVideoModel"]()
        downloadandloadcogvideomodel_1 = downloadandloadcogvideomodel.loadmodel(
            model="THUDM/CogVideoX-5b",
            precision="fp32",
            fp8_transformer="disabled",
            compile="disabled",
            enable_sequential_cpu_offload=False,
        )
        print(f"Model loading time: {time.time() - model_load_start:.2f}s")

        # Load CLIP
        clip_load_start = time.time()
        print("\nLoading CLIP model...")
        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        cliploader_20 = cliploader.load_clip(
            clip_name="t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors",
            type="sd3",
        )
        print(f"CLIP loading time: {time.time() - clip_load_start:.2f}s")

        # Initialize other components
        cogvideotextencode = NODE_CLASS_MAPPINGS["CogVideoTextEncode"]()
        cogvideosampler = NODE_CLASS_MAPPINGS["CogVideoSampler"]()
        cogvideodecode = NODE_CLASS_MAPPINGS["CogVideoDecode"]()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

        for q in range(1):
            # Text encoding
            encode_start = time.time()
            print("\nEncoding text prompts...")
            cogvideotextencode_30 = cogvideotextencode.process(
                prompt="[Prompt 3: Alexei's Apartment]\nClose-up of Alexei in a dim room, staring at a holographic family photo. His face is grim as the sound of Karmov's voice from a broadcast fills the room.\n",
                strength=1,
                force_offload=True,
                clip=get_value_at_index(cliploader_20, 0),
            )

            cogvideotextencode_31 = cogvideotextencode.process(
                prompt="",
                strength=1,
                force_offload=True,
                clip=get_value_at_index(cliploader_20, 0),
            )
            print(f"Text encoding time: {time.time() - encode_start:.2f}s")

            # Video generation
            generation_start = time.time()
            print("\nGenerating video frames...")
            cogvideosampler_34 = cogvideosampler.process(
                height=480,
                width=720,
                num_frames=120,
                steps=50,
                cfg=6,
                seed=random.randint(1, 2**64),
                scheduler="DPM++",
                denoise_strength=1,
                pipeline=get_value_at_index(downloadandloadcogvideomodel_1, 0),
                positive=get_value_at_index(cogvideotextencode_30, 0),
                negative=get_value_at_index(cogvideotextencode_31, 0),
            )
            print(f"Video generation time: {time.time() - generation_start:.2f}s")

            # Decoding
            decode_start = time.time()
            print("\nDecoding video frames...")
            cogvideodecode_11 = cogvideodecode.decode(
                enable_vae_tiling=False,
                tile_sample_min_height=240,
                tile_sample_min_width=360,
                tile_overlap_factor_height=0.2,
                tile_overlap_factor_width=0.2,
                auto_tile_size=True,
                pipeline=get_value_at_index(cogvideosampler_34, 0),
                samples=get_value_at_index(cogvideosampler_34, 1),
            )
            print(f"Decoding time: {time.time() - decode_start:.2f}s")

            # Video combining
            combine_start = time.time()
            print("\nCombining video frames...")
            vhs_videocombine_33 = vhs_videocombine.combine_video(
                frame_rate=12,
                loop_count=0,
                filename_prefix="CogVideoX5B",
                format="video/h264-mp4",
                pingpong=False,
                save_output=True,
                images=get_value_at_index(cogvideodecode_11, 0),
                unique_id=59682353877633674,
            )
            print(f"Video combining time: {time.time() - combine_start:.2f}s")

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s")

if __name__ == "__main__":
    main()
