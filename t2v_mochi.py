import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


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


def main(prompt: str, output_path: str):
    import_custom_nodes()
    with torch.inference_mode():
        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        cliploader_2 = cliploader.load_clip(
            clip_name="t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors",
            type="sd3",
        )

        mochitextencode = NODE_CLASS_MAPPINGS["MochiTextEncode"]()
        downloadandloadmochimodel = NODE_CLASS_MAPPINGS["DownloadAndLoadMochiModel"]()
        mochisampler = NODE_CLASS_MAPPINGS["MochiSampler"]()
        mochidecodespatialtiling = NODE_CLASS_MAPPINGS["MochiDecodeSpatialTiling"]()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

        for q in range(1):
            mochitextencode_1 = mochitextencode.process(
                prompt=prompt,
                strength=1,
                force_offload=False,
                clip=get_value_at_index(cliploader_2, 0),
            )

            mochitextencode_8 = mochitextencode.process(
                prompt="",
                strength=1,
                force_offload=True,
                clip=get_value_at_index(mochitextencode_1, 1),
            )

            downloadandloadmochimodel_4 = downloadandloadmochimodel.loadmodel(
                model="mochi_preview_dit_GGUF_Q8_0.safetensors",
                vae="mochi_preview_vae_decoder_bf16.safetensors",
                precision="fp8_e4m3fn",
                attention_mode="sdpa",
                cublas_ops=False,
                rms_norm_func="default",
                trigger=get_value_at_index(mochitextencode_8, 0),
            )

            mochisampler_14 = mochisampler.process(
                width=848,
                height=480,
                num_frames=61,
                steps=30,
                cfg=4.5,
                seed=random.randint(1, 2**64),
                model=get_value_at_index(downloadandloadmochimodel_4, 0),
                positive=get_value_at_index(mochitextencode_1, 0),
                negative=get_value_at_index(mochitextencode_8, 0),
            )

            mochidecodespatialtiling_15 = mochidecodespatialtiling.decode(
                enable_vae_tiling=True,
                num_tiles_w=4,
                num_tiles_h=4,
                overlap=16,
                min_block_size=1,
                per_batch=6,
                unnormalize=True,
                vae=get_value_at_index(downloadandloadmochimodel_4, 1),
                samples=get_value_at_index(mochisampler_14, 0),
            )

            vhs_videocombine_9 = vhs_videocombine.combine_video(
                frame_rate=10,
                loop_count=0,
                filename_prefix="Mochi_preview",
                format="video/h264-mp4",
                pingpong=True,
                save_output=True,
                images=get_value_at_index(mochidecodespatialtiling_15, 0),
                unique_id=17849377886753643119,

            )



