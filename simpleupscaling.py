import cv2
import torch
import numpy as np
import os
from tqdm import tqdm
from collections import OrderedDict
from torch.nn import functional as F

class ResidualDenseBlock_5C(torch.nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = torch.nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = torch.nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = torch.nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = torch.nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = torch.nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(torch.nn.Module):
    '''Residual in Residual Dense Block'''
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(torch.nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale
        
        # First conv
        self.conv_first = torch.nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        # Body structure
        self.body = torch.nn.ModuleList()
        for i in range(nb):
            self.body.append(RRDB(nf, gc))
        
        # Conv after body
        self.conv_body = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Upsampling
        self.conv_up1 = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up2 = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_hr = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = torch.nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        
        body_fea = fea
        for block in self.body:
            body_fea = block(body_fea)
            
        trunk = self.conv_body(body_fea)
        fea = fea + trunk
        
        # Upsampling
        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        
        return out

def _load_model(self):
    model = RRDBNet(scale=4)
    model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    
    # Download model if not exists
    local_path = os.path.expanduser('~/.cache/realesrgan_weights.pth')
    if not os.path.exists(local_path):
        import urllib.request
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(model_path, local_path)
    
    loadnet = torch.load(local_path, map_location=torch.device('cpu'), weights_only=True)
    if 'params_ema' in loadnet:
        loadnet = loadnet['params_ema']
    elif 'params' in loadnet:
        loadnet = loadnet['params']
        
    # Convert state dict keys
    state_dict = OrderedDict()
    for k, v in loadnet.items():
        if 'body.' in k:
            # Convert body.0.rdb1 -> body.0
            parts = k.split('.')
            new_key = k.replace('body.', 'body.')
            state_dict[new_key] = v
        else:
            state_dict[k] = v
    
    # Debug prints
    print("Model state dict keys:", model.state_dict().keys())
    print("Loaded state dict keys:", state_dict.keys())
    
    model.load_state_dict(state_dict, strict=False)  # Use strict=False initially
    print("Model loaded successfully")
    
    model.eval()
    model = model.to(self.device)
    return model


class VideoUpscaler:
    def __init__(self, device='cuda', tile_size=128, tile_padding=16):
        self.device = device
        self.target_height = 2160
        self.target_width = 3840
        self.tile_size = tile_size
        self.tile_padding = tile_padding
        self.model = self._load_model()
        
    def _load_model(self):
        model = RRDBNet(scale=4)
        model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        
        # Download model if not exists
        local_path = os.path.expanduser('~/.cache/realesrgan_weights.pth')
        if not os.path.exists(local_path):
            import urllib.request
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(model_path, local_path)
        
        loadnet = torch.load(local_path, map_location=torch.device('cpu'))
        if 'params_ema' in loadnet:
            loadnet = loadnet['params_ema']
        elif 'params' in loadnet:
            loadnet = loadnet['params']
            
        # Convert state dict keys
        state_dict = OrderedDict()
        for k, v in loadnet.items():
            new_key = k.replace('module.', '') if 'module.' in k else k
            state_dict[new_key] = v
            
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model = model.to(self.device)
        return model
        
    def _process_tile(self, tile, tile_padding):
        """Process a tile of the image"""
        with torch.no_grad():
            tile = tile.unsqueeze(0).to(self.device)
            tile = self.model(tile)
            return tile.squeeze(0).cpu()

    def enhance_frame(self, frame):
        """Enhanced frame processing with tiling for memory efficiency"""
        # Pre-process frame
        img = torch.from_numpy(frame).float().div(255.)
        img = img.permute(2, 0, 1).unsqueeze(0)
        
        batch, channel, height, width = img.shape
        output_height = height * 4
        output_width = width * 4
        
        # Initialize output tensor
        output = torch.zeros((channel, output_height, output_width))
        tiles_x = (width + self.tile_size - 1) // self.tile_size
        tiles_y = (height + self.tile_size - 1) // self.tile_size
        
        for i in range(tiles_y):
            for j in range(tiles_x):
                # Extract tile
                y_start = i * self.tile_size
                x_start = j * self.tile_size
                y_end = min((i + 1) * self.tile_size, height)
                x_end = min((j + 1) * self.tile_size, width)
                
                # Add padding
                tile = img[0, :, y_start:y_end, x_start:x_end]
                tile_padded = F.pad(tile, 
                                  [self.tile_padding] * 4, 
                                  mode='reflect')
                
                # Process tile
                tile_output = self._process_tile(tile_padded, self.tile_padding)
                
                # Remove padding and place in output
                tile_output = tile_output[:, 
                                        self.tile_padding*4:-self.tile_padding*4,
                                        self.tile_padding*4:-self.tile_padding*4]
                output[:, 
                      y_start*4:y_end*4,
                      x_start*4:x_end*4] = tile_output
        
        # Post-process
        output = output.permute(1, 2, 0).numpy()
        output = (output.clip(0, 1) * 255).round().astype(np.uint8)
        return output

    def calculate_dimensions(self, original_width, original_height):
        """Calculate dimensions maintaining aspect ratio"""
        aspect_ratio = original_width / original_height
        target_aspect = self.target_width / self.target_height

        if aspect_ratio > target_aspect:
            new_width = self.target_width
            new_height = int(new_width / aspect_ratio)
            pad_top = (self.target_height - new_height) // 2
            pad_bottom = self.target_height - new_height - pad_top
            pad_left = 0
            pad_right = 0
        else:
            new_height = self.target_height
            new_width = int(new_height * aspect_ratio)
            pad_left = (self.target_width - new_width) // 2
            pad_right = self.target_width - new_width - pad_left
            pad_top = 0
            pad_bottom = 0

        return {
            'new_width': new_width,
            'new_height': new_height,
            'pad_top': pad_top,
            'pad_bottom': pad_bottom,
            'pad_left': pad_left,
            'pad_right': pad_right
        }

    def add_padding(self, frame, dimensions):
        """Add letterbox/pillarbox padding"""
        padded = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        y_offset = dimensions['pad_top']
        x_offset = dimensions['pad_left']
        padded[y_offset:y_offset + dimensions['new_height'], 
               x_offset:x_offset + dimensions['new_width']] = frame
        return padded

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dimensions = self.calculate_dimensions(original_width, original_height)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, 
                               (self.target_width, self.target_height))
        
        print(f"Input dimensions: {original_width}x{original_height}")
        print(f"Output dimensions: {self.target_width}x{self.target_height}")
        print(f"Scaling factor: {self.target_width/original_width:.2f}x")
        print(f"Using device: {self.device}")
        print(f"Tile size: {self.tile_size}x{self.tile_size} with {self.tile_padding}px padding")
        
        try:
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # AI upscaling
                    enhanced = self.enhance_frame(frame)
                    
                    # Final resize and padding
                    resized = cv2.resize(enhanced, 
                                       (dimensions['new_width'], dimensions['new_height']),
                                       interpolation=cv2.INTER_LANCZOS4)
                    final_frame = self.add_padding(resized, dimensions)
                    
                    writer.write(final_frame)
                    pbar.update(1)
                    
        finally:
            cap.release()
            writer.release()
            print(f"\nProcessing complete. Output saved to: {output_path}")

def main():
    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize with smaller tile size if using CPU
    tile_size = 128 if device == 'cuda' else 64
    upscaler = VideoUpscaler(device=device, tile_size=tile_size)
    
    input_video = "/home/rajathdb/.cursor-tutor/Mochi_preview_00001.mp4"
    output_video = "output_4k.mp4"
    
    if not os.path.exists(input_video):
        print(f"Error: Input file '{input_video}' not found!")
        return
    
    upscaler.process_video(input_video, output_video)

if __name__ == "__main__":
    main()
