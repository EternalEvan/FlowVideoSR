import argparse
import torch
import imageio
import numpy as np
from diffusers import AutoencoderKLCogVideoX
from torchvision import transforms
from torch.nn.functional import pad




def encode_decode(model_path, video_path, dtype, device, block_size=256):
    model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)
    video_reader = imageio.get_reader(video_path, "ffmpeg")

    frames = [transforms.ToTensor()(frame).to(device).to(dtype) for frame in video_reader]
    video_reader.close()
    
    frames_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # Shape: [C, T, H, W]
    ####
    # frames_tensor = frames_tensor[:, :97, :, :]
    ####
    print(f'frames_tensor.shape = {frames_tensor.shape}')
    # Split into spatial blocks
    blocks, h, w, pad_h, pad_w = split_into_blocks(frames_tensor, block_size)
    print(f'blocks.shape = {blocks.shape}')

    decoded_blocks = []
    with torch.no_grad():
        for i in range(blocks.size(0)):
            batch_blocks = blocks[i].unsqueeze(0)
            encoded_batch = model.encode(batch_blocks)[0].sample()
            print("encoded_batch",encoded_batch.shape)
            decoded_batch = model.decode(encoded_batch).sample
            decoded_blocks.append(decoded_batch)
    
    decoded_blocks = torch.cat(decoded_blocks, dim=0)
    print(f'decoded_blocks.shape = {decoded_blocks.shape}')
    decoded_frames = merge_blocks(decoded_blocks, h, w, pad_h, pad_w)
    return decoded_frames

def encode_for_tensor(model, frames_tensor, dtype, device, block_size=256):
    # model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)
    
    
   # frames_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # Shape: [C, T, H, W]
    ####
    # frames_tensor = frames_tensor[:, :97, :, :]
    ####
    print(f'frames_tensor.shape = {frames_tensor.shape}')
    # Split into spatial blocks
    blocks, h, w, pad_h, pad_w = split_into_blocks(frames_tensor, block_size)
    print(f'blocks.shape = {blocks.shape}')
    encoded_blocks = []
    # decoded_blocks = []
    with torch.no_grad():
        for i in range(blocks.size(0)):
            batch_blocks = blocks[i].unsqueeze(0)
            encoded_batch = model.encode(batch_blocks)[0].sample()
            encoded_blocks.append(encoded_batch)
    #         decoded_batch = model.decode(encoded_batch).sample
    #         decoded_blocks.append(decoded_batch)
    
    # decoded_blocks = torch.cat(decoded_blocks, dim=0)
    # print(f'decoded_blocks.shape = {decoded_blocks.shape}')
    # decoded_frames = merge_blocks(decoded_blocks, h, w, pad_h, pad_w)
    return encoded_blocks

def decode_for_tensor(model, encoded_blocks, dtype, device, block_size=256):
    ####
    # print(f'frames_tensor.shape = {frames_tensor.shape}')
    # # Split into spatial blocks
    # blocks, h, w, pad_h, pad_w = split_into_blocks(frames_tensor, block_size)
    # print(f'blocks.shape = {blocks.shape}')

    decoded_blocks = []
    with torch.no_grad():
        for encoded_batch in encoded_blocks:
            # batch_blocks = blocks[i].unsqueeze(0)
            # encoded_batch = model.encode(batch_blocks)[0].sample()
            decoded_batch = model.decode(encoded_batch).sample
            decoded_blocks.append(decoded_batch)
    
    decoded_blocks = torch.cat(decoded_blocks, dim=0)
    print(f'decoded_blocks.shape = {decoded_blocks.shape}')
    decoded_frames = merge_blocks(decoded_blocks, h, w, pad_h, pad_w)
    return decoded_frames


def split_into_blocks(frames_tensor, block_size):
    """Splits the frames tensor into smaller spatial blocks."""
    channel, num_frames, h, w = frames_tensor.size()
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size

    # Pad the height and width if needed
    frames_tensor = torch.nn.functional.pad(frames_tensor, (0, pad_w, 0, pad_h))
    # Calculate new height and width after padding
    h_padded, w_padded = frames_tensor.size(2), frames_tensor.size(3)

    # Split into blocks of size [block_size, block_size]
    blocks = frames_tensor.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    blocks = blocks.contiguous().view(channel, num_frames, -1, block_size, block_size)
    blocks = blocks.permute(2, 0, 1, 3, 4)
    print(f'1111111111111blocks.shape = {blocks.shape}') #([40, 3, 30, 256, 256])
    # [num_blocks, channel, num_frames, block_size, block_size]。
    
    return blocks, h, w, pad_h, pad_w

def merge_blocks(blocks, h, w, pad_h, pad_w):
    """Merges blocks back into the original frames tensor."""
    block_size = blocks.size(-1)
    channel = blocks.size(1)
    num_frames = blocks.size(2)
    h_padded, w_padded = h + pad_h, w + pad_w
    
    # Calculate number of blocks in height and width dimensions
    num_blocks_h = h_padded // block_size
    num_blocks_w = w_padded // block_size
    
    # Reshape blocks to the grid
    blocks = blocks.view(num_blocks_h, num_blocks_w, channel, num_frames, block_size, block_size)
    
    # Permute and reshape to get the original padded frame tensor
    frames_padded = blocks.permute(2, 3, 0, 4, 1, 5).contiguous().view(channel, num_frames, h_padded, w_padded)
    
    # Crop the frames to remove padding
    frames = frames_padded[:, :, :h, :w]
    
    return frames.unsqueeze(0)  # Add batch dimension




def save_video(tensor, output_path):
    """Saves the video frames to a video file."""
    print(f'tensor.shape = {tensor.shape}')
    frames = tensor.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8)

    # writer = imageio.get_writer(output_path + "/output.mp4", fps=24)
    writer = imageio.get_writer(output_path, fps=24)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

def split_for_batch(frames_tensors, block_size, random_block_idx):
    """Splits the frames tensors([B, C, T, H, W]) into smaller spatial blocks, then randomly takes a block."""
    B, C, T, H, W = frames_tensors.size()
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size

    # Pad the height and width if needed
    frames_tensors = torch.nn.functional.pad(frames_tensors, (0, pad_w, 0, pad_h))
    # Calculate new height and width after padding
    H_padded, W_padded = frames_tensors.size(3), frames_tensors.size(4)

    # Split into blocks of size [block_size, block_size]
    blocks = frames_tensors.unfold(3, block_size, block_size).unfold(4, block_size, block_size)
    blocks = blocks.contiguous().view(B, C, T, -1, block_size, block_size)
    blocks = blocks.permute(0, 3, 1, 2, 4, 5)  # Shape: [B, num_blocks, C, T, block_size, block_size]

    # Randomly select a block for each batch
    random_blocks = []
    for b in range(B):
        num_blocks = blocks.size(1)
        # random_block_idx = random.randint(0, num_blocks - 1)
        random_blocks.append(blocks[b, random_block_idx])

    random_blocks = torch.stack(random_blocks)  # Shape: [B, C, T, block_size, block_size]
    
    return random_blocks, H, W, pad_h, pad_w

def encode_block(model, block, dtype, device):
    """Encodes block [B, C, T, block_size, block_size] using the model."""
    with torch.no_grad():
        encoded_block = model.encode(block)[0].sample()
    return encoded_block


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogVideoX encode/decode demo")
    parser.add_argument("--model_path", type=str, help="The path to the CogVideoX model",default='/mnt/nfs/CogVideoX-5b/vae')
    parser.add_argument("--video_path", type=str, help="The path to the video file (for encoding)",default='/mnt/nfs/YouHQ-Train/animal/0AugFrZPP9U/050400_050459_00.mp4')
    parser.add_argument("--encoded_path", type=str, help="The path to the encoded tensor file (for decoding)")
    parser.add_argument("--output_path", type=str, help="The path to save the output file", default='/root/a800/Open-Sora/vae_output')
    parser.add_argument(
        "--mode", type=str, choices=["encode", "decode", "both"], help="Mode: encode, decode, or both",default='both'
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", help="The data type for computation (e.g., 'float16' or 'float32')"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="The device to use for computation (e.g., 'cuda' or 'cpu')"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="The number of blocks to process in a batch (adjust to manage memory usage)"
    )
    parser.add_argument(
        "--block_size", type=int, default=256, help="The size of the spatial blocks to split frames into"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    if args.mode == "encode":
        assert args.video_path, "Video path must be provided for encoding."
        encoded_output, h, w, pad_h, pad_w = encode_video(args.model_path, args.video_path, dtype, device, args.batch_size, args.block_size)
        torch.save((encoded_output, h, w, pad_h, pad_w), args.output_path + "/encoded.pt")
        print(f"Finished encoding the video and saved it to a file at {args.output_path}/encoded.pt")
    elif args.mode == "decode":
        assert args.encoded_path, "Encoded tensor path must be provided for decoding."
        encoded_output, h, w, pad_h, pad_w = torch.load(args.encoded_path)
        decoded_output = decode_video(args.model_path, encoded_output, h, w, pad_h, pad_w, dtype, device, args.batch_size)
        save_video(decoded_output, args.output_path)
        print(f"Finished decoding the video and saved it to a file at {args.output_path}/output.mp4")
    elif args.mode == "both":
        assert args.video_path, "Video path must be provided for encoding."
        # encoded_output, h, w, pad_h, pad_w = encode_video(args.model_path, args.video_path, dtype, device, args.batch_size, args.block_size)
        # torch.save((encoded_output, h, w, pad_h, pad_w), args.output_path + "/encoded.pt")
        #torch.save((encoded_output, h, w, pad_h, pad_w), "/root/Open-Sora/encoded.pt")
        # decoded_output = decode_video(args.model_path, encoded_output, h, w, pad_h, pad_w, dtype, device, args.batch_size)
        decoded_output = encode_decode(args.model_path, args.video_path, dtype, device, args.block_size)
        save_video(decoded_output, args.output_path)