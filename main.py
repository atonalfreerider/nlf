import torch
import torchvision
import cv2
import os
import argparse
from typing import List
import json
from collections import defaultdict
from tqdm import tqdm

def __get_video_frame_count(video_path: str) -> int:
    """Gets the total number of frames in a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def __extract_frame_batch(video_path: str, workspace_dir: str, start_frame: int, batch_size: int) -> List[str]:
    """Extracts a specific batch of frames from video to workspace directory."""
    os.makedirs(workspace_dir, exist_ok=True)
    frame_paths = []
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for i in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_path = os.path.join(workspace_dir, f"frame_{start_frame + i:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
    
    cap.release()
    return frame_paths

def __load_frames_as_batch(frame_dir):
    """Loads frames from a directory and returns a batched tensor."""
    frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    frames = [torchvision.io.read_image(frame_path).cuda() for frame_path in frame_paths]
    batch = torch.stack(frames)
    return batch, frame_paths

def __convert_tensor_to_list(item):
    """Recursively converts PyTorch tensors to lists."""
    if isinstance(item, torch.Tensor):
        return item.cpu().detach().numpy().tolist()
    elif isinstance(item, dict):
        return {key: __convert_tensor_to_list(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [__convert_tensor_to_list(value) for value in item]
    else:
        return item

def __save_progress(poses_dict: dict, output_json: str):
    """Saves the current progress to JSON file."""
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(poses_dict, f, indent=2)

def main(video_path: str, output_dir: str):
    """Main function to process a video, extract frames, and pass them to the model."""
    # Create workspace directory
    workspace_dir = os.path.join(output_dir, "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Get total number of frames
    total_frames = __get_video_frame_count(video_path)
    print(f"Processing video with {total_frames} frames")
    
    # Load model
    model = torch.jit.load("models/nlf_l_multi.torchscript").cuda().eval()
    
    # Store predictions per frame
    frame_poses = defaultdict(list)
    output_json = os.path.join(output_dir, 'poses3d.json')
    
    # Process in batches without overlap
    batch_size = 50
    start_frame = 0
    
    # Create progress bar
    pbar = tqdm(total=total_frames, desc="Processing frames")
    last_processed = 0
    
    while start_frame < total_frames:
        end_frame = min(start_frame + batch_size, total_frames)
        
        try:
            # Extract batch of frames to workspace
            frame_paths = __extract_frame_batch(video_path, workspace_dir, start_frame, batch_size)
            
            # Load frames as batch
            frame_batch, _ = __load_frames_as_batch(workspace_dir)
            
            # Perform inference
            with torch.inference_mode():
                pred = model.detect_smpl_batched(frame_batch)

            batch_joints3d = __convert_tensor_to_list(pred.get('joints3d', []))
            batch_joints2d = __convert_tensor_to_list(pred.get('joints2d', []))
            batch_boxes = __convert_tensor_to_list(pred.get('boxes', []))
            #betas = __convert_tensor_to_list(pred['betas'])
            
            for frame_idx, frame_path in enumerate(frame_paths):
                frame_num = start_frame + frame_idx
                frame_poses[frame_num] = []

                if frame_idx >= len(batch_joints3d) or frame_idx >= len(batch_joints2d) or frame_idx >= len(batch_boxes):
                    # No detections for this frame
                    continue

                frame_poses_3d = batch_joints3d[frame_idx]
                frame_poses_2d = batch_joints2d[frame_idx]
                frame_boxes = batch_boxes[frame_idx]
                #frame_betas = betas[frame_idx]

                for i, pose in enumerate(frame_poses_3d):
                    frame_data = {
                        'joints3d': pose,
                        'joints2d': frame_poses_2d[i],
                        'box': frame_boxes[i],
                        #'betas': frame_betas[i],
                        'batch': f"batch_{start_frame:04d}_{end_frame:04d}"
                    }
                    frame_poses[frame_num].append(frame_data)
            
            # Clean up workspace
            for frame_path in frame_paths:
                os.remove(frame_path)
            
            # Save progress every 1000 frames
            if end_frame // 1000 > last_processed // 1000:
                poses_dict = {
                    'metadata': {
                        'video_path': video_path,
                        'total_frames': total_frames,
                        'batch_size': batch_size,
                        'last_processed_frame': end_frame
                    },
                    'frames': {str(k): v for k, v in sorted(frame_poses.items())}
                }
                __save_progress(poses_dict, output_json)
                print(f"\nProgress saved at frame {end_frame}")
            
            # Update progress bar
            processed_frames = end_frame - last_processed
            pbar.update(processed_frames)
            last_processed = end_frame
            
        except Exception as e:
            print(f"\nError processing batch {start_frame} to {end_frame}: {str(e)}")
            print("Progress saved up to last successful batch")
            continue
        
        # Move to next batch without overlap
        start_frame = end_frame
    
    # Save final results
    poses_dict = {
        'metadata': {
            'video_path': video_path,
            'total_frames': total_frames,
            'batch_size': batch_size,
            'last_processed_frame': last_processed
        },
        'frames': {str(k): v for k, v in sorted(frame_poses.items())}
    }
    __save_progress(poses_dict, output_json)
    
    # Clean up
    pbar.close()
    try:
        os.rmdir(workspace_dir)
    except:
        print("Warning: Could not remove workspace directory")
    
    print(f"3D poses saved to {output_json}")
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video to extract frames and perform SMPL detection.")
    parser.add_argument("video_input_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_frames_dir", type=str, help="Directory to store extracted frames.")
    args = parser.parse_args()

    main(args.video_input_path, args.output_frames_dir)