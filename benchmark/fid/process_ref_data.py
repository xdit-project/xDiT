import os
import argparse
import logging
import json

from PIL import Image
from torchvision.transforms import CenterCrop
from tqdm import tqdm
from pathlib import Path

def setup_logging(log_file='image_processing.log'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

def process_image(input_path, output_path, mode='resize', size=256, quality=95, 
                 resampling=Image.Resampling.LANCZOS):
    """
    Read image, process (resize or crop) to specified size, and save
    
    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        mode: Processing mode ('resize' or 'crop')
        size: Target size (single int for crop, tuple for resize)
        quality: JPEG save quality (1-100)
        resampling: PIL resampling method (only used for resize)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open image
        img = Image.open(input_path).convert('RGB')
        
        # Process image based on mode
        if mode == 'resize':
            processed_img = img.resize(size, resampling)
        else:  # crop mode
            center_crop = CenterCrop(size)
            processed_img = center_crop(img)
        
        # Save processed image
        processed_img.save(output_path, quality=quality)
        return True
    except Exception as e:
        logging.error(f"Failed to process image {input_path}: {str(e)}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process (resize or crop) images in a directory')
    parser.add_argument('--mode', type=str, default='resize',
                      choices=['resize', 'crop'],
                      help='Processing mode: resize or crop')
    parser.add_argument('--json_file', type=str, default='dataset_coco.json',
                      help='Path to dataset_coco.json file')
    parser.add_argument('--num_samples', type=int, default=30000,
                      help='Number of samples to process (optional)')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for processed images')
    parser.add_argument('--width', type=int, default=256,
                      help='Target width (only for resize mode)')
    parser.add_argument('--height', type=int, default=256,
                      help='Target height (only for resize mode)')
    parser.add_argument('--size', type=int, default=256,
                      help='Target size for crop mode')
    parser.add_argument('--quality', type=int, default=95,
                      help='JPEG save quality (1-100)')
    parser.add_argument('--resampling', type=str, default='lanczos',
                      choices=['lanczos', 'bilinear', 'bicubic', 'nearest'],
                      help='Resampling method (only for resize mode)')
    parser.add_argument('--log_file', type=str, default='image_processing.log',
                      help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Map resampling method strings to PIL constants
    resampling_methods = {
        'lanczos': Image.Resampling.LANCZOS,
        'bilinear': Image.Resampling.BILINEAR,
        'bicubic': Image.Resampling.BICUBIC,
        'nearest': Image.Resampling.NEAREST
    }


    # 读取数据
    with open(args.json_file) as f:
        json_data = json.load(f)

    # Get all image files
    image_files = list(map(lambda x: f"{x['filename']}", json_data['images'][:args.num_samples]))
    print(image_files[0])
    # Initialize counters
    success_count = 0
    fail_count = 0
    
    # Log processing parameters
    logging.info(f"Starting image processing")
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    if args.mode == 'resize':
        logging.info(f"Target size: {args.width}x{args.height}")
    else:
        logging.info(f"Crop size: {args.size}x{args.size}")
    logging.info(f"Total images to process: {len(image_files)}")
    
    # Process images with progress bar
    for filename in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)
        
        # Process based on mode
        if args.mode == 'resize':
            success = process_image(
                input_path, 
                output_path,
                mode='resize',
                size=(args.width, args.height),
                quality=args.quality,
                resampling=resampling_methods[args.resampling]
            )
        else:  # crop mode
            success = process_image(
                input_path,
                output_path,
                mode='crop',
                size=args.size,
                quality=args.quality
            )
            
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Print and log statistics
    stats = f"""
Processing completed:
Mode: {args.mode}
Success: {success_count}
Failed: {fail_count}
Total: {len(image_files)}
Success rate: {success_count/len(image_files)*100:.2f}%
"""
    print(stats)
    logging.info(stats)

if __name__ == "__main__":
    main()