import argparse
import logging
import time
from cleanfid import fid
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fid_computation.log')
        ]
    )

def compute_fid_score(ref_path: str, sample_path: str, device: str = "cuda") -> float:
    """
    Compute FID score
    
    Args:
        ref_path: Path to ref images directory
        sample_path: Path to sample images directory
        device: Computing device ('cuda' or 'cpu')
    
    Returns:
        float: FID score
    
    Raises:
        ValueError: If directory does not exist
    """
    # Check if paths exist
    ref_dir = Path(ref_path)
    gen_dir = Path(sample_path)
    
    if not ref_dir.exists():
        raise ValueError(f"ref images directory does not exist: {ref_path}")
    if not gen_dir.exists():
        raise ValueError(f"sample images directory does not exist: {sample_path}")
    
    logging.info(f"Starting FID score computation")
    logging.info(f"ref images directory: {ref_path}")
    logging.info(f"sample images directory: {sample_path}")
    logging.info(f"Using device: {device}")
    
    start_time = time.time()
    
    try:
        score = fid.compute_fid(
            ref_path,
            sample_path,
            device=device,
            num_workers=8  # Can be adjusted as needed
        )
        
        elapsed_time = time.time() - start_time
        logging.info(f"FID computation completed, time elapsed: {elapsed_time:.2f} seconds")
        return score
        
    except Exception as e:
        logging.error(f"Error occurred during FID computation: {str(e)}")
        raise

def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Compute FID score')
    parser.add_argument('--ref', type=str, required=True,
                      help='Path to ref images directory')
    parser.add_argument('--sample', type=str, required=True,
                      help='Path to sample images directory')
    parser.add_argument('--device', type=str, default="cuda",
                      choices=['cuda', 'cpu'], help='Computing device')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        # Compute FID
        score = compute_fid_score(args.ref, args.sample, args.device)
        
        # Output result
        logging.info(f"FID score: {score:.4f}")
        
    except Exception as e:
        logging.error(f"Program execution failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())