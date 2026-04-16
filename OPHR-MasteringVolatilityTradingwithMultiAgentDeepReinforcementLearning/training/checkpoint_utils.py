"""
Checkpoint utilities 
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


def ensure_checkpoint_dir(checkpoint_dir: str) -> str:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(checkpoint_dir).absolute())


def save_agent(agent, filepath: str, metadata: Optional[Dict[str, Any]] = None, verbose: bool = True):
    # Ensure directory exists
    ensure_checkpoint_dir(os.path.dirname(filepath))
    
    # Save agent
    agent.save(filepath)
    
    # Save metadata if provided
    if metadata is not None:
        metadata_path = filepath.replace('.pt', '_metadata.json')
        metadata['saved_at'] = datetime.now().isoformat()
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    if verbose:
        print(f"✓ Saved agent to: {filepath}")
        if metadata is not None:
            print(f"  Metadata saved to: {metadata_path}")


def load_agent(agent, filepath: str, verbose: bool = True) -> Optional[Dict[str, Any]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # Load agent
    agent.load(filepath)
    
    # Load metadata if exists
    metadata = None
    metadata_path = filepath.replace('.pt', '_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    if verbose:
        print(f"✓ Loaded agent from: {filepath}")
        if metadata is not None:
            print(f"  Metadata loaded from: {metadata_path}")
            if 'saved_at' in metadata:
                print(f"  Saved at: {metadata['saved_at']}")
    
    return metadata


def save_stats(stats: Dict[str, Any], filepath: str, verbose: bool = True):
    ensure_checkpoint_dir(os.path.dirname(filepath))
    import numpy as np
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    stats_serializable = convert_to_serializable(stats)
    stats_serializable['saved_at'] = datetime.now().isoformat()
    
    with open(filepath, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    
    if verbose:
        print(f"✓ Saved statistics to: {filepath}")


def load_stats(filepath: str, verbose: bool = True) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Statistics file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        stats = json.load(f)
    
    if verbose:
        print(f"✓ Loaded statistics from: {filepath}")
        if 'saved_at' in stats:
            print(f"  Saved at: {stats['saved_at']}")
    
    return stats


def get_latest_checkpoint(checkpoint_dir: str, pattern: str = "*.pt") -> Optional[str]:
    from glob import glob
    
    checkpoints = glob(os.path.join(checkpoint_dir, pattern))
    if not checkpoints:
        return None
    
    # Sort by modification time
    latest = max(checkpoints, key=os.path.getmtime)
    return latest


def list_checkpoints(checkpoint_dir: str, pattern: str = "*.pt") -> list:
    from glob import glob
    
    checkpoints = glob(os.path.join(checkpoint_dir, pattern))
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints


def save_replay_buffer(replay_buffer, filepath: str, verbose: bool = True):
    ensure_checkpoint_dir(os.path.dirname(filepath))
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(replay_buffer, f)
        
        # Get file size
        file_size_bytes = os.path.getsize(filepath)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        if verbose:
            buffer_size = len(replay_buffer) if hasattr(replay_buffer, '__len__') else 'unknown'
            print(f"✓ Saved replay buffer to: {filepath}")
            print(f"  Buffer size: {buffer_size} transitions")
            print(f"  File size: {file_size_mb:.2f} MB")
            
            if file_size_mb > 100:
                print(f"  ⚠️  Warning: Large file size (>{file_size_mb:.0f} MB)")
    
    except Exception as e:
        raise RuntimeError(f"Failed to save replay buffer: {e}. "
                         "This may happen if the buffer contains non-serializable objects.")


def load_replay_buffer(filepath: str, verbose: bool = True):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Replay buffer not found: {filepath}")
    
    file_size_bytes = os.path.getsize(filepath)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    try:
        with open(filepath, 'rb') as f:
            replay_buffer = pickle.load(f)
        
        if verbose:
            buffer_size = len(replay_buffer) if hasattr(replay_buffer, '__len__') else 'unknown'
            print(f"✓ Loaded replay buffer from: {filepath}")
            print(f"  Buffer size: {buffer_size} transitions")
            print(f"  File size: {file_size_mb:.2f} MB")
        
        return replay_buffer
    
    except Exception as e:
        raise RuntimeError(f"Failed to load replay buffer: {e}. "
                         "The file may be corrupted or incompatible.")


def cleanup_old_checkpoints(checkpoint_dir: str, pattern: str = "*.pt", keep_last_n: int = 5, verbose: bool = True):
    checkpoints = list_checkpoints(checkpoint_dir, pattern)
    
    if len(checkpoints) <= keep_last_n:
        if verbose:
            print(f"No cleanup needed. Found {len(checkpoints)} checkpoints, keeping {keep_last_n}")
        return
    
    to_remove = checkpoints[keep_last_n:]
    for ckpt in to_remove:
        os.remove(ckpt)
        metadata_path = ckpt.replace('.pt', '_metadata.json')
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
    
    if verbose:
        print(f"✓ Cleaned up {len(to_remove)} old checkpoints")
        print(f"  Kept {keep_last_n} most recent checkpoints")


def get_checkpoint_path(phase: str, name: str, base_dir: str = "checkpoints") -> str:
    checkpoint_dir = os.path.join(base_dir, phase)
    ensure_checkpoint_dir(checkpoint_dir)
    return os.path.join(checkpoint_dir, f"{name}.pt")


def print_checkpoint_info(checkpoint_path: str):
    import torch
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    # File info
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
    mod_time = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
    print(f"File size: {file_size:.2f} MB")
    print(f"Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"\nCheckpoint contents:")
        for key in checkpoint.keys():
            print(f"  - {key}")
            if key in ['epsilon', 'steps', 'episodes']:
                print(f"    Value: {checkpoint[key]}")
        
        # Load metadata if exists
        metadata_path = checkpoint_path.replace('.pt', '_metadata.json')
        if os.path.exists(metadata_path):
            print(f"\nMetadata:")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            for key, value in metadata.items():
                print(f"  - {key}: {value}")
    
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
    
    print(f"{'='*60}\n")

