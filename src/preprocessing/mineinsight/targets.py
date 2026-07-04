"""
Targets Module
==============

Responsible for reading targets_list.yaml, building Mine IDs and No_Mine IDs,
and validating mappings for the MineInsight dataset.
"""

import logging
from typing import Dict, List, Any
import yaml
from pathlib import Path

# Do NOT hardcode absolute paths.
from src.preprocessing.mineinsight.mineinsight_config import RAW_DIR

logger = logging.getLogger(__name__)

# Module-level dictionaries
mine_targets: Dict[int, str] = {}
non_mine_targets: Dict[int, str] = {}

MINE_KEYWORDS = [
    "PFM", "PMN", "M6", "TMA", "TC-", "MON",
    "TM-", "TMM", "PROM", "VS-", "Type", "C-3", "M-35"
]

def _check_if_mine_string(target_name: str) -> bool:
    """Internal helper to determine if a string indicates a Mine."""
    if not isinstance(target_name, str):
        return False
    return any(keyword in target_name for keyword in MINE_KEYWORDS)

def load_targets(yaml_path: Path = None) -> None:
    """
    Read and parse the targets YAML file.
    
    Reads Track1, Track2, and Track3 targets and splits them into
    mine_targets and non_mine_targets dictionaries based on their target names.
    Validates for duplicate IDs, missing names, and invalid IDs.
    
    Args:
        yaml_path: Path to targets_list.yaml. Defaults to RAW_DIR / "targets_list.yaml"
    """
    global mine_targets, non_mine_targets
    
    if yaml_path is None:
        yaml_path = RAW_DIR / "targets_list.yaml"
        
    logger.info(f"Reading targets list from {yaml_path}")
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Targets file not found: {yaml_path}")
        
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        
    if not data:
        raise ValueError(f"Empty or invalid YAML file: {yaml_path}")
        
    # Reset dictionaries on reload
    mine_targets.clear()
    non_mine_targets.clear()
    
    all_parsed_ids: Dict[int, str] = {}
    tracks_to_parse = ["Track1", "Track2", "Track3"]
    
    for track in tracks_to_parse:
        if track not in data:
            logger.warning(f"{track} not found in {yaml_path}")
            continue
            
        track_data = data[track]
        if not isinstance(track_data, list):
            logger.warning(f"Data for {track} is not a list. Skipping.")
            continue
            
        for item in track_data:
            if not isinstance(item, dict):
                raise ValueError(f"Item in {track} is not a dictionary: {item}")
                
            if "id" not in item:
                raise ValueError(f"Missing 'id' in {track} item: {item}")
            if "name" not in item:
                raise ValueError(f"Missing 'name' in {track} item: {item}")
                
            try:
                class_id = int(item["id"])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid class ID in {track}: {item['id']}. Must be an integer.")
                
            target_name = item["name"]
            if target_name is None or not str(target_name).strip():
                raise ValueError(f"Missing or empty target name for class ID {class_id} in {track}.")
                
            target_name = str(target_name).strip()
            
            # Validation: duplicate IDs with conflicting names
            if class_id in all_parsed_ids:
                if all_parsed_ids[class_id] != target_name:
                    raise ValueError(f"Duplicate class ID {class_id} found with conflicting names: "
                                     f"'{all_parsed_ids[class_id]}' and '{target_name}'.")
                continue # Skip exact duplicates silently
                
            all_parsed_ids[class_id] = target_name
            
            if _check_if_mine_string(target_name):
                mine_targets[class_id] = target_name
            else:
                non_mine_targets[class_id] = target_name

def get_mine_ids() -> List[int]:
    """Get a list of all Mine class IDs."""
    return list(mine_targets.keys())

def get_non_mine_ids() -> List[int]:
    """Get a list of all Non-Mine class IDs."""
    return list(non_mine_targets.keys())

def is_mine(class_id: int) -> bool:
    """Check if a given class ID is considered a Mine."""
    return class_id in mine_targets

def is_non_mine(class_id: int) -> bool:
    """Check if a given class ID is considered a Non-Mine."""
    return class_id in non_mine_targets

def get_target_name(class_id: int) -> str:
    """Retrieve the target name for a given class ID."""
    if class_id in mine_targets:
        return mine_targets[class_id]
    if class_id in non_mine_targets:
        return non_mine_targets[class_id]
    raise KeyError(f"Class ID {class_id} not found in parsed targets.")

def print_summary() -> None:
    """Print a formatted summary of the parsed targets."""
    print("=====================================")
    print("MineInsight Target Summary")
    print("=====================================")
    print()
    print(f"Mine classes : {len(mine_targets)}")
    print(f"Non-mine classes : {len(non_mine_targets)}")
    print()
    print("Mine IDs:")
    for cid, name in sorted(mine_targets.items()):
        print(f"  {cid}: {name}")
    print()
    print("Non-mine IDs:")
    for cid, name in sorted(non_mine_targets.items()):
        print(f"  {cid}: {name}")
    print()
    print("Validation: PASS")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    try:
        load_targets()
        print_summary()
    except Exception as e:
        logger.error(f"Validation FAILED: {e}")
