"""
Targets Registry Module for MineInsight v2
==========================================

Parses targets_list.yaml and provides exact mapping of YOLO class IDs (1-57)
to canonical target names, semantic subcategories, and 3-category classes
(Mine, Known, Background).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from src.preprocessing.mineinsight_v2.config import (
    CATEGORY_MINE,
    CATEGORY_KNOWN,
    CATEGORY_BACKGROUND,
    KNOWN_CLASS_IDS,
    MINE_CLASS_IDS,
    BACKGROUND_CLASS_ID,
    TARGETS_YAML_PATH,
)

logger = logging.getLogger(__name__)

# Semantic Subgroup Classification
SEMANTIC_SUBGROUPS: Dict[int, str] = {
    # Known Non-Mine Physical Objects (1-20)
    1: "Beverage_Can_Metal",
    2: "Plastic_Packaging",
    3: "Food_Can_Metal",
    4: "Glass_Container",
    5: "Plastic_Packaging",
    6: "Glass_Container",
    7: "Glass_Container",
    8: "Food_Can_Metal",
    9: "Glass_Bottle",
    10: "Plastic_Container",
    11: "Plastic_Bottle",
    12: "Glass_Bottle",
    13: "Plastic_Bottle",
    14: "Plastic_Bottle",
    15: "Plastic_Bottle",
    16: "Beverage_Can_Metal",
    17: "Metal_Cookware",
    18: "Household_Clutter",
    19: "Paper_Container",
    20: "Electronics_Plastic",
    # Mines (21-57)
    21: "Blast_AP_Mine",
    22: "Scatterable_AP_Mine",
    23: "Scatterable_AP_Mine",
    24: "Scatterable_AP_Mine",
    25: "Scatterable_AP_Mine",
    26: "Scatterable_AP_Mine",
    27: "Scatterable_AP_Mine",
    28: "Scatterable_AP_Mine",
    29: "Scatterable_AP_Mine",
    30: "Scatterable_AP_Mine",
    31: "Scatterable_AP_Mine",
    32: "Scatterable_AP_Mine",
    33: "Scatterable_AP_Mine",
    34: "Scatterable_AP_Mine",
    35: "Scatterable_AP_Mine",
    36: "Scatterable_AP_Mine",
    39: "Metallic_AT_Mine",
    40: "Metallic_AT_Mine",
    41: "Minimum_Metal_AT_Mine",
    42: "Plastic_Cased_AT_Mine",
    43: "Blast_AP_Mine",
    44: "Directional_AP_Mine",
    45: "Metallic_AT_Mine",
    46: "Directional_AP_Mine",
    47: "Blast_AP_Mine",
    48: "Metallic_AT_Mine",
    52: "Bounding_AP_Mine",
    53: "Blast_AP_Mine",
    55: "Blast_AP_Mine",
    56: "Blast_AP_Mine",
    57: "Blast_AP_Mine",
}


@dataclass(frozen=True)
class TargetMeta:
    class_id: int
    name: str
    category: str
    semantic_subgroup: str
    tracks_declared: List[str]


class TargetRegistry:
    """Registry that loads targets_list.yaml and caches target metadata."""

    def __init__(self, yaml_path: Optional[Path] = None):
        self._yaml_path = yaml_path or TARGETS_YAML_PATH
        self._targets: Dict[int, TargetMeta] = {}
        self.load()

    def load(self) -> None:
        """Parse targets YAML and populate target metadata dictionary."""
        if not self._yaml_path.exists():
            raise FileNotFoundError(f"Targets YAML not found: {self._yaml_path}")

        with open(self._yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty or invalid YAML file: {self._yaml_path}")

        self._targets.clear()
        track_presence: Dict[int, List[str]] = {}
        names: Dict[int, str] = {}

        for track_name in ["Track1", "Track2", "Track3"]:
            items = data.get(track_name, [])
            if not isinstance(items, list):
                continue

            for item in items:
                if not isinstance(item, dict) or "id" not in item or "name" not in item:
                    continue

                cid = int(item["id"])
                tname = str(item["name"]).strip()

                if cid not in track_presence:
                    track_presence[cid] = []
                    names[cid] = tname
                track_presence[cid].append(track_name)

        for cid, tname in names.items():
            if cid in KNOWN_CLASS_IDS:
                cat = CATEGORY_KNOWN
            elif cid in MINE_CLASS_IDS:
                cat = CATEGORY_MINE
            else:
                logger.warning(f"Class ID {cid} ('{tname}') is outside canonical 1-57 range.")
                continue

            subgroup = SEMANTIC_SUBGROUPS.get(cid, "Uncategorized")
            self._targets[cid] = TargetMeta(
                class_id=cid,
                name=tname,
                category=cat,
                semantic_subgroup=subgroup,
                tracks_declared=track_presence.get(cid, []),
            )

        logger.info(f"TargetRegistry loaded {len(self._targets)} canonical targets from {self._yaml_path.name}")

    def is_valid_target(self, class_id: int) -> bool:
        """Check if class ID is a known target in the registry."""
        return class_id in self._targets

    def get_category(self, class_id: int) -> str:
        """Return 'Mine', 'Known', or 'Background' for class ID."""
        if class_id == BACKGROUND_CLASS_ID:
            return CATEGORY_BACKGROUND
        if class_id in self._targets:
            return self._targets[class_id].category
        raise KeyError(f"Unknown class ID: {class_id}")

    def get_target_name(self, class_id: int) -> str:
        """Return canonical target name string."""
        if class_id == BACKGROUND_CLASS_ID:
            return "Natural_Terrain_Tile"
        if class_id in self._targets:
            return self._targets[class_id].name
        raise KeyError(f"Unknown class ID: {class_id}")

    def get_meta(self, class_id: int) -> TargetMeta:
        """Return complete TargetMeta for class ID."""
        if class_id in self._targets:
            return self._targets[class_id]
        raise KeyError(f"Unknown class ID: {class_id}")

    def get_all_targets(self) -> Dict[int, TargetMeta]:
        """Return copy of all registered targets."""
        return dict(self._targets)


# Global singleton registry
_DEFAULT_REGISTRY: Optional[TargetRegistry] = None


def get_target_registry() -> TargetRegistry:
    """Get or instantiate default TargetRegistry singleton."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = TargetRegistry()
    return _DEFAULT_REGISTRY
