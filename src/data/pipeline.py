import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from PIL import Image


QUESTION_TYPES = ("perception", "prediction", "planning", "behavior")
CAMERA_ORDER = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    qa_type_counts = Counter(sample["qa_type"] for sample in samples)
    scene_tokens = {sample["scene_token"] for sample in samples}
    frame_tokens = {sample["frame_token"] for sample in samples}
    camera_counts = Counter(
        camera
        for sample in samples
        for camera, rel_path in sample.get("image_paths", {}).items()
        if rel_path
    )
    frames_with_all_cameras = sum(
        1
        for frame_token in frame_tokens
        if any(
            sample["frame_token"] == frame_token
            and all(sample.get("image_paths", {}).get(camera) for camera in CAMERA_ORDER)
            for sample in samples
        )
    )

    return {
        "num_samples": len(samples),
        "num_scenes": len(scene_tokens),
        "num_frames": len(frame_tokens),
        "qa_type_counts": {question_type: qa_type_counts.get(question_type, 0) for question_type in QUESTION_TYPES},
        "camera_counts": {camera: camera_counts.get(camera, 0) for camera in CAMERA_ORDER},
        "frames_with_all_cameras": frames_with_all_cameras,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("DriveLM dataset summary")
    print(f"Samples: {summary['num_samples']}")
    print(f"Scenes: {summary['num_scenes']}")
    print(f"Frames: {summary['num_frames']}")
    print(f"Frames with all six cameras: {summary['frames_with_all_cameras']}")
    print("QA type counts:")
    for question_type, count in summary["qa_type_counts"].items():
        print(f"  {question_type}: {count}")
    print("Camera path counts:")
    for camera, count in summary["camera_counts"].items():
        print(f"  {camera}: {count}")

class DriveLMDataset(Dataset):
    """
    Links DriveLM question-answer pairs to local nuScenes camera images.
    """
    def __init__(self, nuscenes_dir: str, drivelm_json_path: str, split: str = 'train', subset_fraction: float = 1.0):
        """
        Args:
            nuscenes_dir: Path to the root of the NuScenes dataset.
            drivelm_json_path: Path to the DriveLM JSON file.
            split: 'train' or 'val'.
            subset_fraction: Fraction of the dataset to keep (for rapid prototyping).
        """
        self.nuscenes_dir = Path(nuscenes_dir)
        self.split = split
        
        with open(drivelm_json_path, 'r') as f:
            self.drivelm_data = json.load(f)
            
        self.samples = self._prepare_samples()
        
        if subset_fraction < 1.0:
            num_samples = int(len(self.samples) * subset_fraction)
            self.samples = self.samples[:num_samples]
            print(f"Subsampled dataset to {num_samples} samples (fraction: {subset_fraction})")

    def _prepare_samples(self):
        samples = []
        for scene_token, scene_data in self.drivelm_data.items():
            for frame_token, frame_data in scene_data.get('key_frames', {}).items():
                image_paths = frame_data.get('image_paths', {})
                qa_data = frame_data.get('QA', {})
                
                for qa_type, qa_list in qa_data.items():
                    for qa_pair in qa_list:
                        cam_front_rel = image_paths.get('CAM_FRONT')
                        if not cam_front_rel:
                            continue
                        full_img_path = self.nuscenes_dir / cam_front_rel
                        if not full_img_path.exists():
                            continue
                        
                        samples.append({
                            'scene_token': scene_token,
                            'frame_token': frame_token,
                            'qa_type': qa_type,
                            'question': qa_pair.get('Q'),
                            'answer': qa_pair.get('A'),
                            'image_paths': image_paths, 
                            'key_object_infos': frame_data.get('key_object_infos', {})
                        })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cam_front_rel_path = sample['image_paths'].get('CAM_FRONT')
        image = None
        if cam_front_rel_path:
            full_image_path = self.nuscenes_dir / cam_front_rel_path
            if full_image_path.exists():
                image = Image.open(full_image_path).convert('RGB')
                
        return {
            'image': image,
            'question': sample['question'],
            'answer': sample['answer'],
            'qa_type': sample['qa_type'],
            'frame_token': sample['frame_token'],
            'image_paths': sample['image_paths']
        }

def categorize_answer(answer: str) -> str:
    """Coarse classification of DriveLM ground-truth answers used by stratified sampling."""
    a = answer.strip().lower().rstrip(".")
    if a == "yes":
        return "Yes"
    if a == "no":
        return "No"
    if a in {"none", "none, no, none", "none, none, none"}:
        return "None-pattern"
    return "other"


# Per-category × answer-type training plan. Targets were chosen to:
#   * balance Yes/No within perception (which was 3.2:1 Yes-skewed in natural ordering)
#   * give prediction/planning enough None-pattern coverage to learn the comma format
#   * keep total non-behavior at ~750 so behavior at 4x upsample is a meaningful fraction
STRATIFIED_PLAN: dict[str, dict[str, int]] = {
    "perception": {"Yes": 50, "No": 50, "other": 150},
    "prediction": {"Yes": 50, "No": 50, "None-pattern": 50, "other": 100},
    "planning":   {"Yes": 50, "No": 50, "None-pattern": 25, "other": 125},
}
BEHAVIOR_UPSAMPLE = 4


def stratified_samples(
    dataset: "DriveLMDataset",
    nuscenes_dir: Path,
    seed: int = 42,
    plan: dict[str, dict[str, int]] | None = None,
    behavior_upsample: int = BEHAVIOR_UPSAMPLE,
) -> list[dict[str, Any]]:
    """Build a balanced training sample list addressing measured natural-distribution biases.

    See `src/train/README.md` for the diagnosis this addresses. Returns a list of
    sample dicts (with duplicates for behavior upsampling). Capped to availability
    per (category, answer-type) bucket.
    """
    rng = random.Random(seed)
    plan = plan if plan is not None else STRATIFIED_PLAN

    usable = []
    for sample in dataset.samples:
        image_paths = sample.get("image_paths", {})
        if not image_paths:
            continue
        if not any((nuscenes_dir / rel_path).exists() for rel_path in image_paths.values()):
            continue
        usable.append(sample)

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for s in usable:
        key = (s["qa_type"], categorize_answer(s["answer"]))
        buckets.setdefault(key, []).append(s)

    selected: list[dict[str, Any]] = []
    for cat, sub_plan in plan.items():
        for ans_type, target in sub_plan.items():
            available = buckets.get((cat, ans_type), [])
            n = min(target, len(available))
            selected.extend(rng.sample(available, n) if n else [])

    behavior_samples: list[dict[str, Any]] = []
    for ans_type in ("Yes", "No", "None-pattern", "other"):
        behavior_samples.extend(buckets.get(("behavior", ans_type), []))
    selected.extend(behavior_samples * max(1, behavior_upsample))

    rng.shuffle(selected)
    return selected


if __name__ == "__main__":
    print("Initializing DriveLMDataset pipeline...")
    base_dir = Path(__file__).parent.parent.parent
    nuscenes_dir = base_dir / "data" / "nuscenes"
    drivelm_json = base_dir / "data" / "drivelm" / "v1_1_train_nus.json"
    
    dataset = DriveLMDataset(str(nuscenes_dir), str(drivelm_json))
    print_summary(summarize_samples(dataset.samples))
    if len(dataset) > 0:
        sample = dataset[0]
        print("First sample loaded successfully:")
        print(f"Q: {sample['question']}")
        print(f"A: {sample['answer']}")
        print(f"Image found: {sample['image'] is not None}")

