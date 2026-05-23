# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CLEVR scene fidelity verifier backed by Grounding DINO.

Example:
    >>> from PIL import Image
    >>> from nanofm.eval.clevr_verifier import compute_fidelity_score
    >>> caption = (
    ...     "[SOS]Object 1 - Position: x=77 y=51 Shape: cube Color: blue "
    ...     "Material: metal. [EOS]"
    ... )
    >>> image = Image.open("generated_clevr.png").convert("RGB")
    >>> report = compute_fidelity_score(caption, image)
    >>> print(report["score"], report["per_category_breakdown"])
"""

from __future__ import annotations

import inspect
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CACHE_DIR = PROJECT_ROOT / ".cache"
os.environ.setdefault("HF_HOME", str(LOCAL_CACHE_DIR / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_DIR / "matplotlib"))

CLEVR_SHAPES = {"cube", "sphere", "cylinder"}
CLEVR_COLORS = {"blue", "cyan", "purple", "gray", "brown", "red", "green", "yellow"}
CLEVR_MATERIALS = {"metal", "rubber"}
Category = Tuple[str, str]
ParsedObject = Dict[str, Any]
Detection = Dict[str, Any]


class CLEVRCaptionParser(object):
    """
    Parser for synthetic CLEVR scene descriptions.

    Captions are expected to contain object blocks with Shape and Color fields. Malformed
    blocks are skipped when either field is missing or outside the CLEVR vocabulary.
    """

    _token_pattern = re.compile(r"\[(SOS|EOS|PAD)\]", flags=re.IGNORECASE)
    _object_pattern = re.compile(r"\bObject\s+\d+\s*-?", flags=re.IGNORECASE)
    _shape_pattern = re.compile(r"\bShape\s*:\s*([A-Za-z]+)", flags=re.IGNORECASE)
    _color_pattern = re.compile(r"\bColor\s*:\s*([A-Za-z]+)", flags=re.IGNORECASE)
    _material_pattern = re.compile(r"\bMaterial\s*:\s*([A-Za-z]+)", flags=re.IGNORECASE)
    _position_pattern = re.compile(
        r"\bPosition\s*:\s*x\s*=\s*(-?\d+(?:\.\d+)?)\s*y\s*=\s*(-?\d+(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )

    def parse(self, caption: str) -> List[ParsedObject]:
        """
        Extract object dictionaries from a CLEVR caption.

        Args:
            caption: Raw caption string, possibly including special tokens.
        Returns:
            A list of dictionaries with shape, color, material, and position keys.
        """
        clean_caption = self.clean_caption(caption)
        segments = self._object_segments(clean_caption)
        objects = []

        for segment in segments:
            parsed = self._parse_segment(segment)
            if parsed is not None:
                objects.append(parsed)

        return objects

    def clean_caption(self, caption: str) -> str:
        """
        Remove tokenizer special tokens while preserving parseable text.

        Args:
            caption: Raw caption string.
        Returns:
            Caption without [SOS], [EOS], and [PAD] tokens.
        """
        if caption is None:
            return ""
        caption = str(caption)
        caption = re.split(r"\[EOS\]", caption, maxsplit=1, flags=re.IGNORECASE)[0]
        caption = self._token_pattern.sub(" ", caption)
        return re.sub(r"\s+", " ", caption).strip()

    def _object_segments(self, caption: str) -> List[str]:
        starts = list(self._object_pattern.finditer(caption))
        if not starts:
            return [caption] if caption else []

        segments = []
        for idx, match in enumerate(starts):
            end = starts[idx + 1].start() if idx + 1 < len(starts) else len(caption)
            segments.append(caption[match.start():end])
        return segments

    def _parse_segment(self, segment: str) -> Optional[ParsedObject]:
        shape = self._extract_vocab_value(self._shape_pattern, segment, CLEVR_SHAPES)
        color = self._extract_vocab_value(self._color_pattern, segment, CLEVR_COLORS)
        if shape is None or color is None:
            return None

        material = self._extract_vocab_value(self._material_pattern, segment, CLEVR_MATERIALS)
        position_match = self._position_pattern.search(segment)
        position = None
        if position_match is not None:
            position = {
                "x": self._number(position_match.group(1)),
                "y": self._number(position_match.group(2)),
            }

        return {
            "shape": shape,
            "color": color,
            "material": material,
            "position": position,
        }

    @staticmethod
    def _extract_vocab_value(pattern: re.Pattern, text: str, vocab: Iterable[str]) -> Optional[str]:
        match = pattern.search(text)
        if match is None:
            return None
        value = match.group(1).lower()
        return value if value in vocab else None

    @staticmethod
    def _number(value: str) -> Union[int, float]:
        number = float(value)
        return int(number) if number.is_integer() else number


class GroundingDINOVerifier(object):
    """
    Grounding DINO wrapper for detecting expected CLEVR categories.

    Args:
        model_id: HuggingFace model ID for the Grounding DINO checkpoint.
        device: Torch device. Defaults to CUDA when available.
        box_threshold: Detection confidence threshold.
        text_threshold: Text grounding threshold.
    """

    def __init__(
            self,
            model_id: str = "IDEA-Research/grounding-dino-tiny",
            device: Optional[Union[str, torch.device]] = None,
            box_threshold: float = 0.35,
            text_threshold: float = 0.25,
        ):
        self.model_id = model_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = None
        self.model = None

    def detect(
            self,
            image: Union[Image.Image, torch.Tensor],
            expected_categories: Sequence[Category],
        ) -> List[Detection]:
        """
        Detect objects matching the requested CLEVR shape/color categories.

        Args:
            image: PIL image or tensor in CHW/HWC format.
            expected_categories: Sequence of (shape, color) pairs.
        Returns:
            List of detections with bbox, label, confidence, shape, and color.
        """
        categories = unique_categories(expected_categories)
        if not categories:
            return []

        self._load_model()
        pil_image = to_pil_image(image)
        prompt = build_grounding_dino_prompt(categories)
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        result = self._post_process(outputs, inputs, pil_image)[0]
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("text_labels", result.get("labels", []))

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            label_text = str(label)
            shape, color = infer_category_from_label(label_text, categories)
            detections.append({
                "shape": shape,
                "color": color,
                "label": label_text,
                "confidence": float(score.detach().cpu().item() if torch.is_tensor(score) else score),
                "bbox": tensor_to_list(box),
            })
        return detections

    def _load_model(self) -> None:
        if self.processor is not None and self.model is not None:
            return
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Grounding DINO verification requires transformers>=4.40. "
                "Install the nano4M evaluation dependencies first."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def _post_process(self, outputs: Any, inputs: Any, image: Image.Image) -> List[Dict[str, Any]]:
        kwargs = {
            "outputs": outputs,
            "target_sizes": [(image.height, image.width)],
            "text_threshold": self.text_threshold,
        }
        signature = inspect.signature(self.processor.post_process_grounded_object_detection)
        if "threshold" in signature.parameters:
            kwargs["threshold"] = self.box_threshold
        else:
            kwargs["box_threshold"] = self.box_threshold
            kwargs["input_ids"] = inputs.input_ids
        if "input_ids" in signature.parameters and "input_ids" not in kwargs:
            kwargs["input_ids"] = inputs.input_ids
        return self.processor.post_process_grounded_object_detection(**kwargs)


def compute_fidelity_score(
        caption: str,
        image: Union[Image.Image, torch.Tensor],
        verifier: Optional[GroundingDINOVerifier] = None,
    ) -> Dict[str, Any]:
    """
    Compute CLEVR scene fidelity for one caption/image pair.

    Args:
        caption: Ground-truth CLEVR caption.
        image: Generated image as a PIL image or torch tensor.
        verifier: Optional verifier instance. A Grounding DINO verifier is created when omitted.
    Returns:
        Dictionary containing the final score, parsed objects, detections, and score components.
    """
    parser = CLEVRCaptionParser()
    expected = parser.parse(caption)
    expected_categories = [(obj["shape"], obj["color"]) for obj in expected]
    verifier = verifier or GroundingDINOVerifier()
    detected = verifier.detect(image, unique_categories(expected_categories))
    score_info = score_detections(expected, detected)

    return {
        "score": score_info["final_score"],
        "expected": expected,
        "detected": detected,
        "per_category_breakdown": score_info["per_category_breakdown"],
        "category_score": score_info["category_score"],
        "hallucinated": score_info["hallucinated"],
        "total_expected": score_info["total_expected"],
        "hallucination_penalty": score_info["hallucination_penalty"],
        "n_expected": score_info["n_expected"],
        "n_detected": score_info["n_detected"],
    }


def score_detections(expected: Sequence[ParsedObject], detected: Sequence[Detection]) -> Dict[str, Any]:
    """
    Score detected objects against parsed expected objects.

    Args:
        expected: Parsed expected CLEVR objects.
        detected: Detected objects with shape and color fields.
    Returns:
        Score components following the CLEVR verifier formula.
    """
    expected_counts = Counter(
        (obj["shape"], obj["color"])
        for obj in expected
        if obj.get("shape") is not None and obj.get("color") is not None
    )
    detected_counts = Counter(
        (det["shape"], det["color"])
        for det in detected
        if det.get("shape") is not None and det.get("color") is not None
    )

    breakdown = {}
    matches = []
    for category, n_expected in expected_counts.items():
        n_detected = detected_counts.get(category, 0)
        match = min(n_expected, n_detected) / max(n_expected, n_detected)
        key = category_to_key(category)
        breakdown[key] = {
            "shape": category[0],
            "color": category[1],
            "expected": n_expected,
            "detected": n_detected,
            "match": match,
        }
        matches.append(match)

    category_score = float(sum(matches) / len(matches)) if matches else 0.0
    hallucinated = sum(
        n_detected
        for category, n_detected in detected_counts.items()
        if category not in expected_counts
    )
    total_expected = sum(expected_counts.values())
    hallucination_penalty = max(0.0, 1.0 - hallucinated / max(total_expected, 1))
    final_score = clamp01(category_score * hallucination_penalty)

    return {
        "final_score": final_score,
        "per_category_breakdown": breakdown,
        "category_score": category_score,
        "hallucinated": hallucinated,
        "total_expected": total_expected,
        "hallucination_penalty": hallucination_penalty,
        "n_expected": {category_to_key(k): v for k, v in expected_counts.items()},
        "n_detected": {category_to_key(k): v for k, v in detected_counts.items()},
    }


def unique_categories(categories: Sequence[Category]) -> List[Category]:
    """Return categories in first-seen order without duplicates."""
    seen = set()
    unique = []
    for shape, color in categories:
        category = (str(shape).lower(), str(color).lower())
        if category not in seen:
            seen.add(category)
            unique.append(category)
    return unique


def build_grounding_dino_prompt(categories: Sequence[Category]) -> str:
    """Build the period-separated Grounding DINO text prompt."""
    parts = [f"{color} {shape}" for shape, color in unique_categories(categories)]
    return ". ".join(parts).lower() + "."


def infer_category_from_label(label: str, categories: Sequence[Category]) -> Tuple[Optional[str], Optional[str]]:
    """Map a Grounding DINO phrase back to a CLEVR (shape, color) category."""
    normalized = re.sub(r"[^a-z0-9 ]+", " ", label.lower())
    words = set(normalized.split())

    for shape, color in categories:
        if shape in words and color in words:
            return shape, color

    for shape, color in categories:
        phrase = f"{color} {shape}"
        if phrase in normalized:
            return shape, color

    shape = next((word for word in words if word in CLEVR_SHAPES), None)
    color = next((word for word in words if word in CLEVR_COLORS), None)
    return shape, color


def to_pil_image(image: Union[Image.Image, torch.Tensor]) -> Image.Image:
    """
    Convert a PIL image or tensor into an RGB PIL image.

    Args:
        image: PIL image, CHW tensor, HWC tensor, or single-item BCHW tensor.
    Returns:
        RGB PIL image.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if not torch.is_tensor(image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    tensor = image.detach().cpu()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Only single images are supported by compute_fidelity_score")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D image tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] in (1, 3):
        tensor = tensor.permute(1, 2, 0)
    if tensor.shape[-1] == 1:
        tensor = tensor.repeat(1, 1, 3)

    tensor = tensor.float()
    if tensor.numel() and float(tensor.min()) < 0.0:
        tensor = (tensor + 1.0) / 2.0
    if tensor.numel() and float(tensor.max()) <= 1.0:
        tensor = tensor * 255.0
    tensor = tensor.clamp(0, 255).byte()
    return Image.fromarray(tensor.numpy(), mode="RGB")


def tensor_to_list(value: Any) -> List[float]:
    """Convert tensor-like values to a plain list of floats."""
    if torch.is_tensor(value):
        return [float(x) for x in value.detach().cpu().tolist()]
    return [float(x) for x in value]


def category_to_key(category: Category) -> str:
    """Return a stable JSON key for a (shape, color) category."""
    shape, color = category
    return f"{color} {shape}"


def clamp01(value: float) -> float:
    """Clamp a score into [0, 1]."""
    return max(0.0, min(1.0, float(value)))
