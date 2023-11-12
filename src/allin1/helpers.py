import numpy as np
import json
import torch

from dataclasses import asdict
from pathlib import Path
from glob import glob
from typing import List, Union
from .utils import mkpath, compact_json_number_array
from .typings import AllInOneOutput, AnalysisResult, PathLike
from .postprocessing import (
    postprocess_metrical_structure,
    postprocess_functional_structure,
    estimate_tempo_from_beats,
)


def run_inference(
    path: Path,
    spec_path: Path,
    model: torch.nn.Module,
    device: str,
    include_activations: bool,
    include_embeddings: bool,
) -> AnalysisResult:
    spec = np.load(spec_path)
    spec = torch.from_numpy(spec).unsqueeze(0).to(device)

    logits = model(spec)

    metrical_structure = postprocess_metrical_structure(logits, model.cfg)
    functional_structure = postprocess_functional_structure(logits, model.cfg)
    bpm = estimate_tempo_from_beats(metrical_structure["beats"])

    result = AnalysisResult(
        path=path,
        bpm=bpm,
        segments=functional_structure,
        **metrical_structure,
    )

    if include_activations:
        activations = compute_activations(logits)
        result.activations = activations

    if include_embeddings:
        result.embeddings = logits.embeddings[0].cpu().numpy()

    return result


def compute_activations(logits: AllInOneOutput):
    activations_beat = torch.sigmoid(logits.logits_beat[0]).cpu().numpy()
    activations_downbeat = torch.sigmoid(logits.logits_downbeat[0]).cpu().numpy()
    activations_segment = torch.sigmoid(logits.logits_section[0]).cpu().numpy()
    activations_label = torch.softmax(logits.logits_function[0], dim=0).cpu().numpy()
    return {
        "beat": activations_beat,
        "downbeat": activations_downbeat,
        "segment": activations_segment,
        "label": activations_label,
    }


def expand_paths(paths: List[Path]):
    expanded_paths = set()
    for path in paths:
        if "*" in str(path) or "?" in str(path):
            matches = [Path(p) for p in glob(str(path))]
            if not matches:
                raise FileNotFoundError(f"Could not find any files matching {path}")
            expanded_paths.update(matches)
        else:
            expanded_paths.add(path)

    return sorted(expanded_paths)


def check_paths(paths: List[Path]):
    missing_files = []
    for path in paths:
        if not path.is_file():
            missing_files.append(str(path))
    if missing_files:
        raise FileNotFoundError(f"Could not find the following files: {missing_files}")


def rmdir_if_empty(path: Path):
    try:
        path.rmdir()
    except (FileNotFoundError, OSError):
        pass


def save_results(
    results: Union[AnalysisResult, List[AnalysisResult]],
    out_dir: PathLike,
):
    if not isinstance(results, list):
        results = [results]

    out_dir = mkpath(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        out_path = out_dir / result.path.with_suffix(".json").name
        result = asdict(result)
        result["path"] = str(result["path"])

        activations = result.pop("activations")
        if activations is not None:
            np.savez(str(out_path.with_suffix(".activ.npz")), **activations)

        embeddings = result.pop("embeddings")
        if embeddings is not None:
            np.save(str(out_path.with_suffix(".embed.npy")), embeddings)

        json_str = json.dumps(result, indent=2)
        json_str = compact_json_number_array(json_str)
        out_path.with_suffix(".json").write_text(json_str)
