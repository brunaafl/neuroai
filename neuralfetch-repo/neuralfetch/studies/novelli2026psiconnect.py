# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""PsiConnect (OpenNeuro ds006110): the same participants scanned at baseline and
under psilocybin, with fMRI and EEG recorded separately on the same day.

The two modalities are not simultaneous, so they are two studies:
Novelli2026PsiconnectEeg ("Psilocybin-EEG") and Novelli2026PsiconnectFmri
("Psilocybin-fMRI"). Both carry ``session``, ``condition`` (ses-01 = baseline,
ses-02 = psilocybin) and ``group`` (``meditation`` for PC2xx participants, who
did 8 weeks of meditation training before both sessions, else ``control``).
"""

import json
import re
import typing as tp
from pathlib import Path

import pandas as pd

import neuralset.utils as nutils
from neuralfetch import download
from neuralset.events import study

_BIBTEX = """
@article{novelli2026psiconnect,
    title={{PsiConnect}: Multimodal Neuroimaging of Context-Dependent Brain and Behaviour Dynamics under Psilocybin},
    author={Novelli, Leonardo and Stoliker, Devon and Barta, T and others and Razi, Adeel},
    journal={Scientific Data},
    year={2026},
    doi={10.1038/s41597-026-07312-1}
}
@misc{razi2025psiconnect,
    title={PsiConnect},
    author={Razi, Adeel},
    publisher={OpenNeuro},
    doi={10.18112/openneuro.ds006110.v1.2.2},
    url={https://openneuro.org/datasets/ds006110/versions/1.2.2}
}
"""
_CONDITIONS = {"01": "baseline", "02": "psilocybin"}
_TOP_LEVEL = [
    "README", "CHANGES", "dataset_description.json", "participants.json", "participants.tsv",
    "task-*_bold.json", "task-series_eeg.json", "coordsystem.json",
]


class _Psiconnect(study.Study):
    requirements: tp.ClassVar[tuple[str, ...]] = ("boto3", "botocore")
    bibtex: tp.ClassVar[str] = _BIBTEX
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds006110/versions/1.2.2"
    licence: tp.ClassVar[str] = "CC0-1.0"
    DOWNLOAD_INCLUDE: tp.ClassVar[tuple[str, ...]] = ()

    def _download(self, overwrite: bool = False) -> None:
        download.S3(
            study="ds006110",
            dset_dir=self.path,
            bucket="openneuro.org",
            prefix="ds006110/",
            include=[*_TOP_LEVEL, *self.DOWNLOAD_INCLUDE],
        ).download(overwrite=overwrite)

    def _bids_root(self) -> Path:
        return self.path / "download"

    @staticmethod
    def _timeline(fp: Path) -> dict[str, tp.Any]:
        ent = dict(re.findall(r"(sub|ses|task|run)-([a-zA-Z0-9]+)", fp.name))
        keys = {"sub": "subject", "ses": "session", "task": "task", "run": "run"}
        return {keys[k]: v for k, v in ent.items()}

    @staticmethod
    def _info_columns(timeline: dict[str, tp.Any]) -> dict[str, str]:
        return dict(
            condition=_CONDITIONS[timeline["session"]],
            group="meditation" if timeline["subject"].startswith("PC2") else "control",
        )


class Novelli2026PsiconnectEeg(_Psiconnect):
    """PsiConnect EEG: resting, meditation, music and video EEG at baseline and under psilocybin.

    Experimental Design:
        - 64-channel EEG (actiCAP slim, BrainAmp, 500 Hz, FCz reference), seated,
          recorded after the MRI on the same day (not in the scanner)
        - 65 participants, 2 sessions (ses-01 baseline, ses-02 19 mg psilocybin,
          open label); 124 recordings
        - Paradigm (task ``series``, one continuous recording): silent clouds
          video (5 min), then eyes closed rest (5 min), guided meditation
          (5 min) and music (~7 min)

    Data Format:
        - BIDS (OpenNeuro ds006110 v1.2.2), raw BrainVision EEG
        - The authors' cleaned EEG (RELAX, described in the paper under
          ``derivatives/EEG``) is not in v1.2.2
        - Events: Eeg (``preproc='raw'``), Stimulus (the EEG markers,
          ``description`` e.g. ``Video/V  1``)

    Notes:
        - Markers are meant to mark block starts and ends, but their order does
          not match the published block order (e.g. PC001 ses-01:
          R, G, R, G, G, F, F, V, V) and they are not turned into blocks yet.
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("Psilocybin-EEG",)
    description: tp.ClassVar[str] = (
        "EEG from 65 participants at baseline and under psilocybin during rest, "
        "meditation, music and video."
    )
    DOWNLOAD_INCLUDE: tp.ClassVar[tuple[str, ...]] = ("sub-*/ses-*/eeg/*",)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for fp in sorted(self._bids_root().glob("sub-*/ses-*/eeg/*_eeg.vhdr")):
            if fp.with_suffix(".eeg").exists():
                yield self._timeline(fp)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        sub, ses, task = timeline["subject"], timeline["session"], timeline["task"]
        folder = self._bids_root() / f"sub-{sub}" / f"ses-{ses}" / "eeg"
        prefix = f"sub-{sub}_ses-{ses}_task-{task}"
        eeg = dict(type="Eeg", start=0.0, filepath=str(folder / f"{prefix}_eeg.vhdr"), preproc="raw")
        events = [pd.DataFrame([eeg])]
        events_fp = folder / f"{prefix}_events.tsv"
        if events_fp.exists():
            marks = nutils.read_bids_events(events_fp).rename(columns={"onset": "start"})
            marks["description"] = marks.trial_type.astype(str).str.strip()
            marks = marks[~marks.description.str.startswith(("New Segment", "Comment"))]
            marks["type"] = "Stimulus"
            events.append(marks[["type", "start", "duration", "description"]])
        out = pd.concat(events, ignore_index=True)
        return out.assign(**self._info_columns(timeline))


class Novelli2026PsiconnectFmri(_Psiconnect):
    """PsiConnect fMRI: multi-echo fMRI during rest, meditation, music and movie at baseline and under psilocybin.

    Experimental Design:
        - 3T multi-echo fMRI (4 echoes, TR = 0.91 s, multiband 4)
        - 65 participants, 2 sessions (ses-01 baseline, ses-02 19 mg psilocybin,
          open label); 505 runs
        - Runs: rest eyes closed (~8 min), guided meditation eyes closed
          (~6.5 min), music eyes closed (~11.4 min, two tracks), movie
          (silent clouds video, ~6 min)

    Data Format:
        - Loaded: the authors' fMRIPrep 22.0.2 output, optimally combined echoes
          in MNI152NLin2009cAsym (``desc-preproc_bold``); confounds and brain
          masks are downloaded next to it
        - The raw multi-echo BOLD (236 GB) is not downloaded; only its sidecars
        - Events: Fmri (no task events: each run is one condition)
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("Psilocybin-fMRI",)
    description: tp.ClassVar[str] = (
        "Multi-echo fMRI from 65 participants at baseline and under psilocybin "
        "during rest, meditation, music and movie."
    )
    SPACE: tp.ClassVar[str] = "MNI152NLin2009cAsym"
    FMRIPREP: tp.ClassVar[str] = "derivatives/fmriprep-22.0.2"
    DOWNLOAD_INCLUDE: tp.ClassVar[tuple[str, ...]] = (
        "sub-*/ses-*/func/*.json",
        "sub-*/ses-*/*_scans.tsv",
        f"{FMRIPREP}/dataset_description.json",
        f"{FMRIPREP}/sub-*/ses-*/func/*_space-{SPACE}_desc-preproc_bold.*",
        f"{FMRIPREP}/sub-*/ses-*/func/*_space-{SPACE}_desc-brain_mask.*",
        f"{FMRIPREP}/sub-*/ses-*/func/*_desc-confounds_timeseries.*",
    )

    def _bold_glob(self) -> str:
        return f"sub-*/ses-*/func/*_space-{self.SPACE}_desc-preproc_bold.nii.gz"

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for fp in sorted((self._bids_root() / self.FMRIPREP).glob(self._bold_glob())):
            yield self._timeline(fp)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        sub, ses = timeline["subject"], timeline["session"]
        folder = self._bids_root() / self.FMRIPREP / f"sub-{sub}" / f"ses-{ses}" / "func"
        prefix = f"sub-{sub}_ses-{ses}_task-{timeline['task']}_run-{timeline['run']}"
        bold = folder / f"{prefix}_space-{self.SPACE}_desc-preproc_bold.nii.gz"
        tr = float(json.loads(bold.with_name(bold.name.replace(".nii.gz", ".json")).read_text())["RepetitionTime"])
        fmri = dict(type="Fmri", start=0.0, filepath=str(bold), frequency=1.0 / tr)
        return pd.DataFrame([fmri]).assign(**self._info_columns(timeline))
