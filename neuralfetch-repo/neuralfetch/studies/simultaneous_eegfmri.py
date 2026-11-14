# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Simultaneous EEG-fMRI datasets stored in BIDS format 

All derive from the same base class _BidsEEGfMRI

A timeline is one run that has both EEG recording ``eeg/`` and BOLD
run in ``func/``. They are both aligned to the eeg timeline, with the fmri event
starting at the first scanner volume trigger found in the EEG markers.

eeg is raw
"""

import json
import logging
import re
import typing as tp
from functools import lru_cache
from pathlib import Path

import mne
import nibabel
import numpy as np
import pandas as pd

import neuralset.utils as nutils
from neuralfetch import download
from neuralset.events import study

logger = logging.getLogger(__name__)

# BIDS filename key -> timeline key
_ENTITIES = {
    "sub": "subject",
    "ses": "session",
    "task": "task",
    "acq": "acquisition",
    "run": "run",
}


@lru_cache(maxsize=4)
def _read_eeglab_epochs(set_fp: str) -> dict[str, tp.Any]:
    """Header of an epoched EEGLAB .set: shape, channels and, for each epoch,
    the event it is locked to (type and index in the original events)."""
    import scipy.io

    eeg = scipy.io.loadmat(set_fp, squeeze_me=True, struct_as_record=False)["EEG"]
    events = np.atleast_1d(eeg.event)
    latency = np.array([float(e.latency) for e in events])
    epoch = np.array([int(e.epoch) for e in events])
    dist = np.abs(latency - ((epoch - 1) * eeg.pnts - eeg.xmin * eeg.srate + 1))  # to t=0
    lock = [events[np.where(epoch == k, -dist, -np.inf).argmax()] for k in range(1, eeg.trials + 1)]
    return dict(
        sfreq=float(eeg.srate),
        tmin=float(eeg.xmin),
        shape=(int(eeg.nbchan), int(eeg.pnts), int(eeg.trials)),
        ch_names=[str(ch.labels) for ch in eeg.chanlocs],
        lock_types=[str(e.type) for e in lock],
        lock_urevents=[int(e.urevent) - 1 for e in lock],
        urevent_types=[str(e.type) for e in np.atleast_1d(eeg.urevent)],
        urevent_times=[(float(e.latency) - 1) / eeg.srate for e in np.atleast_1d(eeg.urevent)],
    )


class _BidsEegFmri(study.Study):
    """Private base class for simultaneous EEG-fMRI datasets in BIDS format."""

    requirements: tp.ClassVar[tuple[str, ...]] = ("openneuro-py",)

    OPENNEURO_ID: tp.ClassVar[str] = ""
    # files/folders (not) downloaded from OpenNeuro (empty include: everything)
    DOWNLOAD_INCLUDE: tp.ClassVar[tuple[str, ...]] = ()
    DOWNLOAD_EXCLUDE: tp.ClassVar[tuple[str, ...]] = ()
    # EEG annotation marking each fMRI volume
    VOLUME_MARKER: tp.ClassVar[str] = "Response/R128"
    # whether func/*_events.tsv onsets are relative to the first volume (BIDS)
    # or already on the EEG clock
    EVENTS_RELATIVE_TO_FMRI: tp.ClassVar[bool] = True # related to the eeg
    # EEG file formats, by order of preference
    EEG_EXTENSIONS: tp.ClassVar[tuple[str, ...]] = (".vhdr", ".edf", ".set")
    # preprocessing of the EEG shared in eeg/ ("raw", "authors" or "own")
    RAW_EEG_PREPROC: tp.ClassVar[str] = "raw"
    # authors' preprocessed EEG, relative to the BIDS root ({sub_dir}: sub-X[/ses-Y])
    PREPROC_EEG: tp.ClassVar[str | None] = None
    # EEG task name = BOLD task name + suffix (e.g. "ON" for "task-GRADON")
    EEG_TASK_SUFFIX: tp.ClassVar[str] = ""
    # raw EEG header -> separate data file (missing data file: only markers kept)
    RAW_EEG_DATA: tp.ClassVar[dict[str, str]] = {".vhdr": ".eeg"}

    def _download(self, overwrite: bool = False) -> None:

        # Download from openenuro
        download.Openneuro(
            study=self.OPENNEURO_ID,
            dset_dir=self.path,
            include=list(self.DOWNLOAD_INCLUDE),
            exclude=list(self.DOWNLOAD_EXCLUDE),
        ).download(overwrite=overwrite)

    def _bids_root(self) -> Path:
        return self.path / "download"

    def _prefix(self, timeline: dict[str, tp.Any]) -> str:
        return "_".join(
            f"{key}-{timeline[name]}" for key, name in _ENTITIES.items() if name in timeline
        )

    def _folder(self, timeline: dict[str, tp.Any], datatype: str) -> Path:
        # Go to specific folders
        folder = self._bids_root() / f"sub-{timeline['subject']}"
        if "session" in timeline:
            folder = folder / f"ses-{timeline['session']}"
        return folder / datatype

    def _inherited_file(
        self, timeline: dict[str, tp.Any], datatype: str, suffix: str
    ) -> Path | None:
        """File for this run, or the dataset-level file (BIDS inheritance)."""
        candidates = [
            self._folder(timeline, datatype) / f"{self._prefix(timeline)}_{suffix}",
            self._bids_root() / f"task-{timeline['task']}_{suffix}",
        ]
        return next((fp for fp in candidates if fp.exists()), None)

    def _eeg_prefix(self, timeline: dict[str, tp.Any]) -> str:
        task = timeline["task"] + self.EEG_TASK_SUFFIX
        return self._prefix({**timeline, "task": task})

    def _eeg_file(self, timeline: dict[str, tp.Any]) -> Path | None:
        for ext in self.EEG_EXTENSIONS:
            fp = self._folder(timeline, "eeg") / f"{self._eeg_prefix(timeline)}_eeg{ext}"
            if fp.exists():
                return fp
        return None

    def _bold_file(self, timeline: dict[str, tp.Any]) -> Path:
        folder = self._folder(timeline, "func")
        return next(folder.glob(f"{self._prefix(timeline)}_bold.nii*"))

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for bold in sorted(self._bids_root().glob("sub-*/**/func/*_bold.nii*")):
            entities = dict(re.findall(r"(sub|ses|task|run)-([a-zA-Z0-9]+)", bold.name))
            timeline = {_ENTITIES[k]: v for k, v in entities.items()}
            if self._eeg_file(timeline) is not None:  # skip fMRI-only runs
                yield timeline

    def _preproc_eeg_file(self, timeline: dict[str, tp.Any]) -> Path | None:
        if self.PREPROC_EEG is None:
            return None
        sub_dir = self._folder(timeline, "").relative_to(self._bids_root())
        fp = self._bids_root() / self.PREPROC_EEG.format(
            sub_dir=sub_dir, prefix=self._eeg_prefix(timeline)
        )
        return fp if fp.exists() else None

    @staticmethod
    def _preproc_offset(raw: mne.Annotations, pre: mne.Annotations, fp: Path) -> float:
        """Start of the preprocessed EEG on the raw EEG clock.

        Preprocessed files are cropped and may miss some markers, so their
        markers are matched to the raw ones. All shared marker types are used:
        volume triggers alone are periodic and also match when shifted by a TR.
        """
        raw_desc, pre_desc = np.asarray(raw.description), np.asarray(pre.description)
        shared = [d for d in set(raw_desc) & set(pre_desc) if not d.startswith("New Segment")]
        if not shared:
            raise RuntimeError(f"No marker of {fp} found in the raw EEG")

        def n_matched(offset: float) -> int:
            n = 0
            for desc in shared:
                pre_on = pre.onset[pre_desc == desc] + offset
                dist = np.abs(pre_on[:, None] - raw.onset[raw_desc == desc][None])
                n += int((dist.min(axis=1) < 0.01).sum())
            return n

        rarest = min(shared, key=lambda d: int((pre_desc == d).sum()))
        offsets = raw.onset[raw_desc == rarest][:, None] - pre.onset[pre_desc == rarest][None]
        best = max(offsets.ravel(), key=n_matched)
        total = int(np.isin(pre_desc, shared).sum())
        if n_matched(best) < 0.9 * total:
            msg = f"Only {n_matched(best)}/{total} markers of {fp} match the raw EEG"
            raise RuntimeError(msg)
        return float(best)

    def _preproc_eeg_events(self, timeline: dict[str, tp.Any]) -> list[dict[str, tp.Any]]:
        """Eeg events of the authors' preprocessed EEG, on the raw EEG clock."""
        pre_fp = self._preproc_eeg_file(timeline)
        if pre_fp is None:
            return []
        with nutils.ignore_all():
            pre_markers = mne.read_annotations(pre_fp.with_suffix(".vmrk"))
        offset = self._preproc_offset(self._raw_markers(timeline), pre_markers, pre_fp)
        return [dict(type="Eeg", start=offset, filepath=str(pre_fp), preproc="authors")]

    def _own_eeg_file(self, timeline: dict[str, tp.Any]) -> Path:
        """Own preprocessed EEG: the raw EEG path mirrored under preprocessed/, as .fif
        (e.g. preprocessed/sub-01/eeg/sub-01_task-x_eeg.fif). It must keep the raw
        EEG start (time 0 on the timeline clock)."""
        raw_fp = self._eeg_file(timeline)
        assert raw_fp is not None
        rel = raw_fp.relative_to(self._bids_root())
        return self._bids_root() / "preprocessed" / rel.with_suffix(".fif")

    def _readable_eeg_file(self, fp: Path) -> Path:
        """Some BrainVision headers point to the original (non-BIDS) data
        file names: write a fixed copy of the header next to the study."""
        if fp.suffix != ".vhdr":
            return fp
        text = fp.read_text(encoding="latin-1")
        targets = {"DataFile": fp.with_suffix(".eeg"), "MarkerFile": fp.with_suffix(".vmrk")}
        current = dict(re.findall(r"^(DataFile|MarkerFile)=(.*?)\s*$", text, flags=re.M))
        if all((fp.parent / current.get(k, "")).is_file() for k in targets):
            return fp
        fixed = self.path / "fixed_vhdr" / fp.name
        if not fixed.exists():
            for key, target in targets.items():
                text = re.sub(rf"^{key}=.*$", f"{key}={target}", text, flags=re.M)
            fixed.parent.mkdir(exist_ok=True)
            fixed.write_text(text, encoding="latin-1")
        return fixed

    def _raw_markers(self, timeline: dict[str, tp.Any]) -> mne.Annotations:
        """Markers of the raw EEG (read from the .vmrk when possible)."""
        eeg_fp = self._eeg_file(timeline)
        assert eeg_fp is not None
        marker_fp = eeg_fp.with_suffix(".vmrk")  # markers only, no need to read EEG data
        with nutils.ignore_all():
            return mne.read_annotations(marker_fp if marker_fp.exists() else eeg_fp)

    def _volume_onsets(self, timeline: dict[str, tp.Any]) -> np.ndarray:
        """Onsets (in s, EEG clock) of the fMRI volume triggers."""
        annots = self._raw_markers(timeline)
        onsets = annots.onset[np.asarray(annots.description) == self.VOLUME_MARKER]
        if not len(onsets):
            found = sorted(set(annots.description))
            msg = f"No {self.VOLUME_MARKER!r} volume marker for {timeline} (found {found})"
            raise RuntimeError(msg)
        return onsets

    def _task_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        """Task events from func/*_events.tsv (onsets relative to the first volume)."""
        fp = self._inherited_file(timeline, "func", "events.tsv")
        if fp is None:
            return pd.DataFrame()
        events = nutils.read_bids_events(fp).rename(columns={"onset": "start"})
        events = events.dropna(subset=["start"])  # empty lines (ds002336 task-fmriNF)
        events["duration"] = events.duration.fillna(0.0)
        if "trial_type" in events.columns:
            events["description"] = events.trial_type.astype(str)
            if pd.api.types.is_integer_dtype(events.trial_type):
                events["code"] = events.trial_type
        events["type"] = "Stimulus"
        return events

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        bold_fp = self._bold_file(timeline)
        bold_json = self._inherited_file(timeline, "func", "bold.json")
        assert bold_json is not None
        tr = float(json.loads(bold_json.read_text())["RepetitionTime"])

        onsets = self._volume_onsets(timeline)
        fmri_start = float(onsets[0])
        n_volumes = nibabel.load(bold_fp).shape[-1]
        if len(onsets) != n_volumes:
            logger.warning(
                "%s: %d volume markers in EEG but %d volumes in %s",
                self.__class__.__name__,
                len(onsets),
                n_volumes,
                bold_fp.name,
            )

        # one Eeg event per available EEG version, tagged with "preproc"
        eegs = []
        raw_fp = self._eeg_file(timeline)
        assert raw_fp is not None
        data_suffix = self.RAW_EEG_DATA.get(raw_fp.suffix)
        if data_suffix is None or raw_fp.with_suffix(data_suffix).exists():  # not only markers
            fp = str(self._readable_eeg_file(raw_fp))
            eegs.append(dict(type="Eeg", start=0.0, filepath=fp, preproc=self.RAW_EEG_PREPROC))
        eegs.extend(self._preproc_eeg_events(timeline))
        if not eegs:
            raise RuntimeError(f"No EEG data for {timeline} in {self.path}")

        # own preprocessed EEG: same start as the raw EEG, under preprocessed/
        own_fp = self._own_eeg_file(timeline)
        if own_fp.exists():
            eegs.append(dict(type="Eeg", start=0.0, filepath=str(own_fp), preproc="own"))

        fmri = dict(type="Fmri", start=fmri_start, filepath=str(bold_fp), frequency=1.0 / tr)
        stims = self._task_events(timeline)
        if len(stims) and self.EVENTS_RELATIVE_TO_FMRI:
            stims["start"] += fmri_start
        return pd.concat([pd.DataFrame([*eegs, fmri]), stims], ignore_index=True)


class Pereira2020Disentangling(_BidsEegFmri):
    """Confidence EEG-fMRI: simultaneous EEG-fMRI during perceptual decisions and confidence reports.

    Participants judged which of two flickering dot patterns contained more dots
    (or observed a decision being made) and then reported their confidence.

    Experimental Design:
        - Simultaneous 64-channel EEG (BrainAmp MR, 5000 Hz) and 3T fMRI (TR = 1.28 s)
        - 20 participants, 1 session, up to 7 runs of the main task
        - Paradigm: dot-numerosity discrimination + confidence rating

    Data Format:
        - BIDS (OpenNeuro ds002158), raw EEG in BrainVision format
        - Authors' preprocessed EEG (``derivatives/eegprep``, 250 Hz, MR
          artifacts removed, average reference) is only available as 4 s epochs
          (-2 to 2 s around each response), all runs merged in one file per
          subject: each epoch is an Eeg event (``preproc="authors"``) placed at
          its time in the raw EEG, so EEG is only available around responses
        - Preprocessed fMRI (``derivatives/fmriprep``) is not downloaded
        - The BrainVision headers reference the original file names; a fixed copy
          is written to ``fixed_vhdr/`` on load
        - Events: Eeg, Fmri, Stimulus (one per trial, with ``trial_type``,
          ``confidence``, ``choice_accuracy``, ...)
    """

    bibtex: tp.ClassVar[str] = """
    @article{pereira2020disentangling,
        title={Disentangling the origins of confidence in speeded perceptual judgments through multimodal imaging},
        author={Pereira, Michael and Faivre, Nathan and Iturrate, I{\\~n}aki and Wirthlin, Marco and Serafini, Luana and Martin, St{\\'e}phanie and Desvachez, Arnaud and Blanke, Olaf and Van De Ville, Dimitri and Mill{\\'a}n, Jos{\\'e} del R},
        journal={Proceedings of the National Academy of Sciences},
        volume={117},
        number={15},
        pages={8382--8390},
        year={2020},
        doi={10.1073/pnas.1918335117}
    }
    """
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds002158"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Simultaneous EEG-fMRI from 20 participants during perceptual decisions "
        "and confidence reports."
    )
    OPENNEURO_ID: tp.ClassVar[str] = "ds002158"
    # anat/fmap are not used (and sub-26's T1w returns HTTP 404 on OpenNeuro)
    DOWNLOAD_EXCLUDE: tp.ClassVar[tuple[str, ...]] = (
        "derivatives/",
        "sub-*/ses-*/anat",
        "sub-*/ses-*/fmap",
    )

    def _epochs_file(self, timeline: dict[str, tp.Any]) -> Path:
        tl = {k: v for k, v in timeline.items() if k != "run"}  # one file for all runs
        folder = self._bids_root() / "derivatives" / "eegprep"
        folder = folder / self._folder(tl, "eeg").relative_to(self._bids_root())
        return folder / f"{self._prefix(tl)}_eeg_preproc.set"

    def _preproc_eeg_events(self, timeline: dict[str, tp.Any]) -> list[dict[str, tp.Any]]:
        set_fp = self._epochs_file(timeline)
        if not set_fp.exists():
            return []
        info = _read_eeglab_epochs(str(set_fp))
        # Runs are merged in the file (in any order, some may be missing): find
        # the merged segment with the same time-locking events (responses) as
        # the raw EEG of this run, same types and same intervals
        lock_types = sorted(set(info["lock_types"]))
        annots = self._raw_markers(timeline)
        raw_desc = np.array([desc.split("/")[-1] for desc in annots.description])
        raw_onsets = annots.onset[np.isin(raw_desc, lock_types)]
        raw_types = list(raw_desc[np.isin(raw_desc, lock_types)])
        types, times = np.array(info["urevent_types"]), np.array(info["urevent_times"])
        segments = np.cumsum(types == "boundary")
        locks = np.where(np.isin(types, lock_types))[0]
        n = len(raw_types)
        offset = None  # time of the merged file -> time in the raw EEG of the run
        for k in range(len(locks) - n + 1):
            sel = locks[k : k + n]
            if (
                segments[sel[0]] == segments[sel[-1]]
                and list(types[sel]) == raw_types
                and np.allclose(np.diff(times[sel]), np.diff(raw_onsets), atol=0.02)
            ):
                offset = float(np.median(raw_onsets - times[sel]))
                break
        if offset is None:
            msg = "%s: no epochs of %s match the raw EEG of %s"
            logger.warning(msg, self.__class__.__name__, set_fp.name, timeline)
            return []
        duration = info["shape"][1] / info["sfreq"]
        events = []
        for epoch, urevent in enumerate(info["lock_urevents"]):
            if not sel[0] <= urevent <= sel[-1]:
                continue
            loader = study.SpecialLoader(method=self._load_epoch, timeline=timeline, epoch=epoch)
            events.append(
                dict(
                    type="Eeg",
                    start=times[urevent] + offset + info["tmin"],
                    duration=duration,
                    frequency=info["sfreq"],
                    filepath=loader.to_json(),
                    preproc="authors",
                )
            )
        return events

    def _load_epoch(self, timeline: dict[str, tp.Any], epoch: int) -> mne.io.RawArray:
        set_fp = self._epochs_file(timeline)
        info = _read_eeglab_epochs(str(set_fp))
        data = np.memmap(
            set_fp.with_suffix(".fdt"), dtype="<f4", mode="r", shape=info["shape"], order="F"
        )
        ch_types = ["ecg" if ch.upper() == "ECG" else "eeg" for ch in info["ch_names"]]
        mne_info = mne.create_info(info["ch_names"], info["sfreq"], ch_types)
        return mne.io.RawArray(data[:, :, epoch] * 1e-6, mne_info, verbose=False)  # µV -> V


class Lioi2020SimultaneousXp2(_BidsEegFmri):
    """EEG-fMRI-NF XP2: simultaneous EEG-fMRI during motor-imagery neurofeedback.

    Experimental Design:
        - Simultaneous 64-channel EEG (BrainAmp MR, 5000 Hz) and 3T fMRI (TR = 1 s)
        - 17 participants (XP2; XP1 is OpenNeuro ds002336)
        - Paradigm: 20 s block design alternating rest and right-hand motor
          imagery, with uni- (1dNF) or bi-dimensional (2dNF) neurofeedback;
          motor-imagery runs without feedback before/after (MIpre, MIpost)

    Data Format:
        - BIDS (OpenNeuro ds002338), raw EEG in BrainVision format
        - Block events from the dataset-level ``task-*_events.tsv``
        - EEG: authors' preprocessed EEG (``derivatives/sub-*/eeg_pp``, 200 Hz,
          MR gradient and pulse artifacts removed); only the markers of the raw
          EEG (``.vhdr``/``.vmrk``) are downloaded, to align it with the fMRI
    """

    bibtex: tp.ClassVar[str] = """
    @article{lioi2020simultaneous,
        title={Simultaneous {EEG-fMRI} during a neurofeedback task, a brain imaging dataset for multimodal data integration},
        author={Lioi, Giulia and Cury, Claire and Perronnet, Lorraine and Mano, Marsel and Bannier, Elise and L{\\'e}cuyer, Anatole and Barillot, Christian},
        journal={Scientific Data},
        volume={7},
        number={1},
        pages={173},
        year={2020},
        doi={10.1038/s41597-020-0498-3}
    }
    """
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds002338"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Simultaneous EEG-fMRI from 17 participants during motor imagery neurofeedback."
    )
    OPENNEURO_ID: tp.ClassVar[str] = "ds002338"
    DOWNLOAD_INCLUDE: tp.ClassVar[tuple[str, ...]] = (
        "*.json",
        "*.tsv",
        "sub-*/func",
        "sub-*/eeg/*.vhdr",
        "sub-*/eeg/*.vmrk",
        "sub-*/eeg/*.eeg",

    )
    DOWNLOAD_EXCLUDE: tp.ClassVar[tuple[str, ...]] = (
        "derivatives/",
    )
    PREPROC_EEG: tp.ClassVar[str | None] = "derivatives/{sub_dir}/eeg_pp/d_{prefix}_eeg_pp.vhdr"


class Lioi2020SimultaneousXp1(Lioi2020SimultaneousXp2):
    """EEG-fMRI-NF XP1: same setup as XP2 (``Lioi2020SimultaneousXp2``), to be pooled with it.

    Experimental Design:
        - Simultaneous 64-channel EEG (BrainAmp MR, 5000 Hz) and 3T fMRI (TR = 2 s)
        - 10 participants
        - Paradigm: 20 s blocks of rest and right-hand motor execution (motorloc)
          or motor imagery without feedback (MIpre, MIpost) or with EEG, fMRI
          or EEG-fMRI neurofeedback (eegNF, fmriNF, eegfmriNF); EEG and fMRI
          are recorded in all runs

    Data Format:
        - BIDS (OpenNeuro ds002336), same layout as ds002338; preprocessed EEG
          files have no ``d_`` prefix
        - sub-xp102 motorloc has no preprocessed EEG; the preprocessed EEG of
          sub-xp101 eegfmriNF and sub-xp106 MIpost does not match the raw EEG
          (markers disagree, no cross-correlation peak): raw EEG is loaded instead
        - EEG usually has up to 6 volume triggers less than the BOLD (missing
          at the end; the first trigger is the first volume)
    """

    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds002336"
    description: tp.ClassVar[str] = (
        "Simultaneous EEG-fMRI from 10 participants during motor imagery neurofeedback (XP1)."
    )
    OPENNEURO_ID: tp.ClassVar[str] = "ds002336"
    DOWNLOAD_INCLUDE: tp.ClassVar[tuple[str, ...]] = (
        *Lioi2020SimultaneousXp2.DOWNLOAD_INCLUDE,
        "sub-*/eeg/*.eeg"
    )
    DOWNLOAD_EXCLUDE: tp.ClassVar[tuple[str, ...]] = (
        *Lioi2020SimultaneousXp2.DOWNLOAD_EXCLUDE,
    )
    
    PREPROC_EEG: tp.ClassVar[str | None] = "derivatives/{sub_dir}/eeg_pp/{prefix}_eeg_pp.vhdr"

    def _preproc_eeg_events(self, timeline: dict[str, tp.Any]) -> list[dict[str, tp.Any]]:
        try:
            return super()._preproc_eeg_events(timeline)
        except RuntimeError as e:  # preprocessed EEG cannot be placed on the raw clock
            logger.warning("%s: preprocessed EEG not loaded (%s)", self.__class__.__name__, e)
            return []


class Daly2019Electroencephalography(_BidsEegFmri):
    """Music EEG-fMRI: joint EEG-fMRI during affective music listening.

    Experimental Design:
        - Simultaneous 30-channel EEG (1000 Hz, EDF) and 3T fMRI (TR = 2 s)
        - 21 participants
        - Paradigm: listening to classical and generated music while
          continuously reporting felt affect; washout runs

    Data Format:
        - BIDS (OpenNeuro ds002725), EEG in EDF format
        - EEG has no annotations: triggers are in ``events.tsv``, which is the
          same file in ``eeg/`` and ``func/`` (onsets on the EEG clock)
        - Scanner TTLs (code 10) can include a few test pulses before the run:
          the fMRI start is the first TTL of the last continuous TTL train
        - Events: Eeg, Fmri, Stimulus (``code`` = trigger value, e.g. 768 for
          music trial start, see ``events.json``)
    """

    bibtex: tp.ClassVar[str] = """
    @article{daly2019electroencephalography,
        title={Electroencephalography reflects the activity of sub-cortical brain regions during approach-withdrawal behaviour while listening to music},
        author={Daly, Ian and Williams, Duncan and Hwang, Faustina and Kirke, Alexis and Miranda, Eduardo R and Nasuto, Slawomir J},
        journal={Scientific Reports},
        volume={9},
        number={1},
        pages={9415},
        year={2019},
        doi={10.1038/s41598-019-45105-2}
    }
    """
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds002725"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Joint EEG-fMRI from 21 participants listening to affective music."
    )
    OPENNEURO_ID: tp.ClassVar[str] = "ds002725"
    VOLUME_MARKER: tp.ClassVar[str] = "10"  # TTL pulse from the scanner
    EVENTS_RELATIVE_TO_FMRI: tp.ClassVar[bool] = False

    def _volume_onsets(self, timeline: dict[str, tp.Any]) -> np.ndarray:
        fp = self._inherited_file(timeline, "eeg", "events.tsv")
        assert fp is not None
        events = nutils.read_bids_events(fp)
        onsets = events.onset[events.trial_type.astype(str) == self.VOLUME_MARKER]
        onsets = onsets.to_numpy()
        breaks = np.where(np.diff(onsets) > 5.0)[0]  # gaps between TTL trains
        return onsets[breaks[-1] + 1 :] if len(breaks) else onsets


class Liwicki2025Synchronous(_BidsEegFmri):
    """Inner speech EEG-fMRI: synchronous EEG-fMRI during inner speech.

    Experimental Design:
        - Simultaneous 64-channel EEG (BrainAmp MR plus, 5000 Hz) and 3T fMRI (TR = 2 s)
        - 3 participants, 2 sessions (EEG only recorded in session 2)
        - Paradigm: 8 words, each repeated 40 times, imagined (inner speech)

    Data Format:
        - BIDS (OpenNeuro ds006033), EEG in BrainVision format, pulse artifact
          already corrected (BrainVision Analyzer)
        - Volume triggers are ``Scanner/Scan Start`` (``R128`` are slice triggers)
        - Events: Eeg, Fmri, Stimulus (``description`` = word or rest)

    Notes:
        - The EEG ``events.tsv`` onsets are ~20 s earlier than the fMRI ones;
          the fMRI ``events.tsv`` is used here.
        - sub-01 has 900 volume triggers but 889 BOLD volumes; the fMRI start
          is set to the first trigger (to be checked).
    """

    bibtex: tp.ClassVar[str] = """
    @misc{liwicki2025synchronous,
        title={Synchronous {EEG} and {fMRI} dataset on inner speech},
        author={Simistira Liwicki, Foteini},
        publisher={OpenNeuro},
        year={2025},
        url={https://openneuro.org/datasets/ds006033}
    }
    """
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds006033"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Simultaneous EEG-fMRI from 3 participants during inner speech."
    )
    OPENNEURO_ID: tp.ClassVar[str] = "ds006033"
    VOLUME_MARKER: tp.ClassVar[str] = "Scanner/Scan Start"
    RAW_EEG_PREPROC: tp.ClassVar[str] = "raw"  # pulse artifact corrected


class Telesford2023Open(_BidsEegFmri):
    """NATVIEW_EEGFMRI: simultaneous EEG-fMRI during naturalistic viewing.

    Experimental Design:
        - Simultaneous 64-channel EEG (BrainAmp MR, 5000 Hz) and 3T fMRI (TR = 2.1 s)
        - 22 participants, up to 2 sessions
        - Paradigm: flickering checkerboard, resting state, inscapes and movie
          clips (Despicable Me, The Present, Monkey videos, ...)

    Data Format:
        - BIDS-like raw data from the INDI S3 bucket
          (``s3://fcp-indi/data/Projects/NATVIEW_EEGFMRI/raw_data``)
        - EEG in EEGLAB ``.set`` format (the BrainVision ``.eeg`` files are not
          shared); volume triggers are read from the ``.vmrk`` files
        - The authors' preprocessed EEG (``preproc_data``, 250 Hz, MR artifacts
          removed) is downloaded to ``download/derivatives/preproc_data`` but not
          loaded: segments were cut out of it (15-17 ``boundary`` events per run)
          and the remaining pieces cannot be reliably re-aligned to the fMRI
        - Only EEG, BOLD and their sidecars are downloaded
        - Events: Eeg, Fmri (stimulus markers are in the EEG annotations)
    """

    requirements: tp.ClassVar[tuple[str, ...]] = ("boto3",)
    bibtex: tp.ClassVar[str] = """
    @article{telesford2023open,
        title={An open-access dataset of naturalistic viewing using simultaneous {EEG-fMRI}},
        author={Telesford, Qawi K and Gonzalez-Moreira, Eduardo and Xu, Ting and Tian, Yiwen and Colcombe, Stanley J and Cloud, Jessica and Russ, Brian E and Falchier, Arnaud and Nentwich, Maximilian and Madsen, Jens and Parra, Lucas C and Schroeder, Charles E and Milham, Michael P and Franco, Alexandre R},
        journal={Scientific Data},
        volume={10},
        number={1},
        pages={554},
        year={2023},
        doi={10.1038/s41597-023-02458-8}
    }
    """
    url: tp.ClassVar[str] = "https://fcon_1000.projects.nitrc.org/indi/retro/nat_view.html"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "Simultaneous EEG-fMRI from 22 participants during naturalistic viewing."
    )
    EEG_EXTENSIONS: tp.ClassVar[tuple[str, ...]] = (".set",)

    def _download(self, overwrite: bool = False) -> None:
        s3 = dict(dset_dir=self.path, bucket="fcp-indi")
        prefix = "data/Projects/NATVIEW_EEGFMRI/"
        raw_include = ["*.json", "*_events.tsv", "*_eeg.*", "*_bold.nii.gz"]
        download.S3(
            study="natview", prefix=prefix + "raw_data", include=raw_include, **s3
        ).download(overwrite=overwrite)
        download.S3(
            study="natview-preproc",
            prefix=prefix + "preproc_data",
            output_dir=self._bids_root() / "derivatives" / "preproc_data",
            include=["*/eeg/*_eeg.set", "*/eeg/*_eeg.json", "*/eeg/*_events.tsv"],
            **s3,
        ).download(overwrite=overwrite)



class Gu2023Simultaneous(_BidsEegFmri):
    """Sleep EEG-fMRI: simultaneous EEG-fMRI during rest and sleep.

    Experimental Design:
        - Simultaneous 32-channel EEG (Brain Products, 5000 Hz) and 3T fMRI (TR = 2.1 s)
        - 33 participants
        - Paradigm: two 10 min resting-state runs (before/after a visuomotor
          task) and several 15 min sleep runs

    Data Format:
        - BIDS (OpenNeuro ds003768), raw EEG in BrainVision format; no
          preprocessed EEG is shared (MR artifacts still present)
        - The authors' preprocessed fMRI (``sourcedata/fMRI_processed_file_1``)
          and eye tracking are not downloaded
        - Volume triggers: ``R128``
        - Events: Eeg, Fmri, SleepStage (30 s epochs from
          ``sourcedata/sub-*-sleep-stage.tsv``, scored from the first fMRI
          volume; ``uncertain``/``unscorable`` epochs are left out)
        - EEG markers ``S  1`` during sleep runs are button presses signalling
          wakefulness (in the EEG annotations)
    """

    bibtex: tp.ClassVar[str] = """
    @article{gu2023simultaneous,
        title={Simultaneous {EEG} and functional {MRI} data during rest and sleep from humans},
        author={Gu, Yameng and Sainburg, Lucas E and Han, Feng and Liu, Xiao},
        journal={Data in Brief},
        volume={48},
        pages={109059},
        year={2023},
        doi={10.1016/j.dib.2023.109059}
    }
    """
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds003768"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Simultaneous EEG-fMRI from 33 participants during rest and sleep."
    )
    OPENNEURO_ID: tp.ClassVar[str] = "ds003768"
    # authors' preprocessed fMRI (217 GB, fMRI is loaded raw) and eye tracking
    DOWNLOAD_EXCLUDE: tp.ClassVar[tuple[str, ...]] = (
        "sourcedata/fMRI_processed_file_1",
        "sourcedata/EyeTracker_data",
    )
    _STAGES: tp.ClassVar[dict[str, str]] = {"W": "W", "1": "N1", "2": "N2", "3": "N3", "R": "R"}

    def _task_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        fp = self._bids_root() / "sourcedata" / f"sub-{timeline['subject']}-sleep-stage.tsv"
        if not fp.exists():
            return pd.DataFrame()
        stages = pd.read_csv(fp, sep="\t", dtype=str)
        # sub-32: one line is separated by a space ("810 W")
        bad = stages.epoch_start_time_sec.str.contains(" ", na=False)
        if bad.any():
            split = stages.loc[bad, "epoch_start_time_sec"].str.split(" ", n=1, expand=True)
            stages.loc[bad, "epoch_start_time_sec"] = split[0]
            stages.loc[bad, "30-sec_epoch_sleep_stage"] = split[1]
        stages = stages[stages.session == f"task-{timeline['task']}_run-{timeline['run']}"]
        stage = stages["30-sec_epoch_sleep_stage"].str.strip().map(self._STAGES)
        return pd.DataFrame(
            dict(
                type="SleepStage",
                start=stages.epoch_start_time_sec.astype(float),  # from the first volume
                duration=30.0,
                stage=stage,
            )
        ).dropna(subset=["stage"])


class Cha2025Sustained(_BidsEegFmri):
    """gradCPT EEG-fMRI: simultaneous EEG-fMRI during sustained attention tasks.

    Experimental Design:
        - Simultaneous 64-channel EEG (Brain Products BrainCap MR, 63 EEG + ECG,
          5000 Hz) and 3T fMRI (TR = 2 s); DWI also acquired (not loaded)
        - 28 participants (sub-003 withdrew)
        - Paradigm: eyes open/closed rest, 15 Hz checkerboard, gradCPT,
          imagery and gradCPT with imagery

    Data Format:
        - BIDS (OpenNeuro ds006040). EEG task names add ``ON`` (scanner on) to
          the BOLD task name (``task-GRADON`` <-> ``task-GRAD``); ``OFF`` runs
          (no fMRI) are not loaded
        - EEG: authors' preprocessed EEG (``derivatives/sub-*/eeg/*_desc-anc0qrsrm``,
          EEGLAB, 5000 Hz, gradient and QRS/pulse artifacts removed); only the
          headers of the raw EEG (``.set``, with markers) are downloaded, to align it
        - Volume triggers: ``T  1`` (``TR`` in the preprocessed EEG; sub-025
          CB/GRAD first TR identified manually by the authors)
        - Events: Eeg, Fmri. No ``events.tsv``: task start/stop markers are in
          the EEG annotations (``S245``/``S254`` start, ``S155`` stop) and trial
          information in ``beh/`` (downloaded, not loaded)
    """

    bibtex: tp.ClassVar[str] = """
    @misc{cha2025sustained,
        title={Sustained Attention Task (gradCPT) Dataset using simultaneous {EEG-fMRI} and {DTI}},
        author={Cha, Younghwa and Lee, Yeji and Ji, Eunhee and Han, SoHyun and Min, Sunhyun and Kim, Hyoungkyu and Cho, Minseo and Lee, Hae Seong and Park, Youngjai and Moon, Joon-Young},
        publisher={OpenNeuro},
        year={2025},
        doi={10.18112/openneuro.ds006040.v1.0.2}
    }
    """
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds006040"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Simultaneous EEG-fMRI from 28 participants during rest, checkerboard and "
        "sustained attention tasks."
    )
    OPENNEURO_ID: tp.ClassVar[str] = "ds006040"
    DOWNLOAD_INCLUDE: tp.ClassVar[tuple[str, ...]] = (
        "*.json",
        "*.tsv",
        "README_dataset.txt",
        "sub-*/func",
        "sub-*/beh",
        "sub-*/eeg/*ON_run-*_eeg.set",
        "sub-*/eeg/*ON_run-*_eeg.fdt"
    )
    VOLUME_MARKER: tp.ClassVar[str] = "T  1"
    EEG_EXTENSIONS: tp.ClassVar[tuple[str, ...]] = (".set",)
    EEG_TASK_SUFFIX: tp.ClassVar[str] = "ON"
    RAW_EEG_DATA: tp.ClassVar[dict[str, str]] = {".set": ".fdt"}
    PREPROC_EEG: tp.ClassVar[str | None] = (
        "derivatives/{sub_dir}/eeg/{prefix}_desc-anc0qrsrm_eeg.set"
    )

    def _eeg_prefix(self, timeline: dict[str, tp.Any]) -> str:
        # EEG runs are not zero-padded (sub-015 CB BOLD is run-01, its EEG run-1)
        return super()._eeg_prefix({**timeline, "run": str(int(timeline["run"]))})

    def _raw_markers(self, timeline: dict[str, tp.Any]) -> mne.Annotations:
        try:
            return super()._raw_markers(timeline)
        except (TypeError, ValueError):  # sub-019 IMGRADON run-2: corrupted raw .set
            pre_fp = self._preproc_eeg_file(timeline)
            if pre_fp is None:
                raise
            logger.warning("Unreadable raw EEG markers for %s, using %s", timeline, pre_fp.name)
            with nutils.ignore_all():
                annots = mne.read_annotations(pre_fp)
            # same length as the raw EEG; volume triggers are named "TR" by the authors
            desc = [self.VOLUME_MARKER if d == "TR" else d for d in annots.description]
            return mne.Annotations(annots.onset, annots.duration, desc)

    def _preproc_eeg_events(self, timeline: dict[str, tp.Any]) -> list[dict[str, tp.Any]]:
        # markers are inside the EEGLAB .set (no .vmrk)
        pre_fp = self._preproc_eeg_file(timeline)
        if pre_fp is None:
            return []
        with nutils.ignore_all():
            pre_markers = mne.read_annotations(pre_fp)
        offset = self._preproc_offset(self._raw_markers(timeline), pre_markers, pre_fp)
        return [dict(type="Eeg", start=offset, filepath=str(pre_fp), preproc="authors")]


class Kucyi2026Multisession(_BidsEegFmri):
    """Experience sampling EEG-fMRI: simultaneous EEG-fMRI with online experience sampling.

    Experimental Design:
        - Simultaneous 31-channel EEG + ECG + 4 carbon wire loops (BrainAmp MR,
          5000 Hz) and 3T fMRI (TR = 2 s, multiband 3)
        - 24 participants, 2 sessions each
        - Paradigm: one gradCPT run and three resting runs with intermittent
          experience sampling (13 thought dimensions rated per probe) per session

    Data Format:
        - BIDS (OpenNeuro ds007216). EEG: authors' preprocessed EEG
          (``derivatives/sub-*/ses-*/eeg/*_eeg_Bergen_CWreg.set``, EEGLAB, 250 Hz,
          gradient artifact removed and carbon wire loop regression); only the
          headers and markers of the raw EEG are downloaded, to align it
        - Volume triggers: ``T  1``. Resting runs were stopped manually, so the
          EEG usually has 1-2 extra triggers at the end (not matching a volume);
          the first trigger is always the first volume
        - Events: Eeg, Fmri, Stimulus (``func/*_events.tsv``, onsets relative
          to the first volume; ``events_corrected.tsv`` with mind-blank
          corrections is used when provided). Experience sampling ratings:
          ``description`` is the rated dimension (``item``), ``response`` the
          rating; gradCPT: one event per stimulus with ``cue`` and ``response``
    """

    bibtex: tp.ClassVar[str] = """
    @misc{kucyi2026multisession,
        title={A multi-session simultaneous {EEG-fMRI} dataset with online experience sampling},
        author={Kucyi, Aaron and Shareef-Trudeau, Lotus and Braun, David and Peng, Huiling and Bounyarith, Tiara and Li, Janet Z},
        publisher={OpenNeuro},
        year={2026},
        doi={10.18112/openneuro.ds007216.v1.1.0}
    }
    """
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds007216"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Simultaneous EEG-fMRI from 24 participants over two sessions during "
        "gradCPT and rest with experience sampling."
    )
    OPENNEURO_ID: tp.ClassVar[str] = "ds007216"
    # raw EEG data, intermediate preprocessing steps, anat/fmap and macOS "._" files
    DOWNLOAD_EXCLUDE: tp.ClassVar[tuple[str, ...]] = (
        "*_eeg_events.*",
        "*_Bergen.set",
        "*_Bergen.fdt",
        "sub-*/ses-*/anat",
        "sub-*/ses-*/fmap",
        "**/._*",
    )
    VOLUME_MARKER: tp.ClassVar[str] = "T  1"

    def _preproc_eeg_file(self, timeline: dict[str, tp.Any]) -> Path | None:
        # sub-023 ses-002 files are named *_eeg_events_events_Bergen_CWreg.set
        folder = self._bids_root() / "derivatives" / self._folder(timeline, "eeg").relative_to(
            self._bids_root()
        )
        return next(folder.glob(f"{self._eeg_prefix(timeline)}_eeg*_Bergen_CWreg.set"), None)

    def _raw_markers(self, timeline: dict[str, tp.Any]) -> mne.Annotations:
        # "Tr/T  1" in the .vmrk, "T  1" in the preprocessed EEG
        annots = super()._raw_markers(timeline)
        desc = [d.split("/", 1)[-1] for d in annots.description]
        return mne.Annotations(annots.onset, annots.duration, desc)

    def _inherited_file(
        self, timeline: dict[str, tp.Any], datatype: str, suffix: str
    ) -> Path | None:
        if suffix == "events.tsv":  # corrected ratings replace events.tsv (v1.1.0)
            fp = super()._inherited_file(timeline, datatype, "events_corrected.tsv")
            if fp is not None:
                return fp
        return super()._inherited_file(timeline, datatype, suffix)

    def _task_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        events = super()._task_events(timeline)
        if "item" in events.columns:  # experience sampling: rated dimension
            events["description"] = events["item"].astype(str)
        # "offset" (= onset + duration) would clash with the neuralset event field
        return events.drop(columns="offset", errors="ignore")

    def _preproc_eeg_events(self, timeline: dict[str, tp.Any]) -> list[dict[str, tp.Any]]:
        # markers are inside the EEGLAB .set (no .vmrk)
        pre_fp = self._preproc_eeg_file(timeline)
        if pre_fp is None:
            return []
        with nutils.ignore_all():
            pre_markers = mne.read_annotations(pre_fp)
        offset = self._preproc_offset(self._raw_markers(timeline), pre_markers, pre_fp)
        return [dict(type="Eeg", start=offset, filepath=str(pre_fp), preproc="authors")]


class Pisauro2017Neural(_BidsEegFmri):
    """Value-based decisions EEG-fMRI: simultaneous EEG-fMRI during food choices.

    Experimental Design:
        - Simultaneous 64-channel EEG (5000 Hz, shared at 1000 Hz) and 3T fMRI (TR = 2.5 s)
        - 22 participants, 2 runs each
        - Paradigm: choice between two food items (left/right) rated beforehand

    Data Format:
        - OpenNeuro ds002734 (BIDS fMRI; EEG not in BIDS): only the authors'
          preprocessed EEG is shared, as MATLAB files
          (``sub-*/EEG/EEG_data_subXX_run_YY.mat``, ``Y`` = 64 x time in µV,
          gradient + BCG artifacts removed, 0.5-40 Hz); no channel names, no
          markers
        - Alignment from ``sub-*/EEG/EEG_events_subXX.mat``: event times are in
          ms on the EEG clock and the first volume of the shared BOLD is at
          ``t0`` + 25 s (checked against ``func/*_events.tsv`` stimulus onsets)
        - Events: Eeg, Fmri, Stimulus (``description`` = ``stim`` with ``rt``,
          ``choose_left``, ``correct``, ``rating_left``/``rating_right``,
          ``difficulty``; ``resp`` at the response). The ``resp`` rows of
          ``func/*_events.tsv`` have the stimulus onset, so events come from
          the EEG files
    """

    bibtex: tp.ClassVar[str] = """
    @article{pisauro2017neural,
        title={Neural correlates of evidence accumulation during value-based decisions revealed via simultaneous {EEG-fMRI}},
        author={Pisauro, M Andrea and Fouragnan, Elsa and Retzler, Chris and Philiastides, Marios G},
        journal={Nature Communications},
        volume={8},
        number={1},
        pages={15808},
        year={2017},
        doi={10.1038/ncomms15808}
    }
    """
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds002734"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Simultaneous EEG-fMRI from 22 participants during value-based food choices."
    )
    OPENNEURO_ID: tp.ClassVar[str] = "ds002734"
    DOWNLOAD_EXCLUDE: tp.ClassVar[tuple[str, ...]] = ("sub-*/anat",)
    RAW_EEG_PREPROC: tp.ClassVar[str] = "authors"
    RAW_EEG_DATA: tp.ClassVar[dict[str, str]] = {".mat": ".eeg"}  # raw EEG is not shared
    EVENTS_RELATIVE_TO_FMRI: tp.ClassVar[bool] = False  # events built on the EEG clock
    FMRI_DELAY: tp.ClassVar[float] = 25.0  # first shared volume, after t0 (s)

    def _eeg_file(self, timeline: dict[str, tp.Any]) -> Path | None:
        sub, run = timeline["subject"], timeline["run"]
        fp = self._folder(timeline, "EEG") / f"EEG_data_sub{sub}_run_{run}.mat"
        return fp if fp.exists() else None

    def _eeg_events(self, timeline: dict[str, tp.Any]) -> tp.Any:
        import scipy.io

        fp = self._folder(timeline, "EEG") / f"EEG_events_sub{timeline['subject']}.mat"
        events = scipy.io.loadmat(fp, squeeze_me=True, struct_as_record=False)["events"]
        return events[int(timeline["run"]) - 1]

    def _volume_onsets(self, timeline: dict[str, tp.Any]) -> np.ndarray:
        # no volume markers: first volume from t0, then one per TR
        n_volumes = nibabel.load(self._bold_file(timeline)).shape[-1]
        tr = float(json.loads(self._inherited_file(timeline, "func", "bold.json").read_text())["RepetitionTime"])  # type: ignore
        return self._eeg_events(timeline).t0 / 1000 + self.FMRI_DELAY + tr * np.arange(n_volumes)

    def _preproc_eeg_events(self, timeline: dict[str, tp.Any]) -> list[dict[str, tp.Any]]:
        import scipy.io

        fp = self._eeg_file(timeline)
        n_samples = dict((n, s) for n, s, _ in scipy.io.whosmat(fp))["Y"][1]
        freq = float(scipy.io.loadmat(fp, variable_names=["fd"], squeeze_me=True)["fd"])
        loader = study.SpecialLoader(method=self._load_eeg, timeline=timeline)
        return [
            dict(
                type="Eeg",
                start=0.0,
                duration=n_samples / freq,
                frequency=freq,
                filepath=loader.to_json(),
                preproc="authors",
            )
        ]

    def _load_eeg(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        import scipy.io

        mat = scipy.io.loadmat(self._eeg_file(timeline), variable_names=["Y", "fd"], squeeze_me=True)
        names = [f"EEG{k + 1:02d}" for k in range(mat["Y"].shape[0])]  # channel names not shared
        info = mne.create_info(names, float(mat["fd"]), ch_types="eeg")
        return mne.io.RawArray(mat["Y"] * 1e-6, info, verbose=False)  # µV -> V

    def _task_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        ev = self._eeg_events(timeline)
        trials = dict(
            rt=ev.rt / 1000,
            choose_left=ev.chooseL.astype(bool),
            correct=ev.corr.astype(bool),
            rating_left=ev.ratL,
            rating_right=ev.ratR,
            difficulty=ev.diffV,
        )
        stim = pd.DataFrame(dict(start=ev.tstim / 1000, duration=1.25, description="stim", **trials))
        resp = pd.DataFrame(dict(start=ev.respt / 1000, duration=0.0, description="resp", **trials))
        events = pd.concat([stim, resp], ignore_index=True).sort_values("start")
        events["type"] = "Stimulus"
        return events.reset_index(drop=True)


@lru_cache
def _gradient_onset(fp: str) -> tuple[float, float]:
    """Start and stop (s) of the MR gradient artifact in a raw in-scanner EEG
    (longest block where the >100 Hz power is above half its 90th percentile)."""
    import scipy.signal

    with nutils.ignore_all():
        raw = mne.io.read_raw_eeglab(fp, preload=True, verbose=False)
    data = raw.pick("eeg").get_data()
    sos = scipy.signal.butter(4, 100, "highpass", fs=raw.info["sfreq"], output="sos")
    power = (scipy.signal.sosfiltfilt(sos, data, axis=1) ** 2).mean(0)
    step = int(raw.info["sfreq"] / 100)  # 10 ms bins
    env = np.sqrt(power[: len(power) // step * step].reshape(-1, step).mean(1))
    above = np.flatnonzero(env > 0.5 * np.percentile(env, 90))
    blocks = np.split(above, np.flatnonzero(np.diff(above) > 100) + 1)  # gaps > 1 s
    block = max(blocks, key=len)
    return block[0] / 100, (block[-1] + 1) / 100


class GallegoRudolf2022Preservation(_BidsEegFmri):
    """EEG-fMRI artifact dataset: EEG outside the scanner, inside without and with fMRI.

    Experimental Design:
        - 32-channel EEG + ECG (1000 Hz) and 3T GE fMRI (TR = 2 s), 20 participants
        - Paradigm: eyes-closed resting state (10 min) and an eyes open / eyes
          closed block task (20 s blocks), simultaneously with fMRI; the same EEG
          was also recorded outside the scanner and inside without scanning

    Data Format:
        - Mendeley Data (doi:10.17632/crhybxpdy6.2): two BIDS trees,
          ``BIDS_dataset_EEG`` (EEGLAB) and ``BIDS_dataset_MRI``
        - Timelines: session 1 BOLD (recorded with the EEG cap) ``eoec`` <->
          EEG ``fmrieoec`` and ``rest`` <-> EEG ``fmrirestingec``. Not loaded (no
          paired recording): EEG ``outside``, ``inside``, ``fmrirestingeo`` (130 s)
          and session 2 BOLD (without cap)
        - EEG: authors' gradient-corrected EEG
          (``derivatives/Gradient_artifact_corrected/*_gac_eeg.set``, BCG still
          present, same length as raw) and raw EEG. Channel names are not shared
          (``E1``-``E32``, ``ECG``)
        - No markers in the EEG: the volumes are found from the gradient artifact
          in the raw EEG. It lasts 5 TRs (sub-001, sub-002) or 10 TRs (others)
          longer than the saved BOLD: 5 unsaved volumes at the end and, in most
          runs, 5 at the start. First volume = artifact end - (n_volumes + 5) TR,
          snapped to the artifact onset grid (checked on the eoec runs with the
          alpha EC/EO contrast of the EEG, and on sub-001 with the BOLD response)
        - Runs without raw EEG data on Mendeley are not loaded (sub-003 eoec;
          rest of sub-002, 003, 004, 005, 008); sub-019 has no rest BOLD
        - Events: Eeg, Fmri, Stimulus (``EO``/``EC`` blocks of ``eoec``)
    """

    bibtex: tp.ClassVar[str] = """
    @article{gallegorudolf2022preservation,
        title={Preservation of {EEG} spectral power features during simultaneous {EEG-fMRI}},
        author={Gallego-Rudolf, Jonathan and Corsi-Cabrera, Mar{\\'i}a and Concha, Luis and Ricardo-Garcell, Josefina and Pasaye-Alcaraz, Erick},
        journal={Frontiers in Neuroscience},
        volume={16},
        pages={951321},
        year={2022},
        doi={10.3389/fnins.2022.951321}
    }
    """
    url: tp.ClassVar[str] = "https://data.mendeley.com/datasets/crhybxpdy6/2"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "Simultaneous EEG-fMRI from 20 participants during eyes-closed rest and an "
        "eyes open / eyes closed task, with EEG also recorded outside the scanner."
    )
    MENDELEY_ID: tp.ClassVar[str] = "crhybxpdy6"
    MENDELEY_VERSION: tp.ClassVar[int] = 2
    EEG_TASKS: tp.ClassVar[dict[str, str]] = {"eoec": "fmrieoec", "rest": "fmrirestingec"}
    EEG_EXTENSIONS: tp.ClassVar[tuple[str, ...]] = (".set",)
    RAW_EEG_DATA: tp.ClassVar[dict[str, str]] = {".set": ".fdt"}
    EXTRA_TRS: tp.ClassVar[int] = 5

    def _download(self, overwrite: bool = False) -> None:
        import urllib.request

        api = f"https://data.mendeley.com/public-api/datasets/{self.MENDELEY_ID}"
        headers = {"User-Agent": "curl/8.5.0"}  # default Python user agent is refused

        def get(url: str) -> tp.Any:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=headers)))

        folders = {f["id"]: f for f in get(f"{api}/folders/{self.MENDELEY_VERSION}")}

        def folder_path(fid: str) -> str:
            parts = []
            while fid in folders:
                parts.append(folders[fid]["name"])
                fid = folders[fid].get("parent_id", "")
            return "/".join(reversed(parts[:-1]))  # drop the top "Simultaneous EEG-fMRI" folder

        eeg_tasks = "|".join(self.EEG_TASKS.values())
        for fid in folders:
            for f in get(f"{api}/files?folder_id={fid}&version={self.MENDELEY_VERSION}"):
                rel = f"{folder_path(fid)}/{f['filename']}"
                if "/anat/" in rel or "/ses-002/" in rel:  # no paired EEG (ses-002: no cap)
                    continue
                if "_eeg." in rel and not re.search(rf"task-({eeg_tasks})_", rel):
                    continue  # EEG outside/inside the scanner, without fMRI
                out = self._bids_root() / rel
                if out.exists() and out.stat().st_size == f["size"] and not overwrite:
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                url = f["content_details"]["download_url"]
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as r:
                    out.write_bytes(r.read())

    def _mri_root(self) -> Path:
        return self._bids_root() / "BIDS_dataset_MRI"

    def _folder(self, timeline: dict[str, tp.Any], datatype: str) -> Path:
        folder = self._mri_root() / f"sub-{timeline['subject']}" / f"ses-{timeline['session']}"
        return folder / datatype

    def _inherited_file(
        self, timeline: dict[str, tp.Any], datatype: str, suffix: str
    ) -> Path | None:
        candidates = [
            self._folder(timeline, datatype) / f"{self._prefix(timeline)}_{suffix}",
            self._mri_root() / f"task-{timeline['task']}_{suffix}",
        ]
        return next((fp for fp in candidates if fp.exists()), None)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for bold in sorted(self._mri_root().glob("sub-*/ses-*/func/*_bold.nii*")):
            entities = dict(re.findall(r"(sub|ses|task|run)-([a-zA-Z0-9]+)", bold.name))
            timeline = {_ENTITIES[k]: v for k, v in entities.items()}
            if self._eeg_file(timeline) is not None:  # session 1 only
                yield timeline

    def _eeg_file(self, timeline: dict[str, tp.Any]) -> Path | None:
        task = self.EEG_TASKS.get(timeline["task"])
        if timeline.get("session") != "001" or task is None:
            return None
        sub = f"sub-{timeline['subject']}"
        fp = self._bids_root() / "BIDS_dataset_EEG" / sub / "eeg" / f"{sub}_task-{task}_eeg.set"
        # the raw data is needed to find the fMRI onset (missing for 6 runs on Mendeley)
        return fp if fp.with_suffix(".fdt").exists() else None

    def _preproc_eeg_file(self, timeline: dict[str, tp.Any]) -> Path | None:
        raw_fp = self._eeg_file(timeline)
        assert raw_fp is not None
        folder = self._bids_root() / "BIDS_dataset_EEG" / "derivatives" / "Gradient_artifact_corrected"
        fp = folder / raw_fp.name.replace("_eeg.set", "_gac_eeg.set")
        return fp if fp.exists() else None

    def _preproc_eeg_events(self, timeline: dict[str, tp.Any]) -> list[dict[str, tp.Any]]:
        pre_fp = self._preproc_eeg_file(timeline)
        if pre_fp is None:
            return []
        # same length as the raw EEG (no markers to match)
        return [dict(type="Eeg", start=0.0, filepath=str(pre_fp), preproc="authors")]

    def _volume_onsets(self, timeline: dict[str, tp.Any]) -> np.ndarray:
        raw_fp = self._eeg_file(timeline)
        assert raw_fp is not None
        bold_json = self._inherited_file(timeline, "func", "bold.json")
        assert bold_json is not None
        tr = float(json.loads(bold_json.read_text())["RepetitionTime"])
        n_volumes = nibabel.load(self._bold_file(timeline)).shape[-1]
        start, stop = _gradient_onset(str(raw_fp))
        if stop - start < n_volumes * tr - 0.5:
            raise RuntimeError(f"Gradient artifact of {raw_fp.name} shorter than the BOLD")
        # the scanner runs EXTRA_TRS volumes after the last saved one (and, in most
        # runs, 5 unsaved volumes before the first one)
        first = stop - (n_volumes + self.EXTRA_TRS) * tr
        if start > 0:  # artifact onset recorded: snap to its volume grid
            first = start + max(round((first - start) / tr), 0) * tr
        return first + tr * np.arange(n_volumes)


class VanDerMeer2016Eyes(_BidsEegFmri):
    """CWL EEG-fMRI: eyes open / eyes closed with carbon wire loop motion channels.

    Experimental Design:
        - 8 participants, 4 on a 3T Siemens Trio (TR = 1.95 s) and 4 on a 3T
          Siemens Verio (TR = 2 s); 30 EEG channels + EOG + ECG + 6 carbon wire
          loops (CW1-CW6), 5000 Hz
        - Paradigm: ~30 s blocks of eyes open / eyes closed, with flashes at the
          end of each block, recorded with the helium pump on and off
          (``acq-pumpon`` / ``acq-pumpoff``)

    Data Format:
        - NITRC ``cwleegfmri_data``: one tarball per participant (not BIDS),
          reorganised into a BIDS-like tree at download (``sub-01``-``sub-04`` =
          ``trio1``-``trio4``, ``sub-05``-``sub-08`` = ``verio5``-``verio8``,
          see ``participants.tsv``)
        - EEG: raw in-scanner EEG (5000 Hz) and the authors' MR-corrected EEG
          (``derivatives/mrcorrected``, gradient + pulse artifact removed,
          1000 Hz, same length as the raw). EEG inside the scanner without fMRI
          (``task-noscan``) and outside (``task-outside``) is kept but not loaded
        - Not loaded: MNI-normalised BOLD (``derivatives/normalized``), SPM models
          (``derivatives/spm_model``), T1, motion parameters
        - No scanner triggers in the EEG, but ``mri`` markers (added by the
          authors) on the volume grid: the first one is 2 TR (dummy scans) after
          the gradient artifact onset in all 16 runs, and the artifact repeats
          exactly every 1.950 s (Trio; the NIfTI header says 1.9575) / 2 s (Verio).
          First volume = first ``mri`` marker. The EEG often stops before the end
          of the scan (gradient artifact up to the last EEG sample): 0-10 final
          volumes have no marker and no EEG
        - Events: Eeg, Fmri, Stimulus (``eyes_open`` / ``eyes_closed`` blocks from
          the ``beo``/``eeo`` and ``bec``/``eec`` markers, ``flash``)
    """

    bibtex: tp.ClassVar[str] = """
    @article{vandermeer2016eyes,
        title={Eyes open -- eyes closed {EEG/fMRI} data set including dedicated ``Carbon Wire Loop'' motion detection channels},
        author={van der Meer, Johan N. and Pampel, Andr{\\'e} and van Someren, Eus J. W. and Ramautar, Jennifer R. and van der Werf, Ysbrand D. and Gomez-Herrero, German and Lepsien, J{\\"o}ran and Hellrung, Lydia and Hinrichs, Hermann and M{\\"o}ller, Harald E. and Walter, Martin},
        journal={Data in Brief},
        volume={7},
        pages={990--994},
        year={2016},
        doi={10.1016/j.dib.2016.03.001}
    }
    """
    url: tp.ClassVar[str] = "https://www.nitrc.org/projects/cwleegfmri_data/"
    licence: tp.ClassVar[str] = "MIT"
    description: tp.ClassVar[str] = (
        "Simultaneous EEG-fMRI from 8 participants during eyes open / eyes closed "
        "blocks, with carbon wire loop motion channels and the helium pump on or off."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ()
    ARCHIVE_URL: tp.ClassVar[str] = "https://cwleegfmri_data.projects.nitrc.org/downloads/{}.tar.gz"
    # original subject name -> (BIDS subject, TR in s)
    SUBJECTS: tp.ClassVar[dict[str, tuple[str, float]]] = {
        **{f"trio{i}": (f"0{i}", 1.95) for i in range(1, 5)},
        **{f"verio{i}": (f"0{i}", 2.0) for i in range(5, 9)},
    }
    EEG_EXTENSIONS: tp.ClassVar[tuple[str, ...]] = (".set",)
    RAW_EEG_DATA: tp.ClassVar[dict[str, str]] = {}  # data inside the .set
    PREPROC_EEG: tp.ClassVar[str | None] = (
        "derivatives/mrcorrected/{sub_dir}/eeg/{prefix}_desc-mrcorrected_eeg.set"
    )
    VOLUME_MARKER: tp.ClassVar[str] = "mri"
    EVENTS_RELATIVE_TO_FMRI: tp.ClassVar[bool] = False

    @staticmethod
    def _bids_name(rel: str, sub: str) -> str | None:
        """BIDS path (relative to download/) of a CWL_Data file, None if not kept."""
        name = Path(rel).name
        acq = "pumpon" if "hpump-on" in name else "pumpoff"
        func = f"sub-{sub}_task-eoec_acq-{acq}"
        if rel.startswith("mri/t1/"):
            return f"sub-{sub}/anat/sub-{sub}_T1w.nii"
        if rel.startswith("mri/epi_raw/"):
            return f"sub-{sub}/func/{func}_bold.nii"
        if rel.startswith("mri/epi_motionparams/"):
            return f"sub-{sub}/func/{func}_motion.txt"
        if rel.startswith("mri/epi_normalized/") and name.endswith(".nii"):
            return f"derivatives/normalized/sub-{sub}/func/{func}_space-MNI_bold.nii"
        if rel.startswith("spm_model_in-scan/"):
            acq = "pumpon" if name.endswith("_on.mat") else "pumpoff"
            return f"derivatives/spm_model/sub-{sub}/sub-{sub}_task-eoec_acq-{acq}_model.mat"
        if rel.startswith("eeg/in-scan/") and "mrcorrected" in name:
            return f"derivatives/mrcorrected/sub-{sub}/eeg/{func}_desc-mrcorrected_eeg.set"
        if rel.startswith("eeg/in-scan/"):
            return f"sub-{sub}/eeg/{func}_eeg.set"
        if rel.startswith("eeg/in-noscan/"):
            return f"sub-{sub}/eeg/sub-{sub}_task-noscan_acq-{acq}_eeg.set"
        if rel.startswith("eeg/out/"):
            when = "before" if "before" in name else "after"
            return f"sub-{sub}/eeg/sub-{sub}_task-outside_acq-{when}_eeg.set"
        return None  # e.g. .mat of the normalisation

    def _download(self, overwrite: bool = False) -> None:
        import shutil
        import subprocess
        import tarfile

        root = self._bids_root()
        archives = self.path / "archives"
        archives.mkdir(parents=True, exist_ok=True)
        rows = []
        for name, (sub, tr) in self.SUBJECTS.items():
            rows.append(dict(participant_id=f"sub-{sub}", original=name,
                             scanner=name.rstrip("0123456789").capitalize(), tr=tr))
            if (root / f"sub-{sub}" / "func").exists() and not overwrite:
                continue
            tgz = archives / f"{name}.tar.gz"
            # curl: the host's certificate does not match its name, Python refuses it
            subprocess.run(["curl", "-sSfL", "-C", "-", "-o", str(tgz),
                            self.ARCHIVE_URL.format(name)], check=True)
            tmp = archives / name
            with tarfile.open(tgz) as tf:
                tf.extractall(tmp, filter="data")
            for fp in sorted((tmp / "CWL_Data").rglob("*")):
                rel = fp.relative_to(tmp / "CWL_Data").as_posix()
                out = self._bids_name(rel, sub) if fp.is_file() else None
                if out is None:
                    continue
                out_fp = root / out
                out_fp.parent.mkdir(parents=True, exist_ok=True)
                if out_fp.suffix == ".txt":  # SPM realignment parameters -> tsv
                    cols = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
                    motion = pd.read_csv(fp, sep=r"\s+", header=None, names=cols)
                    motion.to_csv(out_fp.with_suffix(".tsv"), sep="\t", index=False)
                    continue
                shutil.move(fp, out_fp)
                if out_fp.name.endswith("_bold.nii") and "derivatives" not in out:
                    sidecar = dict(RepetitionTime=tr, TaskName="eoec")
                    out_fp.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
            shutil.rmtree(tmp)
            tgz.unlink()
        pd.DataFrame(rows).to_csv(root / "participants.tsv", sep="\t", index=False)
        archives.rmdir()

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for bold in sorted(self._bids_root().glob("sub-*/func/*_task-eoec_*_bold.nii")):
            entities = dict(re.findall(r"(sub|task|acq)-([a-zA-Z0-9]+)", bold.name))
            timeline = {_ENTITIES[k]: v for k, v in entities.items()}
            if self._eeg_file(timeline) is not None:
                yield timeline

    def _preproc_eeg_events(self, timeline: dict[str, tp.Any]) -> list[dict[str, tp.Any]]:
        pre_fp = self._preproc_eeg_file(timeline)
        if pre_fp is None:
            return []
        with nutils.ignore_all():
            pre_markers = mne.read_annotations(pre_fp)
        # EEGLAB "boundary" (segment joins, not events) would be the rarest shared marker
        markers = [a[np.asarray(a.description) != "boundary"]
                   for a in (self._raw_markers(timeline), pre_markers)]
        offset = self._preproc_offset(*markers, pre_fp)
        return [dict(type="Eeg", start=offset, filepath=str(pre_fp), preproc="authors")]

    def _task_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        annots = self._raw_markers(timeline)
        desc, onset = np.asarray(annots.description), annots.onset
        rows = []
        for begin, end, name in [("beo", "eeo", "eyes_open"), ("bec", "eec", "eyes_closed")]:
            ends = onset[desc == end]
            for start in onset[desc == begin]:
                after = ends[ends > start]
                if len(after):
                    rows.append(dict(start=start, duration=after[0] - start, description=name))
        rows += [dict(start=t, duration=0.0, description="flash") for t in onset[desc == "fl"]]
        events = pd.DataFrame(rows).sort_values("start", ignore_index=True)
        events["type"] = "Stimulus"
        return events
