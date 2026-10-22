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
_ENTITIES = {"sub": "subject", "ses": "session", "task": "task", "run": "run"}


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
    # preprocessing of the EEG shared in eeg/ ("raw" or "authors")
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
        "derivatives/fmriprep",
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


class Lioi2020Simultaneous(_BidsEegFmri):
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
        "derivatives/sub-*/eeg_pp",
    )
    PREPROC_EEG: tp.ClassVar[str | None] = "derivatives/{sub_dir}/eeg_pp/d_{prefix}_eeg_pp.vhdr"


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
    RAW_EEG_PREPROC: tp.ClassVar[str] = "authors"  # pulse artifact corrected


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
        "derivatives/sub-*/eeg/*ON_run-*_desc-anc0qrsrm_eeg.*",
        "derivatives/sub-*/beh",
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
        "*_eeg.eeg",
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
