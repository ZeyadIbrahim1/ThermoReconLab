# Phase 5 Task 1 — External Dataset and Feasibility Audit

Access date for all web sources and local retrievals: **2026-07-18**.

## 1. Scope

This audit covers suitability of the ULRI vehicle-fire dataset and IRT-PVC for an optional research extension. It does not generate datasets, implement neural models or losses, train models, evaluate ML models, or start Task 2.

## 2. Phase 5 research question

Can external thermal data legally and scientifically support research on thermal representations or inverse-field estimation, within approximately 8 GB VRAM and the stated storage limits, without weakening the validated classical package?

## 3. Classical-versus-external physics distinction

ThermoReconLab's classical benchmark is deterministic two-dimensional steady-state heat-source reconstruction. Vehicle fires are three-dimensional, strongly transient, radiative/convective, geometry-dependent experiments. Plate incident heat flux is neither the package's volumetric/source-field `q` nor a steady-state solution target. Any bridge requires an explicit new forward model, target definition, coordinates, boundary conditions, and validation protocol.

## 4. Sources consulted and access dates

| Source title | URL | Information obtained |
|---|---|---|
| Official ULRI Figshare record/API | https://ulri.figshare.com/articles/dataset/30438392 and https://api.figshare.com/v2/articles/30438392 | Identity, authors, dates, version, license, total size, manifest, archive sizes, experiment summary |
| Dataset `Readme.md` (file 59028188) | https://ndownloader.figshare.com/files/59028188 | Experiment procedure, hierarchy, ignition convention, file meanings, plate codes |
| Dataset instrumentation files (59028182, 59028185) | https://ndownloader.figshare.com/files/59028182 and https://ndownloader.figshare.com/files/59028185 | Panel availability/distance, gauge arrangement, acquisition frequency, plan view |
| Dataset vehicle information (59028179) | https://ndownloader.figshare.com/files/59028179 | Vehicle categories and experiment identifiers |
| Sauer et al., *Data in Brief* 65 (2026), 112471 | https://doi.org/10.1016/j.dib.2026.112471 | Peer-reviewed context, 18 tests, measurements and use cases |
| FSRI vehicle-fire study | https://fsri.org/research/fire-safety-batteries-and-electric-vehicles | Official study context |
| Dehghani & DiDomizio, *SoftwareX* 28 (2024), 101934 | https://doi.org/10.1016/j.softx.2024.101934 | HFITS inverse-analysis purpose and point-gauge validation summary |
| Official HFITS repository/manual | https://github.com/ulfsri/HFITS and https://github.com/ulfsri/HFITS/blob/main/MANUAL.pdf | GPL-3.0 boundary; preprocessing and inverse heat-transfer role |
| Creative Commons BY-NC 4.0 legal code | https://creativecommons.org/licenses/by-nc/4.0/legalcode | Attribution, adaptation, sharing and noncommercial conditions |
| Wei et al., *Applied Sciences* 13 (2023), 13093 | https://doi.org/10.3390/app132413093 | IRT-PVC acquisition, specimens, labels, preprocessing, data URL and academic-use statement |
| Author-provided IRT-PVC Depth Kaggle record | https://www.kaggle.com/datasets/ziangwei/irt-pvc-depth | 38 files/measurements, 11.27 GB display, CC BY 4.0 badge, academic/research-use statement |

No HFITS source code was copied or adapted.

## 5. ULRI dataset identity

Exact title: **Measurement data from full-scale fire experiments of battery electric vehicles and internal combustion engine vehicles**. Authors: Nathaniel G. Sauer, Matthew J. DiDomizio, Richard M. Kesler, Shruti Ghanekar, Parham Dehghani, Gavin P. Horn, and Adam Barowy. DOI supplied by the prompt: `10.60752/102376.30438392`; current record DOI: `10.60752/102376.30438392.v1`. Host: UL Research Institutes Figshare.

## 6. Repository metadata

Published/created 2025-10-24 17:43:04 UTC; modified 2026-07-01 20:01:24 UTC; version 1; public, not embargoed. Exact size: **104,701,790,352 bytes** (approximately 97.51 GiB). Eighteen experiments cover EV/ICEV free burns, ordinary-water suppression, one encapsulator-agent suppression, blanket-only, and blanket-plus-water tests.

Identifiers: `G-HK-F`, `E-CB-F`, `E-NL-F`, `E-HI-F`, `E-TM-F`, `E-FM-F`, `E-HK-F`, `G-HK-F2`, `G-TR-F`, `E-HK-S`, `E-CB-S`, `E-TM-S`, `E-FM-S`, `E-FM-B`, `E-TM-B`, `E-FM-SA`, `E-FM-BS`, `E-TM-BS`.

## 7. Dataset license analysis

The official API object's `license` field states **CC BY-NC 4.0**, links to the Creative Commons license, and is the exact source of the dataset-license statement. It permits sharing and adaptation with attribution for noncommercial purposes. Commercial use is restricted. Academic ML research is permitted only insofar as the use is noncommercial and all license conditions are met.

Required attribution includes appropriate credit, a license link, and indication of changes; the official citation should be retained. Raw redistribution and processed-subset redistribution are conditionally permitted for noncommercial purposes with attribution and no additional restrictions. Generated figures may likewise be published noncommercially with attribution and change/provenance disclosure.

Whether a trained checkpoint legally contains an adaptation or extract of licensed expression/data is fact-dependent and **Unresolved**; distribution requires human licensing review. Meaning of “NonCommercial,” mixed academic/industry funding, hosted services, and downstream checkpoint use also require human review.

## 8. Machine-learning-use analysis

The license does not prohibit noncommercial ML research. Scientific permission is narrower: HFITS-derived fields can be inverse-derived reference targets, not independent direct ground truth. A model trained on them may learn HFITS assumptions, smoothing, registration, and preprocessing. Commercial training or deployment is outside the license grant without separate permission.

## 9. Redistribution analysis

Raw data: conditionally redistributable under CC BY-NC 4.0, but unnecessary and discouraged here. Processed subsets: conditionally redistributable with attribution/change notices and noncommercial restriction. Figures: conditionally publishable. Checkpoints: unresolved pending legal review. Repository safeguards exclude external archives, HDF5, outputs, logs, caches, and checkpoints.

## 10. Repository file manifest

Root metadata files are `Readme.pdf` (193,809 B), `Readme.md` (11,520 B), `vehicle_info.csv` (1,469 B; README calls this `vehicle_info.md`, an upstream inconsistency), `instrumentation.pdf` (91,308 B), and `instrumentation.csv` (2,114 B).

| Experiment archive | Bytes | Panel fields? |
|---|---:|---|
| G-HK-F.zip | 11,551,433,362 | Yes |
| E-CB-F.zip | 13,251,222,833 | Yes |
| E-NL-F.zip | 9,281,384,007 | Yes |
| E-HI-F.zip | 8,728,701,124 | Yes |
| E-TM-F.zip | 7,134,990,863 | Yes; smallest suitable |
| E-FM-F.zip | 8,235,628,420 | Yes |
| E-HK-F.zip | 10,448,873,064 | Yes |
| G-HK-F2.zip | 16,708,355,420 | Yes |
| G-TR-F.zip | 16,028,811,786 | Yes |
| E-HK-S.zip | 144,963,378 | No |
| E-CB-S.zip | 33,181,630 | No |
| E-TM-S.zip | 100,817,265 | No |
| E-FM-S.zip | 42,884,435 | No |
| E-FM-B.zip | 1,170,647,125 | No |
| E-TM-B.zip | 284,430,423 | No |
| E-FM-SA.zip | 222,654,460 | No |
| E-FM-BS.zip | 885,162,980 | No |
| E-TM-BS.zip | 447,347,557 | No |

## 11. Experiment hierarchy

Each archive represents one experiment and can contain `information.md`, `events.csv`, `data_timeseries.csv`, `data_massloss.csv`, `data_heatflux.zip`, `data_ftir.zip`, air-sampling files, and reports, depending on instrumentation. `data_heatflux.zip` is documented as 12 HDF5 files: six `T_XX` plate-temperature fields and matched six `HF_XX` incident-radiative-heat-flux fields. `XX` is `DF`, `DM`, `DR`, `PF`, `PM`, or `PR`: driver/passenger side and front/middle/rear position.

## 12. Selected experiment rationale

Candidate: **E-TM-F** (EV, Tesla Model 3 Long Range, free burn), because official instrumentation metadata says free burns have heat-flux panels and its 7,134,990,863-byte archive is the smallest of those experiments. It cannot be selected for download in Task 1 because it alone exceeds the 6 GB cumulative binary/archive cap. The much smaller E-CB-S archive is scientifically unsuitable: instrumentation explicitly reports no heat-flux panels.

Selected plate pair: **Unresolved — no archive downloaded**. `DF` would be a reasonable first candidate only after inspecting the archive manifest, not before.

## 13. Download log

Before experiment download decision: official URL `https://ndownloader.figshare.com/files/59026160`; filename `E-TM-F.zip`; compressed size 7,134,990,863 B; destination would be `<external data root>\ulri_vehicle_fire\raw`; expected extracted size **Unresolved**; free-space query returned an unreliable zero in the managed shell, while the task supplied approximately 129 GB free; purpose: one matched plate-temperature/HFITS-derived-flux pair plus metadata/gauge data; smallest-suitable reason: all smaller experiment archives have no panels. **Stopped before download because 7.135 GB exceeds the 6 GB hard cap.**

Only four metadata files were downloaded. `Readme.pdf` was not downloaded because `Readme.md` is documented as identical.

## 14. Downloaded and extracted sizes

Cumulative downloaded: **106,411 bytes** (`Readme.md`, `vehicle_info.csv`, `instrumentation.pdf`, `instrumentation.csv`). Cumulative extracted: **0 bytes**. No experiment archive, external HDF5, or IRT-PVC data was downloaded.

## 15. Checksums

SHA-256: `Readme.md` `C82BC42F20E66F57152C2B3EE508A26F50DA2A64702BE6D57D9614D3E6F3BABE`; `vehicle_info.csv` `D5438A860D995E9A0798CEE8992C6BD326BF13EAF15AE3FC6DAFB179E8A3738D`; `instrumentation.pdf` `88FF3D589998A3CCA1046810D40CA605EC1257B525496407D6DFEAF2B692CF11`; `instrumentation.csv` `E95C1A93BF8EFA2337B3958B2B08592E4FE75CCD70B064D8568AA9FC6A85CD29`.

## 16. HDF5 structure

Official README verifies 12 files per panel-equipped experiment but does not document internal groups. Groups, dataset keys, shapes, dtypes, chunks, compression, and logical sizes are **Unresolved pending an eligible archive or separately accessible file**. The reusable inspector reports these recursively without loading full arrays.

## 17. HDF5 keys and attributes

Root/group/dataset attributes, units, time and coordinate representation, missing/invalid conventions are **Unresolved**. Filename semantics alone must not be substituted for internal-key evidence.

## 18. Temperature/flux pairing

Pairing is documented by shared `XX` plate code: `T_XX` temperature and `HF_XX` incident radiative heat flux. Exact extensions, internal keys, shapes, dtypes, axis order, frame counts, valid ranges, non-finite values, cropping, masks, and transformations remain unresolved because the smallest eligible archive violates the cap.

## 19. Temporal alignment

Experiment tabular time series are shifted to ignition at `t = 0 s`; free-burn DAQ frequency is 1 or 2 Hz depending on experiment. Whether HDF5 timestamps use that convention, their units/values, and whether matched T/HF timestamps agree or require interpolation are unresolved. Transient inverse processing can introduce temporal smoothing and lag.

## 20. Spatial alignment

Panel codes and nominal distances are documented; free-burn panels are 1.83 m or 2.74 m from the vehicle depending on experiment. Coordinate arrays, plate physical dimensions, pixel units, orientations, coordinate-grid equality, cropping/masking, prior interpolation, resampling need, and registration uncertainty remain unresolved. No visual resampling is justified.

## 21. Units

README describes `T_XX` as temperature and `HF_XX` as incident radiative heat flux, but exact HDF5 units and coordinate units are unresolved without attributes/manual evidence tied to these files. Figures must say “units unresolved” unless attributes verify them.

## 22. Missing data

Air-sampling files use `<LOD` and `NA`, but this does not establish HDF5 conventions. HDF5 fill values, NaN/Inf conventions, masked regions, saturation, and invalid pixels are unresolved.

## 23. Point-gauge validation feasibility

Conventional heat-flux gauges exist in walls for free burns, and instrumentation supplies broad configuration/distance information; `data_timeseries.csv` should contain gauge series. Gauge identifiers, exact coordinates relative to each plate grid, coordinate system, units, HDF5 time relationship, sampling alignment, and mapping rule were not established from retrieved metadata. No comparison is performed and no mapping is invented. If exact registration is later documented, nearest-neighbour or declared interpolation could compare field estimates and gauges with bias/RMSE/MAE/time-offset reporting, while emphasizing they are different measurement systems.

## 24. Target provenance

The full-field flux is an **external reference heat-flux field**, specifically an **HFITS-derived heat-flux estimate** reconstructed from infrared plate temperatures. No independently measured full-field heat-flux target was verified. Conventional point gauges may offer partially independent validation only after spatial and temporal registration is documented.

## 25. HFITS scientific role

HFITS preprocesses thermograms and performs inverse heat-transfer analysis for planar surfaces. Its publication reports agreement in a validation experiment with Schmidt–Boelter gauges (reported RMSE 0.5 kW m⁻² at 18 kW m⁻² incident radiative flux), but that does not make every vehicle-fire field independent ground truth. Supervision would inherit material-property, boundary-condition, convection, emissivity, smoothing, cropping and inverse-regularization assumptions; dataset-specific preprocessing details remain unresolved.

## 26. HFITS licensing boundary

The official repository states **GPL-3.0**. Documentation, algorithm descriptions, and publications were inspected. No source was copied, translated, vendored, or reproduced. Any future interoperability must preserve a clean boundary and receive licensing review; Task 1 implements only a general HDF5 metadata inspector.

## 27. IRT-PVC review

Official paper: Wei, Z., Osman, A., Valeske, B., Maldague, X., “A Dataset of Pulsed Thermography for Automated Defect Depth Estimation,” *Applied Sciences* 13(24), 13093 (2023), DOI `10.3390/app132413093`. Author-provided host: Kaggle, `irt-pvc-depth`. The Kaggle record displays **CC BY 4.0**, while both record and paper also state “academic and research use only”; these terms conflict because CC BY 4.0 permits commercial use. Exact governing license is therefore **Unresolved pending license clarification** despite the badge.

The depth dataset contains 38 PT measurements/sequences from 38 PVC specimens (19 earlier round-hole specimens plus 19 rectangular-indent specimens), each nominally `256 × 320 × 1810`, recorded at 320 × 256 pixels, 10 Hz, for 181 s with a FLIR SC5000 after two 4 kJ flashes. Specimens are 100 × 100 × 5 mm. Defects are round/cylindrical and rectangular/square, nominal lateral sizes 2–10 mm, with labeled depth classes; the paper describes depths from 2.5–4.5 mm from the inspected surface, while Kaggle's pixel-value legend is phrased as 0.5–2.5 cm and is an unresolved inconsistency. Labels cover location, shape/size and depth using manual PCT and refined 3D-CAD-assisted mappings. Raw `.mat` sequences and labels are reported; experiments in the paper derive PPT, PCT and TSR representations.

It is potentially useful for thermal representation pretraining, temporal-feature learning, defect localization, and out-of-domain tests after license resolution. It differs radically in material, scale, boundary conditions, excitation, camera geometry and target task from vehicle-fire plates. Its labels are defect geometry/position/size/depth—not direct source `q` and not heat flux—so direct source-field or heat-flux supervision is unsuitable. No IRT-PVC files were downloaded (displayed size 11.27 GB).

## 28. Storage feasibility

The full ULRI repository is infeasible/unnecessary for Task 1 and large relative to D: capacity. C: is appropriate for a controlled future subset. E-TM-F compressed size exceeds the Task 1 cap but could fit the reported C: free space only if a later step explicitly changed the applicable transfer ceiling; extracted size was unresolved at this point in the audit. IRT-PVC's displayed 11.27 GB also argues against Task 1 download.

## 29. Computational feasibility

The inspector's bounded slices are CPU/RAM safe. An 8 GB GPU could plausibly support cropped/patch-based research later, but feasibility cannot be quantified without shapes, dtypes, frame counts, batch design and a Task 2 protocol. Full transient fields should not be assumed to fit GPU memory. This is not authorization to implement or train a model.

## 30. Scientific limitations

Only nine free burns have plate fields, limiting independent experiment count and risking leakage if frames/plates rather than experiments are split. Vehicle model, side, plate position and test condition create correlated domains. Fire transience, radiation/convection coupling, occlusion/smoke, emissivity, environmental reflections, saturation, moving flames, suppression and boundary conditions violate the classical steady-state assumptions. Appearance cannot establish accuracy; numerical metrics and experiment-held-out validation are required.

## 31. Licensing limitations

ULRI is noncommercial-only and checkpoints need legal review. IRT-PVC has conflicting CC BY 4.0 and academic-only statements. HFITS code is GPL-3.0 and cannot be incorporated under this task. These constraints must be resolved before redistribution or commercial collaboration.

## 32. Unresolved questions

- Internal HDF5 hierarchy, keys, shapes, dtypes, axes, chunks, compression, attributes and units.
- Exact HFITS preprocessing/settings used per vehicle experiment, masks, interpolation and uncertainty.
- Exact plate coordinate system, geometry, orientation and registration.
- HDF5 timestamp representation and synchronization with ignition/gauges.
- Exact gauge identifiers/coordinates/units and plate-grid mapping.
- Whether separately downloadable nested `data_heatflux.zip` files can be exposed without the experiment ZIP.
- Legal treatment of trained checkpoints and mixed/commercial use.
- IRT-PVC governing license and depth-unit inconsistency.

## 33. Requirements before Task 2

Obtain an official sub-file or explicit permission/adjusted cap for E-TM-F; inspect exactly one matched pair lazily; verify keys, units, axes, timestamps, coordinates, masks and transformations; establish gauge registration or explicitly exclude gauge validation; obtain HFITS preprocessing metadata; secure human licensing decisions for processed subsets/checkpoints and clarify IRT-PVC terms; define experiment-level splits, provenance labels and a transient-physics research question. Task 2 must not start until these gates are reviewed.

## 34. Go/no-go recommendation

**No-go for Task 2 now.** Continue only with a bounded metadata-resolution step. ULRI is conditionally promising as noncommercial external-reference data, not independent direct ground truth. IRT-PVC is scientifically limited to representation/defect roles and remains license-unresolved. Synthetic ThermoReconLab data remains the only direct source-field supervision candidate, subject to a future explicitly authorized task.

| Dataset | Direct supervised training | Pretraining | Fine-tuning | External testing | Gauge validation | Unsuitable roles |
|---|---|---|---|---|---|---|
| ULRI/HFITS vehicle-fire data | Conditionally suitable | Conditionally suitable | Conditionally suitable | Conditionally suitable | Unresolved pending metadata | Direct independent full-field ground truth; direct classical `q` supervision: Unsuitable |
| IRT-PVC | Unsuitable | Unresolved pending license | Unresolved pending license | Unresolved pending license | Not applicable | Direct source-field and heat-flux supervision: Unsuitable |
| ThermoReconLab synthetic data | Suitable | Suitable | Suitable | Conditionally suitable | Not applicable | Claims of real-fire generalization without external validation: Unsuitable |

## External access resolution — 2026-07-18

This bounded follow-up tested the official `E-TM-F.zip` URL without downloading the full archive. A GET request for bytes `0-1023` followed the Figshare `302` redirect and returned `206 Partial Content`, `Accept-Ranges: bytes`, `Content-Range: bytes 0-1023/7134990863`, and a 1,024-byte body. The S3 response reported `Last-Modified: Fri, 24 Oct 2025 16:36:12 GMT` and multipart ETag `84ddb0dabe6ccb6f9618703b001c9497-7`. The Figshare response supplied `Content-Disposition: attachment;filename=E-TM-F.zip`. A redirected HEAD reached an expired 10-second signed URL as `403`, so the successful GET range—not `Accept-Ranges` alone—is the evidence for range support.

Only **1,667,580 response-body bytes** were transferred: a 1,024-byte range probe, a 131,072-byte tail, a 512-byte outer member-header range, a 1,534,844-byte metadata prefix, and a 128-byte nested member-header range. The header-only request transferred no body. Temporary files are outside Git at `<external data root>\ulri_vehicle_fire\access_resolution`.

The outer archive is ZIP64. Its ZIP64 EOCD reports eight entries and a 577-byte central directory at absolute archive bytes `7,134,990,188–7,134,990,764`. Relevant entries are:

| Member | Method | Compressed bytes | Uncompressed bytes | CRC-32 | Local-header offset |
|---|---:|---:|---:|---|---:|
| `E-TM-F/data_timeseries.csv` | 8 (deflate) | 1,502,752 | 3,899,943 | `b6459eb8` | 340 |
| `E-TM-F/information.md` | 8 (deflate) | 252 | 348 | `389fa03b` | 1,534,467 |
| `E-TM-F/data_heatflux.zip` | 0 (stored) | 7,133,455,186 | 7,133,455,186 | `47431015` | 1,534,770 |
| `E-TM-F/events.csv` | 8 (deflate) | 111 | 136 | `d60ec751` | 7,134,990,030 |

The stored nested ZIP begins at outer byte `1,534,844`, so its bytes are directly addressable in the outer object. Its ZIP64 EOCD reports 12 entries and a 978-byte central directory at outer bytes `7,134,988,954–7,134,989,931`. All members use deflate:

| Plate | Temperature file (compressed / uncompressed bytes) | Heat-flux file (compressed / uncompressed bytes) |
|---|---:|---:|
| PR | `T_PR.h5` 546,600,025 / 624,369,440 | `HF_PR.h5` 571,627,318 / 624,369,440 |
| PM | `T_PM.h5` 550,415,722 / 626,903,840 | `HF_PM.h5` 574,736,221 / 626,903,840 |
| PF | `T_PF.h5` 552,762,657 / 631,251,040 | `HF_PF.h5` 580,778,742 / 631,251,040 |
| DR | `T_DR.h5` 604,980,981 / 688,912,568 | `HF_DR.h5` 633,796,677 / 688,912,568 |
| DM | `T_DM.h5` 610,537,743 / 700,716,800 | `HF_DM.h5` 640,073,890 / 700,716,800 |
| DF | `T_DF.h5` 617,124,814 / 710,709,056 | `HF_DF.h5` 650,018,294 / 710,709,056 |

The smallest matched pair is **PR**. Its compressed transfer is **1,118,227,343 bytes** and extracted size is **1,248,738,880 bytes**. Exact compressed-data spans in the outer archive are `T_PR.h5`: `1,534,913–548,134,937`, and `HF_PR.h5`: `3,483,957,366–4,055,584,683`. These spans exclude local headers and data descriptors and can be independently range-requested, then raw-deflate decompressed while validating the central-directory CRC values. Keeping both compressed streams and extracted HDF5 files simultaneously requires 2,366,966,223 bytes plus filesystem overhead; at least 2.6 GB free is recommended.

Selective metadata retrieval was demonstrated. `information.md`, `events.csv`, and `data_timeseries.csv` were extracted and checksum-verified from the bounded ranges. `information.md` identifies E-TM-F as a 2022 Tesla Model 3 Long Range free-burn experiment with 2.74 m wall spacing, heat-flux gauges in walls, and 2 Hz acquisition. `data_timeseries.csv` contains gauge-like columns including `D1`–`D4`, `P1`–`P4`, `F`, and `R`; exact units and pixel-to-gauge registration remain unresolved. `events.csv` records ignition at experiment-relative `0:00:00`, plus IR start/end clock times, but does not itself establish HDF5 time alignment.

The official Figshare API/documentation provides article-file downloads, not archive-member downloads. The record exposes `E-TM-F.zip` as one Figshare file; no separately published `data_heatflux.zip`, HDF5 members, official mirror, or author-provided subset was found. Selectivity therefore depends on verified HTTP ranges against the official stored object rather than an archive-member API.

**Outcome A — Selective access feasible.** One PR pair can be obtained without the 7.13 GB outer archive, but its 1.118 GB compressed transfer exceeds this resolution step's 250 MB limit and was not executed during that bounded step. Any later transfer should apply an explicit budget large enough for the full requested ranges, reserve at least 2.6 GB, stream-decompress raw deflate, and verify CRC-32 (`T_PR.h5`: `1a403370`; `HF_PR.h5`: `139ed3a6`) before HDF5 inspection. This resolves archive access mechanics only; it does not resolve units, alignment, provenance, licensing, or the Task 2 gates.

## Selective PR pair verification and inspection — 2026-07-18

A subsequent selective verification transfer retrieved only the PR pair. It transferred approximately 1.118 GB and did not download the complete 7.135 GB experiment archive. This subsequent transfer exceeded the earlier 250 MB access-resolution ceiling; it must not be described as occurring within or being authorized by that bounded step. The transfer used the official Figshare URL, fresh redirect resolution for every 64 MiB chunk, mandatory HTTP 206 responses, exact `Content-Range` validation, and exact per-chunk byte counts. A fresh 1,024-byte probe passed first. C: had 117,584,678,912 bytes free before transfer. `h5py` version 3.16.0 was used.

Current-step network transfer was exactly **1,118,228,367 bytes**: 1,024 probe bytes, 546,600,025 bytes for `T_PR.h5`, and 571,627,318 bytes for `HF_PR.h5`. Including the preceding access-resolution step, cumulative response-body transfer was 1,119,895,947 bytes. `T_PR.h5` took 34.641 s and `HF_PR.h5` took 33.860 s. Each used nine requests; there were no retries or resume events. No other plate was downloaded.

Raw DEFLATE members were streamed to temporary files and decompressed without whole-member memory loading. Verification results:

| File | Compressed bytes | Uncompressed/final bytes | CRC-32 | HDF5 signature |
|---|---:|---:|---|---|
| `T_PR.h5` | 546,600,025 | 624,369,440 | `1a403370` passed | `89 48 44 46 0d 0a 1a 0a` passed |
| `HF_PR.h5` | 571,627,318 | 624,369,440 | `139ed3a6` passed | `89 48 44 46 0d 0a 1a 0a` passed |

Only after all checks passed was each `.h5.part` atomically renamed. The corresponding verified compressed temporary stream was then removed to minimize disk use. Final external files are `<external data root>\ulri_vehicle_fire\E-TM-F\PR\T_PR.h5` and `<external data root>\ulri_vehicle_fire\E-TM-F\PR\HF_PR.h5`.

### HDF5 structure and metadata

Both files have no groups and empty root attributes. Each contains 2,200 root-level, contiguous, uncompressed `float64` datasets with no dataset attributes. Every dataset has shape `(246, 144)`, no HDF5 chunk shape, and logical size 283,392 bytes; all frames together represent 623,462,400 logical array bytes per file. Temperature keys run from `/surface_temperature_batch0_frame000000` to `/surface_temperature_batch0_frame002199`; heat-flux keys run from `/estimated_flux_batch0_frame000000` to `/estimated_flux_batch0_frame002199`.

The actual key content identifies surface-temperature frames and estimated-flux frames. There is no single primary 3-D dataset: each ordered frame dataset is part of the primary series. Root/dataset attributes, units, time values, x/y coordinates, plate metadata, experiment metadata, missing-value sentinel, calibration attributes, and preprocessing attributes are **Unresolved or not present in the inspected HDF5 metadata**. The 2 Hz value belongs to `information.md`/tabular DAQ context and is not asserted as the HDF5 frame rate.

### Complete frame-wise statistics

Statistics were accumulated one `(246, 144)` frame at a time; no complete series was loaded.

| Quantity | Temperature series | HFITS-derived heat-flux estimate |
|---|---:|---:|
| Frames | 2,200 | 2,200 |
| Spatial shape | 246 × 144 | 246 × 144 |
| Finite values | 77,932,800 | 77,932,800 |
| Non-finite / NaN / +Inf / -Inf | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| Zero values | 0 | 0 |
| Minimum | 290.7653286678624 | -5.394931229011633 |
| Maximum | 589.930922940593 | 23.197300204852855 |
| Mean | 440.3212665540147 | 6.51117780847623 |
| Population standard deviation | 59.52952388901683 | 3.341834202701121 |
| Constant / empty frames | 0 / 0 | 0 / 0 |

Units are unresolved, so these magnitudes are deliberately unitless in this report. No explicit non-finite mask or zero sentinel was observed. Selected frames 220, 550 (maximum spatial mean flux), and 1980 showed no constant/empty frames or obvious missing masks. Visual appearance is not validation and no saturation threshold is documented.

### Pair alignment and figures

Both series have equal frame counts, equal frame-index ranges, and equal spatial shapes. Thus no cropping or numerical resampling is required to place same-index arrays side by side. Matching filenames/indices support a structural pairing inference. There is no explicit time array, physical coordinate grid, orientation attribute, transform, or registration record in either file; timestamp equality, physical-grid equality, absolute orientation, missing timestamp offsets, and direct physical alignment remain unresolved. Equal shape/frame count must not be presented as metadata-verified temporal or spatial registration.

Audit outputs under `research/ai/outputs/audit/` are `aligned_PR_frame_000220.png`, `aligned_PR_frame_000550.png`, `aligned_PR_frame_001980.png`, `PR_temporal_spatial_means.png`, `PR_temporal_spatial_peaks.png`, and `PR_pair_audit_summary.json`. Figures use pixel indices, independent color scales, unresolved-unit labels, no interpolation, and no smoothing.

### Gauge and license conclusions

`information.md`, `events.csv`, `data_timeseries.csv`, and official instrumentation metadata establish wall gauges and tabular acquisition context, but do not give an exact PR-plate pixel mapping, gauge coordinates in the plate grid, HDF5 timestamps, or documented gauge units sufficient for comparison. **Gauge-to-pixel registration remains unresolved; no pointwise validation was performed.** No temporal gauge comparison was made because HDF5 time alignment is not established.

The official Figshare record states CC BY-NC 4.0. Noncommercial research and ML use are conditionally permitted with appropriate attribution, a license link, and change notices. Noncommercial raw/processed redistribution is conditionally permitted under those terms; commercial use is restricted, and checkpoint treatment remains unresolved pending human review. Official record: https://ulri.figshare.com/articles/dataset/30438392; legal code: https://creativecommons.org/licenses/by-nc/4.0/legalcode.

HFITS is GPL-3.0 software according to its official repository and publication: https://github.com/ulfsri/HFITS and https://doi.org/10.1016/j.softx.2024.101934. Only published data was inspected; HFITS was used as scientific context. No HFITS implementation code was copied, translated, linked, vendored, or entered into ThermoReconLab. The inspected flux series remains an **HFITS-derived heat-flux estimate**, not independently measured full-field ground truth and not ThermoReconLab source `q`.

### Updated role decision

| Role | Classification | Reason |
|---|---|---|
| Direct supervised training | Conditionally suitable | Only for an explicitly labeled `external_heat_flux` target learning the inverse-derived HFITS reference; not direct `q` or independent truth |
| Fine-tuning | Conditionally suitable | Same provenance/license caveats; experiment-level separation required |
| External validation | Conditionally suitable | Can test agreement with the HFITS-derived reference, not independent physical correctness |
| Point-gauge validation | Unresolved pending metadata | No exact gauge-to-pixel/time registration |
| Temporal modelling | Conditionally suitable | Ordered 2,200-frame series exists, but timestamps and sampling interval are absent from HDF5 |
| Current steady-state Poisson loss | Unsuitable | Transient incident surface heat flux differs from steady-state source `q` and required physics/coordinates are absent |

Any future ML pipeline using this pair should use `task_type="external_heat_flux"`. **Task 2 remains no-go for now** because units, timestamps, physical coordinates/orientation, gauge registration, and checkpoint licensing remain unresolved. The verified pair completes Task 1 feasibility evidence but does not authorize dataset generation or model work.
