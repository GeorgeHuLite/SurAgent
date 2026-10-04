# SurAgent-Bench

![Paper status: under review](https://img.shields.io/badge/Paper-under%20review-orange)

**A benchmark for interconnected surgical vision-language reasoning.**

This repository accompanies *SurAgent: Benchmarking Multi-Agent Collaboration for Vision-Language Reasoning in Surgical Video*. Its data contribution, **SurAgent-Bench**, connects visual perception, surgical action understanding, and clinical language generation through eight tasks annotated on every frame.

![Overview of SurAgent and SurAgent-Bench](overview.png)

## Data Contributions

- **Shared annotations across eight tasks:** Each frame links instruments, locations, actions, anatomical targets, surgical triplets, phases, descriptions, and reports. This supports evaluation of how perceptual findings inform downstream reasoning.
- **Large-scale evaluation:** 34,499 surgical video frames and 275,992 question-answer pairs provide a foundation for evaluating surgical vision-language models and collaborating agents.
- **Additional clinical data:** A private cohort collected through clinical partnerships contributes 15,000 frames from 100 patients, complementing 19,499 frames from public surgical datasets.
- **Compositional reasoning:** Target and triplet recognition assess instrument-action-tissue relationships. Single- and multiple-answer questions capture concurrent entities and events.
- **Clinical grounding:** Expert annotations and a long-tailed label distribution support evaluation of common and rare findings, cross-task consistency, and error propagation.

## Dataset Overview

- **Scale:** 34,499 frames and 275,992 question-answer pairs.
- **Public sources:** Cholec80, CholecT50, and CholecTrack20.
- **Private cohort:** Endoscopic surgery videos from 100 patients, aged 22–91 years.
- **Coverage:** Six visual understanding tasks and two open-language tasks on every frame.

## Task Coverage

**Visual understanding**

- **Instrument Recognition:** Identify visible surgical tools.
- **Instrument Grounding:** Locate tools using bounding boxes.
- **Verb Recognition:** Identify surgical actions.
- **Target Recognition:** Identify tissues or structures being acted upon.
- **Triplet Recognition:** Compose instrument-verb-target relationships.
- **Phase Recognition:** Identify the current surgical stage.

**Open-language generation**

- **Open Description:** Describe the scene and tool-tissue interactions.
- **Surgical Report Generation:** Organize observations, clinical significance, and next steps.

## Data Construction

1. **Collect and sample:** Combine public surgical datasets with a clinical cohort; downsample raw videos from 25 fps to 1 fps to reduce temporal redundancy.
2. **Annotate and verify:** Clinical experts annotate and verify selected frames and task dependencies. Out-of-body frames are anonymized.
3. **Build visual questions:** Generate questions from expert annotations, with sampled answer options and challenging distractor boxes for instrument grounding. Triplets require composing instrument, verb, and target information.
4. **Construct language references:** Generate descriptions and reports with GLM-4.5V, conditioned on images and visual ground-truth annotations, then refine the outputs into clinically relevant narratives.

## Data Availability

- **Benchmark release:** The paper states that the benchmark and source code will be released upon manuscript acceptance.
- **Public source datasets:** Available through the [CAMMA dataset portal](https://camma.unistra.fr/datasets/).
- **Private clinical videos:** Raw videos cannot be publicly released yet due to privacy and ethical restrictions.
