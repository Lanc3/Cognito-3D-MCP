# System overview

## Confidence

High for shape generation; medium for texture generation, which is outside the initial training target.

Evidence uses `CONFIRMED`, `INFERRED`, and `UNKNOWN` labels.

## Purpose

Hunyuan3D-2 separates geometry and texture generation. The 2mv checkpoint is a multiview image-to-geometry model: one to four canonical RGBA/RGB views condition a flow-matching transformer that generates a compact 3D shape representation, then a VAE decoder extracts a mesh.

## Primary User Roles

- Library caller using the Python pipeline.
- Local Gradio user generating and previewing GLB files.
- API or Blender-addon client submitting images and receiving meshes.
- Model developer preparing paired meshes/views and fine-tuning the shape DiT.

## Core Capabilities

- One-to-four-view shape generation with canonical tags `front`, `left`, `back`, and `right`.
- Classifier-free guidance using zero-valued conditioning tokens.
- VAE decoding through dense, hierarchical, or FlashVDM volume evaluation.
- Optional downstream texture synthesis using a separate model family.

## High-Level Runtime

The checkpoint contains 1,645 tensors grouped as `model` (652), `vae` (266), and `conditioner` (727). The base config fixes shape latents at `[3072, 64]`, DINO conditioning width at 1,536, DiT hidden width at 1,024, 16 double-stream blocks, and 32 single-stream blocks.

## Confirmed Facts

- The DINO model is put in evaluation mode and frozen by `ImageEncoder`.
- Each 518px view yields 1,370 DINO tokens including the class token.
- Canonical sinusoidal view embeddings are added before view tokens are concatenated.
- The model predicts a 64-channel velocity for every one of 3,072 shape tokens.
- Euler integration runs from noise at flow time 0 to data at flow time 1.

## Unknowns

Original training-scale data, quality filters, render randomization, and ablation-driven hyperparameters are not shipped with this repository.

## Source Files Referenced

`models/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv/config.yaml`, `hy3dgen/shapegen/models/conditioner.py`, `hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py`, `hy3dgen/shapegen/pipelines.py`, `hy3dgen/shapegen/schedulers.py`.
