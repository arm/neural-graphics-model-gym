<!---
SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->

# NSS Data Capture Guide
> **NOTE** \
> Details about how the dataset should be structured can be found in [docs/nss/nss_dataset_specification.md](nss_dataset_specification.md).

## Table of Contents
1. [Requirements](#requirements)
2. [Data Capture Steps](#data-capture-steps)
3. [Recommendations](#recommendations)
    * [Sequences](#sequences)
    * [Dataset Size](#dataset-size)
    * [Types of Content to Capture](#types-of-content-to-capture)
    * [Check Data Licensing](#check-data-licensing)
4. [Data Capture Theory](#data-capture-theory)
    * [Examples](#examples)
        * [Ground truth color (4x4 box filter)](#ground-truth-color-4x4-box-filter)
        * [Ground Truth Motion (4x4 nearest depth)](#ground-truth-motion-4x4-nearest-depth)
        * [Jittered input decimation](#jittered-input-decimation)
        * [Quantized Halton Sequence](#quantized-halton-sequence)


## Requirements
Before capturing frames, check the requirements for using NSS in your game. Your game must:

* Have support for jittered rendering.
* Render motion vectors for most objects.
* Have added or reused rendering options for very high resolution output.

We recommend integrating the [runtime portion of NSS](https://github.com/arm/neural-graphics-sdk-for-game-engines) and verify that it runs correctly before proceeding with fine-tuning.

Locate the point in your engine's rendering pipeline before most post-processing effects are applied. For more information on the correct point in the rendering pipeline, see FidelityFX Super Resolution 2.3.3 (FSR2) | GPUOpen Manuals.

## Data Capture Steps
1. Capture the very-high-resolution (CaptureResolution)  color, motion vector, and depth buffers.
2. To create the low-resolution (SrcResolution) input frames, pass the color, motion vector, and depth buffers into one or more shaders which decimate and jitter them.
3. To generate the high-quality (TargetResolution) ground-truth frames, downsample the color as recommended in the [Data Capture Theory](#data-capture-theory) section of this document.
4. Save these textures to disk in the layout as defined in the [Dataset Specification](nss_dataset_specification.md). You do not need to continue the rendering pipeline after this step.
5. After you render the required number of frames, write a JSON file with required metadata. You can see an [example](../../tests/datasets/test_exr/0002.json) in the Neural Graphics Model Gym.
6. Optionally, to capture frames of a pre-authored sequence, use **Replay** functionality in your game engine to simplify the capture process and make it repeatable.

## Recommendations
### Sequences
Neural Super Sampling (NSS) is a recurrent model that uses previous predictions and outputs to generate new outputs.

You must train NSS in a recurrent manner. The training input consists of a sequence of t_train consecutive frames. Typically, t_train is set to a value such as 16.

To generate the sequences of t_train frames, you must capture frame sequences from your game. The length of these captured sequences are labelled as t_captured, where t_captured > t_train. Once collections of longer sequences have been captured, then the code within the model-gym handles the generation of batches of data of length t_train for the training process.

We recommend capturing many short sequences of frames rather than a few long ones. This approach simplifies dataset curation and increases training set diversity. We typically set t_captured to approximately 100.
### Dataset Size
In general, larger datasets improve machine learning model performance.

* We use a dataset of >50,000 frames, divided into ~300 sequences to **train** Neural Super Sampling (NSS) **from scratch**.
 To **finetune** NSS for your game, ~5,000 frames are required to achieve good results.
### Types of Content to Capture
When collecting training data for Neural Super Sampling (NSS), we recommend using a wide variety of scenes that reflect the types of content NSS must handle in your game.

Based on our experience developing NSS, we recommend capturing at least the following types of content:

| Content Number    | Content Type  | Goal | Comments |
| -------- | ------- | ------- | ------- |
| 1 | Static Scenes | Ensure NSS can learn accumulation behaviour. | - |
| 2 | Real Gameplay | Present NSS with realistic dynamic scenes that represent actual gameplay. | - |
| 3 | Static camera, moving light | Ensure NSS learns correct rectification and accumulation behaviour. | Also includes view-dependent lighting; can combine with moving cameras for harder content. |
| 4 | Dynamic camera motion | Ensure NSS learns correct rectification and accumulation behaviour. | - |
| 5 | Dynamic character motion | Ensure NSS learns correct rectification and accumulation behaviour. | - |
| 6 | Thin features | Present harder content to NSS. Ensures NSS learns to accumulate and not rectify. | Good to have both static and moving content. |
| 7 | High-contrast thin features | Present harder content to NSS. Ensures NSS learns to accumulate and not rectify. | Good to have both static and moving content. |
| 8 | Particle effects | Present harder content to NSS. Ensures NSS learns to accumulate and not rectify. | Moving content not tracked by motion vectors. Includes water effects like waterfalls. |
| 9 | High-frequency textures | Present harder content to NSS. Ensures NSS learns to accumulate and not rectify. | For example, grass, moss. |
| 10 | Reflections | Present harder content to NSS. Moving content in reflections lacks motion vectors; an extension of particle effects. | - |
| 11 | Foliage  | Present harder content to NSS. Ensures NSS learns to accumulate and not rectify. | - |
| 12 | Transparent objects | Present harder content to NSS. Ensures correct accumulation and rectification behaviour. | - |

We recommend that a training dataset is comprised as follows:
* **60-70% of the total set** of captured frames should be from gameplay that covers **content types 1-5**.
* The remaining **30-40% of the captured frames** should be from the harder content that is covered by **content types 6-12**.

Although we recommend capturing specific types of content, it is most important to collect data that reflects actual gameplay. Ideally, the captured data includes a mix of content, for example, a moving camera, a moving character, particle effects, and complex textures.
> **NOTE** \
> If you have identified visual artifacts while using NSS in your game, you must capture the problematic scenes and include them as part of your training set.
### Check data licensing
When capturing sequences, it is the user's responsibility to ensure licenses on all their assets are legally acceptable for ML training.

## Data Capture Theory
To capture data from a game engine we render each frame, and corresponding Geometry buffer, at native 8k (`7680 × 4320`) resolution, the same 8K textures are then used to construct both the inputs and ground truth frames for Neural Super Sampling training.

Ground truth frames are constructed by downsampling the 8K textures to 1080p. For ground truth color (`ground_truth` image), we apply a 4x4 box filter, to form a 1080p color texture with 16 samples-per-pixel.
For ground truth motion (`motion_gt` image) we inspect the 8k `depth` image within 4x4 tiles and decimate the motion vector by selecting the vector with the closest depth to the camera within each tile (called dilation).

To generate inputs, we dynamically decimate the 8K textures, based on a jitter vector, to construct jittered Super Sampling input textures. We do this by using a [Halton sequence](https://en.wikipedia.org/wiki/Halton_sequence), where we select (decimate) a single sample within each NxN tile of the 8k textures, e.g., for a 2x2 upscaling ratio, that has a target resolution of 1080p, and an input resolution of 540p, we take a single sample within each 8x8 tile. Tile size can be calculated as follows:

```python
CaptureResolution = (7680, 4320) # 8k
TargetResolution = (1920, 1080) # 1080p
Ratio = (2.0, 2.0)
SrcResolution = int(TargetResolution / Ratio) # (960, 540) 540p
TileSize = CaptureResolution / SrcResolution # (8, 8)
```

This approach is visualized for a given 8x8 tile of an 8k frame in the diagram below.

![Decimation](../figures/decimation_diagram.png)

In this example, the "jitter vector" is used to select an index position within each NxN tile of the source frame, we then apply a decimating point-sample at this index to generate 1 SPP aliased input textures.
The index is calculated by quantizing a low discrepancy sequence, such as Halton sequence.

This methodology ensures the closest possible attainable sample consistency, as both the input and ground truth are derived from the same source, whilst also maintaining a considerable degree of similarity between the 1 SPP aliased input and typical jittered rendering, which is the expected inference-time input.

![Jitter Pattern](../figures/jitter_pattern.png)
### Examples
#### Ground truth color (4x4 box filter)

For ground truth color (`ground_truth` image), we apply a 4x4 box filter, to form a 1080p color texture with 16 samples-per-pixel.

```cpp
RWTexture2D<float4> _Result;

float3 TonemapKaris(float3 color)
{
    // 1 / (1 + max(rgb)) see https://graphicrants.blogspot.com/2013/12/tone-mapping.html
    return color / (1 + max(max(color.r, color.g), color.b));
}

float3 InvertTonemapKaris(float3 color)
{
	// 1 / (1 - max(rgb))
	return color / (1.0 - max(max(color.r, color.g), color.b));
}

[numthreads(16, 16, 1)] void CSMain(uint3 dispatchThreadId
                                    : SV_DispatchThreadID) {
    uint2 inputIdx = dispatchThreadId.xy * 4;
    float3 color = 0.0f;

    // each 1080p pixel covers a block of 4 x 4 8K pixels
    for (uint y = 0; y < 4; ++y) {
        for (uint x = 0; x < 4; ++x) {
            uint2 idx = inputIdx + uint2(x, y);
            float3 hdr = max(0.0f, LoadCameraColor(idx).rgb);
            float3 tonemapped = TonemapKaris(hdr) / 16.0f;
            color += tonemapped;
        }
    }

    // write out the result in linear HDR
    _Result[dispatchThreadId.xy] = float4(InvertTonemapKaris(color), 1.0);
}
```

#### Ground truth motion (4x4 nearest depth)

For ground truth motion (`motion_gt` image) we inspect the 8k `depth` image within 4x4 tiles and decimate the motion vector by selecting the vector with the closest depth to the camera within each tile (called dilation).

```cpp
RWTexture2D<half2> _Result;

[numthreads(16, 16, 1)] void CSMain(uint3 dispatchThreadId
                                    : SV_DispatchThreadID) {
    uint2 inputIdx = dispatchThreadId.xy * 4;
    uint2 closestIdx = inputIdx;
    float maxDepth = 0;

    // Each 1080p pixel covers a block of 4 x 4 8K pixels
    for (uint y = 0; y < 4; ++y) {
        for (uint x = 0; x < 4; ++x) {
            uint2 idx = inputIdx + uint2(x, y);
            float depth = LoadCameraDepth(idx);

            // Some game engines use reverse depth so 1 is near, 0 is far
            if (depth > maxDepth) {
                closestIdx = idx;
                maxDepth = depth;
            }
        }
    }

    _Result[dispatchThreadId.xy] = LoadMotionVectors(closestIdx);
}
```

#### Jittered input decimation

Example shows decimation for creating jittered color, the same approach is taken for jittered depth and motion too.

```cpp

RWTexture2D<half4> _Result;

[numthreads(16, 16, 1)] void CSMain(uint3 dispatchThreadId
                                    : SV_DispatchThreadID) {
    uint2 inputIdx = dispatchThreadId.xy;
    float2 coord  = GetTexCoord(inputIdx) + _Jitter.xy;
    float4 color = float4(_CameraColorTexture.Sample(point_clamp_sampler, coord).rgb, 1.0);
    _Result[dispatchThreadId.xy] = color;
}
```

#### Quantized Halton sequence

```csharp
Vector2 GenerateRandomOffset(int sampleIndex, int quantGridSize = 8)
{
    Vector2 offset = new Vector2();

    // get offset between 0 and 1
    offset = m_HaltonPoints[sampleIndex];

    if (++sampleIndex >= m_numJitterSamples)
        sampleIndex = 0;

    // optionally quantize
    if (quantGridSize > 1)
    {
        offset.x = (float)((Math.Floor(offset.x * quantGridSize) + 0.5f) / quantGridSize);
        offset.y = (float)((Math.Floor(offset.y * quantGridSize) + 0.5f) / quantGridSize);
    }

    // we want an offset between -0.5 and +0.5 from the centre of the pixel
    offset.x -= 0.5f;
    offset.y -= 0.5f;
    return offset;
}
```