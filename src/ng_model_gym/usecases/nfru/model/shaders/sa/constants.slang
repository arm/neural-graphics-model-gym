// SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
// its affiliates <open-source-office@arm.com></text>
// SPDX-License-Identifier: Apache-2.0

#ifndef CONSTANTS
#define CONSTANTS
/*------------.
| :: FLAGS :: |
'------------*/

/*
    Output in RGBA8 image format
    Default: False
*/
#ifndef RGBA8_FORMAT
    #define RGBA8_FORMAT 0
#endif

/*
    Inverted depth (1 = close to camera)
    Default: False
*/
#ifndef INVERTED_DEPTH
    #define INVERTED_DEPTH 0
#endif

/*
    Assume linear colour input instead of tonemapped
    Default: False
*/
#ifndef LINEAR_COLOUR
    #define LINEAR_COLOUR 0
#endif

/*
    Invert mv direction (mv = -mv)
    Default: False
*/
#ifndef REVERSE_MV
    #define REVERSE_MV 0
#endif

/*
    Enable v5 dynamic mask
    Default: True
*/
#ifndef NEW_DYNAMIC_MASK
    #define NEW_DYNAMIC_MASK 0 // CHANGE: 0 by default for training
#endif

/*
    Enable debug texture dumping
    Default: False
*/
#ifndef DEBUG
    #define DEBUG 0
#endif

/*
    !IGNORE - only for shader accurate testing!
*/
#ifndef SLANG_ACCURATE
    #define SLANG_ACCURATE 0
#endif

/*
    !IGNORE - only for performance testing!
*/
#ifndef DISABLE_RANDOM_OOB
    #define DISABLE_RANDOM_OOB 0
#endif


/*--------------.
| :: CONSTANTS :: |
'--------------*/

/* --- core constants --- */
#ifndef DEPTH_LOG_K
    #define DEPTH_LOG_K 1200.0
#endif

#ifndef FLT_MIN
    #define FLT_MIN 1.175494351e-38
#endif

#ifndef FLT_EPS
    #define FLT_EPS 1e-7
#endif

#ifndef HALF_MAX
    #define HALF_MAX 65504.0
#endif

#ifndef HALF_MIN
    #define HALF_MIN 0.00006103515625
#endif


#if NEW_DYNAMIC_MASK
    #ifndef MV_SIMILARITY_NOISE_THRESHOLD
        #define MV_SIMILARITY_NOISE_THRESHOLD 1.0
    #endif

    #ifndef MV_SIMILARITY_THRESHOLD
        #define MV_SIMILARITY_THRESHOLD 0.3
    #endif
#else
    #ifndef MV_SIMILARITY_NOISE_THRESHOLD
        #define MV_SIMILARITY_NOISE_THRESHOLD 0.001
    #endif

    #ifndef MV_SIMILARITY_THRESHOLD
        #define MV_SIMILARITY_THRESHOLD 0.01
    #endif
#endif


#ifndef MAX_VAL
    #define MAX_VAL 1023.0
#endif

#ifndef BITS_X
    #define BITS_X 7
#endif

#ifndef BITS_Y
    #define BITS_Y 7
#endif

#ifndef BITS_EXP
    #define BITS_EXP 4
#endif

#ifndef DEPTH_MAX
    #define DEPTH_MAX 1.1
#endif

#ifndef ZERO_MOTION_EPS
    #define ZERO_MOTION_EPS 1e-5
#endif

#ifndef LINEAR_COLOUR_TONEMAP_EXPOSURE
    #define LINEAR_COLOUR_TONEMAP_EXPOSURE 2.0
#endif

#ifndef INVERTED_DEPTH_WIDTH
    #define INVERTED_DEPTH_WIDTH 960
#endif

#ifndef INVERTED_DEPTH_HEIGHT
    #define INVERTED_DEPTH_HEIGHT 540
#endif

#ifndef FAR_PLANE_COLLISION_BIAS
    #define FAR_PLANE_COLLISION_BIAS 0.1
#endif

#ifndef HOLE_FILL_SAMPLE_COUNT
    #define HOLE_FILL_SAMPLE_COUNT 9
#endif

#define HOLE_FILL_OFFSETS_INT32 { \
    int32_t2(+0, +0), \
    int32_t2(+1, +0), \
    int32_t2(+0, +1), \
    int32_t2(+0, -1), \
    int32_t2(-1, +0), \
    int32_t2(-1, +1), \
    int32_t2(+1, +1), \
    int32_t2(-1, -1), \
    int32_t2(+1, -1) \
}

#define HOLE_FILL_OFFSETS_INT { \
    int2(+0, +0), \
    int2(+1, +0), \
    int2(+0, +1), \
    int2(+0, -1), \
    int2(-1, +0), \
    int2(-1, +1), \
    int2(+1, +1), \
    int2(-1, -1), \
    int2(+1, -1) \
}

#ifndef DISOCCLUSION_DEPTH_SEPARATION_SCALE
    #define DISOCCLUSION_DEPTH_SEPARATION_SCALE 1.37e-05
#endif

#ifndef DISOCCLUSION_REFERENCE_HEIGHT
    #define DISOCCLUSION_REFERENCE_HEIGHT 1080.0
#endif

#ifndef DISOCCLUSION_REFERENCE_WIDTH
    #define DISOCCLUSION_REFERENCE_WIDTH 1920.0
#endif

#ifndef RANDOM_OOB_SEED_STRIDE
    #define RANDOM_OOB_SEED_STRIDE 10000u
#endif

#ifndef NETWORK_INPUT_RGB_M1_MV_BASE
    #define NETWORK_INPUT_RGB_M1_MV_BASE 0
#endif

#ifndef NETWORK_INPUT_RGB_P1_MV_BASE
    #define NETWORK_INPUT_RGB_P1_MV_BASE 3
#endif

#ifndef NETWORK_INPUT_RGB_M1_FLOW_BASE
    #define NETWORK_INPUT_RGB_M1_FLOW_BASE 6
#endif

#ifndef NETWORK_INPUT_RGB_P1_FLOW_BASE
    #define NETWORK_INPUT_RGB_P1_FLOW_BASE 9
#endif

#ifndef NETWORK_INPUT_DEPTH_BASE
    #define NETWORK_INPUT_DEPTH_BASE 12
#endif

#ifndef NETWORK_INPUT_DISOCCLUSION_BASE
    #define NETWORK_INPUT_DISOCCLUSION_BASE 14
#endif

#endif // CONSTANTS
