#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>

#define FIXED_POINT	9	// Fixed point scaling factor, set to 0 when using floating point
#define NUMBER_MIN	-32768	// Max value for this numeric type
#define NUMBER_MAX	32767	// Min value for this numeric type
typedef int16_t number_t;		// Standard size numeric type used for weights and activations
typedef int32_t long_number_t;	// Long numeric type used for intermediate results

#ifndef min
static inline long_number_t min(long_number_t a, long_number_t b) {
	if (a <= b)
		return a;
	return b;
}
#endif

#ifndef max
static inline long_number_t max(long_number_t a, long_number_t b) {
	if (a >= b)
		return a;
	return b;
}
#endif

#if FIXED_POINT > 0 // Scaling/clamping for fixed-point representation
static inline long_number_t scale_number_t(long_number_t number) {
	return number >> FIXED_POINT;
}
static inline number_t clamp_to_number_t(long_number_t number) {
	return (number_t) max(NUMBER_MIN, min(NUMBER_MAX, number));
}
#else // No scaling/clamping required for floating-point
static inline long_number_t scale_number_t(long_number_t number) {
	return number;
}
static inline number_t clamp_to_number_t(long_number_t number) {
	return (number_t) number;
}
#endif


#endif //__NUMBER_H__
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       16000
#define CONV_FILTERS        8
#define CONV_KERNEL_SIZE    20
#define CONV_STRIDE         10

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    1
#define CONV_FILTERS      8
#define CONV_KERNEL_SIZE  20


const int16_t conv1d_bias[CONV_FILTERS] = {132, -142, -53, -9, -213, -11, -19, 380}
;

const int16_t conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-35, 17, 18, 1, 69, -75, 58, -106, 38, -112, 51, 2, 1, 17, -37, 65, -65, 40, -53, 6}
}
, {{-103, -16, 3, 12, -25, 16, -82, -10, -103, -15, -75, 14, -31, 36, 0, 74, 17, 45, 96, 176}
}
, {{-70, -88, -55, -34, -78, -50, 34, -116, -19, -4, -57, 12, -79, -71, 26, -11, 7, -20, 25, -16}
}
, {{17, 72, 63, 58, 58, 4, 47, -68, 116, -20, 70, -14, 62, 50, 8, 53, 33, 79, -10, 156}
}
, {{47, 78, -10, 46, -21, 25, 29, -43, 24, -66, 4, -87, 5, -105, -51, -41, -78, 64, -93, 3}
}
, {{81, -98, -68, 126, -82, -95, 66, 137, 72, -171, 2, 35, 5, -44, 115, -31, 23, -115, -91, 142}
}
, {{-64, -69, -32, -68, -41, -1, -17, 34, 25, -27, -17, 11, -2, 10, 0, -59, -91, -67, -93, -80}
}
, {{69, 127, 19, -115, -92, 12, 45, 97, 44, -21, 4, -13, -39, 65, 135, 110, -5, -3, -55, -86}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  8
#define INPUT_SAMPLES   1599
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      8
#define INPUT_SAMPLES       799
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    8
#define CONV_STRIDE         4

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_1_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_1(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    8
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  8


const int16_t conv1d_1_bias[CONV_FILTERS] = {-207, 126, 12, 142, 5, -503, 84, 24, -8, -57, 20, -95, -163, -8, 0, 93}
;

const int16_t conv1d_1_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-50, 51, 123, 56, 13, -72, -120, -35}
, {-139, -19, 66, -28, -70, 26, -13, -29}
, {-2, -3, 139, 81, 57, -111, -134, -67}
, {-44, 21, -100, -101, 55, 90, 86, 35}
, {-16, 84, 165, -7, -105, -5, -49, -112}
, {-127, -6, 40, 28, 6, -4, 16, -124}
, {62, 94, 83, 65, 12, -110, -50, -124}
, {-126, -45, -38, -70, -4, 66, 62, 12}
}
, {{-20, -131, -63, 122, -62, -23, 29, 73}
, {-29, -12, -131, 21, -3, -56, -135, -19}
, {-154, -129, -79, 40, 44, -111, 17, 41}
, {107, 30, -13, -2, -87, 11, 156, -179}
, {31, -35, -51, 61, -18, -102, 37, 132}
, {-144, -51, -8, 94, -141, -175, -86, 11}
, {26, -14, 59, 60, -16, -1, 40, 112}
, {88, 7, 22, -78, -52, 43, 36, -141}
}
, {{62, 13, -62, 58, 67, 83, 93, 72}
, {-112, -124, 32, 152, 14, -62, -172, -31}
, {10, -32, -80, -41, -100, -37, 59, 9}
, {-10, 119, 94, 77, 41, -103, -126, -29}
, {-32, -105, -42, 36, -11, -14, -19, 135}
, {-104, -198, -14, 161, 84, -65, -172, -143}
, {-71, 55, 56, -23, -126, 5, -40, 118}
, {-1, -38, 98, 164, -63, -36, -131, -106}
}
, {{-103, -68, 2, -85, -43, 80, 85, 0}
, {-190, 19, 114, 142, 68, -41, -70, -82}
, {91, -3, -85, -127, -58, 118, 115, 34}
, {-143, -42, 106, 110, 86, -17, 9, -97}
, {-110, 53, -39, 63, 23, 85, 80, -4}
, {-378, -170, -63, -1, -165, -171, -208, -253}
, {40, 47, -121, -166, -66, 31, 56, 16}
, {-137, 19, 107, 62, 61, -21, -55, -77}
}
, {{-53, -53, -125, -80, -72, -87, -58, -28}
, {47, 86, 26, -4, 17, 64, 82, 15}
, {-29, -51, -3, 42, -25, -57, -44, -26}
, {46, 125, 83, -43, -73, -77, -58, -158}
, {-59, -61, -50, 2, -57, -19, 26, 10}
, {-7, -40, 43, 49, 67, 179, 122, 137}
, {94, 38, 26, -48, 9, 79, 31, 65}
, {-53, 99, 40, 120, 27, -6, 112, -31}
}
, {{-107, 41, 52, 39, 83, 72, 48, 10}
, {-61, -144, -13, 4, -13, 13, 67, 89}
, {-99, -104, 66, 51, 15, 27, -47, -88}
, {30, 172, 111, 23, -107, -134, -38, 36}
, {-190, -62, 152, 147, 125, 6, -35, -33}
, {39, -40, -53, -29, 21, 5, 13, 11}
, {-79, 73, 30, 40, 4, -21, -55, -36}
, {38, 42, -27, -1, -117, -122, -55, 9}
}
, {{28, 52, 47, 39, -13, -78, -22, 48}
, {19, 1, -73, 1, 13, 4, 123, 91}
, {95, 36, 23, 28, -32, -163, -44, 32}
, {9, -23, -34, -118, 18, 43, -17, 46}
, {89, -2, 20, 19, -99, -56, 58, -3}
, {-11, -66, -277, -27, 5, -112, 16, -68}
, {86, 28, 76, 95, -102, -65, -19, -43}
, {-11, -48, -63, -49, -46, 47, 40, 34}
}
, {{100, -68, -121, 93, -152, -7, -38, -22}
, {4, -24, -31, 95, -53, -60, 18, 1}
, {-44, 0, -125, 117, -17, -45, -144, 84}
, {-123, -45, -26, -44, 42, -20, 91, 85}
, {90, -86, -55, 99, -150, -103, -7, 160}
, {-75, -174, -179, 226, 27, -67, -54, 45}
, {-50, -130, 65, 6, -141, 18, -161, 187}
, {-89, -59, 23, 92, 45, -21, 79, 46}
}
, {{-6, -72, 28, 99, 7, 117, 39, -92}
, {56, 63, 75, 0, 18, 105, 14, -56}
, {-5, -8, -100, 66, 54, 57, 90, -2}
, {60, 13, -165, 14, -82, 85, 28, -32}
, {-4, -1, 24, 19, 59, 105, 53, -36}
, {15, -75, -139, -42, -217, -43, 27, -130}
, {-84, -131, -10, -22, -46, -13, -39, 22}
, {97, -8, -37, 25, -74, -19, -1, -161}
}
, {{-137, -20, 8, 88, 113, 89, -24, -58}
, {-142, -181, 56, 96, 113, 71, 5, -45}
, {4, -15, -40, 67, 21, -7, -45, 7}
, {-45, 33, 99, 6, -63, -48, -51, 3}
, {-75, -127, 157, 150, 84, -14, -1, -47}
, {-282, -367, -77, 286, 215, -97, -229, -116}
, {-49, -86, -140, -61, -61, -25, -65, 24}
, {-115, 0, 95, 122, 73, -13, -2, -35}
}
, {{-108, -92, 9, 137, -5, 50, 54, -141}
, {-56, -42, -22, 133, 34, 8, -94, -34}
, {-67, -48, -9, 118, 133, 3, 13, -124}
, {-8, -117, 17, -45, 51, 42, 14, -96}
, {-43, -50, -16, 145, 102, -29, -62, -83}
, {-114, -64, -20, 69, -26, -82, -81, -79}
, {-44, -58, 83, 21, 75, 15, -108, -3}
, {-93, -79, 47, -15, 120, 81, 57, -73}
}
, {{-35, -16, -5, -80, -53, 135, -30, -153}
, {28, 61, -33, -93, -81, 43, 59, 59}
, {-42, 15, -54, -10, -119, 155, 66, -47}
, {20, 32, -16, 27, 4, -5, -106, -27}
, {26, 41, -69, -168, 0, 164, 27, -37}
, {17, 68, -93, -317, -136, 31, -156, -252}
, {111, -104, -67, -73, 10, 141, 33, -111}
, {22, 44, 33, -33, -31, 60, -52, 20}
}
, {{-18, 78, -43, -34, 56, -32, -117, 0}
, {-11, 45, 35, 82, 72, 32, 17, 108}
, {93, 1, -38, -60, 71, 11, -90, -113}
, {-7, 28, -25, -24, -134, -12, 85, 64}
, {35, 29, -2, 109, 119, -21, -54, 13}
, {-99, -125, 26, 9, 56, -41, -105, 26}
, {51, -1, -48, 46, -58, 44, 24, 4}
, {-66, -33, -9, -40, -30, 80, 19, 43}
}
, {{30, -29, 74, 12, 66, 28, 24, 65}
, {-122, -198, 96, -50, -73, -36, 87, 190}
, {7, 0, 19, -33, -53, -20, 21, -86}
, {-58, -127, 82, 87, -61, 31, 75, -11}
, {54, -69, 106, 18, -3, 47, 41, -78}
, {0, -1, 98, 45, -43, -4, 108, 65}
, {72, 53, 84, -93, 25, -19, -23, -94}
, {-64, -235, -39, 43, -75, -174, -24, -37}
}
, {{36, -57, 91, 34, 103, 105, 29, 6}
, {-52, -44, -185, 55, 38, -114, 33, -66}
, {48, 65, 27, -51, -105, 84, -67, 87}
, {-40, 2, 32, -2, 87, -54, 44, -73}
, {19, 2, 38, -32, 25, 18, 11, 39}
, {83, 177, 234, 148, 152, 249, 140, 211}
, {-57, -14, -83, 17, -83, 3, 31, -43}
, {3, 31, -40, -3, 132, -132, -47, -125}
}
, {{3, -29, -69, -21, 33, 48, 94, 38}
, {-87, -89, 122, 48, -48, -245, -34, 48}
, {-5, -83, -142, -101, 96, 153, 37, 51}
, {-31, 0, 74, 75, -152, -129, -169, -86}
, {-25, -80, -31, 86, 84, 5, 31, 40}
, {-37, 21, 131, 123, 46, 48, 73, 16}
, {-103, -125, -74, -12, 103, 103, 111, -16}
, {-81, 47, 49, 112, -90, -80, -113, 108}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  16
#define INPUT_SAMPLES   198
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_1_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_1(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       99
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    4
#define CONV_STRIDE         2

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_2_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_2(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  4


const int16_t conv1d_2_bias[CONV_FILTERS] = {50, 1, 55, -48, 178, 497, -1, -211, -244, -15, 1, -58, -356, -82, -127, -50, 90, -51, -533, 38, -62, -113, -73, -33, 262, 161, -158, -222, 152, 211, 121, 22}
;

const int16_t conv1d_2_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-1, 135, 21, 35}
, {-48, 0, -202, -167}
, {109, -63, 32, -91}
, {76, 9, 0, 89}
, {-44, -54, 5, 125}
, {-189, -121, -22, -125}
, {14, 39, 18, 11}
, {4, -160, 16, -45}
, {134, 1, -59, 122}
, {133, -23, 3, 3}
, {47, 120, 34, 68}
, {41, 167, -4, -9}
, {-49, -134, -22, 12}
, {52, -142, -85, 47}
, {-24, -40, 1, 83}
, {-70, -105, 4, -74}
}
, {{103, 59, 48, 110}
, {-96, 85, 53, 97}
, {-260, -29, -21, -261}
, {-33, -41, 10, -47}
, {-53, 68, 0, -73}
, {-7, 54, 97, 105}
, {-60, -177, -83, 31}
, {32, -36, -4, 16}
, {-56, -12, -92, -32}
, {122, 147, 80, 106}
, {-16, 75, 0, -55}
, {22, -43, 73, 47}
, {64, -20, -7, 17}
, {-29, -118, 48, 13}
, {78, 69, -10, -163}
, {40, 100, -1, -87}
}
, {{21, 31, 36, 107}
, {76, 0, 172, -123}
, {103, -77, 53, 77}
, {21, -203, -62, -10}
, {110, 92, 97, 33}
, {-165, -68, -139, -147}
, {-136, 49, -32, 22}
, {89, -16, -155, 15}
, {-136, -68, -88, -150}
, {23, 92, 25, -130}
, {-165, 48, 49, -6}
, {-44, -17, -185, -47}
, {-136, -82, -208, -27}
, {45, 89, -58, 28}
, {0, 5, -17, 67}
, {93, -22, 57, -93}
}
, {{-39, -150, 0, -190}
, {79, -42, 97, -125}
, {63, 81, 112, -37}
, {93, 1, -42, 16}
, {-51, 54, -93, -146}
, {36, 123, 178, -17}
, {84, -2, 116, -102}
, {-98, -227, -253, -323}
, {13, -65, -54, -109}
, {207, 123, 143, 149}
, {-78, 41, -164, -84}
, {-56, 64, -83, 1}
, {-12, 28, -28, 3}
, {-43, -156, -78, 78}
, {27, -36, -21, -20}
, {-70, 112, -136, 116}
}
, {{32, 39, 74, 8}
, {38, 140, 35, 75}
, {74, 41, -18, 43}
, {155, -117, 20, 28}
, {-85, -132, -71, 5}
, {-223, -12, -65, -38}
, {-79, -92, -26, -101}
, {-12, -76, 56, -36}
, {-32, 15, 8, 58}
, {-198, -66, -59, -165}
, {-120, 117, 93, -2}
, {41, -7, 63, -42}
, {8, -45, 87, 42}
, {-1, -83, 166, 70}
, {-46, -209, -83, -103}
, {-44, 4, -126, -122}
}
, {{-7, -7, -56, 90}
, {60, 29, 0, 22}
, {-98, -46, -52, -310}
, {-31, -67, -69, -69}
, {49, -66, -44, 53}
, {-43, -166, -114, -12}
, {97, 24, -195, -46}
, {-22, 70, 54, 21}
, {63, 31, -68, -78}
, {103, 83, 105, 145}
, {57, -217, -122, 33}
, {80, 91, 110, 93}
, {36, -54, -10, -50}
, {50, 15, -30, -30}
, {88, 48, -118, -35}
, {0, -6, -153, -121}
}
, {{-179, -147, -32, -78}
, {-15, -239, -92, 24}
, {9, 160, -3, 124}
, {-148, -105, -172, -107}
, {25, -96, 111, 35}
, {-53, -153, -11, -158}
, {-19, -122, -51, -65}
, {-150, -83, -48, -6}
, {-21, -51, 114, -13}
, {153, 140, 180, 194}
, {179, 88, 50, 122}
, {-143, -4, 150, 22}
, {63, 24, 38, -90}
, {-13, -79, -62, 17}
, {58, -2, 46, 60}
, {131, 22, -45, -99}
}
, {{222, 80, 112, 127}
, {-127, -151, -103, -95}
, {-101, -124, -31, 82}
, {-82, -138, -19, -17}
, {19, 57, 133, 91}
, {-14, -54, -18, 25}
, {-67, 0, -17, 59}
, {-47, 78, -23, 54}
, {-24, 15, 33, -33}
, {32, -32, -56, 16}
, {71, 43, 21, 1}
, {35, -17, 9, -159}
, {-44, -32, -35, -47}
, {-26, 37, 14, 112}
, {-40, 22, 44, -16}
, {61, 90, -10, 128}
}
, {{99, 6, -25, -50}
, {9, 9, 10, -56}
, {-158, -6, -118, -22}
, {37, 28, 107, 74}
, {15, -17, -22, -98}
, {8, -7, -55, 56}
, {-37, 74, 135, 10}
, {-45, -37, 33, 71}
, {-2, 0, -82, 24}
, {-8, -81, 8, -28}
, {52, 108, 54, 35}
, {-117, -71, 70, 119}
, {79, -133, -110, -61}
, {-101, -111, -15, -44}
, {-37, -26, -70, -128}
, {59, 88, 71, 22}
}
, {{92, 138, 151, 70}
, {62, -33, -49, -57}
, {-19, 58, 12, -21}
, {-23, -63, -30, -41}
, {-30, -62, 4, 52}
, {12, 52, -15, 16}
, {-77, -96, -112, -138}
, {103, 138, 14, 88}
, {-71, 32, -83, -110}
, {-13, -10, 7, 9}
, {6, -129, -95, -41}
, {27, 118, 183, 112}
, {-31, 3, 59, 107}
, {157, 18, -50, -54}
, {-65, -1, -6, 18}
, {-32, -100, -140, -65}
}
, {{48, -59, -58, 82}
, {-75, 97, -30, -105}
, {121, 72, -139, -136}
, {32, 50, 71, 35}
, {69, 87, -131, 17}
, {33, 1, 3, -33}
, {-65, -90, 107, 92}
, {132, 38, -53, 5}
, {-63, 30, -107, -115}
, {-124, -79, -152, -103}
, {-173, -134, 53, -80}
, {80, -6, -47, -31}
, {-17, -214, -7, 57}
, {51, -43, 0, 61}
, {-65, -55, -100, -125}
, {-123, 120, 44, -78}
}
, {{-39, -57, -29, 36}
, {-107, -80, -1, -189}
, {-36, -99, -41, -35}
, {90, 43, 167, 160}
, {-8, 49, 16, -74}
, {35, 168, 24, -188}
, {24, 45, 70, -63}
, {58, 65, 16, 7}
, {50, -11, 78, -53}
, {-101, 10, 31, 38}
, {-88, -2, 109, 66}
, {-102, -235, -218, -135}
, {-48, -85, -90, -149}
, {-24, -121, 0, -19}
, {33, 64, 41, 34}
, {-98, 61, 136, 59}
}
, {{-53, 100, 151, 62}
, {-16, 15, -56, 2}
, {79, -39, -22, 86}
, {-89, -167, 37, 78}
, {-30, -38, -12, 57}
, {-187, -82, 69, 121}
, {-137, -99, -28, 26}
, {89, 55, 89, 57}
, {-214, -161, 53, 41}
, {-11, -134, -97, -33}
, {-1, 107, 108, 82}
, {-150, -107, 30, 33}
, {-88, -23, 16, 25}
, {75, -115, -88, -6}
, {38, 39, 1, -16}
, {-44, -11, 41, 54}
}
, {{-34, 8, 42, 71}
, {-71, -35, -29, -12}
, {-30, -88, -154, -70}
, {88, -37, -169, -19}
, {-23, -18, -5, -79}
, {-11, 26, -42, -14}
, {31, 34, 27, -17}
, {15, 90, 134, 14}
, {42, -6, 69, 84}
, {3, 1, -108, 17}
, {-146, -158, -72, 92}
, {29, 48, -163, -79}
, {56, 58, 29, 23}
, {39, 43, 80, 0}
, {-60, -198, -145, -46}
, {-156, -131, 57, 70}
}
, {{115, 80, -97, -70}
, {-14, -45, -75, 59}
, {33, 19, 61, 219}
, {-124, -157, 6, 0}
, {-56, 12, 40, -93}
, {-12, 135, 86, -15}
, {-103, -13, -63, -31}
, {-9, -60, 65, 87}
, {23, 79, 137, -12}
, {-142, -27, 66, -117}
, {-2, 38, 10, 55}
, {102, 120, 85, -28}
, {-184, -83, 55, -39}
, {-143, -120, 67, 127}
, {-14, 20, -10, -33}
, {-67, 30, -12, 94}
}
, {{-68, -90, -150, -59}
, {69, 54, 5, 34}
, {-58, -12, -27, -26}
, {20, 97, 73, 64}
, {56, 70, 98, -5}
, {-11, -36, -6, -94}
, {4, 11, 41, 94}
, {-70, -61, 5, 61}
, {-48, 29, 116, 63}
, {-35, -104, -95, -131}
, {90, 22, 46, 95}
, {9, 27, 4, 43}
, {-13, 44, 48, -11}
, {-2, 19, -83, 90}
, {-31, -53, 6, 19}
, {11, -84, -49, -33}
}
, {{-20, -165, -97, -100}
, {22, 0, -150, -36}
, {-152, 224, 22, 15}
, {-3, -88, -142, -103}
, {30, -43, -165, 41}
, {-118, -23, -173, -172}
, {51, -1, -76, -34}
, {101, 96, 68, -74}
, {46, -40, -34, 78}
, {34, 53, 45, 22}
, {-131, -161, -138, -36}
, {30, 87, 40, -65}
, {-8, -29, 64, -1}
, {-6, -2, 153, 49}
, {-40, 28, 184, 143}
, {-180, 34, 113, 102}
}
, {{-114, -320, -187, -149}
, {28, 33, 67, 122}
, {91, 74, -113, -306}
, {72, 48, -92, -208}
, {42, 6, 21, 54}
, {-117, -136, -37, 60}
, {-72, -12, 64, 94}
, {-17, -16, -79, -83}
, {-25, -63, -25, 55}
, {-109, -25, -132, -145}
, {-40, -175, -111, 45}
, {79, 136, 8, 63}
, {52, 10, -103, -22}
, {24, 74, 37, -50}
, {-54, 8, 27, 15}
, {1, -36, -152, 51}
}
, {{18, 26, 15, 42}
, {11, 2, 26, 76}
, {156, 89, 89, 76}
, {31, -91, 46, 23}
, {83, -69, -58, -24}
, {159, -7, 63, 114}
, {-4, 10, -138, -40}
, {-135, -13, 19, -209}
, {42, -35, -29, -27}
, {-78, 24, -101, 185}
, {15, -51, -62, -67}
, {51, 14, 13, -37}
, {43, 22, 10, -61}
, {-52, 0, 51, -64}
, {11, 51, -15, 60}
, {108, -54, 68, 102}
}
, {{24, 117, 74, -8}
, {22, 85, 141, 118}
, {-84, -30, -225, -109}
, {-1, 58, -39, -48}
, {-10, -96, 1, 22}
, {12, -32, 94, 70}
, {-60, -74, 37, -15}
, {-11, -43, -36, 94}
, {44, -19, -144, -158}
, {45, 85, -86, 60}
, {-31, -1, 2, 4}
, {132, 119, 82, -137}
, {68, -120, -31, 67}
, {-26, -66, -46, 44}
, {-25, -106, -137, 35}
, {-67, -112, -66, 140}
}
, {{-75, -14, -50, -24}
, {92, -3, 41, -42}
, {-8, -153, -91, 23}
, {-121, -135, -55, -15}
, {42, 0, -31, 63}
, {110, 87, 116, 166}
, {-28, 73, 86, -66}
, {88, 95, 114, -39}
, {-10, 12, 0, -14}
, {-13, -128, -65, -55}
, {-78, 16, -31, -24}
, {17, 21, 25, 89}
, {9, 56, -34, -61}
, {-99, -56, 29, -52}
, {46, 20, -46, 25}
, {121, 39, -57, -11}
}
, {{23, 0, 3, -29}
, {25, 19, -54, -57}
, {39, 144, 124, 3}
, {52, 0, -113, -38}
, {-58, 120, 127, 99}
, {-16, 77, -22, -36}
, {-27, 27, 87, -40}
, {-99, -80, -74, 7}
, {-68, 15, 95, 0}
, {66, 54, -13, -138}
, {-176, -169, -96, -147}
, {-185, -134, -36, -100}
, {1, -80, 3, 104}
, {25, -65, -102, 91}
, {56, 84, -32, -9}
, {-43, -73, -59, 82}
}
, {{-64, -19, -41, 177}
, {54, 6, 58, 71}
, {97, 118, 159, -74}
, {-146, 85, 162, -22}
, {-60, 79, 4, -4}
, {-114, -154, -86, -9}
, {-75, -80, 11, 30}
, {90, -17, 95, 9}
, {-50, 30, 46, -89}
, {0, 170, 77, 74}
, {-60, -27, -84, -28}
, {10, -15, -27, -121}
, {-44, 18, -155, -188}
, {-18, -94, 16, -139}
, {12, 27, -113, -68}
, {59, -55, 66, -18}
}
, {{146, -130, -192, -144}
, {-73, -9, -46, -30}
, {126, -23, 77, -272}
, {-67, 8, 37, 8}
, {54, -78, -45, -99}
, {78, 81, -105, -69}
, {-126, -29, -21, 23}
, {18, 61, 22, -34}
, {-9, 84, 96, 79}
, {-8, 246, 102, 125}
, {-70, 13, 92, 74}
, {7, 23, -56, -57}
, {-19, 27, -33, 36}
, {-65, 77, 51, -29}
, {-42, -95, -139, -8}
, {-70, 13, 184, 121}
}
, {{41, 154, 95, -29}
, {-65, -15, 8, -26}
, {-171, -261, -174, 15}
, {39, 24, -92, -84}
, {18, 89, 31, -34}
, {42, -41, -43, -59}
, {137, 88, -71, -118}
, {-120, -28, 44, 30}
, {81, 38, -50, -81}
, {88, 71, 24, 73}
, {25, 54, 28, -4}
, {-27, -96, 19, -39}
, {69, 49, -12, -69}
, {90, 20, 18, 28}
, {3, -36, -106, -105}
, {-89, -141, -96, 31}
}
, {{-144, 20, 71, -13}
, {-106, 18, 125, -5}
, {-52, -56, 37, -31}
, {-74, 92, 122, -39}
, {-172, -202, -101, -135}
, {-164, 96, 146, -48}
, {-25, -20, 50, 101}
, {-144, 106, 93, 75}
, {-5, -176, -112, -47}
, {39, -3, -76, 107}
, {-50, -205, -213, -17}
, {-177, -8, 97, 22}
, {-35, -198, -181, -43}
, {44, 7, 22, 80}
, {3, 124, 126, 74}
, {43, 19, 59, -37}
}
, {{67, -13, 57, 22}
, {8, 22, -34, 12}
, {-161, -98, -110, -57}
, {-35, 97, 77, 52}
, {-136, 102, -28, -44}
, {10, -95, -20, 109}
, {6, 20, -148, -104}
, {24, 110, 214, 15}
, {26, 62, -8, 20}
, {-48, -62, 28, 11}
, {-63, -69, -30, 7}
, {97, -14, -59, 65}
, {-4, -56, 107, -33}
, {-16, -120, -59, -123}
, {103, 16, 3, 42}
, {-100, 13, 83, 72}
}
, {{5, 8, 56, -31}
, {50, 49, -1, 13}
, {69, -32, -20, 51}
, {-29, -67, 38, 94}
, {-13, 46, -3, -104}
, {-6, -20, -99, -56}
, {-5, 29, -24, -3}
, {-23, 52, 80, 15}
, {22, 0, -87, 42}
, {-46, 51, 74, 8}
, {-123, -47, -2, -50}
, {-174, -155, -187, -46}
, {40, 16, 79, -1}
, {89, 1, 141, 134}
, {28, -53, 31, 80}
, {-17, -35, 71, -72}
}
, {{-112, -29, -18, -80}
, {51, -170, -208, 194}
, {207, -167, -175, 83}
, {-87, -96, -134, 17}
, {-53, -24, -8, 78}
, {115, 72, -123, 16}
, {-88, 57, -28, -25}
, {-211, 182, 217, -92}
, {87, -40, -59, -11}
, {-28, 44, 46, -107}
, {60, 107, 5, -134}
, {-64, -44, -29, -47}
, {-18, 65, 43, -9}
, {-170, 1, -55, -113}
, {-14, -94, -110, -45}
, {25, -59, -128, 134}
}
, {{135, 47, 81, 178}
, {-34, 59, -61, -33}
, {-7, -72, 65, -21}
, {7, -36, 142, -17}
, {-171, -203, -40, -1}
, {-31, 34, 152, -17}
, {58, 27, 41, -57}
, {-48, -33, 186, 51}
, {14, 86, -112, -71}
, {5, -65, -97, -104}
, {-152, -82, -82, 13}
, {-118, 60, -25, 147}
, {-107, 40, 23, -22}
, {65, -64, -76, -129}
, {-212, -50, -239, -71}
, {-16, -80, -13, -92}
}
, {{48, -49, -64, 3}
, {-42, -23, 69, 22}
, {-25, 36, 111, 77}
, {-45, 70, -35, 112}
, {-38, 4, -104, -4}
, {-88, 25, -45, -165}
, {-54, 39, -21, -12}
, {-75, -31, 63, 62}
, {-31, 0, 26, 94}
, {350, 127, 24, 59}
, {65, 69, 74, 54}
, {-21, -12, -21, 116}
, {-56, -56, -37, 46}
, {-79, 21, -95, 38}
, {-255, -162, -207, -139}
, {-27, 13, 62, -65}
}
, {{-169, 26, -46, -50}
, {-68, -13, 0, -39}
, {-6, 193, 92, -6}
, {7, -33, 71, -56}
, {-73, -225, -95, -41}
, {-56, -260, -66, -226}
, {-21, 1, 35, -7}
, {-58, -25, 52, 29}
, {-38, 62, 96, 9}
, {185, 72, -75, -99}
, {-107, 20, 44, 6}
, {-56, -44, 132, 68}
, {37, -31, 18, -39}
, {40, -3, -28, -51}
, {-21, 64, 10, 5}
, {13, 163, 135, 50}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   48
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_2_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_2(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       24
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    2
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_3_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_3(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    32
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  2


const int16_t conv1d_3_bias[CONV_FILTERS] = {83, 0, 58, 198, 70, 51, 141, 177, -3, 252, -173, 145, 191, 65, 20, 128, 130, 48, 135, 166, 36, 228, 129, 221, -156, 213, -53, -207, -83, 109, 320, 211, 263, 90, 96, 252, -89, 91, 122, 97, 56, 221, 105, 126, 150, 53, 188, 48, 5, 85, 3, 115, 159, 276, -194, -73, -216, -24, -121, -218, 110, 148, -218, 25}
;

const int16_t conv1d_3_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{100, 120}
, {-29, -59}
, {49, -68}
, {-48, 43}
, {110, -55}
, {89, -127}
, {-10, 28}
, {-27, 82}
, {39, 0}
, {-20, -96}
, {-83, -152}
, {5, 4}
, {-120, -49}
, {43, 76}
, {-165, 52}
, {-113, 75}
, {-145, -67}
, {63, 55}
, {-124, -62}
, {91, -54}
, {-68, -18}
, {-16, 60}
, {-13, -3}
, {-75, 84}
, {-180, -17}
, {-60, 115}
, {103, -139}
, {78, 69}
, {99, -1}
, {95, -29}
, {9, -65}
, {53, 97}
}
, {{47, 119}
, {-41, -118}
, {-64, -99}
, {31, -36}
, {-114, 51}
, {-99, -91}
, {86, -67}
, {-37, -81}
, {-80, 37}
, {0, 53}
, {52, -75}
, {-45, -180}
, {-55, -43}
, {17, -99}
, {-60, 87}
, {65, 90}
, {146, 157}
, {37, 0}
, {8, 78}
, {-145, 33}
, {-45, 0}
, {-16, 62}
, {104, -30}
, {-9, 72}
, {57, -78}
, {39, -24}
, {-38, -59}
, {67, -17}
, {-9, -130}
, {-132, -163}
, {-94, -9}
, {-26, 21}
}
, {{18, 137}
, {-13, -166}
, {29, 47}
, {-88, -91}
, {-35, 52}
, {88, -21}
, {-76, -47}
, {-111, -78}
, {-29, 15}
, {-55, 48}
, {-213, -146}
, {18, -36}
, {-41, 167}
, {-99, -44}
, {-21, 92}
, {-4, 40}
, {36, 2}
, {-12, 15}
, {91, 41}
, {0, 43}
, {-44, -14}
, {-59, 48}
, {-175, -11}
, {36, 93}
, {206, -10}
, {58, 15}
, {62, -68}
, {-28, 8}
, {46, 85}
, {25, -41}
, {14, 23}
, {22, -75}
}
, {{-33, 76}
, {66, 109}
, {-215, 16}
, {0, -197}
, {-232, -378}
, {-28, 72}
, {13, 44}
, {127, 14}
, {-165, 18}
, {20, 11}
, {-255, -202}
, {-90, -48}
, {0, -184}
, {61, -2}
, {-37, -52}
, {39, 15}
, {98, -17}
, {-11, -59}
, {-132, -22}
, {91, 27}
, {-79, 49}
, {-78, -14}
, {12, -48}
, {90, 78}
, {-12, -106}
, {18, -71}
, {15, 4}
, {-117, 95}
, {8, -85}
, {7, 104}
, {25, -13}
, {-1, 105}
}
, {{-5, -10}
, {-150, -41}
, {-243, 172}
, {-18, 46}
, {6, 100}
, {193, 36}
, {-45, 23}
, {-11, -70}
, {-25, 55}
, {-173, -52}
, {-101, -104}
, {-170, -1}
, {-62, -32}
, {29, -168}
, {-43, 53}
, {22, 22}
, {-243, -127}
, {46, -64}
, {-1, -63}
, {29, 92}
, {-36, 77}
, {-128, -146}
, {-43, 63}
, {46, -35}
, {128, -14}
, {7, 163}
, {42, -22}
, {-68, -7}
, {123, 123}
, {25, 111}
, {114, 88}
, {-87, -64}
}
, {{33, -31}
, {-33, -142}
, {24, 14}
, {78, 95}
, {-134, 105}
, {-30, 9}
, {161, -37}
, {-52, -28}
, {39, 86}
, {-72, 0}
, {-98, -153}
, {-22, -141}
, {-143, -101}
, {-56, 92}
, {-357, -174}
, {15, -48}
, {8, -45}
, {-113, -54}
, {75, -20}
, {-258, 97}
, {-25, -9}
, {-13, -191}
, {113, 162}
, {-69, -108}
, {119, 38}
, {-20, -18}
, {112, 68}
, {71, 2}
, {-268, -177}
, {57, -5}
, {8, 8}
, {-33, -1}
}
, {{76, 90}
, {-26, -95}
, {50, 44}
, {-71, 8}
, {-122, -121}
, {92, 116}
, {132, -42}
, {5, -40}
, {156, 53}
, {-96, -144}
, {-9, -43}
, {-5, 138}
, {-133, -17}
, {-17, -66}
, {8, 40}
, {-26, 29}
, {-99, -30}
, {-218, -240}
, {-5, -160}
, {29, 141}
, {-16, -83}
, {-229, -103}
, {-36, -94}
, {158, 1}
, {-4, 61}
, {5, 55}
, {-53, -71}
, {-132, -81}
, {-63, -58}
, {-71, 183}
, {-105, 13}
, {0, 23}
}
, {{29, -68}
, {-64, 21}
, {7, 67}
, {89, -134}
, {-87, -8}
, {-48, 60}
, {147, 129}
, {51, -74}
, {-113, 87}
, {-226, -48}
, {202, -223}
, {-164, -79}
, {3, 99}
, {95, -54}
, {-54, 27}
, {36, -25}
, {157, -105}
, {90, -99}
, {11, -33}
, {-17, -24}
, {12, 29}
, {-35, -232}
, {-129, 19}
, {79, 57}
, {-65, 15}
, {-5, -128}
, {78, -53}
, {-14, 5}
, {12, -196}
, {-56, -141}
, {-114, 26}
, {52, -33}
}
, {{76, 2}
, {-65, -71}
, {-102, 15}
, {-143, -217}
, {100, -56}
, {-12, 94}
, {30, 283}
, {-4, 86}
, {54, -17}
, {109, -126}
, {27, 22}
, {126, 51}
, {-163, -139}
, {58, 41}
, {-29, -280}
, {31, 30}
, {-306, 26}
, {5, -52}
, {-80, -103}
, {4, -30}
, {7, 63}
, {-4, -1}
, {-27, -231}
, {16, -69}
, {-61, 84}
, {-145, -277}
, {56, 70}
, {-69, -116}
, {47, 48}
, {-3, -36}
, {-40, -157}
, {-7, -74}
}
, {{58, -105}
, {58, 20}
, {16, -12}
, {-73, -68}
, {-112, -37}
, {204, 156}
, {41, -157}
, {31, 3}
, {66, 30}
, {22, -96}
, {-244, -67}
, {62, 133}
, {108, -20}
, {-8, -76}
, {30, -26}
, {-36, -347}
, {-118, -231}
, {-82, -531}
, {71, -43}
, {-87, -135}
, {-7, 6}
, {-145, 41}
, {69, -40}
, {-42, -18}
, {139, 32}
, {35, 10}
, {16, -33}
, {-148, 2}
, {-116, -19}
, {14, 8}
, {65, -92}
, {21, -9}
}
, {{-73, -236}
, {126, -28}
, {-22, -71}
, {-13, -36}
, {-8, 149}
, {-54, 22}
, {-131, -58}
, {41, -170}
, {-157, -168}
, {28, 73}
, {64, -29}
, {176, 54}
, {177, -107}
, {-99, -64}
, {20, 2}
, {-13, 5}
, {-20, -248}
, {-54, 36}
, {-39, -152}
, {85, 52}
, {-68, -15}
, {15, 48}
, {90, 20}
, {-270, -50}
, {35, 6}
, {68, 192}
, {46, -85}
, {-37, -20}
, {-57, -46}
, {97, -176}
, {-58, 125}
, {-70, 12}
}
, {{10, -11}
, {58, -33}
, {-182, 96}
, {111, -75}
, {44, 64}
, {69, 122}
, {-138, 207}
, {-103, 12}
, {-112, 112}
, {-11, -37}
, {119, -26}
, {111, 36}
, {-306, -57}
, {-21, 5}
, {-165, 2}
, {-11, -42}
, {-71, -135}
, {22, 7}
, {-6, 15}
, {61, 115}
, {-38, -96}
, {19, -174}
, {-18, 59}
, {-5, -93}
, {124, 130}
, {-37, 49}
, {-43, -15}
, {42, -29}
, {106, 78}
, {-17, -54}
, {-54, 50}
, {-18, -75}
}
, {{65, 20}
, {83, 7}
, {178, 76}
, {-81, 40}
, {-70, 101}
, {148, 19}
, {88, 28}
, {163, -195}
, {91, 114}
, {-64, 62}
, {-328, 78}
, {51, -183}
, {-142, -225}
, {27, 5}
, {-98, -145}
, {-134, 32}
, {100, -5}
, {-148, 22}
, {-201, -248}
, {-58, 90}
, {-64, 13}
, {-12, -10}
, {71, -147}
, {-85, -198}
, {3, -226}
, {7, 100}
, {-133, -39}
, {-176, -55}
, {-64, 226}
, {-22, -133}
, {-46, 84}
, {76, -13}
}
, {{29, 115}
, {-14, -137}
, {-14, -6}
, {-4, -87}
, {-159, -129}
, {-10, -69}
, {48, -14}
, {8, 3}
, {142, -107}
, {70, 188}
, {-19, -32}
, {49, 49}
, {-142, -34}
, {135, -240}
, {18, -35}
, {-75, -116}
, {8, 205}
, {-14, -154}
, {-46, 62}
, {-5, -102}
, {-84, -86}
, {-210, 27}
, {-84, -274}
, {25, -21}
, {67, 102}
, {-249, -3}
, {-43, 4}
, {-211, -78}
, {127, 49}
, {81, 125}
, {-88, -84}
, {-60, 50}
}
, {{69, 84}
, {-42, 50}
, {-113, 29}
, {12, -7}
, {-187, -171}
, {-59, 134}
, {-170, -54}
, {-18, 11}
, {67, 57}
, {-51, -11}
, {53, -13}
, {12, 82}
, {5, -37}
, {9, -52}
, {-66, 52}
, {28, 27}
, {24, -63}
, {-44, -52}
, {27, 57}
, {23, 40}
, {9, 33}
, {11, -9}
, {47, 25}
, {-25, 69}
, {-42, -30}
, {172, -24}
, {-42, -86}
, {-11, 19}
, {88, -26}
, {-6, -16}
, {-108, -88}
, {73, -35}
}
, {{-44, 9}
, {-77, -89}
, {-247, -161}
, {71, -38}
, {125, 0}
, {34, 84}
, {-214, 185}
, {-9, -9}
, {22, -79}
, {76, -40}
, {80, -102}
, {-24, 13}
, {0, -42}
, {43, 5}
, {83, 30}
, {77, -174}
, {-221, -96}
, {6, -47}
, {20, 12}
, {63, -10}
, {68, -1}
, {-56, -224}
, {141, 22}
, {0, 111}
, {4, -102}
, {-151, 49}
, {-126, -14}
, {9, -135}
, {149, 106}
, {42, 111}
, {18, -72}
, {-45, -179}
}
, {{-93, -44}
, {-198, 27}
, {197, 133}
, {21, 130}
, {74, 22}
, {-45, -127}
, {180, -14}
, {-135, 4}
, {-200, 14}
, {-170, 1}
, {13, 0}
, {-94, 96}
, {-27, 76}
, {-11, -13}
, {88, 13}
, {-115, 27}
, {189, 7}
, {-28, -145}
, {-398, -40}
, {56, 32}
, {-67, -4}
, {-55, -81}
, {-132, 124}
, {26, 30}
, {-198, -83}
, {-105, 139}
, {-61, 48}
, {-131, -103}
, {337, 55}
, {-248, -9}
, {-18, 64}
, {-99, 159}
}
, {{-1, 24}
, {54, 4}
, {-84, -95}
, {-50, 10}
, {-222, 41}
, {-260, -216}
, {-162, 8}
, {39, -123}
, {0, -26}
, {-22, -125}
, {-174, -16}
, {80, -8}
, {145, -63}
, {-72, -69}
, {78, -69}
, {-100, -127}
, {-128, 87}
, {-98, 36}
, {-23, -1}
, {63, -36}
, {-19, -77}
, {87, 37}
, {-201, 107}
, {210, 5}
, {-40, 40}
, {-209, 169}
, {22, 16}
, {56, 101}
, {74, -185}
, {104, -24}
, {26, -107}
, {-5, 202}
}
, {{49, 74}
, {28, -8}
, {183, -181}
, {-23, 16}
, {-46, 60}
, {4, -115}
, {146, 5}
, {-3, -6}
, {-63, 87}
, {-45, -104}
, {-74, -15}
, {1, -39}
, {-66, -111}
, {-46, -47}
, {-53, -16}
, {-9, -31}
, {36, 95}
, {-136, 144}
, {-34, -99}
, {-143, 177}
, {-37, -176}
, {11, 116}
, {-118, 216}
, {35, 43}
, {-75, -74}
, {-31, -154}
, {-58, 133}
, {-55, 62}
, {77, -90}
, {51, 52}
, {73, 87}
, {17, 2}
}
, {{-68, 36}
, {3, 71}
, {-30, -340}
, {-9, -24}
, {77, -34}
, {-4, 28}
, {26, 43}
, {79, -65}
, {25, -21}
, {27, 18}
, {-24, 3}
, {0, 80}
, {-60, 30}
, {53, 98}
, {49, 54}
, {15, -59}
, {-8, -88}
, {-102, -24}
, {70, -134}
, {37, 27}
, {-38, -21}
, {90, -94}
, {-84, 11}
, {-24, -184}
, {-3, -121}
, {31, 143}
, {-5, -46}
, {4, -27}
, {-112, 24}
, {95, 47}
, {123, 13}
, {61, -68}
}
, {{34, 72}
, {-22, -131}
, {123, -27}
, {-89, -51}
, {-326, -52}
, {-48, -85}
, {-22, 50}
, {-35, 77}
, {-49, 47}
, {-137, 0}
, {-28, -37}
, {109, -51}
, {-74, 4}
, {-164, -187}
, {-97, -215}
, {0, 71}
, {-40, 0}
, {-210, -146}
, {0, -60}
, {48, 106}
, {20, 65}
, {-57, 27}
, {-23, 46}
, {42, -25}
, {53, 129}
, {-82, 81}
, {-112, 124}
, {-31, 19}
, {-6, 89}
, {-162, 36}
, {-231, -3}
, {13, -34}
}
, {{-150, 33}
, {-162, 27}
, {-91, -34}
, {9, 86}
, {-144, 161}
, {38, 148}
, {-87, -51}
, {-99, -118}
, {-98, 8}
, {-132, -133}
, {-56, -64}
, {-1, -44}
, {68, -60}
, {-51, -80}
, {-183, 38}
, {-18, 68}
, {108, 221}
, {56, -19}
, {-27, -120}
, {13, 23}
, {71, -49}
, {-63, 65}
, {55, 66}
, {-113, -53}
, {73, 15}
, {-74, -13}
, {15, -43}
, {-7, 50}
, {37, 56}
, {-106, -169}
, {-3, -66}
, {-148, -188}
}
, {{-116, -89}
, {152, -65}
, {-96, -16}
, {44, 110}
, {18, 109}
, {-58, -30}
, {-27, -95}
, {-129, -64}
, {67, -58}
, {-8, 25}
, {131, -34}
, {-1, -1}
, {-44, -182}
, {-110, 4}
, {-73, -190}
, {4, -47}
, {6, -116}
, {53, 37}
, {-32, 20}
, {-119, 24}
, {61, 48}
, {59, -5}
, {-145, -16}
, {85, 68}
, {128, 11}
, {92, 32}
, {83, 32}
, {-28, -16}
, {-105, 108}
, {-77, -94}
, {-32, 117}
, {-69, 27}
}
, {{-28, -102}
, {104, 44}
, {167, -56}
, {-23, 164}
, {30, -212}
, {-28, 11}
, {-28, -6}
, {-38, -244}
, {13, -132}
, {44, 28}
, {-13, 25}
, {64, -52}
, {45, -8}
, {224, -201}
, {99, 14}
, {-134, -129}
, {15, -17}
, {-18, -330}
, {-25, 44}
, {91, 38}
, {-30, -76}
, {35, -124}
, {6, -55}
, {-126, 127}
, {-74, -119}
, {62, 75}
, {-14, 35}
, {-34, 16}
, {39, 56}
, {49, -53}
, {112, 59}
, {0, -95}
}
, {{-227, 48}
, {-65, 29}
, {99, 163}
, {-59, -58}
, {-37, 20}
, {61, -25}
, {43, -115}
, {-44, -116}
, {56, 119}
, {-103, 99}
, {81, 52}
, {-48, 22}
, {84, 63}
, {-8, 77}
, {-45, -20}
, {-32, 49}
, {-5, -122}
, {34, -2}
, {-90, -55}
, {-69, 29}
, {-2, -17}
, {58, -121}
, {-113, -52}
, {0, 14}
, {-109, 83}
, {-186, -33}
, {-49, -162}
, {-114, -18}
, {57, 81}
, {132, 190}
, {46, 125}
, {9, -84}
}
, {{-163, -33}
, {105, 25}
, {-36, -195}
, {33, 33}
, {-50, -23}
, {-43, -214}
, {-191, 33}
, {76, -91}
, {32, 56}
, {-110, -199}
, {-129, -250}
, {18, -36}
, {-15, 111}
, {-53, 10}
, {-165, 18}
, {1, -54}
, {104, -223}
, {-91, 47}
, {-100, 44}
, {12, -189}
, {-44, 44}
, {18, -53}
, {-104, 2}
, {91, -10}
, {97, 116}
, {26, 6}
, {-17, -109}
, {115, 26}
, {-60, -267}
, {-1, 66}
, {-187, -173}
, {4, -132}
}
, {{-158, -14}
, {-130, 80}
, {-71, 43}
, {-176, 68}
, {152, -49}
, {-2, -131}
, {-21, 4}
, {-65, 57}
, {-155, -1}
, {-214, 22}
, {25, -23}
, {33, -45}
, {104, 48}
, {195, -47}
, {76, -26}
, {-175, -98}
, {-16, -126}
, {-225, 39}
, {-196, 87}
, {97, -95}
, {-236, -55}
, {-110, 77}
, {61, -18}
, {200, 52}
, {63, 44}
, {78, -40}
, {-102, 11}
, {-68, 63}
, {-27, -72}
, {140, 123}
, {-27, -33}
, {178, 4}
}
, {{6, -143}
, {46, -30}
, {121, 20}
, {64, 16}
, {-199, -105}
, {-105, -320}
, {111, 102}
, {57, -66}
, {3, 49}
, {-55, -300}
, {50, 120}
, {-33, 44}
, {-57, -199}
, {114, -121}
, {63, -270}
, {-137, -156}
, {154, 89}
, {-50, 120}
, {-90, -90}
, {-50, -206}
, {-30, -43}
, {49, 84}
, {152, 71}
, {207, 96}
, {-109, 76}
, {-147, -195}
, {-40, 3}
, {-2, 26}
, {119, 56}
, {-277, -5}
, {34, 112}
, {-13, -28}
}
, {{-32, -31}
, {44, 78}
, {-83, 74}
, {-234, -165}
, {-12, -28}
, {94, -4}
, {43, 2}
, {-73, 30}
, {87, 95}
, {-94, 97}
, {12, 36}
, {76, 26}
, {-9, 29}
, {77, -193}
, {-71, -6}
, {-59, -120}
, {94, -4}
, {-13, 180}
, {-31, -1}
, {-130, 9}
, {102, -58}
, {146, 12}
, {-137, 111}
, {-164, -69}
, {8, -71}
, {-108, 14}
, {99, -141}
, {41, -151}
, {27, -31}
, {-86, -38}
, {-52, -85}
, {112, -58}
}
, {{-94, -26}
, {-154, -33}
, {47, 130}
, {29, -20}
, {11, 2}
, {-68, 3}
, {-84, -14}
, {-41, -40}
, {-45, 103}
, {6, -6}
, {-23, 86}
, {-93, -177}
, {122, 30}
, {-4, 66}
, {-139, 2}
, {-20, 1}
, {57, 79}
, {-34, 85}
, {-80, -17}
, {-78, -111}
, {0, -19}
, {-87, 71}
, {146, -178}
, {-34, -25}
, {82, -29}
, {36, 114}
, {78, -78}
, {-152, 57}
, {88, -44}
, {-29, 81}
, {75, 46}
, {94, 57}
}
, {{15, 27}
, {-26, -54}
, {-249, -144}
, {41, -81}
, {65, 71}
, {-75, 192}
, {23, 88}
, {-23, 0}
, {71, -2}
, {-85, -30}
, {-74, -140}
, {35, 25}
, {65, -84}
, {-27, 36}
, {-129, 100}
, {-15, -29}
, {-407, 45}
, {-113, -83}
, {-130, 67}
, {1, 89}
, {-259, -86}
, {96, 70}
, {133, -8}
, {-111, -37}
, {-75, -24}
, {45, -108}
, {-19, -41}
, {-133, 85}
, {14, 36}
, {46, -15}
, {91, -77}
, {36, 20}
}
, {{34, 92}
, {69, 9}
, {-105, 19}
, {-20, 36}
, {-70, -86}
, {11, 86}
, {-89, -249}
, {42, 45}
, {40, 9}
, {84, 93}
, {-212, -132}
, {-128, -10}
, {0, 32}
, {43, 60}
, {17, 76}
, {-45, 20}
, {-29, -84}
, {-55, -119}
, {-73, 48}
, {13, 79}
, {0, -23}
, {-51, 18}
, {-20, 163}
, {-47, -83}
, {-4, 48}
, {-105, -58}
, {-32, -14}
, {45, 28}
, {-178, -124}
, {-134, 95}
, {-177, -49}
, {26, 29}
}
, {{97, -92}
, {115, 70}
, {-42, -15}
, {11, 14}
, {40, 80}
, {-6, 40}
, {-131, -111}
, {-121, -91}
, {69, -36}
, {-103, 38}
, {-116, 71}
, {-61, -88}
, {-14, 50}
, {-40, -37}
, {-157, -63}
, {-82, -163}
, {-7, 9}
, {101, 134}
, {106, -74}
, {-81, 8}
, {-30, -8}
, {45, -175}
, {33, -180}
, {-29, -271}
, {103, 26}
, {111, 55}
, {-43, -96}
, {24, -5}
, {64, 160}
, {-84, -33}
, {0, 26}
, {5, -49}
}
, {{-184, -29}
, {-61, 106}
, {-200, -178}
, {0, 106}
, {-99, 230}
, {69, 72}
, {-253, 16}
, {38, 8}
, {-6, -12}
, {-96, -65}
, {-90, 92}
, {55, 32}
, {97, 4}
, {-67, -44}
, {-238, -93}
, {22, 1}
, {-36, 98}
, {44, 129}
, {-61, 81}
, {-150, -35}
, {-39, 46}
, {3, -75}
, {-113, -178}
, {20, 118}
, {13, -65}
, {-162, 80}
, {92, -46}
, {-23, -176}
, {52, 40}
, {-116, -39}
, {-59, 14}
, {-82, -19}
}
, {{-20, -76}
, {-16, 75}
, {79, -19}
, {-4, 139}
, {-95, 36}
, {83, 43}
, {-40, 59}
, {44, -19}
, {-24, 11}
, {-81, -7}
, {57, -82}
, {34, -102}
, {80, 32}
, {-56, -66}
, {34, -49}
, {-181, 56}
, {-52, 107}
, {-28, 83}
, {-50, -26}
, {144, 55}
, {-32, 6}
, {-135, 56}
, {5, -44}
, {86, 5}
, {-394, -76}
, {15, 104}
, {-148, 34}
, {-6, 94}
, {59, 69}
, {-28, -245}
, {-84, 63}
, {-21, -12}
}
, {{-55, -35}
, {33, -29}
, {131, 73}
, {13, 130}
, {-80, 172}
, {-25, 66}
, {-122, -74}
, {-28, 81}
, {-142, 37}
, {-75, -27}
, {83, 4}
, {-5, -46}
, {-20, -62}
, {-42, -99}
, {-139, -217}
, {-7, -51}
, {-50, 51}
, {-94, 93}
, {-17, 44}
, {-108, 69}
, {57, 25}
, {-12, -26}
, {56, -187}
, {35, -85}
, {57, 17}
, {-81, -64}
, {-121, 101}
, {-140, 41}
, {148, -209}
, {-9, -100}
, {12, 46}
, {15, -15}
}
, {{33, 42}
, {-25, 193}
, {39, 55}
, {-51, 16}
, {-14, -52}
, {-190, -173}
, {223, 4}
, {-132, -113}
, {20, 115}
, {-149, -189}
, {-34, -151}
, {-85, -126}
, {-41, -3}
, {-35, 8}
, {-13, -66}
, {59, 102}
, {70, 113}
, {46, -68}
, {-83, 60}
, {72, -61}
, {-151, -127}
, {30, -15}
, {6, -166}
, {36, 120}
, {-96, 46}
, {105, 21}
, {20, 151}
, {-6, -46}
, {-78, -231}
, {-112, -90}
, {-13, 37}
, {-167, -125}
}
, {{138, 78}
, {-94, -107}
, {-76, 59}
, {-159, 108}
, {4, -120}
, {48, -6}
, {-34, 81}
, {13, 9}
, {2, 119}
, {-60, -7}
, {-67, -40}
, {-88, -148}
, {-145, 41}
, {-31, 38}
, {6, 107}
, {-34, -18}
, {10, 93}
, {-133, -48}
, {-236, -12}
, {-3, 12}
, {-133, -96}
, {0, 115}
, {-105, 41}
, {-114, 62}
, {107, 84}
, {-96, -266}
, {67, 88}
, {-37, -15}
, {62, 69}
, {144, 172}
, {-8, 1}
, {0, -67}
}
, {{-25, -92}
, {-30, 39}
, {-22, -42}
, {184, 105}
, {43, -81}
, {99, 23}
, {-106, -171}
, {40, 89}
, {-53, -75}
, {24, 39}
, {34, -1}
, {81, 99}
, {12, 33}
, {-6, -64}
, {-76, -192}
, {-157, -372}
, {32, 24}
, {165, -231}
, {3, -53}
, {-241, -342}
, {47, -44}
, {44, 111}
, {14, 12}
, {-92, -85}
, {81, -58}
, {-161, -98}
, {-72, -127}
, {-122, -106}
, {55, -31}
, {-79, 21}
, {23, 64}
, {52, 89}
}
, {{-105, -283}
, {-29, -40}
, {40, 103}
, {25, 78}
, {-47, -2}
, {58, -14}
, {-23, -31}
, {0, 2}
, {-34, 2}
, {31, 65}
, {-112, 140}
, {38, -113}
, {-71, -60}
, {-18, -121}
, {70, -49}
, {51, -154}
, {7, -19}
, {-19, 100}
, {-9, -63}
, {-113, -98}
, {-8, 0}
, {91, 19}
, {0, -89}
, {97, -173}
, {69, 45}
, {80, -79}
, {52, 5}
, {19, 89}
, {111, 122}
, {-98, -219}
, {-46, -71}
, {164, 112}
}
, {{-36, -73}
, {117, -238}
, {15, 4}
, {41, 41}
, {-91, 35}
, {138, -225}
, {-11, -71}
, {-78, -134}
, {79, -16}
, {-106, 124}
, {33, -168}
, {54, 56}
, {102, -9}
, {-75, -295}
, {-59, 38}
, {6, -66}
, {83, 33}
, {134, -27}
, {-15, -21}
, {-180, -44}
, {-144, -110}
, {126, 26}
, {41, 67}
, {43, -165}
, {20, -195}
, {-1, 24}
, {92, -69}
, {73, -17}
, {-195, 41}
, {122, 0}
, {-164, -62}
, {38, 83}
}
, {{-110, 125}
, {-46, -39}
, {146, -125}
, {-94, -129}
, {50, -2}
, {30, -10}
, {151, 52}
, {-54, -7}
, {25, -133}
, {62, 74}
, {-32, 59}
, {91, 61}
, {-58, -139}
, {6, -10}
, {-85, 26}
, {-21, -1}
, {-64, -15}
, {-48, 40}
, {-171, -66}
, {54, 101}
, {72, -2}
, {65, -24}
, {0, 64}
, {-154, -83}
, {-23, 133}
, {-169, 15}
, {66, -7}
, {44, -53}
, {20, 71}
, {115, 31}
, {-67, -155}
, {-56, 6}
}
, {{-83, -21}
, {-97, 24}
, {76, -203}
, {41, 87}
, {22, 24}
, {-4, 51}
, {118, 98}
, {-204, -304}
, {-13, -43}
, {148, 3}
, {-73, 2}
, {-458, -295}
, {50, 132}
, {-74, 24}
, {-52, -33}
, {62, -1}
, {-101, -150}
, {-36, -62}
, {-240, -10}
, {33, 6}
, {-25, -115}
, {14, 35}
, {35, 15}
, {102, 74}
, {-99, 122}
, {-104, -380}
, {-21, 61}
, {-80, -18}
, {-166, -13}
, {-27, -119}
, {71, 41}
, {-189, -174}
}
, {{12, -34}
, {52, 77}
, {-136, -6}
, {-43, -88}
, {-63, 27}
, {30, -10}
, {16, 30}
, {-43, -93}
, {-58, -19}
, {-87, 37}
, {166, 54}
, {252, 188}
, {-84, 0}
, {122, 0}
, {-34, -94}
, {22, -55}
, {-223, -158}
, {41, 44}
, {-36, -121}
, {65, 19}
, {-127, -89}
, {-16, 64}
, {-59, 111}
, {54, 58}
, {-18, -94}
, {58, 41}
, {5, -81}
, {-17, 91}
, {16, -3}
, {34, 5}
, {42, 28}
, {-73, -62}
}
, {{43, -89}
, {-56, -117}
, {-111, -135}
, {10, 16}
, {98, -53}
, {25, 34}
, {141, -90}
, {-42, -10}
, {26, -4}
, {-5, 17}
, {-141, 35}
, {71, -8}
, {20, 44}
, {-20, -68}
, {-23, 55}
, {3, 39}
, {18, -40}
, {-8, -32}
, {-58, -168}
, {48, 33}
, {-104, -54}
, {59, -35}
, {29, -154}
, {114, 3}
, {106, 177}
, {171, -93}
, {-8, -143}
, {53, -17}
, {-22, -158}
, {12, 72}
, {116, 62}
, {82, -1}
}
, {{-51, -38}
, {77, 47}
, {87, 45}
, {112, -39}
, {61, -73}
, {-39, -22}
, {-31, 82}
, {-66, -9}
, {-44, -93}
, {-54, -2}
, {-3, -9}
, {-22, -15}
, {63, -68}
, {-76, -153}
, {-38, -117}
, {0, 29}
, {-106, 66}
, {-37, 105}
, {-87, -56}
, {147, 86}
, {-154, 4}
, {-25, 77}
, {97, -137}
, {-26, 42}
, {-34, -53}
, {149, 65}
, {-39, 102}
, {-42, 48}
, {91, 36}
, {-41, -28}
, {-15, -78}
, {31, 28}
}
, {{-59, -21}
, {-236, 43}
, {-23, -194}
, {-13, -26}
, {5, 33}
, {11, -55}
, {-98, 79}
, {-316, 35}
, {-51, -148}
, {-194, -31}
, {-38, -136}
, {-124, -213}
, {-219, 104}
, {-35, -90}
, {-44, -30}
, {8, -45}
, {28, 136}
, {70, 129}
, {39, 50}
, {-21, -77}
, {-31, -108}
, {35, 133}
, {-63, -44}
, {-34, 104}
, {13, -57}
, {37, 84}
, {47, -22}
, {58, 24}
, {39, -77}
, {-50, 131}
, {-23, 28}
, {-18, 65}
}
, {{103, 5}
, {-77, 17}
, {-163, 28}
, {-48, -251}
, {21, 94}
, {-5, 52}
, {-32, 282}
, {-49, 34}
, {7, 149}
, {61, -98}
, {105, 2}
, {78, -29}
, {-68, 77}
, {38, 40}
, {-22, -231}
, {-21, -134}
, {-51, 71}
, {-14, 9}
, {-46, -30}
, {-9, 80}
, {1, 81}
, {10, -144}
, {-34, 0}
, {-31, -50}
, {140, -22}
, {33, -90}
, {-106, -94}
, {31, -167}
, {-15, -65}
, {100, -41}
, {-37, 22}
, {-103, -18}
}
, {{-27, 34}
, {62, 6}
, {74, 167}
, {-5, -51}
, {-39, -28}
, {-93, 81}
, {93, 110}
, {-82, -66}
, {-44, 113}
, {14, 149}
, {-86, -58}
, {-51, 11}
, {61, 24}
, {-72, -8}
, {38, -53}
, {-71, -61}
, {-84, -48}
, {-196, -159}
, {-41, -97}
, {141, 130}
, {-13, -19}
, {37, -146}
, {-19, 104}
, {64, 44}
, {-33, 77}
, {-151, 119}
, {15, 65}
, {31, -114}
, {-81, 65}
, {-120, 102}
, {-83, -57}
, {81, 141}
}
, {{0, 125}
, {-33, -64}
, {-3, 19}
, {118, 10}
, {-95, -190}
, {177, -193}
, {51, 127}
, {-157, 59}
, {-103, 21}
, {37, -123}
, {14, 77}
, {83, 6}
, {-61, 12}
, {1, -15}
, {36, 89}
, {-48, 79}
, {17, 8}
, {-282, -117}
, {-87, 4}
, {-2, -29}
, {-1, -55}
, {98, 53}
, {18, 125}
, {-109, 146}
, {18, -203}
, {-10, -53}
, {7, -139}
, {-11, -11}
, {87, -4}
, {160, -103}
, {-169, 15}
, {-260, 149}
}
, {{83, 38}
, {106, 32}
, {80, 20}
, {-174, -278}
, {36, -52}
, {182, 169}
, {76, 94}
, {57, -3}
, {110, -24}
, {-43, -10}
, {-97, -60}
, {-204, -212}
, {-30, -162}
, {-54, -52}
, {78, 87}
, {-71, -39}
, {59, 86}
, {16, -35}
, {15, -191}
, {44, -18}
, {-102, -78}
, {-28, -24}
, {82, -83}
, {42, 52}
, {85, 127}
, {56, -8}
, {-8, 67}
, {70, 15}
, {167, -8}
, {-221, -64}
, {-9, -92}
, {88, -61}
}
, {{46, -49}
, {85, 93}
, {34, 47}
, {-135, -122}
, {90, -94}
, {83, 129}
, {80, 70}
, {35, -42}
, {23, -218}
, {32, -40}
, {-78, -48}
, {18, -133}
, {-10, -14}
, {-32, -31}
, {-197, 53}
, {-119, -83}
, {53, -23}
, {-63, -175}
, {-112, -51}
, {34, 51}
, {-84, 26}
, {-53, 47}
, {-132, 23}
, {1, 28}
, {29, 198}
, {34, 43}
, {-3, 116}
, {-2, 61}
, {44, 62}
, {72, 131}
, {128, -112}
, {253, -96}
}
, {{52, 42}
, {-43, -31}
, {169, 34}
, {-170, -252}
, {40, 11}
, {102, 90}
, {-18, -10}
, {-59, -121}
, {48, 53}
, {-125, 9}
, {-199, -187}
, {-73, 42}
, {-7, 148}
, {-7, 4}
, {17, 91}
, {-47, 21}
, {-12, -49}
, {-81, -40}
, {-196, -193}
, {20, 28}
, {-13, -48}
, {-5, -32}
, {-383, -47}
, {37, 152}
, {155, -6}
, {77, 53}
, {-25, -19}
, {-175, -35}
, {-256, 28}
, {87, -88}
, {20, 176}
, {42, 59}
}
, {{91, 81}
, {-80, 14}
, {-73, -1}
, {118, 100}
, {-3, 26}
, {-115, 11}
, {131, 31}
, {60, -51}
, {-46, 114}
, {-231, -157}
, {-285, -42}
, {-98, -106}
, {64, -127}
, {-161, -185}
, {-2, -176}
, {41, 33}
, {-27, 1}
, {12, 66}
, {45, 17}
, {38, -119}
, {-82, -155}
, {84, -204}
, {-17, 68}
, {12, -92}
, {28, -69}
, {-252, -51}
, {29, -122}
, {88, 12}
, {-112, -155}
, {-126, -168}
, {3, -129}
, {-6, 22}
}
, {{56, 42}
, {-124, 13}
, {-13, -67}
, {-45, -26}
, {-19, 105}
, {-226, -112}
, {59, 107}
, {-29, -24}
, {95, 139}
, {-67, 35}
, {-308, -59}
, {-126, -86}
, {28, 3}
, {60, -28}
, {-73, 76}
, {1, 47}
, {-37, -119}
, {-5, -8}
, {15, -45}
, {113, 50}
, {-172, -217}
, {-105, -62}
, {33, 0}
, {-86, 87}
, {113, 2}
, {95, 110}
, {-150, -85}
, {-86, -49}
, {-43, -172}
, {21, 0}
, {54, -6}
, {25, -127}
}
, {{-150, -66}
, {-132, 82}
, {88, 36}
, {54, 155}
, {-14, -23}
, {-133, -28}
, {134, -434}
, {-87, -66}
, {39, 69}
, {99, 57}
, {175, 4}
, {41, 83}
, {10, -102}
, {12, -78}
, {66, -245}
, {-24, 68}
, {-63, 124}
, {-20, -42}
, {-36, -7}
, {-19, 104}
, {-51, -97}
, {-56, 11}
, {-57, 2}
, {-39, -59}
, {-114, 22}
, {-102, -16}
, {-6, 10}
, {-81, -24}
, {148, -219}
, {-71, 44}
, {-198, -253}
, {4, 38}
}
, {{113, -70}
, {94, 28}
, {-157, -73}
, {-85, -67}
, {70, 89}
, {-273, -175}
, {67, -97}
, {7, -5}
, {-72, 98}
, {23, 54}
, {-35, 135}
, {-180, 74}
, {99, 57}
, {-67, -44}
, {-42, 15}
, {-115, 54}
, {13, 4}
, {-49, 9}
, {-75, -11}
, {-97, -34}
, {-152, -58}
, {-21, 69}
, {61, -60}
, {158, -36}
, {81, -81}
, {-146, 62}
, {-150, -16}
, {-70, 49}
, {77, 161}
, {128, -81}
, {69, 21}
, {-52, 75}
}
, {{-18, -43}
, {-2, 27}
, {-15, 124}
, {122, -131}
, {-150, -67}
, {-116, -99}
, {84, -248}
, {12, -97}
, {-79, 140}
, {-8, 6}
, {-21, -60}
, {-62, 29}
, {-141, -127}
, {-18, 141}
, {-65, 69}
, {11, 45}
, {-129, -67}
, {-99, -246}
, {64, 13}
, {-42, -100}
, {-11, -2}
, {72, -55}
, {-1, -101}
, {-167, -163}
, {58, -32}
, {-113, -20}
, {130, 37}
, {31, -53}
, {81, -330}
, {-11, -61}
, {-86, 152}
, {-41, 27}
}
, {{19, -22}
, {-61, 57}
, {-79, -229}
, {-90, 156}
, {-158, -27}
, {-7, -44}
, {-106, -139}
, {-42, 133}
, {-89, 73}
, {65, 101}
, {-114, 21}
, {66, 27}
, {-126, -188}
, {16, 28}
, {-105, -87}
, {-20, 70}
, {-6, -81}
, {37, -20}
, {-37, 71}
, {65, 68}
, {-35, 20}
, {6, 9}
, {-89, -157}
, {-36, 26}
, {-14, 6}
, {82, -131}
, {45, 57}
, {12, -39}
, {-111, -19}
, {-44, -27}
, {-76, -31}
, {38, 50}
}
, {{38, 130}
, {17, 8}
, {-97, -241}
, {-12, -204}
, {-73, -87}
, {99, -6}
, {17, 38}
, {-46, -192}
, {-26, 128}
, {-129, -36}
, {107, 42}
, {14, 107}
, {81, -145}
, {68, -12}
, {-87, -79}
, {13, -25}
, {-74, 20}
, {13, -87}
, {11, -137}
, {-72, 2}
, {24, 53}
, {-224, -13}
, {208, -170}
, {-77, 44}
, {35, -54}
, {32, 84}
, {-7, -13}
, {66, -60}
, {-78, -160}
, {162, 86}
, {70, 84}
, {-135, -7}
}
, {{7, 56}
, {-93, -4}
, {-31, -42}
, {-142, 29}
, {-117, 87}
, {54, -13}
, {94, 119}
, {34, -65}
, {-100, 12}
, {-38, 71}
, {-121, -75}
, {-94, 1}
, {6, -54}
, {-28, -32}
, {-17, 61}
, {-138, 143}
, {-216, -45}
, {-169, 45}
, {-31, 0}
, {70, 3}
, {-56, -41}
, {-79, 45}
, {16, 103}
, {57, 79}
, {-40, 67}
, {-5, 212}
, {-145, -19}
, {-36, 47}
, {-258, 15}
, {-11, -118}
, {-29, 101}
, {-21, 66}
}
, {{-111, -150}
, {-26, -68}
, {15, 27}
, {-239, 48}
, {33, 115}
, {118, -124}
, {-26, 72}
, {19, -29}
, {-91, 53}
, {-1, 38}
, {-109, 104}
, {-122, 27}
, {25, 51}
, {-100, -10}
, {-48, -1}
, {-75, -169}
, {-46, -15}
, {87, -259}
, {77, -67}
, {125, 102}
, {5, 52}
, {-118, 1}
, {7, 15}
, {-132, -95}
, {-148, -90}
, {-199, 58}
, {-37, 69}
, {-100, 138}
, {-47, 22}
, {41, 27}
, {-74, 109}
, {-64, -35}
}
, {{-27, 85}
, {47, 119}
, {19, -266}
, {-45, -60}
, {28, -85}
, {-82, -33}
, {-17, -91}
, {-84, -3}
, {69, 95}
, {-72, 52}
, {-179, -86}
, {83, -79}
, {107, 16}
, {-322, -18}
, {5, 30}
, {-137, 149}
, {3, 41}
, {-55, -93}
, {-9, -164}
, {2, 39}
, {-34, 111}
, {66, -69}
, {-8, -89}
, {45, 106}
, {2, 57}
, {19, -24}
, {-68, 25}
, {-85, -67}
, {-150, 23}
, {-8, 65}
, {30, -60}
, {-58, 35}
}
, {{91, 42}
, {81, -70}
, {17, 19}
, {127, 96}
, {14, 92}
, {-26, -57}
, {69, 32}
, {62, -47}
, {-54, 88}
, {-24, -23}
, {-71, 82}
, {-95, -78}
, {78, -8}
, {-20, 44}
, {-140, -128}
, {26, -129}
, {-63, -37}
, {-17, 42}
, {32, -88}
, {-1, -59}
, {-34, 4}
, {72, 75}
, {-53, -43}
, {-50, -389}
, {-1, -8}
, {-67, -151}
, {27, 48}
, {-36, 52}
, {35, 29}
, {131, -57}
, {-19, 43}
, {46, -19}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    averagepool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   23
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t average_pooling1d_output_type[INPUT_CHANNELS][POOL_LENGTH];

void average_pooling1d(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned short x;
  long_number_t avg, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
      tmp = 0;
      for (x = 0; x < POOL_SIZE; x++) {
        tmp += input[k][(pos_x*POOL_STRIDE)+x];
      }
#ifdef ACTIVATION_RELU
      if (tmp < 0) {
        tmp = 0;
      }
#endif
      avg = tmp / POOL_SIZE;
      output[k][pos_x] = clamp_to_number_t(avg);
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_DIM [5][64]
#define OUTPUT_DIM 320

//typedef number_t *flatten_output_type;
typedef number_t flatten_output_type[OUTPUT_DIM];

#define flatten //noop (IN, OUT)  OUT = (number_t*)IN

#undef INPUT_DIM
#undef OUTPUT_DIM

/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_SAMPLES 320
#define FC_UNITS 5
#define ACTIVATION_LINEAR

typedef number_t dense_output_type[FC_UNITS];

static inline void dense(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]) {			                // OUT

  unsigned short k, z; 
  long_number_t output_acc; 

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0; 
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ( kernel[k][z] * input[z] ); 

    output_acc = scale_number_t(output_acc);

    output_acc = output_acc + bias[k]; 


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = clamp_to_number_t(output_acc);
#elif defined(ACTIVATION_RELU)
    // ReLU
    if (output_acc < 0)
      output[k] = 0;
    else
      output[k] = clamp_to_number_t(output_acc);
#endif
  }
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 320
#define FC_UNITS 5


const int16_t dense_bias[FC_UNITS] = {-145, 21, 137, -24, 19}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{-34, -38, -93, -80, -99, 61, 52, 38, 13, 16, -70, -70, -22, -37, -41, 71, 36, 16, 97, 55, 23, -21, 103, 17, 73, -11, 22, -15, -31, -95, 143, 61, 49, 32, 53, 204, 48, 104, 75, 106, 60, 80, 62, 89, 68, 0, 8, -13, -36, -17, -1, 14, 5, -46, -53, -9, 0, 10, -12, 4, 75, 73, 43, 31, 52, 122, 152, 155, 205, 235, -14, 42, -40, -3, -20, -57, -53, 32, -45, -69, -102, -34, -89, -67, -10, -68, -2, -27, 6, -48, 49, 44, 37, -34, -34, -12, -74, -13, 0, -36, 59, 90, 77, 16, 70, -178, -123, -174, -163, -158, -4, -23, -5, -11, 32, 159, 103, -8, 37, 88, 112, 151, 104, 48, 39, 36, 19, -40, 9, -74, 2, 85, 9, 40, 56, -77, -18, -9, -28, -55, 26, 29, -12, 29, 13, 58, 37, 23, -16, -51, -57, -152, 4, -8, 1, 19, 58, 73, 11, -5, -138, -111, -11, 15, -24, -47, -76, -95, -66, -157, 54, -26, 44, 6, -8, 24, -26, 1, 35, 0, -4, 14, 37, -15, -20, 19, -2, 42, 43, 42, 0, -53, -71, -67, -85, -17, -30, -9, -85, -39, -26, 69, 112, 39, 52, -10, -39, -7, 6, 7, -133, -52, -125, -117, -124, -102, -133, -74, -102, -66, -105, 31, -59, -71, -76, 90, 71, 33, 52, 77, 0, 12, 4, -6, 65, -91, -67, -22, -50, -1, 0, 73, 40, -11, 50, 51, 31, 33, 87, 28, 32, -3, 23, 29, 130, 44, 29, 57, 98, 54, -91, -92, -55, -9, -60, -38, 58, -11, 33, 79, -117, -77, -128, -160, -146, 99, 121, 87, 123, 99, 16, 9, 40, 10, -9, 54, 10, 55, 62, 148, 33, 6, 17, 37, 51, 62, 76, 1, 52, 74, -126, -109, -114, -85, -117, -119, -80, -99, -14, -41, 50, 29, 39, 136, 164, -76, -101, -58, -35, -126}
, {72, 30, 99, 96, 29, -40, -65, -72, -121, -136, -80, -50, 43, -51, 2, 65, 14, 31, 47, 110, 107, 81, 117, 16, 67, 35, -9, 86, -8, 57, -199, -95, -214, -159, -130, 60, 74, -16, 19, 105, 57, 72, 38, 68, 15, -26, -66, -110, -128, -119, -123, -57, -19, -115, -98, 71, 81, 60, 74, 58, 294, 164, 173, 162, 186, 88, 88, 23, 120, -23, 34, 30, 64, 69, 16, 153, 123, 81, 94, 136, 81, -146, 153, -41, -44, 208, 92, 80, 125, 212, 32, 2, 30, 95, 46, 81, 35, -6, -67, 42, -29, 11, -81, -13, 15, -11, 12, 9, -30, 5, -133, -95, -73, -59, -59, -58, -111, -26, -28, -38, -109, -73, -55, -33, -3, -22, 51, -9, 46, -50, -87, -202, 66, -57, -187, -234, -17, -51, -121, -203, 64, 43, 61, 47, 140, 79, 40, -108, -22, -41, 84, 176, -88, 14, -15, -98, -30, 7, 34, 3, -196, 57, -240, -155, -129, -42, 32, -27, 17, -88, -60, 17, 46, 2, -6, 78, 9, 45, 10, 12, -42, -38, -91, -94, -146, -136, -166, -121, -162, -163, -15, -93, 34, 7, -101, -42, -89, -34, 39, 0, -181, 83, 18, -37, -48, 45, 64, 49, 20, 37, 59, 17, 81, 75, 33, 123, 102, 115, 99, 112, 72, 41, 14, 22, 16, 66, -80, -105, -50, -15, -206, -82, -147, -263, -322, 104, 84, 160, 94, 137, -66, -116, -44, -99, -86, -13, -72, -8, -15, -20, -119, -96, -37, -58, -36, -46, -53, -124, -69, -58, 30, 155, -6, 41, 16, -181, -152, -259, -145, -191, -7, 33, 71, 13, 54, -54, -104, 16, -107, -90, 135, 23, 40, 55, 102, -82, -30, -56, -60, 10, 42, 13, -47, -24, 19, -42, -93, 12, 29, 11, 83, 116, 39, 140, 119, -16, -66, -19, 26, -30, -39, -21, -22, -71, -96, 6, 11, 40, 50, 6}
, {61, 51, 18, 41, -6, 80, 80, 61, 63, 74, -39, 38, 1, 58, 68, 39, 19, -22, 55, 30, -181, -212, -140, -81, -68, 154, 73, 110, 52, 84, 52, 36, 43, 70, 80, -73, -40, -63, -18, -89, -117, -91, -161, -82, -20, 143, 79, 97, 78, 103, 153, 101, 5, 56, 18, -74, -36, -61, -11, -36, -228, -188, -177, -113, -68, -177, -179, -118, -251, -297, -68, 77, 26, -21, -36, -39, -48, 27, -7, -44, -223, -191, -292, -272, -322, 12, 3, -42, -3, -17, 131, 132, 160, 87, 154, -28, 47, 7, 30, 109, -12, 7, -12, -6, -44, -43, 54, 21, -8, 44, -110, -3, 15, 13, 21, 91, -15, 29, -53, -15, -110, -40, -78, -113, -154, 78, 57, 76, 78, 130, 8, 54, -2, 45, 162, 175, 173, 167, 220, 230, -117, -34, -68, -40, 8, -68, -19, -17, -39, -82, 141, 144, 217, 184, 101, -47, 47, 73, 13, 34, 130, -13, 179, 97, 138, 72, -33, -98, -106, -71, 30, 67, 11, 89, 77, -139, -63, -31, -176, -84, -22, -26, -19, -67, -71, 104, 149, 110, 128, 148, -1, 7, 1, 15, 30, -39, -27, -55, -31, -100, 54, 17, -100, -31, -104, 60, 29, 103, 60, 24, -200, -72, -149, -133, -85, -116, -133, -161, -133, -114, -16, 43, 71, -18, -60, 20, -20, -7, -2, -8, -1, -23, -22, 6, 23, 25, -36, -115, -125, -19, 52, 66, 65, 125, 89, 65, 41, 58, 80, 57, -79, -69, -80, -86, -109, -94, -119, -73, -74, 47, -13, -72, -48, -60, -49, 145, 159, 99, 83, 142, -104, -84, -96, -141, -150, -66, -58, -33, 142, 58, -110, -115, -119, -132, -179, 141, 66, 55, 115, 61, 2, -132, -90, -75, -109, 38, -10, -28, -55, -173, 22, -8, 58, 63, 82, 140, 121, 75, 19, 189, -137, -132, -158, -112, -85, 31, -41, -5, 52, -64}
, {-3, 32, -19, 39, -26, 16, 5, 30, -16, -48, 30, -20, -42, -18, 23, -186, -177, -150, -127, -139, 23, -26, 33, 16, 79, -16, -6, 68, 25, 29, -88, 78, -32, 3, -32, 78, 23, -45, -74, -19, 30, 18, -3, 72, 41, 35, 33, 78, 23, 97, -101, -123, -171, -70, -64, 22, 27, 92, 15, 21, 16, -48, -25, -32, 24, 57, -76, -34, -72, -17, 31, -13, -54, -5, -66, -18, -75, -59, -80, -65, 16, 255, 96, 141, 123, -243, -28, -146, -47, -115, -31, -10, -15, 27, -42, -22, 17, -63, -23, -111, -53, -68, -29, -43, -53, 58, 34, 40, 27, 64, 131, 53, 78, 77, 108, -128, -16, 51, 54, 77, -28, 44, 120, 28, 37, -136, -142, -177, -17, 43, 360, 103, 158, 107, 119, -40, -38, 13, -28, -109, 108, 107, 62, 2, 35, 66, 12, 42, 56, 7, -279, -207, -150, -201, -243, -62, -65, -62, -107, -104, 177, 123, 119, 125, 107, -12, 53, 41, 54, 33, 69, -48, 11, 5, 59, 72, 109, 36, 50, 83, 110, 143, 157, 151, 210, -100, -144, -47, -51, -97, 131, 159, 192, 154, 110, 23, 63, 107, 47, 55, 150, -71, -3, 8, 99, -11, 7, 8, -8, 26, 16, 10, 19, 91, 39, -93, -97, -72, -109, -36, -4, 48, -15, -54, -43, 20, 54, 73, -6, 8, 68, 70, -21, 17, 28, -82, -44, 25, 53, -4, -100, -72, -120, -175, -155, -219, -160, -224, -241, -195, -67, 46, -19, 15, 31, -22, -111, -43, -61, -52, -114, -130, -207, -157, -206, -31, -23, -5, -13, 6, 86, 45, 79, 54, 58, -1, -35, -106, -54, 25, -27, -33, 20, 2, 43, 57, 58, -31, -26, -110, -45, -21, -28, -42, -18, -199, -147, -165, -138, -291, -7, 25, -3, -41, 0, 103, -31, -24, 12, 3, 92, 137, 70, 128, 97, 24, 31, -27, 35, 3}
, {-71, -70, -92, -29, -41, 5, 8, -2, -7, -9, 20, 69, 68, 18, 51, -91, -101, -89, -99, -106, -70, -96, -25, -8, -26, -96, -147, -86, -113, -152, 77, 84, 14, 75, 48, -150, -67, -114, -50, -34, -2, -107, -92, -58, -139, -36, -48, -22, -53, -37, 84, 52, -7, 26, 51, -85, -69, -133, -116, -62, -259, -195, -252, -194, -278, -121, -70, -178, -93, -65, 55, 45, 21, 27, 33, -7, -10, 19, 6, 87, 35, 46, 85, 108, 119, 172, 78, 31, 16, 45, -129, -164, -191, -112, -187, 41, -51, 23, 41, 3, 76, 60, 55, 71, 153, 110, 107, 119, 47, 89, 29, -1, 6, 2, 3, -187, -53, -146, -59, -194, 1, 68, 32, -96, -71, -50, 4, 36, 45, -37, -217, -16, -261, -95, -101, -48, -196, -11, 21, -9, -59, -82, -147, -150, -61, 28, 29, 77, 88, 48, -5, -9, -12, -45, -70, 32, 10, 51, 39, 48, 32, -204, 24, -158, -82, 74, 17, 77, 99, 134, -126, -118, -106, -157, -133, 35, -9, 41, 13, -33, 14, 17, 7, -28, -10, 51, 82, 30, 39, -24, -176, -115, -37, -96, -95, 66, 64, 22, 46, 59, -144, -30, -94, -62, -167, -243, -116, -112, -99, -109, 83, 50, 83, 55, 100, 54, 47, 78, 43, 67, 46, 107, 83, 45, 32, -86, -116, -51, -76, -113, 40, 75, 24, 81, 97, -52, -2, 19, -1, -61, 87, 25, 79, -5, 26, 38, 90, 78, 63, 42, 88, 45, 89, 58, 93, 81, 86, 92, 159, 155, 105, 78, 74, 110, 152, 53, 66, 34, 39, -7, 74, 114, 96, 77, 163, -31, 64, -29, 40, -37, 42, -10, -1, -13, -57, -98, -98, -110, -147, -20, -31, -1, 9, 31, -3, 12, 75, 83, 9, 58, -59, -63, -96, 5, -8, -5, -14, -24, -18, -21, -56, 0, -47, 25, -7, 126, 19, 8, 65, 99}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define MODEL_OUTPUT_SAMPLES 5
#define MODEL_INPUT_SAMPLES 16000 // node 0 is InputLayer so use its output shape as input shape of the model
#define MODEL_INPUT_CHANNELS 1

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  //dense_output_type dense_output);
  number_t output[MODEL_OUTPUT_SAMPLES]);

#endif//__MODEL_H__
/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"

 // InputLayer is excluded
#include "conv1d.c"
#include "weights/conv1d.c" // InputLayer is excluded
#include "max_pooling1d.c" // InputLayer is excluded
#include "conv1d_1.c"
#include "weights/conv1d_1.c" // InputLayer is excluded
#include "max_pooling1d_1.c" // InputLayer is excluded
#include "conv1d_2.c"
#include "weights/conv1d_2.c" // InputLayer is excluded
#include "max_pooling1d_2.c" // InputLayer is excluded
#include "conv1d_3.c"
#include "weights/conv1d_3.c" // InputLayer is excluded
#include "average_pooling1d.c" // InputLayer is excluded
#include "flatten.c" // InputLayer is excluded
#include "dense.c"
#include "weights/dense.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_output_type dense_output) {

  // Output array allocation
  static union {
    conv1d_output_type conv1d_output;
    conv1d_1_output_type conv1d_1_output;
    conv1d_2_output_type conv1d_2_output;
    conv1d_3_output_type conv1d_3_output;
  } activations1;

  static union {
    max_pooling1d_output_type max_pooling1d_output;
    max_pooling1d_1_output_type max_pooling1d_1_output;
    max_pooling1d_2_output_type max_pooling1d_2_output;
    average_pooling1d_output_type average_pooling1d_output;
    flatten_output_type flatten_output;
  } activations2;


  //static union {
//
//    static input_1_output_type input_1_output;
//
//    static conv1d_output_type conv1d_output;
//
//    static max_pooling1d_output_type max_pooling1d_output;
//
//    static conv1d_1_output_type conv1d_1_output;
//
//    static max_pooling1d_1_output_type max_pooling1d_1_output;
//
//    static conv1d_2_output_type conv1d_2_output;
//
//    static max_pooling1d_2_output_type max_pooling1d_2_output;
//
//    static conv1d_3_output_type conv1d_3_output;
//
//    static average_pooling1d_output_type average_pooling1d_output;
//
//    static flatten_output_type flatten_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  conv1d(
     // First layer uses input passed as model parameter
    input,
    conv1d_kernel,
    conv1d_bias,
    activations1.conv1d_output
  );
 // InputLayer is excluded 
  max_pooling1d(
    
    activations1.conv1d_output,
    activations2.max_pooling1d_output
  );
 // InputLayer is excluded 
  conv1d_1(
    
    activations2.max_pooling1d_output,
    conv1d_1_kernel,
    conv1d_1_bias,
    activations1.conv1d_1_output
  );
 // InputLayer is excluded 
  max_pooling1d_1(
    
    activations1.conv1d_1_output,
    activations2.max_pooling1d_1_output
  );
 // InputLayer is excluded 
  conv1d_2(
    
    activations2.max_pooling1d_1_output,
    conv1d_2_kernel,
    conv1d_2_bias,
    activations1.conv1d_2_output
  );
 // InputLayer is excluded 
  max_pooling1d_2(
    
    activations1.conv1d_2_output,
    activations2.max_pooling1d_2_output
  );
 // InputLayer is excluded 
  conv1d_3(
    
    activations2.max_pooling1d_2_output,
    conv1d_3_kernel,
    conv1d_3_bias,
    activations1.conv1d_3_output
  );
 // InputLayer is excluded 
  average_pooling1d(
    
    activations1.conv1d_3_output,
    activations2.average_pooling1d_output
  );
 // InputLayer is excluded 
  flatten(
    
    activations2.average_pooling1d_output,
    activations2.flatten_output
  );
 // InputLayer is excluded 
  dense(
    
    activations2.flatten_output,
    dense_kernel,
    dense_bias, // Last layer uses output passed as model parameter
    dense_output
  );

}
