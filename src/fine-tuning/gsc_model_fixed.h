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


const int16_t conv1d_bias[CONV_FILTERS] = {128, -144, -64, -20, -218, -11, -22, 374}
;

const int16_t conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-40, 21, 17, 6, 67, -70, 58, -101, 38, -109, 54, 6, 6, 22, -32, 69, -61, 44, -50, 10}
}
, {{-98, -17, 5, 13, -26, 17, -76, -9, -100, -16, -67, 12, -26, 35, 0, 71, 18, 43, 94, 173}
}
, {{-59, -77, -42, -21, -70, -38, 40, -104, -9, 7, -45, 25, -66, -61, 31, -1, 12, -9, 28, -5}
}
, {{14, 62, 60, 49, 52, -1, 39, -74, 103, -26, 64, -24, 58, 40, 6, 43, 30, 69, -17, 144}
}
, {{49, 81, -11, 49, -25, 30, 26, -38, 25, -61, 9, -80, 14, -98, -40, -33, -69, 72, -86, 12}
}
, {{77, -96, -70, 130, -80, -92, 62, 140, 72, -169, 2, 37, 6, -42, 117, -28, 24, -112, -88, 146}
}
, {{-59, -65, -30, -63, -37, 5, -8, 40, 26, -22, -18, 13, -1, 12, 2, -55, -89, -63, -90, -75}
}
, {{68, 120, 9, -123, -93, 6, 39, 92, 37, -28, 6, -20, -42, 58, 129, 104, -13, -9, -67, -94}
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


const int16_t conv1d_1_bias[CONV_FILTERS] = {-212, 127, 12, 136, 8, -498, 80, 25, -7, -59, 18, -92, -168, -12, 0, 87}
;

const int16_t conv1d_1_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-51, 46, 121, 50, 9, -71, -117, -32}
, {-144, -18, 59, -36, -77, 19, -21, -35}
, {-6, -4, 135, 76, 54, -111, -129, -66}
, {-50, 15, -106, -102, 50, 83, 77, 31}
, {-15, 79, 162, -13, -108, -1, -43, -117}
, {-131, -10, 38, 26, 7, -6, 19, -122}
, {59, 92, 79, 60, 11, -105, -42, -124}
, {-129, -52, -35, -72, -6, 60, 55, 6}
}
, {{-26, -131, -59, 119, -67, -16, 31, 73}
, {-29, -12, -128, 28, -1, -60, -126, -18}
, {-160, -133, -80, 40, 43, -108, 18, 40}
, {108, 31, -11, -1, -87, 10, 160, -176}
, {23, -32, -51, 59, -15, -98, 33, 133}
, {-149, -51, -7, 92, -137, -174, -89, 11}
, {19, -17, 59, 60, -16, 0, 39, 111}
, {88, 10, 21, -78, -50, 43, 36, -132}
}
, {{64, 21, -63, 53, 63, 81, 95, 71}
, {-116, -126, 36, 155, 16, -59, -175, -25}
, {13, -25, -78, -45, -101, -40, 58, 9}
, {-11, 116, 92, 75, 39, -102, -125, -28}
, {-30, -96, -49, 34, -14, -17, -17, 135}
, {-103, -190, -14, 156, 87, -72, -175, -140}
, {-64, 61, 54, -30, -126, 2, -42, 118}
, {-4, -42, 96, 163, -65, -33, -130, -106}
}
, {{-106, -76, -1, -85, -42, 75, 81, -7}
, {-196, 17, 108, 134, 61, -47, -69, -85}
, {84, -8, -91, -124, -49, 115, 110, 25}
, {-142, -46, 100, 105, 79, -20, 8, -91}
, {-114, 44, -42, 68, 20, 79, 74, -16}
, {-383, -178, -65, 0, -170, -177, -212, -266}
, {32, 42, -123, -160, -59, 28, 51, 10}
, {-139, 13, 104, 60, 54, -24, -60, -73}
}
, {{-53, -51, -118, -73, -65, -85, -58, -23}
, {46, 82, 25, -6, 23, 70, 81, 19}
, {-25, -43, 8, 48, -23, -59, -45, -23}
, {45, 124, 80, -44, -67, -69, -55, -154}
, {-51, -54, -48, 3, -59, -18, 29, 10}
, {-7, -35, 50, 51, 70, 178, 126, 142}
, {101, 49, 35, -45, 8, 76, 32, 67}
, {-52, 98, 35, 118, 27, -2, 116, -29}
}
, {{-102, 43, 51, 42, 88, 75, 53, 16}
, {-57, -137, -11, 6, -9, 14, 69, 90}
, {-93, -99, 71, 56, 20, 34, -43, -81}
, {36, 178, 113, 27, -104, -128, -36, 39}
, {-182, -63, 155, 149, 127, 10, -33, -32}
, {40, -43, -54, -29, 21, 9, 14, 16}
, {-75, 77, 35, 43, 9, -16, -51, -31}
, {39, 47, -28, -4, -116, -123, -54, 8}
}
, {{20, 46, 44, 37, -19, -88, -25, 43}
, {17, -2, -77, 0, 9, 0, 114, 85}
, {92, 32, 20, 25, -36, -163, -49, 34}
, {8, -27, -38, -123, 14, 42, -16, 43}
, {85, -3, 17, 11, -102, -61, 56, -3}
, {-17, -69, -280, -33, -4, -115, 12, -71}
, {81, 25, 74, 91, -104, -67, -24, -41}
, {-15, -49, -70, -55, -52, 45, 40, 28}
}
, {{108, -62, -111, 96, -144, -5, -34, -17}
, {2, -27, -32, 93, -50, -58, 20, -4}
, {-40, 4, -118, 122, -12, -37, -130, 92}
, {-121, -44, -27, -46, 42, -23, 88, 77}
, {94, -84, -55, 108, -149, -93, -4, 169}
, {-70, -168, -174, 236, 31, -61, -49, 56}
, {-46, -124, 69, 12, -140, 25, -150, 197}
, {-88, -61, 25, 90, 47, -22, 77, 47}
}
, {{-8, -85, 30, 100, 13, 121, 36, -90}
, {58, 66, 77, -2, 16, 108, 16, -49}
, {-11, -17, -97, 63, 55, 56, 85, -7}
, {62, 14, -165, 13, -83, 88, 32, -25}
, {-5, -7, 22, 19, 60, 101, 50, -41}
, {11, -74, -137, -39, -209, -38, 28, -133}
, {-94, -133, -9, -24, -45, -14, -44, 18}
, {100, -3, -38, 30, -74, -24, 2, -153}
}
, {{-140, -26, 12, 81, 105, 85, -18, -65}
, {-141, -183, 56, 98, 113, 66, 2, -45}
, {-6, -19, -44, 59, 12, -8, -40, 6}
, {-44, 32, 99, 11, -58, -50, -51, 4}
, {-78, -125, 159, 143, 85, -10, 0, -50}
, {-284, -369, -71, 290, 216, -94, -223, -117}
, {-56, -88, -141, -69, -69, -22, -61, 23}
, {-113, 2, 98, 129, 81, -8, -3, -30}
}
, {{-112, -93, 10, 139, -9, 52, 64, -144}
, {-54, -43, -19, 131, 30, 2, -92, -31}
, {-73, -52, -12, 112, 128, 3, 17, -128}
, {-6, -115, 15, -41, 46, 34, 10, -97}
, {-50, -50, -18, 144, 103, -24, -62, -84}
, {-114, -61, -19, 69, -25, -81, -75, -81}
, {-51, -62, 80, 16, 73, 18, -107, -5}
, {-92, -78, 49, -7, 123, 84, 55, -74}
}
, {{-31, -13, 2, -72, -42, 143, -25, -145}
, {31, 67, -27, -90, -84, 45, 63, 65}
, {-44, 17, -50, -7, -107, 162, 70, -44}
, {23, 33, -17, 26, -2, -10, -110, -24}
, {28, 43, -67, -158, 7, 163, 33, -33}
, {26, 68, -92, -310, -127, 40, -150, -242}
, {112, -102, -65, -66, 20, 146, 34, -108}
, {26, 45, 35, -32, -34, 58, -52, 23}
}
, {{-19, 75, -46, -37, 54, -35, -118, -1}
, {-17, 39, 27, 76, 63, 25, 16, 104}
, {89, 0, -38, -62, 67, 11, -89, -121}
, {-13, 21, -31, -30, -139, -20, 81, 64}
, {33, 31, -7, 105, 116, -21, -56, 7}
, {-105, -129, 23, 8, 54, -43, -101, 22}
, {48, 0, -48, 45, -62, 46, 23, 2}
, {-69, -39, -13, -45, -33, 75, 16, 44}
}
, {{31, -30, 69, 13, 65, 24, 20, 63}
, {-124, -198, 88, -56, -73, -34, 87, 190}
, {3, -7, 14, -33, -59, -25, 16, -92}
, {-63, -123, 78, 80, -62, 26, 75, -11}
, {47, -72, 105, 15, -8, 46, 39, -82}
, {0, -1, 96, 47, -45, -9, 108, 61}
, {69, 47, 81, -93, 19, -20, -28, -100}
, {-65, -232, -43, 37, -70, -177, -26, -38}
}
, {{30, -57, 89, 33, 104, 104, 25, 1}
, {-46, -38, -190, 52, 35, -113, 40, -63}
, {42, 59, 25, -50, -106, 83, -70, 81}
, {-34, 12, 33, -2, 87, -51, 49, -67}
, {13, 1, 45, -32, 26, 17, 8, 35}
, {79, 175, 229, 149, 147, 245, 132, 208}
, {-61, -19, -79, 18, -82, 2, 26, -48}
, {0, 38, -39, -2, 133, -131, -46, -118}
}
, {{-1, -36, -68, -27, 27, 42, 86, 28}
, {-90, -95, 111, 42, -48, -244, -26, 44}
, {-9, -86, -139, -103, 90, 146, 31, 42}
, {-37, -8, 61, 66, -152, -124, -162, -86}
, {-32, -82, -28, 77, 75, 1, 20, 37}
, {-38, 15, 128, 114, 36, 38, 64, 5}
, {-106, -128, -72, -15, 97, 97, 101, -24}
, {-89, 40, 41, 102, -90, -79, -109, 110}
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


const int16_t conv1d_2_bias[CONV_FILTERS] = {44, -1, 58, -54, 180, 497, 3, -208, -238, -19, 2, -62, -354, -80, -122, -50, 81, -47, -526, 37, -52, -118, -73, -38, 257, 160, -154, -219, 148, 211, 123, 16}
;

const int16_t conv1d_2_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{0, 133, 21, 33}
, {-51, -6, -204, -169}
, {103, -56, 31, -93}
, {63, 10, -4, 85}
, {-48, -57, 0, 123}
, {-191, -127, -28, -131}
, {13, 36, 20, 8}
, {0, -164, 18, -43}
, {129, -4, -70, 117}
, {130, -31, -3, 1}
, {42, 118, 28, 70}
, {31, 164, -10, -11}
, {-53, -136, -30, 6}
, {47, -147, -79, 41}
, {-26, -42, 1, 79}
, {-77, -100, 1, -76}
}
, {{96, 51, 51, 101}
, {-94, 81, 50, 94}
, {-256, -30, -28, -260}
, {-29, -45, 0, -47}
, {-54, 65, -2, -78}
, {-11, 50, 97, 97}
, {-66, -179, -81, 24}
, {26, -39, -7, 16}
, {-57, -15, -90, -37}
, {127, 144, 75, 103}
, {-23, 66, -2, -65}
, {25, -48, 73, 41}
, {63, -25, -8, 14}
, {-37, -121, 47, 9}
, {71, 65, -10, -167}
, {31, 94, 2, -91}
}
, {{15, 27, 33, 103}
, {74, 6, 173, -121}
, {101, -76, 56, 72}
, {17, -207, -61, -5}
, {113, 94, 100, 29}
, {-165, -65, -136, -148}
, {-138, 50, -30, 16}
, {91, -11, -161, 13}
, {-138, -66, -87, -149}
, {31, 98, 23, -125}
, {-163, 44, 53, -12}
, {-48, -20, -178, -51}
, {-129, -79, -208, -26}
, {48, 94, -54, 37}
, {2, 10, -12, 68}
, {94, -17, 56, -100}
}
, {{-42, -147, 0, -193}
, {88, -43, 98, -125}
, {65, 77, 117, -40}
, {95, 3, -39, 16}
, {-57, 45, -99, -155}
, {30, 116, 167, -24}
, {81, -3, 113, -110}
, {-104, -231, -256, -324}
, {10, -69, -66, -118}
, {211, 117, 142, 144}
, {-78, 38, -155, -84}
, {-57, 67, -92, 1}
, {-18, 28, -29, 3}
, {-44, -148, -80, 76}
, {25, -43, -24, -28}
, {-69, 108, -137, 108}
}
, {{36, 46, 76, 12}
, {42, 140, 32, 71}
, {81, 43, -17, 40}
, {156, -114, 20, 27}
, {-83, -131, -67, 7}
, {-221, -10, -61, -36}
, {-73, -85, -21, -95}
, {-9, -74, 58, -35}
, {-31, 17, 16, 60}
, {-193, -68, -54, -167}
, {-118, 125, 96, 3}
, {41, -1, 68, -42}
, {11, -44, 91, 46}
, {1, -77, 169, 66}
, {-41, -206, -80, -98}
, {-44, 3, -123, -122}
}
, {{-9, -8, -54, 86}
, {57, 24, -2, 22}
, {-103, -46, -45, -313}
, {-37, -70, -66, -72}
, {46, -65, -38, 58}
, {-36, -170, -108, -16}
, {91, 24, -196, -46}
, {-18, 71, 56, 27}
, {54, 28, -68, -78}
, {102, 79, 107, 146}
, {56, -224, -116, 28}
, {79, 89, 117, 88}
, {32, -52, -7, -48}
, {52, 21, -25, -31}
, {97, 55, -115, -26}
, {-1, -6, -152, -117}
}
, {{-181, -141, -33, -75}
, {-17, -241, -90, 20}
, {7, 161, -6, 128}
, {-147, -102, -171, -106}
, {27, -91, 116, 42}
, {-53, -150, -9, -149}
, {-18, -116, -50, -66}
, {-149, -81, -43, -5}
, {-21, -46, 114, -11}
, {159, 144, 177, 200}
, {179, 95, 49, 117}
, {-151, 1, 149, 28}
, {65, 33, 39, -87}
, {-14, -74, -56, 21}
, {66, 1, 50, 65}
, {127, 20, -42, -95}
}
, {{215, 82, 103, 119}
, {-123, -148, -109, -93}
, {-101, -129, -35, 85}
, {-86, -142, -21, -13}
, {22, 60, 135, 90}
, {-14, -56, -16, 24}
, {-69, 1, -25, 55}
, {-46, 78, -33, 62}
, {-20, 16, 31, -33}
, {32, -27, -56, 17}
, {69, 48, 18, -5}
, {30, -15, 8, -158}
, {-44, -34, -32, -49}
, {-28, 38, 10, 110}
, {-37, 30, 48, -11}
, {60, 82, -10, 119}
}
, {{102, 8, -19, -47}
, {16, 9, 9, -60}
, {-152, -2, -113, -19}
, {46, 31, 104, 75}
, {17, -13, -19, -96}
, {12, -1, -54, 57}
, {-36, 76, 143, 13}
, {-45, -47, 34, 65}
, {1, 1, -79, 27}
, {2, -72, 2, -29}
, {53, 105, 57, 36}
, {-112, -69, 68, 118}
, {80, -135, -107, -59}
, {-96, -114, -10, -42}
, {-30, -22, -68, -117}
, {67, 93, 77, 29}
}
, {{88, 131, 153, 70}
, {58, -29, -51, -57}
, {-15, 60, 9, -20}
, {-22, -62, -32, -42}
, {-35, -63, -3, 48}
, {11, 45, -18, 12}
, {-80, -102, -109, -140}
, {95, 131, 15, 81}
, {-74, 38, -88, -115}
, {-4, -14, -2, 1}
, {5, -136, -99, -42}
, {27, 122, 181, 112}
, {-29, -4, 54, 98}
, {157, 13, -46, -52}
, {-66, -2, -4, 24}
, {-33, -96, -136, -62}
}
, {{44, -57, -63, 85}
, {-66, 96, -23, -104}
, {116, 70, -135, -136}
, {31, 53, 82, 46}
, {70, 84, -133, 20}
, {27, -1, 8, -32}
, {-65, -83, 104, 92}
, {139, 40, -45, 11}
, {-63, 29, -104, -113}
, {-120, -80, -151, -104}
, {-171, -126, 53, -70}
, {74, -1, -44, -26}
, {-16, -214, -4, 60}
, {49, -43, 2, 65}
, {-66, -52, -100, -131}
, {-122, 113, 42, -77}
}
, {{-39, -64, -37, 29}
, {-108, -80, -6, -188}
, {-42, -100, -45, -42}
, {86, 41, 164, 155}
, {-13, 42, 5, -80}
, {27, 162, 17, -190}
, {24, 43, 68, -67}
, {56, 59, 10, 3}
, {46, -15, 77, -52}
, {-101, 5, 25, 42}
, {-89, -7, 107, 60}
, {-108, -235, -212, -141}
, {-50, -98, -94, -157}
, {-25, -128, -4, -27}
, {27, 61, 35, 28}
, {-100, 52, 125, 48}
}
, {{-50, 98, 151, 62}
, {-10, 12, -61, 4}
, {73, -37, -28, 85}
, {-89, -162, 34, 78}
, {-29, -36, -14, 59}
, {-186, -78, 71, 121}
, {-129, -101, -28, 26}
, {95, 56, 87, 57}
, {-213, -155, 51, 45}
, {-6, -138, -97, -28}
, {-1, 108, 107, 83}
, {-144, -107, 29, 35}
, {-81, -24, 18, 28}
, {79, -115, -86, -8}
, {37, 42, 2, -13}
, {-48, -11, 43, 49}
}
, {{-26, 17, 55, 70}
, {-66, -24, -22, -9}
, {-33, -77, -152, -69}
, {91, -33, -169, -20}
, {-24, -14, -4, -75}
, {-7, 23, -39, -9}
, {33, 42, 40, -15}
, {15, 97, 142, 13}
, {42, -3, 69, 81}
, {3, 5, -111, 23}
, {-138, -151, -65, 93}
, {27, 44, -173, -80}
, {58, 58, 30, 23}
, {45, 44, 83, 3}
, {-51, -193, -141, -39}
, {-151, -129, 65, 72}
}
, {{124, 83, -86, -63}
, {-13, -44, -72, 60}
, {38, 23, 65, 224}
, {-119, -147, 6, 3}
, {-51, 17, 44, -86}
, {-3, 143, 94, -9}
, {-95, -9, -54, -22}
, {-4, -53, 69, 94}
, {27, 83, 144, -5}
, {-144, -31, 73, -107}
, {-1, 40, 15, 61}
, {101, 123, 84, -29}
, {-179, -82, 56, -36}
, {-139, -122, 66, 125}
, {-8, 24, -5, -30}
, {-59, 38, -11, 99}
}
, {{-66, -82, -152, -59}
, {74, 58, 3, 34}
, {-58, -19, -27, -28}
, {15, 90, 70, 63}
, {54, 73, 98, -8}
, {-11, -40, -10, -95}
, {5, 14, 35, 92}
, {-66, -69, -1, 54}
, {-48, 25, 112, 58}
, {-32, -113, -98, -141}
, {98, 23, 44, 92}
, {-1, 20, -4, 42}
, {-19, 42, 44, -14}
, {-2, 18, -84, 90}
, {-30, -53, 8, 20}
, {10, -80, -52, -31}
}
, {{-22, -172, -98, -98}
, {20, 0, -152, -35}
, {-164, 221, 17, 14}
, {-11, -95, -138, -100}
, {22, -49, -173, 30}
, {-121, -26, -179, -179}
, {48, -11, -81, -37}
, {98, 97, 63, -80}
, {39, -44, -37, 72}
, {32, 62, 40, 22}
, {-123, -165, -135, -29}
, {22, 76, 31, -71}
, {-14, -32, 58, -8}
, {-8, -12, 145, 45}
, {-45, 19, 175, 134}
, {-183, 33, 104, 98}
}
, {{-105, -313, -180, -141}
, {25, 37, 73, 129}
, {89, 75, -108, -306}
, {76, 54, -99, -207}
, {43, 8, 21, 57}
, {-108, -134, -33, 67}
, {-65, 3, 83, 105}
, {-16, 0, -69, -75}
, {-21, -55, -17, 65}
, {-106, -22, -137, -138}
, {-36, -172, -106, 59}
, {90, 141, 16, 75}
, {58, 19, -94, -11}
, {39, 83, 42, -49}
, {-48, 10, 26, 10}
, {6, -35, -156, 50}
}
, {{29, 39, 22, 42}
, {11, 10, 24, 81}
, {155, 100, 91, 83}
, {28, -78, 45, 29}
, {84, -63, -53, -16}
, {163, 0, 67, 120}
, {6, 20, -126, -37}
, {-134, -3, 26, -199}
, {45, -30, -28, -17}
, {-74, 20, -95, 195}
, {22, -44, -59, -63}
, {48, 26, 12, -31}
, {50, 31, 21, -51}
, {-41, 1, 57, -68}
, {12, 53, -18, 56}
, {113, -46, 72, 105}
}
, {{22, 117, 75, -8}
, {21, 83, 140, 121}
, {-84, -30, -228, -106}
, {-2, 57, -44, -48}
, {-14, -98, 0, 23}
, {8, -35, 87, 67}
, {-60, -78, 36, -17}
, {-9, -44, -40, 88}
, {40, -25, -152, -159}
, {44, 81, -87, 54}
, {-33, -5, 1, 0}
, {122, 112, 80, -142}
, {65, -124, -33, 63}
, {-36, -69, -53, 37}
, {-26, -103, -138, 32}
, {-71, -112, -71, 138}
}
, {{-77, -12, -45, -19}
, {99, 0, 43, -33}
, {-13, -144, -88, 29}
, {-118, -126, -62, -19}
, {46, 6, -26, 70}
, {112, 91, 111, 170}
, {-28, 77, 95, -60}
, {93, 107, 119, -35}
, {-4, 17, 4, -10}
, {-11, -126, -70, -56}
, {-79, 18, -26, -16}
, {28, 28, 25, 97}
, {19, 61, -31, -53}
, {-92, -46, 38, -38}
, {51, 27, -35, 32}
, {120, 43, -52, -2}
}
, {{16, -1, -3, -29}
, {22, 24, -49, -54}
, {34, 142, 123, -3}
, {49, -5, -108, -40}
, {-62, 112, 119, 89}
, {-26, 75, -31, -35}
, {-32, 25, 83, -37}
, {-97, -84, -69, 0}
, {-71, 10, 89, -3}
, {63, 46, -21, -146}
, {-168, -171, -94, -148}
, {-181, -139, -32, -101}
, {-1, -89, -4, 100}
, {19, -64, -108, 89}
, {46, 75, -40, -19}
, {-46, -76, -67, 74}
}
, {{-59, -19, -38, 176}
, {61, 9, 52, 71}
, {106, 121, 157, -69}
, {-145, 88, 159, -14}
, {-60, 79, 5, -6}
, {-112, -154, -88, -10}
, {-74, -75, 16, 30}
, {97, -18, 90, 5}
, {-55, 26, 47, -92}
, {0, 171, 81, 80}
, {-57, -30, -86, -32}
, {7, -13, -34, -125}
, {-46, 14, -154, -185}
, {-19, -91, 17, -134}
, {15, 30, -119, -67}
, {65, -48, 67, -12}
}
, {{139, -132, -198, -155}
, {-77, -12, -54, -35}
, {122, -28, 68, -273}
, {-74, 1, 26, 12}
, {52, -81, -50, -103}
, {71, 76, -112, -78}
, {-132, -33, -26, 13}
, {22, 53, 21, -35}
, {-13, 81, 89, 72}
, {-12, 241, 93, 126}
, {-78, 12, 82, 68}
, {4, 18, -66, -59}
, {-24, 24, -36, 29}
, {-67, 71, 47, -35}
, {-49, -102, -142, -11}
, {-78, 10, 178, 111}
}
, {{35, 149, 80, -37}
, {-67, -23, 0, -27}
, {-175, -272, -174, 16}
, {30, 14, -104, -85}
, {14, 81, 30, -37}
, {38, -52, -45, -62}
, {127, 83, -86, -128}
, {-122, -29, 35, 23}
, {74, 26, -58, -84}
, {83, 66, 29, 67}
, {15, 48, 23, -7}
, {-28, -100, 16, -43}
, {63, 46, -17, -74}
, {87, 18, 10, 28}
, {5, -36, -108, -106}
, {-91, -151, -99, 34}
}
, {{-150, 19, 73, -15}
, {-100, 19, 128, -7}
, {-56, -58, 30, -30}
, {-77, 85, 114, -41}
, {-168, -201, -104, -133}
, {-166, 95, 142, -47}
, {-35, -25, 48, 100}
, {-144, 106, 93, 72}
, {3, -175, -118, -49}
, {45, -11, -72, 111}
, {-55, -206, -212, -26}
, {-169, -15, 93, 19}
, {-34, -200, -179, -41}
, {39, 6, 25, 82}
, {0, 122, 124, 69}
, {44, 17, 57, -30}
}
, {{68, -9, 59, 21}
, {17, 24, -37, 13}
, {-149, -94, -103, -58}
, {-30, 100, 78, 45}
, {-135, 104, -24, -42}
, {9, -91, -17, 111}
, {9, 25, -144, -106}
, {22, 116, 211, 3}
, {29, 62, -9, 20}
, {-52, -52, 30, 2}
, {-64, -66, -32, 2}
, {97, -16, -58, 59}
, {-5, -50, 110, -35}
, {-12, -115, -56, -118}
, {110, 21, 5, 50}
, {-93, 18, 84, 77}
}
, {{2, 13, 64, -22}
, {56, 45, -1, 12}
, {73, -31, -14, 51}
, {-28, -64, 38, 94}
, {-7, 50, 0, -98}
, {2, -20, -95, -51}
, {-11, 33, -22, 2}
, {-18, 53, 83, 18}
, {27, 1, -84, 46}
, {-46, 50, 71, 11}
, {-119, -41, 3, -47}
, {-171, -160, -190, -48}
, {35, 18, 77, 7}
, {81, 2, 142, 134}
, {33, -44, 37, 81}
, {-6, -32, 76, -72}
}
, {{-118, -40, -24, -85}
, {42, -170, -203, 183}
, {200, -170, -177, 74}
, {-96, -103, -137, 11}
, {-57, -26, -11, 74}
, {111, 69, -127, 10}
, {-91, 46, -36, -26}
, {-211, 178, 213, -93}
, {84, -39, -61, -10}
, {-30, 43, 47, -114}
, {59, 96, 7, -135}
, {-74, -41, -35, -46}
, {-17, 60, 39, -7}
, {-175, -5, -55, -109}
, {-20, -98, -106, -53}
, {19, -63, -126, 129}
}
, {{134, 43, 84, 182}
, {-45, 50, -57, -36}
, {-9, -70, 67, -18}
, {11, -32, 146, -17}
, {-169, -202, -43, -1}
, {-35, 32, 155, -19}
, {56, 22, 42, -55}
, {-52, -39, 192, 48}
, {9, 85, -115, -72}
, {10, -68, -105, -115}
, {-145, -86, -79, 16}
, {-115, 62, -30, 146}
, {-103, 37, 18, -25}
, {61, -71, -68, -123}
, {-208, -46, -236, -63}
, {-21, -81, -10, -82}
}
, {{46, -45, -71, 6}
, {-38, -28, 66, 24}
, {-26, 30, 110, 80}
, {-53, 66, -35, 110}
, {-34, 7, -99, -1}
, {-86, 28, -43, -166}
, {-53, 42, -20, -10}
, {-73, -33, 66, 67}
, {-25, 2, 29, 105}
, {351, 129, 30, 68}
, {62, 68, 70, 49}
, {-21, -16, -20, 116}
, {-55, -50, -31, 46}
, {-76, 29, -87, 43}
, {-257, -164, -207, -142}
, {-30, 10, 68, -66}
}
, {{-169, 22, -53, -51}
, {-69, -20, 11, -40}
, {0, 187, 94, -9}
, {6, -40, 71, -60}
, {-74, -226, -96, -46}
, {-58, -265, -65, -228}
, {-24, -3, 33, -7}
, {-59, -23, 52, 29}
, {-37, 60, 99, 3}
, {190, 76, -78, -97}
, {-104, 20, 32, 5}
, {-61, -46, 131, 67}
, {34, -33, 15, -40}
, {35, -15, -37, -56}
, {-24, 59, 2, -1}
, {10, 156, 134, 40}
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


const int16_t conv1d_3_bias[CONV_FILTERS] = {90, -13, 55, 192, 68, 48, 146, 173, -2, 252, -185, 145, 188, 64, 17, 123, 131, 42, 127, 163, 32, 230, 127, 224, -159, 205, -51, -209, -86, 111, 323, 202, 263, 87, 96, 250, -85, 80, 113, 97, 57, 229, 105, 126, 148, 57, 178, 43, 5, 77, -1, 116, 157, 275, -195, -69, -225, -27, -128, -221, 113, 145, -223, 23}
;

const int16_t conv1d_3_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{111, 120}
, {-27, -47}
, {48, -71}
, {-43, 45}
, {106, -53}
, {93, -132}
, {-11, 28}
, {-22, 83}
, {38, 2}
, {-16, -93}
, {-91, -147}
, {5, 2}
, {-115, -45}
, {42, 83}
, {-164, 63}
, {-112, 84}
, {-141, -66}
, {64, 65}
, {-118, -59}
, {93, -56}
, {-62, -13}
, {-7, 64}
, {-12, 3}
, {-73, 93}
, {-175, -10}
, {-58, 110}
, {109, -134}
, {81, 73}
, {103, 4}
, {95, -29}
, {10, -60}
, {57, 99}
}
, {{46, 112}
, {-45, -128}
, {-64, -98}
, {27, -40}
, {-105, 61}
, {-102, -95}
, {79, -73}
, {-46, -94}
, {-80, 42}
, {-10, 41}
, {56, -62}
, {-48, -184}
, {-52, -49}
, {12, -98}
, {-74, 74}
, {46, 77}
, {137, 144}
, {24, -11}
, {1, 68}
, {-143, 30}
, {-61, -12}
, {-29, 51}
, {107, -21}
, {-11, 71}
, {50, -73}
, {38, -35}
, {-51, -73}
, {56, -30}
, {-14, -122}
, {-125, -164}
, {-98, -4}
, {-21, 26}
}
, {{15, 132}
, {-17, -160}
, {30, 40}
, {-83, -101}
, {-43, 52}
, {88, -23}
, {-77, -47}
, {-112, -81}
, {-30, 5}
, {-61, 41}
, {-223, -151}
, {19, -40}
, {-46, 164}
, {-93, -48}
, {-26, 84}
, {-13, 34}
, {43, 5}
, {-9, 12}
, {88, 34}
, {-3, 23}
, {-45, -19}
, {-62, 50}
, {-180, -16}
, {33, 84}
, {204, -11}
, {62, 12}
, {61, -74}
, {-34, 4}
, {43, 70}
, {17, -47}
, {10, 16}
, {25, -82}
}
, {{-34, 74}
, {61, 105}
, {-215, 17}
, {-1, -194}
, {-229, -382}
, {-29, 75}
, {10, 41}
, {121, 10}
, {-160, 17}
, {12, 4}
, {-249, -204}
, {-95, -45}
, {-1, -183}
, {52, -8}
, {-48, -61}
, {30, 6}
, {94, -19}
, {-22, -66}
, {-136, -28}
, {85, 21}
, {-87, 44}
, {-80, -17}
, {9, -44}
, {83, 77}
, {-12, -110}
, {15, -74}
, {13, 8}
, {-121, 92}
, {8, -88}
, {3, 106}
, {15, -17}
, {1, 100}
}
, {{-8, -16}
, {-142, -47}
, {-237, 170}
, {-19, 43}
, {7, 94}
, {194, 29}
, {-55, 21}
, {-11, -70}
, {-22, 59}
, {-170, -53}
, {-98, -99}
, {-171, 0}
, {-58, -27}
, {29, -173}
, {-51, 61}
, {19, 23}
, {-245, -136}
, {51, -66}
, {-2, -66}
, {26, 100}
, {-38, 75}
, {-134, -142}
, {-36, 54}
, {40, -34}
, {128, -13}
, {13, 161}
, {46, -22}
, {-65, -8}
, {120, 128}
, {34, 102}
, {109, 87}
, {-93, -74}
}
, {{30, -35}
, {-30, -145}
, {22, 8}
, {76, 88}
, {-135, 104}
, {-31, 7}
, {156, -46}
, {-55, -33}
, {45, 89}
, {-69, 0}
, {-92, -151}
, {-25, -148}
, {-147, -104}
, {-45, 87}
, {-359, -176}
, {15, -48}
, {3, -50}
, {-119, -54}
, {70, -21}
, {-258, 106}
, {-31, -16}
, {-19, -198}
, {115, 158}
, {-70, -113}
, {121, 37}
, {-23, -23}
, {109, 62}
, {69, -4}
, {-270, -176}
, {57, -5}
, {13, 11}
, {-37, 1}
}
, {{76, 90}
, {-27, -92}
, {49, 43}
, {-71, 14}
, {-122, -121}
, {92, 122}
, {137, -43}
, {10, -35}
, {160, 52}
, {-96, -140}
, {-10, -43}
, {-1, 142}
, {-129, -17}
, {-12, -66}
, {12, 43}
, {-23, 33}
, {-94, -30}
, {-218, -240}
, {-1, -156}
, {29, 145}
, {-11, -80}
, {-223, -103}
, {-37, -94}
, {163, 1}
, {-4, 67}
, {5, 54}
, {-48, -66}
, {-128, -77}
, {-64, -58}
, {-68, 183}
, {-105, 14}
, {-1, 22}
}
, {{18, -65}
, {-66, 13}
, {6, 71}
, {80, -127}
, {-86, -6}
, {-50, 58}
, {150, 131}
, {46, -78}
, {-120, 75}
, {-231, -48}
, {198, -233}
, {-166, -82}
, {-4, 93}
, {95, -58}
, {-66, 18}
, {35, -21}
, {153, -96}
, {84, -94}
, {3, -39}
, {-24, -20}
, {10, 24}
, {-34, -229}
, {-134, 18}
, {74, 60}
, {-62, 11}
, {-6, -130}
, {75, -59}
, {-11, 7}
, {0, -206}
, {-58, -147}
, {-121, 25}
, {48, -39}
}
, {{81, 0}
, {-65, -76}
, {-112, 9}
, {-138, -212}
, {92, -52}
, {-7, 96}
, {34, 283}
, {-8, 85}
, {48, -27}
, {111, -126}
, {21, 19}
, {112, 41}
, {-166, -139}
, {55, 37}
, {-31, -282}
, {35, 30}
, {-305, 30}
, {12, -50}
, {-80, -102}
, {2, -20}
, {5, 63}
, {-2, 2}
, {-34, -234}
, {12, -77}
, {-57, 86}
, {-154, -279}
, {55, 72}
, {-73, -119}
, {49, 45}
, {-2, -37}
, {-42, -158}
, {-15, -74}
}
, {{53, -104}
, {60, 24}
, {11, -19}
, {-65, -67}
, {-107, -42}
, {208, 159}
, {46, -147}
, {30, 3}
, {69, 31}
, {26, -92}
, {-238, -73}
, {60, 133}
, {106, -23}
, {-1, -73}
, {30, -23}
, {-34, -347}
, {-116, -229}
, {-83, -529}
, {72, -42}
, {-86, -131}
, {-6, 7}
, {-145, 39}
, {63, -34}
, {-43, -3}
, {138, 29}
, {38, 14}
, {17, -34}
, {-151, -3}
, {-113, -21}
, {17, 11}
, {69, -84}
, {23, -7}
}
, {{-83, -238}
, {119, -37}
, {-31, -74}
, {-13, -34}
, {-10, 148}
, {-55, 6}
, {-136, -65}
, {28, -179}
, {-164, -167}
, {23, 61}
, {61, -29}
, {163, 55}
, {166, -111}
, {-107, -72}
, {12, 4}
, {-17, -6}
, {-26, -253}
, {-47, 37}
, {-48, -156}
, {84, 57}
, {-79, -26}
, {6, 34}
, {96, 22}
, {-270, -55}
, {28, -5}
, {56, 191}
, {40, -92}
, {-49, -27}
, {-59, -44}
, {95, -179}
, {-54, 124}
, {-73, 16}
}
, {{11, -14}
, {59, -38}
, {-188, 95}
, {108, -70}
, {41, 67}
, {68, 116}
, {-143, 209}
, {-102, 13}
, {-115, 114}
, {-10, -36}
, {120, -28}
, {108, 42}
, {-305, -62}
, {-19, 11}
, {-167, 3}
, {-7, -39}
, {-74, -130}
, {28, 6}
, {-7, 17}
, {53, 118}
, {-38, -94}
, {15, -169}
, {-12, 55}
, {-3, -81}
, {125, 131}
, {-39, 43}
, {-42, -9}
, {39, -30}
, {104, 77}
, {-9, -57}
, {-46, 44}
, {-23, -80}
}
, {{64, 20}
, {80, 7}
, {178, 74}
, {-81, 41}
, {-70, 100}
, {151, 14}
, {85, 25}
, {160, -197}
, {89, 113}
, {-67, 58}
, {-326, 78}
, {49, -183}
, {-141, -225}
, {23, 3}
, {-100, -147}
, {-138, 28}
, {98, -4}
, {-152, 17}
, {-202, -250}
, {-58, 90}
, {-67, 9}
, {-13, -11}
, {70, -147}
, {-85, -200}
, {0, -229}
, {5, 99}
, {-138, -42}
, {-179, -57}
, {-66, 226}
, {-22, -135}
, {-46, 84}
, {73, -13}
}
, {{33, 117}
, {-13, -139}
, {-13, -6}
, {-6, -92}
, {-163, -134}
, {-17, -70}
, {49, -16}
, {6, 0}
, {142, -115}
, {67, 187}
, {-21, -36}
, {48, 46}
, {-147, -40}
, {133, -236}
, {17, -37}
, {-81, -113}
, {6, 202}
, {-20, -154}
, {-47, 61}
, {-12, -103}
, {-84, -87}
, {-209, 30}
, {-86, -268}
, {24, -26}
, {65, 97}
, {-257, -12}
, {-48, 0}
, {-204, -77}
, {122, 51}
, {81, 131}
, {-93, -78}
, {-58, 53}
}
, {{69, 79}
, {-45, 47}
, {-117, 28}
, {6, -7}
, {-197, -179}
, {-62, 129}
, {-175, -59}
, {-21, 8}
, {64, 59}
, {-54, -16}
, {51, -20}
, {6, 80}
, {3, -38}
, {1, -63}
, {-73, 47}
, {26, 24}
, {19, -69}
, {-40, -51}
, {21, 52}
, {10, 29}
, {6, 30}
, {10, -11}
, {42, 20}
, {-39, 53}
, {-43, -32}
, {160, -35}
, {-43, -89}
, {-17, 15}
, {86, -31}
, {-13, -18}
, {-106, -86}
, {72, -45}
}
, {{-44, 8}
, {-81, -98}
, {-253, -161}
, {65, -45}
, {113, -12}
, {29, 76}
, {-220, 184}
, {-14, -13}
, {18, -82}
, {67, -41}
, {74, -110}
, {-26, 11}
, {-4, -48}
, {39, 0}
, {79, 28}
, {75, -182}
, {-228, -96}
, {-1, -44}
, {14, 8}
, {56, -12}
, {64, -4}
, {-62, -228}
, {138, 8}
, {-4, 105}
, {-1, -109}
, {-155, 46}
, {-128, -19}
, {-4, -142}
, {141, 97}
, {41, 110}
, {15, -73}
, {-50, -188}
}
, {{-90, -46}
, {-192, 25}
, {201, 129}
, {22, 135}
, {73, 23}
, {-40, -128}
, {182, -13}
, {-130, 3}
, {-199, 6}
, {-175, -4}
, {16, 3}
, {-91, 97}
, {-24, 71}
, {-1, -9}
, {93, 12}
, {-112, 32}
, {188, 12}
, {-31, -141}
, {-401, -43}
, {53, 27}
, {-67, -3}
, {-52, -81}
, {-129, 118}
, {31, 29}
, {-193, -86}
, {-100, 140}
, {-56, 50}
, {-131, -102}
, {337, 52}
, {-249, -10}
, {-24, 62}
, {-94, 158}
}
, {{-1, 29}
, {50, 9}
, {-84, -107}
, {-58, 7}
, {-223, 42}
, {-264, -226}
, {-164, -1}
, {32, -127}
, {1, -18}
, {-21, -116}
, {-180, -14}
, {74, -17}
, {145, -65}
, {-83, -72}
, {71, -68}
, {-108, -130}
, {-136, 80}
, {-104, 30}
, {-29, -10}
, {61, -38}
, {-26, -82}
, {80, 34}
, {-205, 110}
, {202, -4}
, {-42, 46}
, {-215, 156}
, {17, 15}
, {50, 95}
, {67, -188}
, {104, -27}
, {28, -110}
, {-10, 195}
}
, {{45, 73}
, {27, -19}
, {172, -181}
, {-27, 18}
, {-43, 54}
, {0, -118}
, {138, 0}
, {-12, -14}
, {-62, 82}
, {-49, -109}
, {-69, -20}
, {-3, -45}
, {-74, -115}
, {-43, -52}
, {-55, -29}
, {-11, -38}
, {32, 88}
, {-142, 137}
, {-45, -106}
, {-139, 170}
, {-46, -187}
, {4, 107}
, {-125, 221}
, {39, 49}
, {-72, -76}
, {-38, -156}
, {-61, 124}
, {-65, 55}
, {75, -95}
, {50, 55}
, {76, 90}
, {14, 4}
}
, {{-70, 34}
, {0, 67}
, {-37, -348}
, {-14, -27}
, {73, -32}
, {-14, 20}
, {19, 34}
, {79, -66}
, {25, -18}
, {17, 18}
, {-32, 7}
, {0, 82}
, {-69, 25}
, {53, 103}
, {46, 48}
, {16, -60}
, {-23, -94}
, {-108, -36}
, {69, -133}
, {31, 22}
, {-39, -22}
, {89, -95}
, {-85, 4}
, {-30, -182}
, {-4, -121}
, {24, 133}
, {-10, -49}
, {-1, -28}
, {-116, 14}
, {101, 50}
, {125, 17}
, {57, -68}
}
, {{23, 65}
, {-14, -127}
, {127, -25}
, {-94, -54}
, {-330, -46}
, {-46, -87}
, {-17, 58}
, {-32, 78}
, {-58, 41}
, {-138, 0}
, {-23, -32}
, {118, -48}
, {-67, 5}
, {-172, -196}
, {-99, -218}
, {-12, 60}
, {-33, 5}
, {-213, -141}
, {0, -62}
, {47, 102}
, {18, 60}
, {-55, 31}
, {-21, 38}
, {34, -26}
, {51, 130}
, {-82, 75}
, {-107, 122}
, {-28, 18}
, {6, 93}
, {-158, 32}
, {-233, -13}
, {16, -36}
}
, {{-145, 37}
, {-170, 29}
, {-92, -38}
, {10, 84}
, {-142, 156}
, {36, 144}
, {-93, -52}
, {-100, -117}
, {-89, 5}
, {-121, -128}
, {-58, -69}
, {-6, -45}
, {62, -62}
, {-50, -77}
, {-181, 49}
, {-16, 71}
, {108, 220}
, {66, -22}
, {-23, -121}
, {19, 23}
, {74, -45}
, {-63, 67}
, {57, 72}
, {-107, -56}
, {70, 16}
, {-64, -11}
, {13, -40}
, {-8, 54}
, {29, 53}
, {-104, -169}
, {-3, -72}
, {-142, -188}
}
, {{-123, -91}
, {146, -66}
, {-91, -23}
, {45, 106}
, {23, 114}
, {-48, -26}
, {-20, -90}
, {-133, -66}
, {59, -56}
, {-9, 24}
, {117, -34}
, {0, -7}
, {-45, -184}
, {-112, 13}
, {-77, -197}
, {4, -48}
, {9, -115}
, {57, 33}
, {-34, 16}
, {-117, 18}
, {58, 46}
, {61, -5}
, {-152, -23}
, {88, 75}
, {130, 13}
, {93, 25}
, {79, 29}
, {-29, -13}
, {-107, 103}
, {-85, -100}
, {-34, 113}
, {-60, 24}
}
, {{-28, -102}
, {110, 46}
, {163, -50}
, {-14, 164}
, {36, -210}
, {-23, 8}
, {-19, -7}
, {-36, -240}
, {18, -133}
, {51, 32}
, {-11, 25}
, {69, -47}
, {46, -9}
, {223, -205}
, {101, 16}
, {-132, -127}
, {14, -19}
, {-14, -329}
, {-22, 47}
, {96, 41}
, {-26, -74}
, {33, -122}
, {2, -49}
, {-112, 134}
, {-68, -117}
, {63, 74}
, {-6, 40}
, {-28, 19}
, {41, 58}
, {53, -51}
, {118, 56}
, {-3, -93}
}
, {{-229, 40}
, {-65, 30}
, {95, 155}
, {-63, -58}
, {-33, 24}
, {51, -25}
, {40, -118}
, {-45, -122}
, {61, 119}
, {-101, 98}
, {79, 48}
, {-51, 19}
, {85, 55}
, {-14, 82}
, {-46, -26}
, {-38, 47}
, {-11, -125}
, {39, -4}
, {-87, -52}
, {-70, 29}
, {-3, -20}
, {58, -119}
, {-115, -55}
, {0, 9}
, {-112, 80}
, {-196, -37}
, {-48, -164}
, {-105, -19}
, {54, 86}
, {131, 196}
, {46, 123}
, {6, -85}
}
, {{-172, -39}
, {97, 24}
, {-37, -199}
, {26, 37}
, {-48, -12}
, {-36, -220}
, {-189, 32}
, {70, -95}
, {39, 57}
, {-112, -203}
, {-129, -257}
, {16, -40}
, {-25, 112}
, {-54, 2}
, {-176, 15}
, {-6, -66}
, {97, -226}
, {-103, 49}
, {-104, 35}
, {16, -190}
, {-48, 39}
, {2, -59}
, {-98, -4}
, {95, -10}
, {102, 113}
, {14, -1}
, {-23, -112}
, {103, 11}
, {-66, -273}
, {3, 64}
, {-188, -177}
, {1, -138}
}
, {{-151, -13}
, {-130, 79}
, {-66, 49}
, {-179, 66}
, {152, -48}
, {-4, -136}
, {-24, 5}
, {-64, 60}
, {-160, -6}
, {-218, 25}
, {30, -25}
, {39, -47}
, {104, 49}
, {193, -50}
, {79, -31}
, {-175, -95}
, {-15, -122}
, {-218, 39}
, {-191, 89}
, {99, -91}
, {-236, -51}
, {-112, 80}
, {67, -20}
, {201, 52}
, {57, 45}
, {81, -32}
, {-99, 8}
, {-68, 65}
, {-32, -67}
, {138, 119}
, {-28, -40}
, {183, 1}
}
, {{-1, -140}
, {44, -30}
, {129, 20}
, {59, 18}
, {-204, -106}
, {-98, -312}
, {117, 102}
, {57, -70}
, {6, 55}
, {-53, -308}
, {56, 126}
, {-30, 49}
, {-54, -201}
, {126, -123}
, {64, -267}
, {-128, -156}
, {148, 84}
, {-42, 116}
, {-85, -91}
, {-38, -207}
, {-27, -45}
, {40, 77}
, {150, 77}
, {217, 102}
, {-99, 76}
, {-145, -187}
, {-40, -2}
, {-9, 23}
, {133, 53}
, {-270, -4}
, {35, 115}
, {-6, -25}
}
, {{-29, -27}
, {36, 71}
, {-97, 62}
, {-234, -168}
, {-8, -24}
, {83, -13}
, {28, -5}
, {-80, 23}
, {87, 100}
, {-94, 97}
, {10, 32}
, {66, 13}
, {-16, 22}
, {73, -187}
, {-74, -5}
, {-59, -113}
, {89, -7}
, {-15, 178}
, {-40, -9}
, {-132, 12}
, {97, -65}
, {141, 6}
, {-136, 111}
, {-164, -65}
, {0, -73}
, {-111, 13}
, {96, -143}
, {40, -153}
, {20, -37}
, {-80, -39}
, {-49, -81}
, {112, -55}
}
, {{-100, -28}
, {-159, -27}
, {48, 130}
, {35, -25}
, {10, 2}
, {-71, 3}
, {-76, -18}
, {-39, -37}
, {-53, 103}
, {-1, -9}
, {-26, 85}
, {-89, -176}
, {115, 38}
, {-1, 71}
, {-137, 4}
, {-23, -2}
, {63, 83}
, {-26, 81}
, {-79, -15}
, {-80, -116}
, {-1, -20}
, {-80, 76}
, {141, -187}
, {-32, -27}
, {81, -30}
, {43, 119}
, {74, -78}
, {-150, 60}
, {91, -43}
, {-29, 89}
, {69, 44}
, {94, 58}
}
, {{22, 33}
, {-21, -48}
, {-244, -139}
, {46, -80}
, {68, 76}
, {-72, 196}
, {20, 89}
, {-16, 3}
, {80, 5}
, {-77, -28}
, {-66, -134}
, {39, 33}
, {74, -79}
, {-26, 42}
, {-128, 100}
, {-13, -23}
, {-406, 45}
, {-118, -84}
, {-126, 72}
, {4, 94}
, {-257, -82}
, {98, 71}
, {137, -2}
, {-107, -40}
, {-74, -18}
, {50, -113}
, {-18, -40}
, {-131, 87}
, {9, 44}
, {54, -12}
, {97, -73}
, {34, 18}
}
, {{30, 77}
, {65, 3}
, {-109, 17}
, {-16, 37}
, {-78, -97}
, {-2, 70}
, {-93, -258}
, {38, 42}
, {39, 4}
, {69, 79}
, {-214, -143}
, {-122, 0}
, {-3, 25}
, {21, 36}
, {7, 66}
, {-66, -3}
, {-39, -93}
, {-66, -131}
, {-74, 46}
, {6, 62}
, {-5, -28}
, {-55, 12}
, {-25, 154}
, {-67, -99}
, {-17, 28}
, {-115, -58}
, {-36, -21}
, {38, 21}
, {-183, -137}
, {-143, 88}
, {-188, -61}
, {22, 29}
}
, {{93, -99}
, {114, 71}
, {-46, -17}
, {18, 12}
, {38, 85}
, {-8, 42}
, {-130, -113}
, {-122, -91}
, {74, -40}
, {-103, 41}
, {-117, 69}
, {-61, -86}
, {-19, 50}
, {-42, -36}
, {-155, -63}
, {-84, -168}
, {2, 19}
, {100, 141}
, {105, -71}
, {-89, 12}
, {-29, -7}
, {40, -171}
, {31, -187}
, {-28, -276}
, {95, 22}
, {113, 59}
, {-46, -95}
, {25, 0}
, {59, 162}
, {-76, -32}
, {6, 17}
, {3, -52}
}
, {{-182, -30}
, {-65, 103}
, {-201, -168}
, {0, 101}
, {-104, 226}
, {71, 80}
, {-250, 10}
, {33, 4}
, {-13, -15}
, {-96, -71}
, {-99, 89}
, {56, 31}
, {93, 6}
, {-70, -41}
, {-236, -97}
, {26, 0}
, {-40, 99}
, {49, 130}
, {-63, 79}
, {-153, -43}
, {-40, 40}
, {1, -81}
, {-112, -175}
, {19, 114}
, {7, -74}
, {-162, 85}
, {87, -52}
, {-29, -171}
, {48, 30}
, {-119, -39}
, {-55, 9}
, {-81, -22}
}
, {{-15, -76}
, {-15, 76}
, {74, -27}
, {-5, 134}
, {-92, 37}
, {77, 39}
, {-46, 52}
, {41, -21}
, {-23, 15}
, {-80, -3}
, {62, -82}
, {29, -108}
, {79, 32}
, {-52, -60}
, {33, -51}
, {-173, 57}
, {-52, 106}
, {-35, 82}
, {-53, -27}
, {149, 53}
, {-34, 4}
, {-141, 49}
, {3, -46}
, {88, 8}
, {-392, -77}
, {17, 108}
, {-148, 29}
, {-7, 93}
, {59, 68}
, {-27, -242}
, {-85, 54}
, {-20, -8}
}
, {{-58, -39}
, {31, -37}
, {134, 73}
, {6, 131}
, {-76, 170}
, {-31, 72}
, {-125, -76}
, {-32, 76}
, {-145, 31}
, {-75, -25}
, {80, -4}
, {-14, -54}
, {-21, -70}
, {-38, -98}
, {-141, -218}
, {-3, -45}
, {-56, 47}
, {-95, 94}
, {-20, 42}
, {-104, 72}
, {51, 21}
, {-18, -27}
, {51, -193}
, {39, -84}
, {47, 18}
, {-79, -63}
, {-127, 99}
, {-140, 43}
, {135, -212}
, {-15, -95}
, {16, 52}
, {12, -26}
}
, {{37, 38}
, {-26, 195}
, {36, 51}
, {-49, 20}
, {-12, -51}
, {-181, -180}
, {220, 0}
, {-134, -112}
, {16, 114}
, {-151, -192}
, {-35, -153}
, {-87, -123}
, {-44, 0}
, {-28, 16}
, {-14, -59}
, {60, 111}
, {71, 116}
, {48, -62}
, {-79, 62}
, {72, -59}
, {-150, -125}
, {28, -12}
, {10, -163}
, {35, 129}
, {-89, 48}
, {108, 20}
, {19, 151}
, {-6, -42}
, {-82, -231}
, {-112, -86}
, {-16, 46}
, {-161, -126}
}
, {{131, 75}
, {-100, -113}
, {-73, 57}
, {-162, 100}
, {5, -120}
, {39, -4}
, {-41, 71}
, {5, 1}
, {-6, 118}
, {-69, -11}
, {-68, -46}
, {-92, -153}
, {-145, 39}
, {-42, 36}
, {2, 95}
, {-46, -28}
, {9, 84}
, {-138, -54}
, {-245, -23}
, {-5, 9}
, {-145, -109}
, {-9, 105}
, {-103, 37}
, {-122, 54}
, {97, 73}
, {-94, -267}
, {59, 81}
, {-45, -20}
, {56, 65}
, {140, 170}
, {-11, -6}
, {0, -67}
}
, {{-31, -92}
, {-35, 35}
, {-34, -45}
, {182, 102}
, {45, -80}
, {105, 21}
, {-99, -176}
, {31, 79}
, {-54, -75}
, {21, 28}
, {31, -6}
, {71, 94}
, {1, 26}
, {-7, -67}
, {-82, -200}
, {-163, -380}
, {32, 25}
, {169, -232}
, {-6, -60}
, {-243, -351}
, {36, -54}
, {34, 101}
, {7, 16}
, {-92, -79}
, {79, -69}
, {-162, -94}
, {-74, -135}
, {-130, -111}
, {54, -27}
, {-73, 25}
, {29, 70}
, {49, 97}
}
, {{-108, -287}
, {-33, -43}
, {38, 102}
, {28, 76}
, {-43, -2}
, {58, -17}
, {-26, -34}
, {-1, -1}
, {-36, -6}
, {26, 67}
, {-120, 126}
, {39, -110}
, {-71, -62}
, {-17, -130}
, {76, -46}
, {50, -149}
, {8, -24}
, {-12, 109}
, {-10, -63}
, {-120, -102}
, {-10, -1}
, {90, 18}
, {-11, -92}
, {95, -169}
, {58, 32}
, {85, -72}
, {52, 4}
, {21, 90}
, {101, 110}
, {-106, -220}
, {-46, -72}
, {159, 118}
}
, {{-39, -72}
, {109, -234}
, {13, 0}
, {44, 46}
, {-84, 38}
, {127, -229}
, {-16, -71}
, {-79, -133}
, {78, -16}
, {-104, 127}
, {30, -172}
, {54, 56}
, {105, -7}
, {-77, -299}
, {-58, 40}
, {9, -68}
, {87, 37}
, {133, -33}
, {-14, -19}
, {-187, -46}
, {-145, -109}
, {124, 25}
, {35, 68}
, {50, -168}
, {23, -195}
, {1, 30}
, {89, -70}
, {75, -12}
, {-193, 33}
, {121, 2}
, {-157, -55}
, {42, 88}
}
, {{-106, 133}
, {-43, -37}
, {142, -126}
, {-100, -123}
, {51, 0}
, {35, -11}
, {147, 49}
, {-47, -3}
, {27, -128}
, {70, 77}
, {-29, 65}
, {85, 62}
, {-53, -136}
, {20, -6}
, {-78, 37}
, {-8, 4}
, {-68, -10}
, {-45, 46}
, {-170, -60}
, {51, 103}
, {80, 6}
, {71, -21}
, {2, 68}
, {-149, -79}
, {-12, 135}
, {-173, 17}
, {71, 1}
, {49, -49}
, {28, 78}
, {119, 36}
, {-69, -156}
, {-60, 0}
}
, {{-83, -21}
, {-98, 23}
, {77, -203}
, {40, 87}
, {22, 25}
, {-4, 51}
, {118, 98}
, {-205, -304}
, {-13, -43}
, {148, 3}
, {-73, 2}
, {-458, -295}
, {49, 132}
, {-75, 24}
, {-53, -34}
, {62, -2}
, {-100, -149}
, {-35, -61}
, {-241, -10}
, {32, 6}
, {-25, -115}
, {13, 36}
, {35, 15}
, {101, 73}
, {-99, 121}
, {-104, -380}
, {-21, 61}
, {-81, -19}
, {-166, -14}
, {-27, -119}
, {71, 40}
, {-189, -174}
}
, {{13, -33}
, {55, 77}
, {-140, -1}
, {-46, -85}
, {-63, 30}
, {35, -8}
, {12, 21}
, {-41, -94}
, {-54, -20}
, {-87, 33}
, {167, 50}
, {248, 192}
, {-82, 0}
, {126, -2}
, {-35, -95}
, {22, -54}
, {-230, -155}
, {33, 47}
, {-35, -118}
, {57, 12}
, {-128, -89}
, {-15, 68}
, {-54, 114}
, {49, 60}
, {-15, -96}
, {51, 38}
, {4, -83}
, {-15, 90}
, {12, -9}
, {38, 0}
, {36, 22}
, {-75, -68}
}
, {{37, -92}
, {-60, -112}
, {-110, -132}
, {9, 9}
, {91, -48}
, {33, 28}
, {140, -92}
, {-45, -10}
, {23, 0}
, {-9, 13}
, {-148, 29}
, {68, -5}
, {20, 50}
, {-17, -68}
, {-28, 53}
, {1, 39}
, {19, -43}
, {-3, -40}
, {-59, -173}
, {47, 25}
, {-106, -53}
, {60, -35}
, {30, -150}
, {111, 2}
, {108, 176}
, {169, -90}
, {-7, -140}
, {49, -15}
, {-21, -160}
, {10, 77}
, {117, 63}
, {81, 3}
}
, {{-50, -39}
, {81, 53}
, {84, 47}
, {114, -40}
, {70, -70}
, {-41, -22}
, {-33, 83}
, {-63, -8}
, {-37, -93}
, {-52, 3}
, {4, -6}
, {-20, -13}
, {61, -71}
, {-65, -150}
, {-32, -112}
, {6, 30}
, {-106, 65}
, {-33, 104}
, {-83, -51}
, {158, 88}
, {-150, 9}
, {-26, 78}
, {95, -138}
, {-20, 45}
, {-39, -43}
, {152, 68}
, {-36, 102}
, {-41, 52}
, {103, 38}
, {-37, -24}
, {-7, -73}
, {29, 24}
}
, {{-67, -19}
, {-234, 32}
, {-27, -198}
, {-19, -23}
, {10, 38}
, {18, -63}
, {-106, 68}
, {-321, 26}
, {-51, -150}
, {-196, -41}
, {-44, -140}
, {-132, -218}
, {-224, 95}
, {-32, -87}
, {-49, -37}
, {0, -50}
, {21, 130}
, {66, 124}
, {33, 41}
, {-18, -76}
, {-39, -117}
, {25, 126}
, {-60, -54}
, {-37, 101}
, {13, -55}
, {34, 76}
, {41, -28}
, {52, 18}
, {37, -81}
, {-50, 129}
, {-27, 35}
, {-22, 61}
}
, {{112, 8}
, {-79, 15}
, {-169, 27}
, {-49, -260}
, {30, 85}
, {-13, 47}
, {-38, 268}
, {-54, 33}
, {13, 154}
, {61, -109}
, {104, 0}
, {73, -31}
, {-63, 78}
, {39, 49}
, {-25, -224}
, {-24, -142}
, {-48, 60}
, {-9, 4}
, {-48, -33}
, {-7, 85}
, {0, 80}
, {0, -148}
, {-35, 0}
, {-37, -53}
, {137, -19}
, {32, -99}
, {-108, -90}
, {25, -172}
, {-19, -66}
, {105, -41}
, {-34, 22}
, {-106, -21}
}
, {{-24, 37}
, {65, 5}
, {70, 173}
, {0, -51}
, {-43, -32}
, {-87, 80}
, {90, 112}
, {-82, -70}
, {-34, 109}
, {13, 145}
, {-78, -59}
, {-47, 13}
, {57, 22}
, {-74, -10}
, {38, -57}
, {-70, -58}
, {-87, -46}
, {-200, -158}
, {-41, -95}
, {142, 121}
, {-15, -21}
, {41, -146}
, {-22, 111}
, {63, 49}
, {-32, 74}
, {-155, 122}
, {9, 60}
, {27, -117}
, {-89, 64}
, {-112, 108}
, {-83, -47}
, {86, 149}
}
, {{-5, 112}
, {-36, -70}
, {0, 21}
, {113, 5}
, {-91, -195}
, {171, -191}
, {50, 131}
, {-160, 55}
, {-99, 26}
, {31, -122}
, {20, 69}
, {80, 6}
, {-57, 6}
, {-12, -26}
, {27, 77}
, {-60, 68}
, {11, 4}
, {-284, -115}
, {-93, -2}
, {-7, -33}
, {-6, -59}
, {91, 50}
, {27, 119}
, {-120, 138}
, {12, -216}
, {-18, -60}
, {-1, -147}
, {-17, -15}
, {84, -10}
, {165, -104}
, {-177, 6}
, {-255, 147}
}
, {{76, 28}
, {100, 29}
, {79, 18}
, {-175, -276}
, {39, -50}
, {186, 174}
, {85, 101}
, {51, -6}
, {106, -31}
, {-50, -15}
, {-103, -75}
, {-197, -203}
, {-33, -166}
, {-64, -63}
, {68, 77}
, {-74, -45}
, {58, 87}
, {16, -33}
, {9, -194}
, {49, -17}
, {-106, -81}
, {-26, -22}
, {77, -81}
, {39, 55}
, {79, 121}
, {49, -11}
, {-8, 66}
, {67, 12}
, {166, -18}
, {-229, -65}
, {-9, -96}
, {81, -64}
}
, {{38, -60}
, {86, 93}
, {39, 58}
, {-135, -131}
, {91, -95}
, {87, 134}
, {92, 85}
, {36, -39}
, {14, -225}
, {27, -46}
, {-78, -49}
, {26, -122}
, {-14, -17}
, {-41, -45}
, {-194, 46}
, {-114, -83}
, {61, -10}
, {-54, -171}
, {-106, -47}
, {26, 54}
, {-80, 25}
, {-45, 53}
, {-134, 19}
, {-3, 22}
, {30, 195}
, {31, 44}
, {0, 119}
, {0, 66}
, {53, 60}
, {64, 121}
, {122, -117}
, {249, -99}
}
, {{46, 37}
, {-44, -27}
, {167, 33}
, {-173, -252}
, {40, 15}
, {99, 93}
, {-24, -12}
, {-59, -124}
, {53, 53}
, {-129, 9}
, {-199, -194}
, {-73, 42}
, {-13, 146}
, {-8, 9}
, {15, 87}
, {-49, 18}
, {-12, -49}
, {-84, -47}
, {-197, -196}
, {18, 30}
, {-15, -50}
, {-1, -38}
, {-383, -47}
, {35, 151}
, {153, -8}
, {72, 52}
, {-31, -21}
, {-175, -30}
, {-259, 22}
, {87, -82}
, {20, 175}
, {42, 58}
}
, {{85, 74}
, {-84, 14}
, {-76, -7}
, {119, 109}
, {4, 21}
, {-117, 17}
, {126, 28}
, {60, -53}
, {-45, 116}
, {-229, -157}
, {-286, -38}
, {-96, -112}
, {65, -130}
, {-155, -186}
, {0, -180}
, {41, 30}
, {-26, 4}
, {8, 62}
, {45, 17}
, {36, -115}
, {-81, -158}
, {83, -202}
, {-14, 73}
, {13, -89}
, {32, -70}
, {-258, -52}
, {28, -120}
, {87, 10}
, {-106, -159}
, {-130, -171}
, {10, -120}
, {-6, 30}
}
, {{55, 41}
, {-125, 12}
, {-13, -67}
, {-46, -26}
, {-19, 104}
, {-226, -112}
, {59, 106}
, {-30, -25}
, {94, 138}
, {-68, 34}
, {-308, -60}
, {-127, -87}
, {27, 2}
, {59, -28}
, {-74, 75}
, {0, 46}
, {-38, -119}
, {-6, -8}
, {14, -46}
, {112, 50}
, {-173, -218}
, {-105, -62}
, {32, -1}
, {-86, 87}
, {112, 2}
, {95, 109}
, {-151, -86}
, {-86, -49}
, {-44, -172}
, {21, -1}
, {54, -6}
, {25, -127}
}
, {{-150, -63}
, {-129, 87}
, {92, 29}
, {64, 155}
, {-19, -12}
, {-136, -31}
, {134, -428}
, {-84, -62}
, {53, 78}
, {96, 72}
, {180, 9}
, {53, 77}
, {14, -103}
, {1, -80}
, {67, -244}
, {-27, 73}
, {-60, 119}
, {-23, -49}
, {-30, 0}
, {-18, 113}
, {-44, -94}
, {-60, 15}
, {-62, 11}
, {-41, -53}
, {-111, 30}
, {-97, -14}
, {-3, 12}
, {-75, -16}
, {152, -217}
, {-63, 58}
, {-196, -246}
, {7, 42}
}
, {{106, -72}
, {85, 20}
, {-162, -77}
, {-93, -73}
, {63, 83}
, {-269, -175}
, {66, -95}
, {0, -15}
, {-76, 93}
, {20, 49}
, {-45, 127}
, {-183, 69}
, {96, 49}
, {-70, -50}
, {-47, 6}
, {-124, 49}
, {6, -4}
, {-45, 9}
, {-85, -22}
, {-105, -41}
, {-160, -68}
, {-25, 63}
, {52, -60}
, {157, -37}
, {81, -86}
, {-150, 61}
, {-157, -24}
, {-76, 41}
, {69, 149}
, {119, -86}
, {61, 20}
, {-56, 74}
}
, {{-23, -42}
, {-8, 24}
, {-17, 116}
, {118, -127}
, {-153, -64}
, {-125, -100}
, {72, -250}
, {11, -98}
, {-73, 139}
, {-17, 0}
, {-26, -57}
, {-62, 29}
, {-142, -130}
, {-20, 141}
, {-64, 63}
, {5, 40}
, {-136, -73}
, {-115, -255}
, {66, 12}
, {-41, -106}
, {-12, -4}
, {66, -57}
, {4, -100}
, {-162, -156}
, {53, -31}
, {-119, -28}
, {124, 34}
, {24, -59}
, {84, -331}
, {-11, -55}
, {-83, 152}
, {-32, 33}
}
, {{16, -20}
, {-70, 50}
, {-78, -221}
, {-98, 155}
, {-151, -28}
, {-3, -37}
, {-114, -140}
, {-46, 127}
, {-94, 68}
, {54, 93}
, {-116, 21}
, {62, 22}
, {-128, -192}
, {1, 10}
, {-109, -94}
, {-32, 56}
, {-9, -80}
, {29, -23}
, {-43, 65}
, {69, 71}
, {-43, 11}
, {1, 7}
, {-92, -162}
, {-46, 10}
, {-17, -1}
, {81, -127}
, {41, 52}
, {12, -40}
, {-114, -26}
, {-49, -30}
, {-71, -31}
, {32, 48}
}
, {{33, 125}
, {17, 0}
, {-102, -237}
, {-12, -208}
, {-76, -94}
, {91, -12}
, {10, 31}
, {-51, -192}
, {-26, 130}
, {-133, -44}
, {112, 44}
, {15, 111}
, {76, -147}
, {64, -22}
, {-91, -81}
, {15, -28}
, {-74, 20}
, {16, -91}
, {6, -139}
, {-71, 0}
, {21, 52}
, {-224, -17}
, {203, -168}
, {-78, 41}
, {37, -56}
, {27, 88}
, {-9, -15}
, {63, -64}
, {-75, -163}
, {159, 81}
, {66, 82}
, {-137, -7}
}
, {{9, 49}
, {-94, -7}
, {-27, -34}
, {-142, 36}
, {-118, 80}
, {54, -7}
, {87, 127}
, {38, -64}
, {-95, 6}
, {-39, 75}
, {-112, -74}
, {-95, 1}
, {5, -55}
, {-22, -32}
, {-12, 58}
, {-130, 145}
, {-217, -46}
, {-168, 44}
, {-29, 2}
, {69, 6}
, {-55, -37}
, {-78, 48}
, {9, 102}
, {62, 78}
, {-33, 68}
, {-7, 215}
, {-148, -27}
, {-30, 49}
, {-263, 15}
, {-11, -117}
, {-28, 98}
, {-21, 65}
}
, {{-111, -147}
, {-27, -70}
, {12, 19}
, {-245, 51}
, {27, 109}
, {120, -121}
, {-28, 77}
, {15, -33}
, {-96, 45}
, {-3, 37}
, {-105, 100}
, {-121, 23}
, {21, 49}
, {-92, -20}
, {-50, -7}
, {-75, -171}
, {-49, -11}
, {89, -260}
, {73, -69}
, {118, 89}
, {2, 50}
, {-127, -5}
, {9, 15}
, {-125, -90}
, {-148, -95}
, {-200, 61}
, {-39, 65}
, {-105, 130}
, {-44, 19}
, {39, 20}
, {-73, 110}
, {-56, -29}
}
, {{-31, 81}
, {49, 112}
, {14, -271}
, {-50, -64}
, {35, -90}
, {-81, -30}
, {-18, -89}
, {-88, -7}
, {61, 91}
, {-65, 46}
, {-175, -95}
, {75, -73}
, {101, 13}
, {-318, -16}
, {0, 26}
, {-147, 144}
, {-1, 44}
, {-63, -102}
, {-15, -174}
, {5, 37}
, {-42, 103}
, {61, -70}
, {-9, -88}
, {43, 110}
, {0, 57}
, {18, -34}
, {-76, 21}
, {-86, -65}
, {-155, 17}
, {-16, 62}
, {27, -55}
, {-58, 30}
}
, {{80, 37}
, {80, -75}
, {4, 12}
, {125, 100}
, {7, 91}
, {-30, -62}
, {63, 29}
, {56, -53}
, {-56, 93}
, {-21, -23}
, {-80, 77}
, {-100, -78}
, {73, -7}
, {-26, 44}
, {-139, -126}
, {29, -127}
, {-67, -41}
, {-11, 50}
, {28, -92}
, {-4, -60}
, {-37, 4}
, {71, 72}
, {-53, -37}
, {-57, -384}
, {-7, -14}
, {-66, -142}
, {31, 49}
, {-38, 49}
, {29, 27}
, {130, -57}
, {-21, 46}
, {37, -19}
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


const int16_t dense_bias[FC_UNITS] = {-152, 27, 136, -19, 15}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{-35, -33, -104, -79, -93, 54, 48, 35, 0, 7, -84, -74, -22, -51, -47, 46, 30, 8, 90, 48, 14, -15, 109, 18, 75, -6, 13, -10, -36, -92, 143, 60, 49, 25, 53, 190, 38, 94, 65, 106, 57, 81, 51, 85, 66, 0, 1, -10, -33, -22, -5, 12, 4, -47, -64, -7, 7, 10, -15, 6, 79, 73, 44, 31, 52, 117, 156, 158, 197, 233, -16, 37, -41, -5, -26, -57, -51, 34, -44, -72, -101, -40, -84, -71, -9, -71, 2, -32, 6, -51, 51, 39, 39, -33, -34, -16, -74, -18, -3, -40, 61, 90, 74, 18, 74, -183, -127, -174, -168, -157, -6, -25, -4, -13, 21, 152, 105, -14, 36, 81, 108, 155, 98, 40, 37, 31, 19, -40, 7, -83, 5, 81, 8, 41, 59, -73, -10, 0, -22, -51, 31, 35, -2, 28, 8, 50, 41, 18, -32, -64, -62, -157, 0, -5, 2, 6, 50, 71, 8, -2, -141, -104, -8, 16, -28, -41, -83, -96, -73, -159, 49, -27, 36, 1, -15, 29, -23, -1, 37, -3, -3, 16, 32, -18, -17, 12, -8, 43, 45, 47, 2, -56, -75, -63, -75, -16, -25, -9, -83, -34, -24, 71, 109, 34, 49, -20, -39, -18, -7, 4, -134, -52, -125, -117, -123, -101, -128, -82, -107, -73, -107, 29, -62, -77, -81, 93, 76, 38, 43, 80, -9, 4, 3, -10, 53, -87, -65, -19, -49, 0, -10, 65, 46, -22, 47, 57, 26, 29, 86, 32, 35, -6, 21, 33, 134, 45, 32, 59, 103, 59, -91, -86, -60, -10, -58, -33, 51, -12, 24, 84, -118, -77, -129, -162, -146, 104, 123, 81, 128, 100, 10, 12, 34, 11, -16, 48, 1, 50, 57, 133, 26, 0, 11, 27, 43, 59, 77, -5, 48, 67, -129, -110, -123, -89, -112, -118, -74, -91, -13, -41, 37, 20, 31, 125, 151, -72, -96, -64, -34, -122}
, {70, 30, 103, 94, 22, -42, -60, -68, -104, -123, -76, -42, 42, -38, 17, 68, 17, 41, 49, 108, 114, 73, 108, 19, 70, 41, -14, 89, -11, 51, -200, -95, -214, -165, -130, 64, 74, -12, 30, 100, 45, 70, 37, 73, 28, -25, -66, -118, -130, -117, -122, -66, -21, -110, -90, 69, 79, 60, 69, 44, 290, 164, 172, 161, 185, 85, 90, 18, 117, -25, 39, 30, 62, 65, 14, 155, 132, 78, 84, 129, 79, -144, 147, -43, -45, 201, 90, 81, 121, 206, 28, 0, 25, 87, 50, 82, 48, -14, -64, 43, -24, 16, -66, -11, 14, -15, 20, 5, -26, 7, -123, -89, -67, -53, -53, -64, -100, -29, -33, -34, -105, -83, -62, -30, -3, -24, 55, -6, 51, -43, -96, -209, 61, -58, -181, -228, -16, -50, -115, -207, 65, 34, 51, 42, 137, 76, 40, -104, -25, -31, 86, 177, -93, 7, -22, -90, -20, 15, 36, 3, -192, 53, -249, -158, -137, -41, 32, -28, 19, -77, -58, 18, 49, 5, 1, 78, 12, 54, 12, 7, -41, -36, -92, -94, -148, -130, -165, -128, -166, -163, -9, -91, 28, 7, -99, -32, -82, -23, 40, 1, -190, 80, 15, -37, -44, 39, 66, 51, 26, 52, 59, 17, 81, 75, 32, 130, 97, 108, 97, 111, 65, 33, 10, 27, 20, 53, -94, -115, -49, -16, -209, -69, -142, -260, -309, 101, 76, 158, 94, 133, -69, -113, -48, -101, -84, -11, -71, -14, -19, -24, -115, -100, -46, -53, -30, -50, -65, -133, -64, -56, 30, 151, -11, 41, 12, -188, -146, -265, -145, -193, -8, 32, 70, 12, 54, -52, -106, 20, -106, -104, 133, 23, 32, 58, 92, -74, -30, -54, -59, 18, 42, 19, -36, -20, 22, -44, -88, 15, 33, 20, 81, 115, 39, 135, 115, -4, -64, -21, 27, -26, -36, -30, -32, -61, -87, 15, 14, 46, 60, 19}
, {64, 54, 17, 42, -16, 71, 62, 57, 57, 66, -31, 40, 1, 50, 60, 43, 17, -25, 59, 34, -177, -208, -136, -80, -77, 142, 69, 101, 62, 76, 52, 35, 42, 64, 80, -66, -33, -56, -26, -80, -112, -85, -159, -78, -26, 145, 85, 98, 74, 106, 151, 108, 5, 54, 23, -72, -38, -55, -6, -26, -227, -191, -176, -113, -72, -177, -184, -113, -246, -303, -66, 81, 28, -15, -27, -34, -44, 35, 0, -39, -226, -186, -296, -276, -323, 10, -1, -40, -16, -18, 123, 126, 161, 89, 156, -29, 34, 8, 32, 106, -10, 13, -19, -10, -37, -48, 57, 24, -17, 43, -105, 0, 20, 15, 31, 98, -21, 37, -49, -16, -104, -38, -72, -118, -152, 72, 55, 75, 75, 123, 13, 56, 3, 48, 158, 172, 178, 163, 211, 232, -117, -32, -71, -43, 7, -58, -21, -16, -35, -88, 140, 139, 221, 190, 97, -49, 51, 73, 11, 31, 121, -12, 180, 100, 140, 68, -23, -88, -98, -61, 23, 59, 5, 82, 74, -129, -55, -32, -176, -72, -22, -32, -27, -72, -74, 100, 145, 106, 128, 141, -1, 14, 6, 9, 30, -44, -33, -62, -42, -98, 46, 9, -105, -38, -106, 59, 28, 105, 57, 24, -200, -72, -149, -133, -86, -116, -131, -166, -128, -112, -12, 48, 69, -22, -65, 28, -14, -5, -1, -3, -5, -34, -32, 3, 10, 31, -27, -118, -121, -16, 65, 69, 64, 130, 93, 69, 39, 61, 81, 56, -76, -66, -70, -95, -109, -82, -111, -70, -86, 47, -13, -75, -42, -60, -52, 144, 157, 99, 79, 132, -104, -85, -96, -140, -150, -59, -61, -34, 138, 58, -104, -115, -115, -130, -172, 137, 64, 56, 117, 67, 3, -130, -89, -72, -97, 36, -14, -31, -58, -179, 24, -3, 64, 60, 85, 129, 109, 67, 10, 185, -131, -122, -155, -110, -80, 29, -42, -7, 37, -74}
, {4, 26, -10, 42, -23, 33, 16, 36, -7, -46, 33, -14, -32, -10, 29, -173, -176, -149, -128, -136, 22, -26, 36, 15, 82, -24, 2, 70, 16, 28, -88, 77, -32, 10, -32, 85, 33, -43, -77, -22, 40, 23, 1, 70, 35, 22, 23, 74, 17, 87, -94, -113, -163, -65, -61, 27, 25, 90, 15, 21, 15, -48, -25, -32, 22, 62, -79, -32, -65, -11, 30, -8, -49, -2, -59, -18, -78, -65, -76, -51, 13, 257, 102, 141, 121, -232, -35, -138, -35, -104, -29, -4, -22, 24, -48, -16, 21, -52, -21, -100, -52, -67, -36, -54, -55, 66, 31, 47, 33, 61, 124, 55, 75, 75, 109, -121, -31, 55, 53, 80, -33, 38, 112, 35, 32, -133, -142, -171, -14, 45, 357, 99, 153, 101, 113, -41, -45, 4, -35, -107, 100, 101, 52, -1, 32, 68, 7, 43, 64, 0, -283, -202, -152, -200, -249, -61, -72, -62, -98, -99, 177, 116, 116, 122, 109, -13, 52, 39, 54, 22, 79, -47, 20, 9, 60, 68, 102, 30, 44, 85, 108, 142, 166, 157, 210, -93, -138, -39, -45, -97, 125, 151, 192, 145, 98, 25, 56, 103, 44, 54, 150, -63, 0, 16, 104, 3, 13, 10, -3, 19, 16, 10, 19, 91, 39, -99, -103, -67, -106, -32, 3, 58, -10, -55, -38, 19, 53, 71, -5, 1, 83, 71, -15, 29, 35, -80, -42, 22, 50, -4, -100, -79, -125, -177, -158, -223, -153, -208, -241, -189, -63, 50, -15, 7, 25, -13, -106, -42, -64, -55, -114, -136, -212, -157, -206, -39, -22, -1, -9, 7, 85, 44, 79, 54, 58, -1, -42, -98, -64, 33, -20, -38, 27, 0, 43, 56, 63, -29, -24, -98, -40, -19, -29, -36, -15, -195, -154, -159, -139, -294, -11, 33, -3, -33, 0, 95, -27, -25, 18, 5, 102, 151, 86, 130, 100, 25, 26, -29, 33, -4}
, {-78, -73, -100, -34, -35, -4, 0, -10, -20, -13, 18, 57, 59, 11, 43, -89, -99, -95, -97, -105, -75, -104, -27, -12, -29, -93, -146, -93, -119, -148, 77, 84, 15, 69, 48, -153, -75, -117, -45, -32, -4, -118, -87, -60, -142, -30, -42, -17, -46, -31, 80, 43, -17, 18, 42, -93, -75, -136, -114, -63, -259, -198, -254, -195, -276, -118, -77, -185, -94, -66, 50, 43, 18, 25, 27, -15, -25, 16, 0, 69, 37, 39, 79, 111, 122, 172, 84, 25, 18, 46, -124, -161, -184, -103, -189, 38, -61, 22, 38, 0, 67, 49, 52, 78, 147, 109, 102, 113, 36, 90, 23, -9, -4, -4, -5, -190, -50, -152, -57, -192, 4, 75, 34, -98, -73, -46, 0, 28, 40, -38, -218, -12, -267, -98, -108, -56, -199, -8, 18, -16, -56, -75, -135, -139, -51, 17, 31, 79, 91, 54, -1, -13, -18, -49, -63, 35, 7, 45, 34, 44, 34, -201, 28, -158, -77, 70, 12, 71, 94, 124, -130, -113, -106, -154, -133, 25, -16, 38, 16, -37, 16, 15, 2, -30, -14, 43, 75, 23, 32, -30, -182, -110, -34, -89, -96, 56, 63, 21, 53, 53, -138, -32, -90, -65, -170, -248, -125, -105, -94, -114, 83, 50, 83, 55, 100, 45, 47, 85, 36, 61, 46, 97, 83, 47, 28, -86, -114, -47, -71, -111, 31, 69, 22, 69, 88, -62, -9, 23, -3, -63, 76, 18, 76, 1, 22, 28, 89, 69, 68, 35, 77, 44, 86, 63, 90, 65, 82, 94, 160, 149, 105, 72, 70, 110, 151, 53, 65, 35, 43, -2, 74, 114, 96, 77, 163, -38, 69, -33, 42, -48, 36, -10, -1, -19, -53, -97, -92, -109, -148, -31, -30, -5, 2, 27, -14, 8, 70, 77, 3, 52, -56, -74, -94, 6, -11, -1, -14, -22, -16, -24, -63, -6, -53, 22, -10, 114, 16, 8, 66, 95}
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
