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


const int16_t conv1d_bias[CONV_FILTERS] = {128, -144, -63, -21, -219, -14, -20, 373}
;

const int16_t conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-40, 20, 14, 5, 70, -70, 56, -103, 37, -109, 52, 4, 4, 21, -37, 67, -68, 42, -54, 9}
}
, {{-100, -16, 6, 14, -19, 19, -73, -7, -98, -14, -70, 15, -25, 37, -2, 73, 22, 44, 96, 174}
}
, {{-62, -78, -46, -23, -68, -38, 39, -104, -15, 6, -46, 26, -74, -59, 29, 0, 7, -10, 23, -5}
}
, {{14, 62, 61, 49, 56, 0, 45, -73, 103, -26, 61, -24, 58, 41, 11, 44, 32, 70, -18, 145}
}
, {{49, 76, -11, 46, -14, 30, 32, -37, 26, -61, 12, -81, 14, -98, -40, -33, -72, 73, -88, 13}
}
, {{74, -99, -63, 128, -75, -93, 65, 138, 72, -170, -2, 36, 5, -43, 113, -30, 21, -113, -91, 146}
}
, {{-60, -67, -26, -65, -35, 2, -6, 37, 24, -27, -18, 9, -3, 8, 10, -55, -85, -66, -84, -76}
}
, {{61, 121, 5, -122, -98, 9, 36, 94, 37, -26, 3, -20, -42, 59, 125, 104, -12, -7, -61, -91}
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


const int16_t conv1d_1_bias[CONV_FILTERS] = {-212, 127, 14, 135, 7, -498, 81, 23, -7, -59, 21, -94, -168, -13, -2, 86}
;

const int16_t conv1d_1_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-51, 46, 122, 49, 9, -74, -115, -31}
, {-144, -23, 57, -34, -75, 20, -18, -35}
, {-5, -3, 137, 77, 57, -108, -130, -67}
, {-50, 13, -106, -101, 48, 83, 80, 31}
, {-13, 82, 162, -13, -104, 0, -47, -113}
, {-127, -9, 38, 24, 3, -3, 16, -123}
, {60, 93, 81, 63, 14, -105, -46, -123}
, {-126, -51, -40, -72, -8, 60, 57, 6}
}
, {{-25, -130, -58, 123, -64, -19, 37, 74}
, {-27, -12, -133, 27, -2, -58, -132, -25}
, {-160, -132, -76, 42, 44, -109, 18, 41}
, {107, 30, -17, -2, -87, 7, 156, -180}
, {27, -32, -49, 58, -17, -93, 33, 134}
, {-148, -49, 0, 95, -141, -172, -87, 19}
, {20, -16, 61, 61, -17, 0, 41, 112}
, {88, 8, 20, -81, -47, 41, 33, -135}
}
, {{60, 15, -64, 52, 65, 78, 95, 68}
, {-113, -125, 38, 156, 18, -55, -168, -31}
, {12, -27, -75, -39, -99, -39, 58, 11}
, {-6, 121, 94, 74, 41, -100, -123, -31}
, {-33, -106, -49, 35, -14, -16, -16, 136}
, {-107, -197, -17, 155, 86, -73, -176, -141}
, {-66, 60, 57, -22, -125, 3, -41, 119}
, {0, -39, 95, 162, -67, -33, -130, -106}
}
, {{-106, -74, -4, -91, -49, 74, 75, -10}
, {-199, 16, 107, 134, 61, -48, -70, -88}
, {81, -11, -93, -127, -60, 108, 108, 24}
, {-143, -47, 99, 104, 82, -18, 7, -93}
, {-118, 44, -42, 57, 12, 79, 70, -17}
, {-382, -182, -73, -1, -173, -178, -215, -265}
, {31, 40, -121, -166, -70, 22, 49, 7}
, {-141, 13, 101, 57, 53, -20, -56, -75}
}
, {{-53, -51, -116, -73, -69, -85, -59, -24}
, {47, 80, 25, -5, 20, 70, 81, 19}
, {-28, -45, 6, 45, -25, -59, -46, -25}
, {45, 122, 79, -43, -68, -70, -54, -154}
, {-52, -56, -49, 0, -60, -19, 27, 11}
, {-7, -38, 49, 51, 70, 180, 124, 140}
, {98, 49, 35, -47, 8, 76, 30, 66}
, {-53, 98, 34, 116, 28, -5, 117, -31}
}
, {{-105, 44, 55, 44, 87, 74, 55, 16}
, {-57, -141, -13, 8, -10, 15, 70, 92}
, {-95, -98, 72, 57, 21, 33, -43, -86}
, {35, 175, 113, 27, -100, -130, -34, 41}
, {-187, -59, 154, 150, 128, 10, -34, -31}
, {38, -40, -51, -28, 23, 9, 17, 13}
, {-75, 78, 35, 44, 10, -15, -52, -34}
, {40, 47, -28, -2, -115, -121, -51, 11}
}
, {{20, 48, 44, 39, -15, -85, -23, 45}
, {19, -3, -78, -2, 10, 1, 118, 88}
, {92, 33, 20, 26, -37, -163, -52, 31}
, {9, -26, -36, -123, 14, 43, -17, 46}
, {83, -5, 16, 15, -99, -61, 58, -6}
, {-17, -66, -278, -29, 0, -114, 13, -68}
, {82, 24, 74, 93, -103, -67, -27, -43}
, {-10, -51, -68, -54, -51, 45, 40, 30}
}
, {{108, -61, -118, 95, -142, -2, -34, -20}
, {0, -26, -36, 90, -55, -61, 20, -4}
, {-37, 4, -122, 122, -5, -34, -133, 91}
, {-124, -44, -29, -52, 39, -24, 87, 76}
, {96, -83, -53, 107, -146, -96, -6, 166}
, {-71, -169, -173, 231, 38, -62, -49, 50}
, {-46, -126, 69, 14, -136, 25, -152, 195}
, {-92, -64, 25, 84, 43, -24, 75, 44}
}
, {{-8, -83, 30, 102, 13, 120, 38, -93}
, {58, 65, 74, -3, 15, 106, 17, -51}
, {-12, -13, -100, 62, 57, 56, 87, -7}
, {62, 12, -165, 12, -84, 87, 30, -28}
, {-2, -8, 23, 22, 61, 106, 50, -40}
, {14, -69, -134, -41, -208, -38, 33, -133}
, {-93, -133, -12, -25, -43, -13, -42, 18}
, {99, -6, -35, 30, -74, -22, 0, -152}
}
, {{-140, -28, 11, 82, 105, 84, -21, -60}
, {-144, -182, 55, 96, 112, 65, 1, -50}
, {0, -17, -43, 59, 17, -6, -41, 10}
, {-45, 31, 100, 13, -62, -50, -52, 3}
, {-75, -124, 156, 146, 86, -11, 0, -43}
, {-282, -367, -77, 284, 214, -93, -221, -117}
, {-54, -87, -143, -68, -63, -22, -62, 28}
, {-116, 0, 97, 127, 73, -12, -2, -34}
}
, {{-114, -92, 12, 140, -7, 52, 61, -142}
, {-53, -42, -16, 135, 33, 7, -91, -30}
, {-70, -49, -8, 116, 129, 4, 12, -127}
, {-4, -117, 16, -38, 50, 38, 15, -93}
, {-45, -45, -17, 144, 104, -26, -60, -82}
, {-117, -61, -17, 70, -25, -78, -79, -79}
, {-47, -57, 83, 18, 75, 17, -108, -4}
, {-89, -77, 49, -4, 126, 85, 61, -70}
}
, {{-30, -14, 2, -72, -43, 140, -27, -153}
, {30, 63, -30, -94, -84, 43, 61, 65}
, {-44, 14, -48, -1, -110, 158, 67, -52}
, {20, 31, -19, 24, -1, -9, -106, -22}
, {23, 41, -62, -159, 2, 163, 26, -39}
, {22, 68, -88, -305, -126, 40, -152, -250}
, {110, -101, -62, -63, 17, 144, 30, -117}
, {25, 45, 33, -35, -34, 58, -49, 22}
}
, {{-21, 71, -45, -37, 53, -34, -120, -8}
, {-16, 38, 27, 77, 63, 25, 14, 105}
, {89, -1, -37, -61, 65, 12, -91, -120}
, {-12, 19, -36, -30, -138, -20, 82, 64}
, {30, 30, -5, 102, 115, -20, -59, 6}
, {-106, -127, 27, 10, 53, -44, -104, 22}
, {48, -2, -48, 45, -64, 46, 21, 0}
, {-69, -39, -15, -48, -31, 71, 16, 42}
}
, {{30, -30, 72, 10, 63, 22, 19, 58}
, {-132, -198, 87, -52, -73, -34, 88, 188}
, {2, -6, 11, -36, -59, -27, 14, -95}
, {-66, -127, 80, 81, -61, 29, 78, -12}
, {49, -75, 104, 15, -8, 44, 35, -80}
, {-2, 0, 97, 47, -46, -9, 102, 57}
, {67, 47, 78, -98, 17, -23, -31, -100}
, {-67, -233, -42, 40, -72, -174, -26, -38}
}
, {{34, -58, 91, 35, 102, 101, 25, 4}
, {-52, -40, -190, 52, 37, -113, 34, -64}
, {44, 60, 26, -50, -106, 81, -70, 82}
, {-38, 6, 30, -4, 89, -51, 45, -70}
, {14, 3, 45, -30, 20, 16, 6, 34}
, {78, 176, 230, 147, 145, 244, 134, 212}
, {-59, -17, -80, 18, -86, 0, 27, -47}
, {-1, 37, -44, -4, 131, -129, -48, -120}
}
, {{-4, -32, -69, -25, 26, 40, 86, 29}
, {-92, -97, 112, 43, -47, -233, -29, 49}
, {-9, -83, -141, -104, 90, 144, 28, 42}
, {-41, -11, 62, 67, -151, -116, -163, -84}
, {-30, -85, -32, 77, 73, 0, 21, 37}
, {-42, 18, 131, 116, 34, 38, 61, 8}
, {-105, -125, -73, -15, 96, 94, 100, -26}
, {-89, 38, 41, 102, -91, -72, -115, 115}
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


const int16_t conv1d_2_bias[CONV_FILTERS] = {43, 0, 59, -56, 183, 492, 3, -207, -237, -17, 2, -60, -355, -81, -122, -49, 83, -46, -526, 37, -52, -120, -71, -38, 257, 162, -154, -219, 149, 211, 122, 13}
;

const int16_t conv1d_2_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-7, 131, 21, 29}
, {-47, -5, -203, -166}
, {105, -53, 28, -96}
, {65, 12, -5, 83}
, {-48, -56, -1, 122}
, {-198, -126, -28, -129}
, {11, 35, 19, 6}
, {2, -162, 19, -50}
, {130, -2, -69, 114}
, {130, -31, -4, 0}
, {43, 113, 35, 65}
, {28, 164, -12, -13}
, {-53, -136, -30, 5}
, {48, -146, -85, 43}
, {-30, -43, 0, 78}
, {-79, -100, 0, -75}
}
, {{95, 54, 48, 106}
, {-94, 84, 53, 101}
, {-257, -31, -21, -256}
, {-28, -43, 5, -44}
, {-53, 64, -2, -77}
, {-11, 51, 100, 102}
, {-65, -177, -83, 28}
, {32, -40, -4, 17}
, {-56, -14, -90, -36}
, {126, 145, 76, 102}
, {-27, 70, -3, -58}
, {25, -46, 72, 49}
, {61, -25, -10, 14}
, {-39, -124, 42, 10}
, {74, 68, -10, -163}
, {29, 95, 2, -86}
}
, {{13, 28, 29, 104}
, {71, 2, 176, -118}
, {98, -82, 58, 73}
, {18, -203, -59, -4}
, {112, 93, 103, 34}
, {-164, -63, -135, -147}
, {-140, 54, -29, 18}
, {89, -8, -160, 13}
, {-137, -63, -83, -145}
, {29, 98, 32, -114}
, {-168, 52, 52, -6}
, {-44, -15, -180, -47}
, {-128, -74, -207, -24}
, {48, 96, -57, 35}
, {4, 7, -12, 67}
, {93, -22, 58, -100}
}
, {{-44, -145, -1, -192}
, {82, -47, 97, -126}
, {65, 78, 111, -41}
, {91, 4, -46, 19}
, {-61, 42, -100, -158}
, {30, 115, 167, -27}
, {77, -3, 113, -103}
, {-105, -232, -257, -323}
, {4, -73, -63, -116}
, {208, 116, 143, 143}
, {-78, 39, -158, -88}
, {-60, 67, -93, 0}
, {-23, 25, -33, 5}
, {-44, -151, -80, 79}
, {24, -41, -26, -30}
, {-69, 109, -140, 108}
}
, {{35, 49, 77, 12}
, {41, 137, 27, 73}
, {82, 43, -13, 46}
, {161, -114, 24, 31}
, {-79, -131, -64, 11}
, {-219, -12, -58, -34}
, {-76, -81, -20, -97}
, {-12, -75, 59, -38}
, {-25, 14, 17, 66}
, {-195, -72, -58, -160}
, {-115, 124, 96, 1}
, {46, -1, 68, -37}
, {11, -41, 91, 49}
, {-2, -73, 169, 65}
, {-36, -202, -75, -95}
, {-41, 5, -121, -121}
}
, {{-13, -9, -53, 84}
, {56, 24, -3, 17}
, {-110, -41, -48, -305}
, {-35, -69, -70, -70}
, {45, -68, -45, 52}
, {-45, -167, -112, -17}
, {90, 24, -201, -51}
, {-20, 69, 55, 25}
, {54, 28, -72, -84}
, {102, 87, 107, 141}
, {56, -223, -122, 29}
, {83, 92, 113, 88}
, {32, -56, -14, -53}
, {46, 15, -29, -32}
, {91, 50, -115, -25}
, {-2, -10, -155, -119}
}
, {{-185, -141, -33, -74}
, {-19, -238, -89, 27}
, {5, 157, -2, 130}
, {-144, -103, -171, -101}
, {27, -93, 114, 41}
, {-54, -152, -10, -148}
, {-23, -119, -53, -65}
, {-150, -78, -43, -10}
, {-20, -49, 114, -10}
, {158, 145, 179, 195}
, {180, 91, 52, 114}
, {-150, -1, 154, 30}
, {67, 31, 39, -90}
, {-12, -76, -59, 21}
, {64, 0, 48, 64}
, {129, 17, -45, -93}
}
, {{218, 83, 104, 119}
, {-126, -146, -104, -97}
, {-100, -126, -33, 89}
, {-81, -142, -23, -15}
, {23, 60, 135, 92}
, {-13, -54, -15, 23}
, {-69, 3, -21, 54}
, {-46, 78, -26, 58}
, {-21, 18, 32, -35}
, {37, -28, -48, 19}
, {69, 48, 21, -5}
, {33, -18, 7, -159}
, {-41, -31, -33, -50}
, {-28, 35, 13, 106}
, {-35, 28, 50, -10}
, {61, 81, -10, 119}
}
, {{108, 9, -21, -47}
, {12, 14, 5, -54}
, {-157, -5, -109, -15}
, {45, 29, 111, 79}
, {15, -11, -19, -94}
, {8, 0, -50, 61}
, {-29, 79, 140, 13}
, {-43, -41, 35, 70}
, {-1, 9, -81, 29}
, {-4, -72, 12, -26}
, {57, 113, 58, 35}
, {-112, -71, 70, 125}
, {80, -130, -105, -58}
, {-97, -114, -8, -43}
, {-30, -20, -70, -121}
, {64, 89, 76, 29}
}
, {{90, 130, 152, 66}
, {54, -25, -52, -55}
, {-16, 61, 8, -23}
, {-21, -60, -36, -47}
, {-33, -60, -1, 49}
, {12, 45, -22, 11}
, {-78, -102, -109, -139}
, {97, 133, 8, 86}
, {-71, 39, -86, -114}
, {-4, -13, -3, 0}
, {5, -134, -98, -51}
, {27, 116, 175, 108}
, {-28, -2, 56, 102}
, {160, 18, -46, -52}
, {-64, -2, -5, 20}
, {-31, -98, -138, -70}
}
, {{45, -56, -57, 82}
, {-65, 98, -24, -107}
, {121, 71, -140, -132}
, {34, 53, 78, 46}
, {72, 85, -129, 20}
, {32, 0, 7, -27}
, {-66, -87, 111, 88}
, {139, 41, -41, 12}
, {-63, 26, -103, -109}
, {-119, -84, -150, -103}
, {-172, -125, 59, -73}
, {73, -3, -45, -21}
, {-17, -212, -3, 59}
, {47, -43, 3, 65}
, {-63, -53, -102, -125}
, {-121, 113, 42, -71}
}
, {{-41, -63, -35, 31}
, {-106, -81, 0, -188}
, {-41, -103, -43, -40}
, {88, 36, 164, 158}
, {-11, 43, 14, -79}
, {30, 159, 21, -190}
, {25, 45, 68, -67}
, {51, 64, 14, 6}
, {47, -15, 78, -56}
, {-99, 12, 30, 42}
, {-89, -7, 108, 64}
, {-106, -237, -213, -135}
, {-49, -91, -90, -154}
, {-23, -128, -6, -24}
, {29, 59, 36, 30}
, {-106, 50, 128, 54}
}
, {{-52, 98, 150, 63}
, {-14, 15, -55, 3}
, {75, -32, -23, 84}
, {-86, -156, 38, 80}
, {-31, -36, -15, 57}
, {-189, -80, 74, 121}
, {-132, -102, -31, 27}
, {99, 55, 86, 59}
, {-213, -156, 52, 44}
, {-8, -139, -91, -33}
, {-4, 110, 107, 86}
, {-141, -101, 30, 39}
, {-81, -25, 15, 27}
, {76, -119, -89, -7}
, {35, 41, 4, -15}
, {-47, -10, 40, 50}
}
, {{-29, 13, 53, 71}
, {-66, -23, -21, -9}
, {-33, -77, -148, -72}
, {93, -30, -164, -21}
, {-25, -16, -5, -76}
, {-8, 22, -39, -13}
, {32, 40, 39, -15}
, {13, 98, 144, 13}
, {41, -4, 67, 83}
, {1, 1, -106, 20}
, {-141, -154, -61, 94}
, {25, 49, -168, -85}
, {56, 57, 29, 24}
, {41, 40, 82, 1}
, {-56, -193, -143, -43}
, {-158, -131, 66, 70}
}
, {{122, 83, -90, -65}
, {-14, -46, -70, 59}
, {37, 22, 67, 223}
, {-119, -148, 9, 0}
, {-53, 17, 43, -88}
, {-4, 141, 93, -9}
, {-95, -9, -61, -26}
, {-1, -55, 64, 89}
, {29, 82, 144, -9}
, {-144, -31, 69, -113}
, {0, 40, 16, 62}
, {103, 122, 85, -29}
, {-180, -82, 54, -37}
, {-139, -118, 66, 127}
, {-10, 24, -6, -30}
, {-61, 38, -8, 97}
}
, {{-62, -81, -152, -54}
, {67, 53, 0, 29}
, {-58, -16, -20, -26}
, {12, 88, 77, 61}
, {57, 73, 100, -3}
, {-14, -38, -10, -95}
, {8, 16, 35, 96}
, {-63, -69, -1, 55}
, {-50, 29, 110, 59}
, {-33, -111, -94, -138}
, {93, 23, 41, 95}
, {0, 21, -4, 40}
, {-16, 45, 45, -9}
, {-1, 24, -84, 93}
, {-28, -52, 10, 21}
, {12, -77, -48, -29}
}
, {{-21, -168, -96, -93}
, {21, 2, -148, -34}
, {-162, 222, 20, 11}
, {-13, -90, -140, -102}
, {23, -47, -170, 30}
, {-123, -24, -182, -179}
, {49, -9, -78, -33}
, {99, 91, 66, -77}
, {42, -45, -32, 72}
, {27, 61, 39, 21}
, {-122, -167, -137, -30}
, {24, 82, 29, -71}
, {-17, -33, 59, -5}
, {-15, -8, 146, 43}
, {-43, 22, 176, 135}
, {-180, 35, 110, 96}
}
, {{-105, -312, -183, -141}
, {26, 36, 75, 132}
, {91, 68, -108, -303}
, {78, 46, -97, -208}
, {44, 8, 25, 59}
, {-107, -134, -33, 67}
, {-63, 4, 80, 108}
, {-15, -3, -68, -70}
, {-22, -56, -17, 65}
, {-104, -24, -129, -139}
, {-35, -168, -109, 56}
, {91, 142, 17, 76}
, {58, 21, -97, -12}
, {40, 86, 38, -51}
, {-47, 10, 27, 11}
, {8, -35, -156, 51}
}
, {{33, 36, 23, 40}
, {8, 3, 27, 80}
, {150, 101, 87, 81}
, {28, -82, 45, 27}
, {86, -62, -51, -15}
, {162, 0, 69, 122}
, {9, 16, -126, -38}
, {-132, -11, 28, -199}
, {42, -30, -23, -14}
, {-70, 19, -93, 192}
, {21, -49, -60, -63}
, {50, 25, 20, -30}
, {54, 32, 21, -52}
, {-39, 7, 58, -67}
, {12, 54, -20, 57}
, {114, -44, 71, 104}
}
, {{24, 118, 79, -9}
, {20, 87, 135, 120}
, {-85, -32, -233, -105}
, {0, 56, -48, -46}
, {-14, -100, -1, 23}
, {8, -33, 87, 65}
, {-57, -79, 38, -18}
, {-10, -49, -41, 90}
, {42, -23, -154, -160}
, {43, 78, -94, 62}
, {-33, -6, 2, 0}
, {127, 109, 78, -141}
, {64, -126, -31, 64}
, {-33, -65, -46, 37}
, {-24, -106, -138, 32}
, {-68, -110, -71, 137}
}
, {{-80, -12, -45, -17}
, {98, 0, 45, -33}
, {-10, -150, -86, 26}
, {-118, -127, -59, -22}
, {46, 5, -25, 70}
, {113, 92, 113, 170}
, {-30, 75, 92, -58}
, {88, 107, 119, -36}
, {-4, 19, 1, -11}
, {-9, -125, -64, -54}
, {-83, 19, -24, -17}
, {25, 24, 30, 93}
, {17, 63, -28, -53}
, {-93, -46, 37, -37}
, {52, 30, -35, 32}
, {120, 43, -51, -4}
}
, {{15, 5, -5, -32}
, {31, 20, -52, -57}
, {39, 138, 124, -1}
, {46, -9, -109, -32}
, {-61, 109, 122, 88}
, {-28, 74, -28, -34}
, {-32, 33, 83, -40}
, {-97, -88, -67, 0}
, {-70, 5, 92, 0}
, {59, 42, -10, -141}
, {-170, -166, -93, -149}
, {-184, -135, -31, -96}
, {-4, -87, -1, 100}
, {17, -64, -107, 93}
, {47, 77, -44, -17}
, {-39, -76, -66, 77}
}
, {{-60, -23, -35, 175}
, {63, 15, 56, 76}
, {108, 126, 157, -73}
, {-141, 92, 162, -14}
, {-57, 82, 6, -4}
, {-113, -150, -84, -5}
, {-75, -80, 16, 28}
, {95, -16, 91, 12}
, {-51, 29, 47, -91}
, {-2, 172, 78, 80}
, {-53, -33, -87, -29}
, {14, -6, -31, -119}
, {-44, 14, -156, -186}
, {-23, -92, 18, -132}
, {9, 30, -119, -65}
, {63, -48, 66, -11}
}
, {{144, -126, -193, -153}
, {-71, -8, -57, -33}
, {129, -24, 68, -269}
, {-64, 0, 25, 7}
, {51, -80, -51, -103}
, {72, 78, -115, -76}
, {-128, -29, -24, 12}
, {19, 50, 18, -36}
, {-17, 81, 92, 72}
, {-19, 244, 96, 121}
, {-75, 16, 82, 70}
, {4, 13, -67, -61}
, {-26, 22, -32, 30}
, {-64, 74, 53, -39}
, {-45, -96, -141, -11}
, {-70, 12, 179, 114}
}
, {{33, 149, 82, -42}
, {-72, -28, 0, -33}
, {-178, -267, -178, 9}
, {29, 13, -108, -86}
, {14, 80, 31, -36}
, {35, -50, -49, -61}
, {126, 82, -84, -126}
, {-126, -36, 40, 23}
, {73, 26, -56, -84}
, {80, 67, 26, 70}
, {13, 48, 21, -8}
, {-30, -100, 14, -47}
, {64, 46, -14, -72}
, {87, 20, 10, 29}
, {7, -32, -106, -105}
, {-93, -146, -101, 30}
}
, {{-149, 18, 80, -15}
, {-99, 23, 128, -6}
, {-47, -55, 36, -33}
, {-68, 88, 121, -37}
, {-161, -196, -99, -125}
, {-168, 101, 145, -48}
, {-29, -26, 53, 102}
, {-140, 108, 92, 80}
, {6, -174, -117, -42}
, {42, -6, -74, 110}
, {-58, -207, -212, -22}
, {-167, -7, 95, 22}
, {-30, -201, -170, -38}
, {36, 10, 30, 76}
, {2, 128, 126, 71}
, {44, 24, 58, -27}
}
, {{63, -7, 60, 22}
, {11, 25, -34, 11}
, {-152, -98, -103, -54}
, {-33, 97, 78, 48}
, {-133, 105, -24, -40}
, {7, -91, -18, 111}
, {6, 24, -144, -107}
, {16, 114, 210, 4}
, {29, 63, -6, 18}
, {-54, -57, 23, 2}
, {-66, -66, -30, 0}
, {99, -19, -60, 60}
, {-5, -53, 111, -35}
, {-16, -109, -56, -115}
, {110, 23, 7, 53}
, {-92, 19, 83, 82}
}
, {{6, 17, 65, -23}
, {57, 41, 0, 16}
, {72, -31, -14, 52}
, {-28, -69, 40, 98}
, {-8, 49, 0, -97}
, {1, -23, -97, -52}
, {-9, 35, -20, 3}
, {-21, 52, 82, 20}
, {28, 1, -83, 46}
, {-43, 44, 72, 16}
, {-117, -45, 6, -45}
, {-170, -155, -186, -47}
, {36, 18, 78, 6}
, {83, 1, 144, 132}
, {31, -46, 35, 80}
, {-8, -32, 77, -70}
}
, {{-118, -37, -20, -85}
, {51, -176, -205, 188}
, {207, -172, -178, 74}
, {-92, -99, -135, 13}
, {-54, -27, -11, 74}
, {113, 70, -123, 10}
, {-92, 49, -33, -24}
, {-210, 181, 212, -88}
, {86, -41, -60, -10}
, {-27, 43, 46, -111}
, {61, 101, 4, -134}
, {-69, -39, -29, -41}
, {-21, 61, 41, -5}
, {-177, -2, -52, -110}
, {-18, -95, -102, -54}
, {27, -60, -121, 128}
}
, {{132, 44, 80, 184}
, {-42, 55, -57, -34}
, {-7, -72, 68, -21}
, {12, -37, 146, -21}
, {-171, -202, -41, -3}
, {-34, 28, 155, -21}
, {53, 23, 39, -53}
, {-57, -38, 190, 48}
, {11, 83, -111, -76}
, {8, -71, -106, -115}
, {-150, -89, -83, 21}
, {-112, 56, -31, 144}
, {-102, 38, 18, -25}
, {63, -66, -71, -119}
, {-212, -47, -234, -60}
, {-23, -83, -8, -88}
}
, {{43, -43, -73, 4}
, {-39, -27, 64, 22}
, {-20, 32, 111, 76}
, {-44, 69, -35, 108}
, {-34, 9, -100, -2}
, {-83, 26, -43, -168}
, {-55, 49, -24, -8}
, {-73, -33, 63, 72}
, {-23, 4, 30, 106}
, {353, 127, 32, 63}
, {59, 70, 71, 49}
, {-21, -11, -21, 110}
, {-54, -53, -31, 46}
, {-77, 24, -90, 40}
, {-265, -165, -201, -148}
, {-29, 12, 64, -70}
}
, {{-178, 20, -51, -60}
, {-71, -15, 1, -39}
, {-5, 191, 84, -5}
, {4, -40, 61, -57}
, {-78, -230, -105, -49}
, {-59, -267, -68, -226}
, {-33, -7, 31, -14}
, {-62, -29, 49, 28}
, {-42, 54, 90, 3}
, {182, 72, -77, -96}
, {-104, 20, 34, 1}
, {-65, -52, 123, 68}
, {29, -39, 16, -44}
, {34, -13, -32, -58}
, {-25, 59, 2, -2}
, {9, 155, 125, 39}
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


const int16_t conv1d_3_bias[CONV_FILTERS] = {91, -13, 52, 192, 66, 49, 145, 174, 0, 253, -182, 147, 196, 64, 16, 125, 128, 41, 129, 164, 33, 229, 129, 224, -155, 204, -56, -210, -87, 108, 318, 202, 261, 87, 98, 250, -86, 82, 112, 96, 55, 229, 106, 123, 152, 60, 179, 47, 5, 75, -1, 115, 164, 274, -194, -68, -223, -28, -130, -218, 116, 141, -225, 23}
;

const int16_t conv1d_3_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{110, 120}
, {-25, -51}
, {49, -72}
, {-42, 48}
, {110, -56}
, {93, -126}
, {-8, 26}
, {-23, 84}
, {37, 2}
, {-12, -87}
, {-84, -152}
, {12, 4}
, {-116, -44}
, {52, 78}
, {-158, 59}
, {-111, 83}
, {-146, -62}
, {62, 67}
, {-120, -59}
, {98, -52}
, {-63, -14}
, {-8, 65}
, {-11, -1}
, {-71, 92}
, {-172, -7}
, {-64, 112}
, {111, -134}
, {81, 73}
, {110, 2}
, {95, -34}
, {8, -64}
, {55, 100}
}
, {{43, 114}
, {-46, -129}
, {-62, -96}
, {29, -45}
, {-112, 56}
, {-101, -89}
, {77, -71}
, {-46, -94}
, {-78, 39}
, {-13, 39}
, {57, -67}
, {-46, -185}
, {-49, -47}
, {7, -99}
, {-73, 72}
, {46, 76}
, {141, 144}
, {22, -14}
, {0, 66}
, {-150, 27}
, {-61, -13}
, {-28, 53}
, {101, -23}
, {-10, 74}
, {49, -72}
, {39, -36}
, {-51, -74}
, {58, -29}
, {-14, -128}
, {-130, -170}
, {-107, -17}
, {-24, 23}
}
, {{20, 133}
, {-19, -165}
, {24, 34}
, {-90, -96}
, {-41, 48}
, {86, -26}
, {-82, -49}
, {-115, -86}
, {-27, 8}
, {-62, 40}
, {-220, -152}
, {14, -39}
, {-41, 160}
, {-95, -48}
, {-28, 85}
, {-15, 32}
, {39, 8}
, {-11, 9}
, {86, 32}
, {-1, 27}
, {-48, -21}
, {-66, 48}
, {-183, -19}
, {30, 89}
, {204, -20}
, {61, 15}
, {57, -73}
, {-37, 2}
, {43, 70}
, {16, -49}
, {3, 23}
, {19, -78}
}
, {{-38, 74}
, {62, 108}
, {-217, 16}
, {0, -197}
, {-234, -385}
, {-30, 82}
, {8, 42}
, {121, 9}
, {-164, 18}
, {13, 5}
, {-248, -196}
, {-90, -45}
, {0, -185}
, {54, -8}
, {-47, -58}
, {30, 4}
, {99, -20}
, {-23, -62}
, {-137, -28}
, {85, 21}
, {-86, 43}
, {-80, -17}
, {4, -45}
, {85, 75}
, {-12, -111}
, {16, -73}
, {14, 9}
, {-120, 88}
, {7, -85}
, {0, 105}
, {18, -21}
, {1, 102}
}
, {{-5, -9}
, {-145, -46}
, {-244, 169}
, {-21, 46}
, {0, 93}
, {193, 33}
, {-48, 22}
, {-14, -75}
, {-23, 61}
, {-171, -46}
, {-94, -101}
, {-167, -3}
, {-58, -29}
, {24, -175}
, {-47, 56}
, {19, 21}
, {-248, -137}
, {42, -62}
, {-5, -66}
, {23, 97}
, {-41, 74}
, {-135, -151}
, {-39, 60}
, {42, -33}
, {129, -14}
, {9, 156}
, {41, -22}
, {-69, -13}
, {121, 127}
, {28, 102}
, {109, 89}
, {-88, -68}
}
, {{31, -34}
, {-30, -143}
, {25, 5}
, {78, 91}
, {-131, 109}
, {-28, 12}
, {161, -43}
, {-54, -32}
, {45, 85}
, {-66, 2}
, {-95, -150}
, {-23, -145}
, {-145, -108}
, {-45, 91}
, {-357, -172}
, {13, -46}
, {1, -47}
, {-118, -52}
, {71, -21}
, {-261, 107}
, {-27, -15}
, {-16, -193}
, {115, 165}
, {-67, -106}
, {126, 35}
, {-25, -22}
, {106, 65}
, {70, -3}
, {-268, -171}
, {55, -6}
, {12, 12}
, {-31, 0}
}
, {{71, 87}
, {-27, -96}
, {47, 40}
, {-76, 13}
, {-123, -121}
, {92, 120}
, {137, -43}
, {10, -37}
, {160, 55}
, {-97, -142}
, {-14, -46}
, {-1, 142}
, {-128, -21}
, {-12, -72}
, {11, 38}
, {-22, 32}
, {-94, -30}
, {-221, -240}
, {-2, -156}
, {29, 141}
, {-12, -80}
, {-222, -103}
, {-37, -99}
, {162, 1}
, {-4, 67}
, {5, 52}
, {-48, -65}
, {-128, -77}
, {-64, -58}
, {-68, 177}
, {-106, 8}
, {0, 22}
}
, {{19, -65}
, {-67, 19}
, {4, 72}
, {84, -130}
, {-81, -6}
, {-48, 65}
, {152, 132}
, {45, -78}
, {-119, 72}
, {-225, -45}
, {196, -231}
, {-165, -86}
, {-2, 93}
, {94, -64}
, {-65, 18}
, {34, -23}
, {155, -101}
, {87, -102}
, {5, -39}
, {-20, -20}
, {9, 22}
, {-34, -233}
, {-129, 22}
, {75, 57}
, {-65, 14}
, {-5, -126}
, {78, -56}
, {-13, 4}
, {5, -205}
, {-55, -145}
, {-114, 23}
, {52, -38}
}
, {{79, -1}
, {-65, -72}
, {-108, 15}
, {-135, -208}
, {92, -52}
, {-10, 95}
, {37, 287}
, {-6, 88}
, {47, -24}
, {112, -123}
, {24, 21}
, {119, 48}
, {-167, -134}
, {57, 36}
, {-30, -278}
, {35, 30}
, {-306, 34}
, {12, -48}
, {-77, -99}
, {0, -20}
, {7, 65}
, {0, 3}
, {-33, -236}
, {22, -72}
, {-51, 88}
, {-147, -271}
, {56, 76}
, {-68, -117}
, {50, 48}
, {-4, -39}
, {-39, -158}
, {-6, -80}
}
, {{59, -103}
, {61, 22}
, {10, -16}
, {-67, -65}
, {-115, -46}
, {204, 162}
, {48, -151}
, {30, 3}
, {71, 32}
, {28, -95}
, {-237, -74}
, {61, 133}
, {106, -20}
, {-3, -77}
, {30, -23}
, {-31, -344}
, {-113, -231}
, {-85, -527}
, {71, -42}
, {-87, -128}
, {-5, 8}
, {-145, 41}
, {66, -35}
, {-42, -11}
, {136, 29}
, {40, 10}
, {17, -33}
, {-153, -5}
, {-108, -15}
, {22, 9}
, {72, -93}
, {22, -6}
}
, {{-77, -236}
, {120, -36}
, {-29, -74}
, {-13, -32}
, {-10, 144}
, {-50, 12}
, {-137, -63}
, {30, -177}
, {-167, -165}
, {21, 61}
, {63, -31}
, {169, 60}
, {168, -110}
, {-104, -69}
, {11, 2}
, {-13, 0}
, {-25, -255}
, {-46, 35}
, {-47, -155}
, {85, 60}
, {-78, -24}
, {7, 36}
, {95, 31}
, {-269, -51}
, {31, 1}
, {60, 189}
, {37, -89}
, {-47, -26}
, {-59, -43}
, {96, -175}
, {-53, 126}
, {-70, 19}
}
, {{13, -10}
, {58, -36}
, {-185, 98}
, {107, -66}
, {40, 64}
, {70, 119}
, {-135, 215}
, {-103, 13}
, {-114, 114}
, {-5, -34}
, {121, -29}
, {110, 46}
, {-309, -59}
, {-17, 6}
, {-165, 3}
, {-5, -40}
, {-69, -126}
, {27, 12}
, {-4, 16}
, {54, 118}
, {-33, -92}
, {20, -167}
, {-17, 57}
, {0, -78}
, {125, 131}
, {-40, 46}
, {-45, -5}
, {41, -27}
, {104, 80}
, {-10, -60}
, {-47, 47}
, {-25, -78}
}
, {{65, 22}
, {86, 11}
, {183, 81}
, {-80, 41}
, {-70, 105}
, {155, 16}
, {93, 25}
, {167, -190}
, {96, 114}
, {-58, 64}
, {-325, 82}
, {54, -182}
, {-140, -221}
, {26, 8}
, {-95, -144}
, {-136, 35}
, {105, -3}
, {-145, 18}
, {-196, -244}
, {-58, 90}
, {-61, 15}
, {-11, -6}
, {71, -147}
, {-85, -199}
, {3, -225}
, {7, 105}
, {-128, -39}
, {-173, -50}
, {-64, 227}
, {-22, -137}
, {-46, 82}
, {73, -13}
}
, {{28, 115}
, {-14, -136}
, {-10, -2}
, {-6, -98}
, {-157, -129}
, {-15, -73}
, {53, -12}
, {5, 1}
, {141, -113}
, {65, 188}
, {-22, -32}
, {50, 46}
, {-149, -38}
, {136, -235}
, {15, -41}
, {-78, -111}
, {4, 202}
, {-15, -149}
, {-48, 60}
, {-12, -108}
, {-85, -85}
, {-209, 33}
, {-78, -268}
, {23, -25}
, {63, 97}
, {-253, -7}
, {-48, -1}
, {-203, -77}
, {123, 52}
, {80, 129}
, {-86, -90}
, {-62, 50}
}
, {{64, 74}
, {-48, 47}
, {-119, 26}
, {10, -8}
, {-192, -178}
, {-56, 131}
, {-174, -54}
, {-21, 8}
, {64, 54}
, {-54, -15}
, {49, -20}
, {6, 78}
, {2, -43}
, {0, -61}
, {-73, 44}
, {24, 24}
, {22, -69}
, {-46, -59}
, {21, 52}
, {12, 31}
, {5, 29}
, {9, -12}
, {37, 16}
, {-42, 57}
, {-45, -32}
, {157, -39}
, {-45, -89}
, {-17, 14}
, {85, -29}
, {-14, -22}
, {-106, -82}
, {68, -46}
}
, {{-47, 6}
, {-79, -98}
, {-251, -155}
, {63, -40}
, {123, -10}
, {30, 75}
, {-220, 185}
, {-14, -11}
, {20, -80}
, {73, -43}
, {76, -108}
, {-26, 12}
, {2, -48}
, {41, 0}
, {80, 30}
, {76, -180}
, {-226, -99}
, {2, -39}
, {14, 11}
, {58, -10}
, {64, -2}
, {-56, -227}
, {141, 14}
, {-3, 108}
, {1, -106}
, {-153, 43}
, {-127, -21}
, {2, -143}
, {144, 99}
, {40, 102}
, {13, -73}
, {-45, -181}
}
, {{-91, -47}
, {-194, 24}
, {198, 127}
, {20, 129}
, {73, 17}
, {-49, -124}
, {179, -14}
, {-133, 0}
, {-198, 7}
, {-171, 2}
, {13, 0}
, {-91, 96}
, {-28, 69}
, {-3, -16}
, {90, 10}
, {-116, 27}
, {187, 7}
, {-34, -142}
, {-404, -42}
, {49, 26}
, {-71, -4}
, {-57, -84}
, {-138, 126}
, {29, 33}
, {-195, -84}
, {-103, 136}
, {-60, 45}
, {-131, -107}
, {336, 52}
, {-244, -10}
, {-22, 64}
, {-93, 157}
}
, {{-3, 24}
, {49, 8}
, {-84, -109}
, {-60, 5}
, {-227, 47}
, {-255, -217}
, {-167, -1}
, {32, -127}
, {1, -21}
, {-21, -116}
, {-180, -13}
, {73, -20}
, {147, -69}
, {-84, -60}
, {71, -70}
, {-106, -130}
, {-136, 80}
, {-108, 28}
, {-29, -11}
, {60, -35}
, {-28, -82}
, {77, 32}
, {-203, 103}
, {207, -6}
, {-41, 49}
, {-215, 153}
, {13, 16}
, {49, 95}
, {66, -190}
, {99, -27}
, {28, -111}
, {-4, 193}
}
, {{46, 72}
, {26, -16}
, {177, -184}
, {-22, 18}
, {-51, 57}
, {8, -119}
, {138, 0}
, {-9, -8}
, {-65, 86}
, {-46, -109}
, {-71, -18}
, {-3, -47}
, {-73, -111}
, {-42, -53}
, {-57, -26}
, {-11, -41}
, {31, 88}
, {-139, 138}
, {-41, -106}
, {-140, 174}
, {-44, -183}
, {4, 109}
, {-120, 219}
, {42, 43}
, {-69, -80}
, {-38, -161}
, {-60, 128}
, {-62, 56}
, {78, -94}
, {47, 47}
, {77, 87}
, {13, 1}
}
, {{-68, 35}
, {0, 70}
, {-31, -346}
, {-18, -22}
, {76, -37}
, {-9, 20}
, {18, 34}
, {81, -65}
, {22, -19}
, {22, 16}
, {-35, 9}
, {-4, 80}
, {-64, 23}
, {61, 102}
, {43, 55}
, {21, -55}
, {-22, -97}
, {-105, -34}
, {70, -132}
, {32, 27}
, {-39, -19}
, {90, -93}
, {-86, 14}
, {-25, -182}
, {-8, -120}
, {26, 135}
, {-11, -46}
, {-1, -30}
, {-114, 20}
, {97, 49}
, {122, 13}
, {59, -73}
}
, {{20, 67}
, {-14, -127}
, {127, -23}
, {-94, -52}
, {-329, -48}
, {-47, -81}
, {-11, 59}
, {-32, 78}
, {-59, 34}
, {-138, -3}
, {-27, -32}
, {117, -46}
, {-65, 3}
, {-171, -196}
, {-100, -217}
, {-9, 63}
, {-28, 10}
, {-218, -140}
, {0, -62}
, {41, 97}
, {19, 60}
, {-52, 33}
, {-20, 38}
, {32, -31}
, {54, 133}
, {-80, 81}
, {-105, 122}
, {-26, 21}
, {3, 95}
, {-162, 33}
, {-241, -14}
, {11, -36}
}
, {{-144, 37}
, {-170, 26}
, {-94, -38}
, {14, 85}
, {-142, 162}
, {36, 142}
, {-95, -60}
, {-99, -118}
, {-89, 6}
, {-130, -125}
, {-53, -67}
, {-6, -46}
, {62, -63}
, {-52, -73}
, {-178, 51}
, {-17, 67}
, {109, 219}
, {63, -24}
, {-21, -121}
, {20, 21}
, {73, -46}
, {-64, 67}
, {63, 63}
, {-112, -53}
, {66, 17}
, {-70, -15}
, {12, -41}
, {-9, 54}
, {34, 55}
, {-102, -169}
, {-7, -65}
, {-143, -186}
}
, {{-122, -89}
, {149, -67}
, {-89, -15}
, {44, 103}
, {12, 108}
, {-56, -32}
, {-18, -92}
, {-130, -64}
, {58, -58}
, {-12, 21}
, {118, -37}
, {3, -7}
, {-38, -188}
, {-110, 13}
, {-72, -194}
, {6, -43}
, {10, -120}
, {60, 39}
, {-32, 18}
, {-115, 21}
, {60, 49}
, {65, -4}
, {-149, -23}
, {88, 72}
, {124, 15}
, {95, 34}
, {82, 28}
, {-29, -16}
, {-107, 106}
, {-82, -96}
, {-38, 117}
, {-57, 23}
}
, {{-23, -105}
, {106, 48}
, {164, -50}
, {-13, 167}
, {36, -210}
, {-27, 9}
, {-16, -6}
, {-35, -239}
, {18, -133}
, {50, 28}
, {-11, 30}
, {69, -49}
, {48, -9}
, {221, -204}
, {103, 12}
, {-130, -124}
, {20, -15}
, {-10, -324}
, {-24, 47}
, {94, 38}
, {-29, -73}
, {39, -123}
, {2, -51}
, {-115, 134}
, {-63, -124}
, {64, 79}
, {-6, 41}
, {-29, 21}
, {41, 61}
, {54, -52}
, {120, 52}
, {-3, -93}
}
, {{-226, 42}
, {-63, 35}
, {96, 157}
, {-58, -54}
, {-40, 24}
, {57, -26}
, {48, -112}
, {-45, -116}
, {57, 125}
, {-104, 102}
, {82, 59}
, {-48, 15}
, {88, 58}
, {-13, 91}
, {-42, -28}
, {-31, 47}
, {-8, -124}
, {34, -8}
, {-82, -54}
, {-69, 30}
, {0, -21}
, {59, -112}
, {-120, -58}
, {-3, 16}
, {-107, 85}
, {-187, -32}
, {-47, -160}
, {-109, -12}
, {66, 79}
, {138, 195}
, {48, 127}
, {13, -86}
}
, {{-166, -35}
, {98, 22}
, {-37, -201}
, {36, 32}
, {-49, -13}
, {-40, -221}
, {-196, 26}
, {70, -97}
, {40, 58}
, {-114, -203}
, {-123, -257}
, {17, -41}
, {-22, 110}
, {-52, 1}
, {-179, 15}
, {-8, -64}
, {97, -226}
, {-100, 46}
, {-104, 34}
, {8, -191}
, {-50, 37}
, {4, -59}
, {-95, -1}
, {94, -6}
, {101, 113}
, {16, -6}
, {-22, -115}
, {101, 11}
, {-62, -268}
, {3, 66}
, {-186, -169}
, {2, -136}
}
, {{-152, -19}
, {-130, 80}
, {-72, 42}
, {-178, 69}
, {157, -51}
, {-6, -137}
, {-21, -1}
, {-69, 55}
, {-159, -7}
, {-217, 24}
, {27, -29}
, {33, -46}
, {103, 43}
, {194, -48}
, {72, -30}
, {-178, -94}
, {-17, -130}
, {-221, 34}
, {-197, 84}
, {97, -91}
, {-243, -56}
, {-108, 74}
, {57, -18}
, {192, 54}
, {53, 41}
, {77, -37}
, {-102, 7}
, {-64, 60}
, {-33, -79}
, {137, 115}
, {-29, -38}
, {175, 1}
}
, {{2, -145}
, {43, -30}
, {129, 19}
, {53, 18}
, {-198, -106}
, {-99, -315}
, {117, 95}
, {55, -68}
, {6, 58}
, {-51, -310}
, {55, 127}
, {-32, 46}
, {-53, -202}
, {123, -126}
, {62, -272}
, {-128, -152}
, {144, 81}
, {-50, 123}
, {-88, -91}
, {-40, -206}
, {-29, -46}
, {42, 78}
, {154, 79}
, {214, 101}
, {-103, 69}
, {-148, -191}
, {-43, -2}
, {-10, 20}
, {130, 54}
, {-269, -2}
, {31, 116}
, {-10, -27}
}
, {{-34, -28}
, {37, 70}
, {-93, 58}
, {-232, -163}
, {-10, -23}
, {86, -14}
, {34, -7}
, {-81, 22}
, {89, 104}
, {-94, 100}
, {8, 31}
, {65, 13}
, {-12, 25}
, {74, -186}
, {-77, -6}
, {-64, -116}
, {90, -11}
, {-14, 178}
, {-40, -11}
, {-131, 13}
, {96, -66}
, {140, 5}
, {-135, 103}
, {-167, -64}
, {-3, -72}
, {-105, 10}
, {95, -143}
, {40, -157}
, {16, -38}
, {-77, -36}
, {-51, -83}
, {113, -54}
}
, {{-101, -24}
, {-152, -34}
, {47, 129}
, {24, -27}
, {3, 0}
, {-74, 0}
, {-76, -13}
, {-38, -38}
, {-51, 98}
, {-3, -8}
, {-33, 80}
, {-96, -174}
, {115, 29}
, {-4, 72}
, {-143, 7}
, {-25, 0}
, {59, 83}
, {-29, 78}
, {-80, -13}
, {-81, -122}
, {0, -18}
, {-80, 74}
, {143, -186}
, {-36, -21}
, {80, -34}
, {37, 118}
, {77, -79}
, {-148, 65}
, {85, -49}
, {-31, 89}
, {72, 40}
, {91, 54}
}
, {{22, 32}
, {-24, -50}
, {-243, -143}
, {48, -80}
, {70, 69}
, {-78, 193}
, {15, 90}
, {-23, -1}
, {78, 5}
, {-83, -29}
, {-68, -134}
, {42, 22}
, {70, -78}
, {-27, 32}
, {-128, 100}
, {-18, -29}
, {-409, 40}
, {-119, -89}
, {-131, 67}
, {3, 85}
, {-258, -87}
, {95, 65}
, {139, -11}
, {-115, -32}
, {-76, -31}
, {53, -109}
, {-17, -45}
, {-136, 84}
, {9, 38}
, {56, -13}
, {94, -80}
, {41, 16}
}
, {{32, 78}
, {64, 5}
, {-108, 15}
, {-20, 35}
, {-76, -94}
, {2, 75}
, {-93, -257}
, {39, 42}
, {37, 5}
, {73, 80}
, {-213, -140}
, {-123, 0}
, {-4, 23}
, {21, 40}
, {8, 66}
, {-67, -1}
, {-38, -94}
, {-65, -132}
, {-75, 46}
, {1, 62}
, {-6, -28}
, {-55, 11}
, {-26, 155}
, {-65, -95}
, {-18, 28}
, {-117, -62}
, {-36, -21}
, {37, 21}
, {-184, -132}
, {-142, 86}
, {-185, -61}
, {20, 24}
}
, {{96, -94}
, {111, 72}
, {-45, -15}
, {14, 13}
, {40, 79}
, {-8, 44}
, {-126, -110}
, {-123, -91}
, {73, -44}
, {-105, 38}
, {-114, 68}
, {-61, -84}
, {-17, 51}
, {-39, -39}
, {-159, -66}
, {-83, -165}
, {0, 18}
, {100, 137}
, {104, -73}
, {-89, 13}
, {-31, -9}
, {39, -173}
, {36, -184}
, {-31, -276}
, {95, 19}
, {110, 57}
, {-48, -94}
, {24, -6}
, {59, 164}
, {-78, -33}
, {8, 20}
, {6, -49}
}
, {{-186, -33}
, {-64, 104}
, {-199, -174}
, {-5, 100}
, {-115, 221}
, {69, 78}
, {-252, 10}
, {36, 6}
, {-22, -19}
, {-104, -69}
, {-97, 82}
, {57, 27}
, {92, 3}
, {-70, -42}
, {-234, -95}
, {24, -6}
, {-40, 97}
, {41, 128}
, {-61, 81}
, {-161, -40}
, {-39, 44}
, {3, -78}
, {-114, -181}
, {14, 111}
, {3, -75}
, {-167, 84}
, {87, -50}
, {-29, -172}
, {48, 26}
, {-123, -47}
, {-55, 4}
, {-86, -22}
}
, {{-11, -71}
, {-15, 80}
, {78, -22}
, {4, 137}
, {-94, 36}
, {87, 41}
, {-39, 53}
, {42, -18}
, {-15, 22}
, {-74, 2}
, {63, -79}
, {34, -106}
, {81, 38}
, {-55, -60}
, {32, -47}
, {-176, 55}
, {-46, 103}
, {-28, 85}
, {-53, -28}
, {149, 59}
, {-35, 5}
, {-139, 51}
, {1, -40}
, {96, 10}
, {-391, -76}
, {19, 109}
, {-143, 37}
, {-6, 90}
, {62, 68}
, {-26, -244}
, {-79, 62}
, {-15, -6}
}
, {{-55, -35}
, {29, -38}
, {135, 72}
, {8, 127}
, {-84, 169}
, {-28, 67}
, {-123, -72}
, {-32, 75}
, {-148, 31}
, {-73, -25}
, {78, -5}
, {-16, -53}
, {-22, -70}
, {-42, -92}
, {-144, -218}
, {-2, -44}
, {-61, 45}
, {-86, 92}
, {-20, 43}
, {-106, 77}
, {52, 23}
, {-20, -29}
, {57, -195}
, {37, -79}
, {44, 23}
, {-80, -63}
, {-131, 99}
, {-141, 42}
, {133, -208}
, {-11, -98}
, {19, 52}
, {13, -20}
}
, {{37, 47}
, {-31, 196}
, {39, 51}
, {-52, 24}
, {-7, -47}
, {-186, -175}
, {221, 0}
, {-132, -111}
, {15, 118}
, {-156, -192}
, {-31, -151}
, {-84, -121}
, {-46, 3}
, {-28, 10}
, {-13, -55}
, {60, 111}
, {69, 115}
, {51, -63}
, {-79, 62}
, {71, -58}
, {-150, -125}
, {31, -13}
, {15, -170}
, {40, 123}
, {-88, 47}
, {108, 20}
, {20, 151}
, {-6, -47}
, {-86, -231}
, {-112, -84}
, {-21, 45}
, {-163, -125}
}
, {{137, 78}
, {-99, -107}
, {-70, 59}
, {-162, 101}
, {8, -115}
, {43, -2}
, {-38, 70}
, {7, 3}
, {-4, 118}
, {-63, -10}
, {-65, -45}
, {-94, -156}
, {-141, 40}
, {-39, 35}
, {2, 98}
, {-46, -27}
, {7, 83}
, {-136, -51}
, {-245, -20}
, {-2, 7}
, {-146, -105}
, {-8, 104}
, {-99, 44}
, {-121, 59}
, {99, 74}
, {-95, -265}
, {60, 80}
, {-44, -23}
, {56, 66}
, {140, 174}
, {-12, -6}
, {0, -66}
}
, {{-30, -92}
, {-35, 32}
, {-35, -44}
, {185, 101}
, {43, -81}
, {104, 22}
, {-99, -175}
, {27, 78}
, {-56, -78}
, {17, 31}
, {27, -10}
, {71, 91}
, {0, 23}
, {-13, -64}
, {-85, -199}
, {-162, -378}
, {29, 19}
, {172, -236}
, {-9, -63}
, {-244, -353}
, {33, -57}
, {31, 100}
, {8, 16}
, {-86, -84}
, {75, -66}
, {-163, -92}
, {-77, -138}
, {-130, -110}
, {52, -31}
, {-81, 25}
, {27, 70}
, {57, 97}
}
, {{-111, -291}
, {-29, -44}
, {36, 99}
, {23, 72}
, {-54, -4}
, {53, -22}
, {-29, -33}
, {-2, -1}
, {-37, -8}
, {23, 64}
, {-123, 126}
, {36, -116}
, {-74, -66}
, {-19, -127}
, {74, -47}
, {51, -146}
, {6, -25}
, {-10, 111}
, {-12, -64}
, {-118, -102}
, {-11, 0}
, {88, 16}
, {-14, -97}
, {97, -167}
, {56, 38}
, {83, -76}
, {49, 0}
, {20, 89}
, {101, 111}
, {-108, -217}
, {-42, -78}
, {160, 120}
}
, {{-34, -73}
, {108, -236}
, {12, -1}
, {38, 42}
, {-84, 28}
, {129, -225}
, {-13, -68}
, {-82, -136}
, {75, -14}
, {-102, 125}
, {30, -173}
, {50, 59}
, {102, -11}
, {-76, -300}
, {-61, 42}
, {7, -72}
, {82, 37}
, {138, -36}
, {-15, -23}
, {-185, -45}
, {-143, -112}
, {124, 25}
, {33, 65}
, {49, -168}
, {24, -199}
, {-1, 31}
, {90, -65}
, {70, -15}
, {-197, 30}
, {121, -6}
, {-159, -56}
, {37, 82}
}
, {{-104, 127}
, {-44, -39}
, {141, -129}
, {-102, -123}
, {55, -4}
, {31, -5}
, {144, 50}
, {-48, -4}
, {28, -125}
, {71, 78}
, {-32, 61}
, {80, 64}
, {-56, -137}
, {21, 0}
, {-81, 37}
, {-6, 8}
, {-72, -14}
, {-43, 48}
, {-174, -61}
, {57, 106}
, {79, 6}
, {69, -22}
, {0, 66}
, {-151, -71}
, {-11, 136}
, {-176, 12}
, {68, 0}
, {45, -49}
, {33, 81}
, {116, 31}
, {-64, -153}
, {-60, 1}
}
, {{-83, -20}
, {-98, 23}
, {78, -203}
, {40, 88}
, {22, 25}
, {-2, 51}
, {119, 98}
, {-205, -303}
, {-13, -43}
, {149, 4}
, {-73, 2}
, {-458, -295}
, {49, 132}
, {-73, 26}
, {-53, -33}
, {63, -1}
, {-100, -149}
, {-35, -61}
, {-240, -8}
, {33, 7}
, {-23, -114}
, {15, 36}
, {36, 15}
, {101, 73}
, {-98, 122}
, {-104, -380}
, {-20, 63}
, {-79, -17}
, {-165, -13}
, {-27, -119}
, {71, 40}
, {-189, -174}
}
, {{15, -34}
, {51, 76}
, {-142, -5}
, {-46, -90}
, {-63, 25}
, {34, -3}
, {14, 21}
, {-46, -95}
, {-61, -22}
, {-88, 30}
, {164, 49}
, {249, 187}
, {-83, -3}
, {123, -3}
, {-38, -102}
, {18, -59}
, {-231, -158}
, {33, 45}
, {-39, -122}
, {57, 12}
, {-129, -93}
, {-16, 62}
, {-63, 107}
, {50, 57}
, {-19, -99}
, {49, 36}
, {3, -83}
, {-19, 87}
, {11, -10}
, {28, 0}
, {33, 22}
, {-79, -68}
}
, {{37, -83}
, {-56, -113}
, {-112, -135}
, {16, 11}
, {97, -47}
, {31, 34}
, {139, -93}
, {-42, -5}
, {33, 3}
, {-6, 20}
, {-139, 41}
, {70, -5}
, {23, 50}
, {-16, -60}
, {-28, 66}
, {6, 42}
, {18, -46}
, {-1, -38}
, {-58, -169}
, {51, 30}
, {-104, -52}
, {62, -34}
, {29, -153}
, {110, 9}
, {109, 179}
, {169, -91}
, {-4, -142}
, {49, -16}
, {-24, -163}
, {13, 82}
, {115, 67}
, {85, 1}
}
, {{-45, -38}
, {80, 53}
, {90, 51}
, {119, -42}
, {74, -70}
, {-39, -19}
, {-26, 86}
, {-62, -4}
, {-40, -92}
, {-53, 0}
, {0, -6}
, {-13, -7}
, {66, -64}
, {-71, -148}
, {-32, -111}
, {4, 30}
, {-100, 67}
, {-28, 103}
, {-82, -50}
, {154, 89}
, {-147, 10}
, {-22, 79}
, {98, -132}
, {-20, 48}
, {-38, -46}
, {158, 70}
, {-36, 108}
, {-38, 54}
, {102, 37}
, {-38, -25}
, {-10, -73}
, {34, 25}
}
, {{-66, -18}
, {-237, 36}
, {-27, -191}
, {-20, -22}
, {8, 41}
, {17, -55}
, {-107, 67}
, {-320, 30}
, {-55, -152}
, {-200, -44}
, {-46, -130}
, {-127, -215}
, {-220, 98}
, {-37, -88}
, {-48, -37}
, {0, -52}
, {18, 131}
, {63, 128}
, {36, 42}
, {-18, -79}
, {-37, -117}
, {28, 129}
, {-66, -51}
, {-36, 98}
, {11, -57}
, {35, 75}
, {46, -27}
, {51, 19}
, {43, -83}
, {-50, 129}
, {-23, 34}
, {-17, 62}
}
, {{108, 9}
, {-70, 18}
, {-169, 24}
, {-45, -257}
, {31, 91}
, {1, 59}
, {-44, 271}
, {-50, 34}
, {15, 152}
, {71, -96}
, {112, 1}
, {71, -24}
, {-68, 82}
, {43, 52}
, {-25, -216}
, {-21, -136}
, {-53, 63}
, {-11, 7}
, {-44, -29}
, {-6, 89}
, {0, 84}
, {2, -151}
, {-40, 10}
, {-36, -50}
, {141, -25}
, {25, -99}
, {-103, -91}
, {28, -172}
, {-21, -61}
, {109, -34}
, {-35, 21}
, {-109, -23}
}
, {{-29, 35}
, {63, 4}
, {69, 171}
, {0, -53}
, {-38, -31}
, {-92, 77}
, {91, 109}
, {-85, -70}
, {-38, 110}
, {14, 142}
, {-84, -59}
, {-50, 19}
, {56, 23}
, {-77, -9}
, {34, -57}
, {-68, -56}
, {-86, -45}
, {-206, -157}
, {-41, -94}
, {143, 123}
, {-16, -19}
, {41, -146}
, {-19, 110}
, {66, 52}
, {-26, 75}
, {-155, 124}
, {9, 61}
, {28, -113}
, {-86, 68}
, {-113, 104}
, {-80, -52}
, {81, 149}
}
, {{-1, 114}
, {-39, -65}
, {-1, 12}
, {119, 4}
, {-95, -195}
, {170, -187}
, {45, 128}
, {-164, 51}
, {-100, 24}
, {26, -121}
, {19, 74}
, {84, 1}
, {-62, 11}
, {-8, -23}
, {26, 79}
, {-61, 65}
, {12, 3}
, {-284, -119}
, {-92, -4}
, {-9, -36}
, {-8, -61}
, {89, 49}
, {24, 120}
, {-115, 135}
, {10, -213}
, {-18, -62}
, {-3, -149}
, {-18, -20}
, {86, -9}
, {161, -106}
, {-174, 6}
, {-258, 145}
}
, {{78, 28}
, {102, 29}
, {79, 19}
, {-176, -282}
, {39, -50}
, {181, 175}
, {90, 103}
, {50, -7}
, {100, -25}
, {-50, -17}
, {-107, -72}
, {-201, -203}
, {-30, -167}
, {-64, -67}
, {71, 79}
, {-78, -47}
, {62, 87}
, {15, -35}
, {13, -190}
, {43, -25}
, {-105, -79}
, {-29, -26}
, {82, -89}
, {39, 50}
, {78, 119}
, {56, -12}
, {-12, 65}
, {66, 13}
, {166, -11}
, {-222, -64}
, {-13, -89}
, {89, -62}
}
, {{43, -60}
, {84, 92}
, {36, 57}
, {-135, -140}
, {93, -94}
, {81, 134}
, {93, 84}
, {34, -40}
, {14, -232}
, {24, -46}
, {-79, -53}
, {26, -122}
, {-17, -21}
, {-36, -44}
, {-199, 46}
, {-112, -84}
, {61, -14}
, {-55, -173}
, {-107, -46}
, {28, 54}
, {-80, 25}
, {-46, 53}
, {-136, 26}
, {-1, 24}
, {32, 195}
, {34, 43}
, {-4, 117}
, {0, 66}
, {58, 62}
, {70, 129}
, {125, -115}
, {242, -104}
}
, {{52, 44}
, {-38, -27}
, {174, 33}
, {-172, -252}
, {40, 15}
, {102, 90}
, {-18, -11}
, {-54, -119}
, {53, 53}
, {-125, 13}
, {-199, -188}
, {-72, 42}
, {-5, 152}
, {-8, 8}
, {17, 95}
, {-44, 25}
, {-11, -49}
, {-81, -40}
, {-191, -189}
, {19, 33}
, {-8, -44}
, {0, -31}
, {-383, -47}
, {36, 154}
, {160, -5}
, {78, 53}
, {-24, -14}
, {-170, -30}
, {-257, 28}
, {87, -82}
, {20, 181}
, {43, 59}
}
, {{84, 72}
, {-80, 15}
, {-74, -10}
, {122, 109}
, {0, 20}
, {-110, 16}
, {126, 23}
, {60, -54}
, {-40, 114}
, {-235, -148}
, {-291, -36}
, {-97, -107}
, {65, -132}
, {-154, -189}
, {0, -181}
, {38, 32}
, {-27, 1}
, {6, 61}
, {45, 16}
, {36, -114}
, {-80, -158}
, {83, -204}
, {-18, 68}
, {20, -90}
, {33, -69}
, {-249, -53}
, {33, -119}
, {86, 5}
, {-118, -157}
, {-135, -165}
, {9, -130}
, {-7, 19}
}
, {{56, 42}
, {-123, 13}
, {-13, -68}
, {-45, -26}
, {-19, 105}
, {-226, -112}
, {60, 106}
, {-29, -24}
, {95, 139}
, {-68, 35}
, {-308, -59}
, {-126, -87}
, {28, 3}
, {59, -28}
, {-73, 75}
, {0, 47}
, {-37, -119}
, {-6, -8}
, {15, -45}
, {112, 51}
, {-171, -217}
, {-105, -62}
, {33, 0}
, {-86, 86}
, {112, 3}
, {96, 110}
, {-150, -85}
, {-85, -49}
, {-43, -172}
, {21, 0}
, {54, -6}
, {25, -128}
}
, {{-148, -60}
, {-133, 92}
, {93, 30}
, {58, 159}
, {-21, -11}
, {-129, -29}
, {134, -434}
, {-83, -61}
, {49, 82}
, {103, 69}
, {176, 12}
, {51, 79}
, {16, -98}
, {2, -79}
, {68, -243}
, {-25, 78}
, {-60, 115}
, {-23, -47}
, {-29, 2}
, {-17, 113}
, {-45, -94}
, {-59, 19}
, {-63, 11}
, {-38, -57}
, {-107, 30}
, {-98, -19}
, {-1, 11}
, {-77, -17}
, {151, -217}
, {-60, 54}
, {-198, -243}
, {8, 39}
}
, {{101, -76}
, {90, 21}
, {-156, -77}
, {-88, -67}
, {62, 85}
, {-264, -174}
, {67, -97}
, {2, -12}
, {-77, 97}
, {19, 52}
, {-38, 126}
, {-186, 68}
, {96, 55}
, {-69, -47}
, {-45, 10}
, {-124, 53}
, {12, -3}
, {-50, 11}
, {-82, -20}
, {-102, -40}
, {-159, -65}
, {-27, 65}
, {55, -57}
, {161, -42}
, {78, -82}
, {-149, 59}
, {-155, -23}
, {-74, 46}
, {71, 154}
, {119, -82}
, {67, 18}
, {-53, 75}
}
, {{-27, -49}
, {-5, 21}
, {-21, 114}
, {119, -128}
, {-156, -61}
, {-122, -104}
, {64, -252}
, {11, -99}
, {-67, 143}
, {-18, 0}
, {-23, -60}
, {-65, 28}
, {-139, -129}
, {-22, 145}
, {-63, 63}
, {3, 41}
, {-136, -73}
, {-119, -253}
, {66, 11}
, {-40, -104}
, {-12, -4}
, {67, -55}
, {-2, -103}
, {-165, -154}
, {53, -31}
, {-119, -30}
, {124, 31}
, {24, -59}
, {85, -334}
, {-9, -56}
, {-83, 143}
, {-39, 28}
}
, {{13, -26}
, {-72, 47}
, {-81, -231}
, {-96, 153}
, {-158, -21}
, {-9, -37}
, {-113, -135}
, {-47, 125}
, {-99, 64}
, {53, 88}
, {-117, 20}
, {59, 22}
, {-133, -193}
, {0, 15}
, {-112, -97}
, {-35, 53}
, {-8, -81}
, {33, -27}
, {-44, 63}
, {62, 63}
, {-43, 10}
, {0, 5}
, {-95, -157}
, {-45, 11}
, {-18, 0}
, {76, -124}
, {40, 50}
, {9, -42}
, {-120, -28}
, {-54, -33}
, {-72, -33}
, {30, 48}
}
, {{32, 124}
, {18, 0}
, {-101, -237}
, {-13, -210}
, {-77, -94}
, {95, -9}
, {15, 32}
, {-48, -186}
, {-21, 131}
, {-129, -40}
, {113, 45}
, {12, 113}
, {81, -147}
, {69, -14}
, {-84, -80}
, {16, -28}
, {-75, 15}
, {15, -85}
, {11, -137}
, {-68, -2}
, {22, 53}
, {-220, -11}
, {205, -171}
, {-84, 41}
, {38, -52}
, {32, 89}
, {-10, -14}
, {64, -56}
, {-77, -162}
, {161, 87}
, {71, 79}
, {-134, -7}
}
, {{10, 59}
, {-94, -6}
, {-31, -35}
, {-143, 40}
, {-115, 87}
, {56, 0}
, {89, 126}
, {41, -59}
, {-100, 7}
, {-30, 77}
, {-115, -71}
, {-94, -3}
, {7, -55}
, {-23, -32}
, {-8, 64}
, {-129, 148}
, {-217, -45}
, {-163, 46}
, {-28, 4}
, {72, 4}
, {-51, -34}
, {-74, 52}
, {7, 100}
, {62, 78}
, {-32, 68}
, {-7, 216}
, {-143, -20}
, {-28, 54}
, {-263, 14}
, {-11, -122}
, {-32, 99}
, {-24, 66}
}
, {{-110, -145}
, {-33, -76}
, {4, 22}
, {-244, 46}
, {28, 103}
, {119, -117}
, {-25, 70}
, {10, -38}
, {-97, 44}
, {-2, 34}
, {-108, 95}
, {-127, 21}
, {17, 48}
, {-91, -20}
, {-57, -12}
, {-75, -171}
, {-51, -9}
, {89, -264}
, {68, -75}
, {118, 93}
, {0, 45}
, {-128, -5}
, {0, 16}
, {-128, -97}
, {-146, -100}
, {-209, 59}
, {-40, 62}
, {-111, 126}
, {-45, 19}
, {37, 23}
, {-77, 108}
, {-66, -24}
}
, {{-34, 76}
, {43, 109}
, {13, -265}
, {-54, -64}
, {36, -89}
, {-86, -33}
, {-17, -93}
, {-91, -10}
, {57, 88}
, {-71, 40}
, {-179, -99}
, {74, -76}
, {97, 11}
, {-321, -19}
, {0, 23}
, {-147, 141}
, {0, 43}
, {-63, -102}
, {-17, -175}
, {4, 28}
, {-43, 102}
, {63, -71}
, {-9, -89}
, {42, 106}
, {3, 51}
, {17, -31}
, {-74, 21}
, {-90, -59}
, {-149, 13}
, {-16, 61}
, {28, -63}
, {-59, 30}
}
, {{82, 36}
, {77, -75}
, {5, 13}
, {125, 95}
, {12, 93}
, {-29, -55}
, {60, 26}
, {58, -50}
, {-61, 92}
, {-22, -20}
, {-79, 76}
, {-100, -78}
, {73, -8}
, {-25, 50}
, {-141, -124}
, {31, -129}
, {-67, -39}
, {-14, 47}
, {28, -91}
, {-6, -55}
, {-37, 5}
, {71, 72}
, {-48, -47}
, {-53, -389}
, {-7, -14}
, {-75, -154}
, {29, 54}
, {-39, 53}
, {23, 30}
, {131, -61}
, {-25, 42}
, {47, -19}
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


const int16_t dense_bias[FC_UNITS] = {-152, 26, 137, -18, 15}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{-32, -35, -107, -82, -96, 57, 49, 35, 0, 8, -82, -71, -24, -57, -48, 47, 24, 7, 94, 50, 19, -15, 105, 17, 73, -7, 23, -8, -37, -94, 143, 60, 49, 25, 58, 194, 41, 100, 71, 106, 53, 83, 54, 86, 66, -1, -1, -9, -29, -21, -3, 4, 1, -47, -66, -7, 8, 12, -17, 9, 78, 73, 44, 33, 51, 117, 155, 146, 198, 231, -19, 34, -44, -9, -28, -61, -52, 33, -44, -72, -101, -41, -86, -70, -12, -70, 6, -29, 2, -51, 50, 40, 39, -36, -35, -17, -73, -18, 0, -43, 58, 90, 72, 17, 77, -180, -123, -180, -168, -157, -6, -24, -5, -19, 23, 155, 107, -11, 38, 81, 110, 154, 108, 52, 44, 27, 11, -40, 5, -80, 9, 82, 7, 42, 61, -70, -13, -3, -23, -57, 31, 35, -2, 26, 13, 53, 36, 23, -19, -58, -58, -161, -1, -4, 2, 5, 48, 70, 6, -3, -136, -103, -5, 13, -28, -46, -79, -94, -72, -156, 44, -27, 40, 3, -14, 25, -24, 0, 34, 0, 0, 14, 35, -12, -18, 13, -8, 45, 45, 46, 2, -55, -73, -63, -78, -11, -24, -6, -83, -34, -20, 70, 107, 40, 47, -27, -29, -17, -7, 3, -134, -52, -127, -118, -123, -106, -128, -72, -102, -62, -115, 30, -54, -77, -81, 92, 78, 44, 53, 81, -6, 7, 3, -11, 57, -91, -63, -19, -44, -5, -4, 78, 49, -21, 49, 55, 25, 26, 86, 32, 33, -8, 21, 32, 128, 45, 31, 66, 105, 61, -91, -92, -60, -10, -61, -34, 48, -12, 29, 82, -118, -77, -129, -160, -146, 94, 121, 83, 129, 101, 11, 10, 40, 13, -8, 50, 3, 48, 52, 133, 24, 0, 6, 24, 41, 59, 69, 6, 47, 75, -127, -120, -129, -91, -116, -118, -79, -94, -10, -41, 38, 18, 32, 125, 145, -77, -97, -65, -32, -124}
, {67, 25, 99, 96, 28, -35, -66, -66, -114, -122, -74, -42, 42, -41, 14, 67, 15, 42, 50, 109, 113, 74, 106, 19, 62, 44, -14, 87, -10, 56, -199, -96, -214, -165, -135, 58, 72, -11, 29, 101, 50, 68, 41, 71, 26, -28, -66, -119, -129, -118, -124, -63, -20, -109, -99, 70, 80, 66, 68, 45, 297, 164, 170, 159, 185, 87, 90, 14, 114, -28, 39, 31, 61, 63, 10, 158, 128, 83, 86, 129, 84, -143, 145, -37, -45, 202, 85, 79, 122, 202, 26, -1, 27, 95, 52, 81, 46, -14, -63, 47, -21, 13, -70, -4, 16, -16, 14, 7, -28, 4, -121, -90, -70, -52, -55, -62, -98, -29, -35, -38, -113, -85, -64, -31, -5, -21, 52, -8, 49, -44, -93, -206, 58, -61, -178, -226, -19, -54, -119, -200, 68, 36, 54, 45, 134, 80, 36, -108, -31, -33, 87, 180, -81, 6, -21, -93, -17, 12, 36, 5, -191, 55, -247, -159, -137, -40, 32, -25, 19, -80, -56, 18, 42, -1, 4, 75, 17, 51, 13, 6, -39, -36, -91, -90, -148, -132, -164, -127, -167, -161, -16, -89, 34, 5, -98, -34, -84, -23, 39, 2, -185, 81, 17, -41, -52, 43, 65, 53, 29, 44, 58, 17, 79, 74, 31, 131, 99, 108, 98, 109, 68, 34, 9, 27, 19, 60, -91, -117, -50, -22, -204, -73, -144, -266, -309, 99, 81, 160, 89, 141, -68, -108, -43, -103, -89, -5, -66, -8, -19, -22, -117, -98, -42, -48, -29, -49, -62, -131, -63, -56, 29, 155, -11, 41, 19, -182, -155, -262, -150, -197, -8, 32, 70, 13, 54, -49, -109, 20, -111, -89, 133, 22, 40, 58, 98, -75, -24, -56, -58, 16, 45, 12, -37, -17, 22, -42, -96, 16, 33, 19, 86, 117, 39, 132, 120, -6, -54, -15, 27, -22, -29, -23, -25, -61, -84, 13, 16, 50, 64, 19}
, {65, 54, 21, 44, -13, 66, 65, 55, 54, 58, -33, 37, 0, 54, 58, 34, 16, -24, 58, 32, -179, -211, -133, -80, -68, 146, 73, 103, 62, 77, 51, 36, 43, 65, 75, -63, -40, -58, -21, -86, -119, -79, -159, -76, -22, 149, 85, 94, 74, 102, 151, 110, 10, 54, 27, -71, -40, -58, -5, -28, -233, -193, -174, -112, -70, -183, -184, -114, -247, -294, -64, 82, 29, -14, -25, -34, -40, 32, 4, -37, -224, -186, -295, -273, -328, 6, 1, -43, -12, -16, 124, 125, 162, 91, 155, -31, 34, 6, 30, 104, -19, 17, -16, -9, -40, -50, 50, 25, -13, 46, -106, -1, 25, 14, 34, 96, -22, 36, -48, -15, -105, -33, -76, -117, -157, 74, 61, 75, 84, 125, 8, 57, 4, 47, 152, 169, 178, 170, 216, 234, -114, -39, -68, -42, 12, -59, -12, -14, -31, -92, 141, 140, 214, 187, 97, -45, 49, 73, 15, 32, 132, -12, 179, 103, 137, 68, -20, -93, -94, -62, 25, 65, 7, 92, 70, -129, -58, -28, -175, -76, -29, -29, -17, -74, -77, 103, 146, 107, 127, 141, 0, 14, -1, 16, 33, -44, -25, -60, -31, -100, 45, 11, -96, -38, -97, 59, 25, 105, 54, 29, -201, -72, -151, -133, -87, -116, -134, -157, -137, -109, -12, 38, 65, -25, -65, 29, -16, -2, 3, -2, -6, -35, -24, 5, 9, 34, -38, -124, -120, -15, 58, 59, 59, 132, 96, 65, 37, 54, 87, 56, -74, -69, -71, -92, -110, -93, -112, -71, -76, 46, -13, -73, -42, -60, -52, 141, 160, 103, 81, 136, -104, -85, -97, -141, -150, -57, -62, -35, 147, 56, -102, -109, -121, -127, -172, 138, 64, 58, 115, 70, 1, -131, -88, -71, -97, 38, -14, -33, -56, -179, 12, -4, 63, 66, 83, 132, 109, 66, 13, 177, -133, -123, -150, -108, -78, 31, -42, -5, 43, -76}
, {6, 35, -9, 39, -23, 29, 23, 36, -2, -45, 33, -16, -32, -6, 32, -172, -174, -152, -133, -137, 25, -26, 38, 22, 83, -24, -1, 68, 17, 28, -88, 77, -33, 9, -37, 80, 28, -51, -73, -18, 45, 18, 0, 73, 37, 22, 23, 74, 18, 90, -93, -112, -162, -69, -61, 26, 22, 85, 18, 26, 13, -45, -27, -33, 25, 65, -76, -25, -64, -12, 31, -7, -47, 2, -57, -20, -80, -64, -77, -52, 12, 253, 96, 141, 126, -232, -31, -137, -34, -110, -27, -5, -21, 26, -50, -16, 24, -49, -24, -104, -52, -69, -32, -52, -57, 66, 36, 40, 35, 59, 127, 54, 78, 76, 108, -122, -30, 57, 56, 84, -24, 37, 114, 29, 37, -129, -141, -168, -13, 46, 354, 104, 153, 108, 112, -45, -44, 5, -33, -112, 96, 105, 54, -1, 31, 67, 4, 39, 58, 3, -276, -201, -152, -207, -249, -59, -71, -63, -101, -98, 175, 114, 118, 118, 113, -17, 45, 39, 49, 19, 77, -48, 20, 13, 60, 71, 102, 30, 42, 86, 106, 147, 159, 152, 210, -94, -137, -43, -43, -98, 127, 148, 189, 146, 99, 22, 56, 99, 42, 56, 153, -64, 0, 15, 96, 4, 10, 11, 0, 24, 16, 10, 21, 91, 40, -99, -101, -76, -107, -40, 4, 57, -14, -50, -39, 17, 53, 70, -11, 1, 75, 74, -14, 27, 30, -87, -47, 24, 45, -8, -96, -81, -128, -179, -156, -224, -159, -212, -239, -190, -68, 53, -18, 8, 28, -15, -105, -48, -66, -55, -114, -130, -213, -158, -208, -30, -20, -5, -8, 12, 85, 44, 79, 54, 58, -5, -38, -96, -63, 25, -23, -39, 26, 0, 37, 57, 60, -29, -21, -98, -37, -19, -24, -35, -17, -196, -143, -168, -140, -293, -7, 33, -4, -30, -1, 101, -26, -24, 14, 6, 94, 144, 80, 135, 102, 25, 23, -32, 32, -1}
, {-84, -75, -96, -30, -39, -1, -7, -10, -15, -9, 16, 57, 61, 10, 45, -89, -94, -94, -98, -105, -79, -105, -24, -20, -24, -92, -150, -91, -121, -153, 77, 84, 15, 69, 48, -151, -64, -116, -59, -32, -9, -117, -92, -67, -144, -32, -40, -12, -48, -28, 76, 41, -18, 19, 47, -95, -71, -137, -115, -76, -264, -201, -254, -195, -278, -121, -78, -181, -103, -65, 49, 41, 19, 25, 28, -13, -24, 17, -8, 72, 37, 39, 89, 104, 118, 173, 79, 25, 15, 52, -122, -158, -190, -110, -185, 44, -60, 21, 38, 1, 72, 50, 54, 69, 148, 107, 101, 120, 37, 93, 17, -7, -9, 2, -5, -190, -52, -159, -60, -190, -7, 75, 35, -93, -79, -50, 5, 27, 35, -41, -214, -17, -264, -97, -107, -60, -192, -8, 21, -10, -57, -75, -139, -140, -56, 17, 32, 79, 89, 50, -12, -14, -18, -40, -62, 34, 6, 50, 34, 40, 25, -206, 18, -154, -77, 79, 12, 73, 95, 129, -126, -116, -104, -161, -133, 29, -16, 39, 17, -37, 13, 7, 4, -33, -9, 45, 74, 22, 31, -29, -175, -108, -30, -93, -97, 57, 60, 21, 50, 52, -151, -32, -101, -67, -169, -249, -130, -110, -99, -114, 83, 50, 82, 54, 100, 46, 42, 82, 37, 61, 45, 98, 74, 46, 33, -83, -114, -58, -76, -108, 38, 72, 13, 73, 95, -53, 0, 23, -1, -64, 82, 18, 82, -1, 18, 25, 90, 75, 54, 24, 80, 44, 86, 57, 90, 73, 81, 93, 153, 149, 105, 78, 69, 110, 151, 52, 68, 32, 43, -3, 74, 114, 97, 77, 163, -36, 71, -39, 37, -47, 36, -13, -7, -23, -54, -98, -97, -108, -145, -32, -31, 0, 2, 23, -13, 9, 78, 78, 3, 52, -55, -68, -88, 2, -11, -6, -19, -24, -17, -20, -59, -6, -55, 14, -15, 117, 18, 7, 55, 95}
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
