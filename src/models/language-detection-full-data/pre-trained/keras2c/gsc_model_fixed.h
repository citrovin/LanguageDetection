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


const int16_t conv1d_bias[CONV_FILTERS] = {0, 0, 0, 0, 0, 0, 0, 0}
;

const int16_t conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{72, -31, -10, -59, -6, -9, -55, 16, -10, 64, 72, 88, -5, 44, 51, 3, 66, -52, 61, 4}
}
, {{74, -14, -67, -58, 13, -16, 72, 75, 12, -46, -47, -62, -56, -64, -38, 92, -85, 52, -58, 23}
}
, {{-93, 38, 57, 60, 29, -44, -48, 69, 91, 32, 74, 71, 84, -39, -73, -44, -35, -9, -9, -29}
}
, {{-87, 72, -14, -73, -58, 14, 29, -44, -52, 21, 5, -77, 68, 34, 41, -56, 51, -53, 88, -27}
}
, {{67, 91, 48, -1, -26, -70, -42, 71, 68, -16, -64, -29, 22, 90, 64, -61, 61, 59, -84, -63}
}
, {{-54, -37, 75, 47, 26, 0, -3, 56, 79, -26, 41, 23, -17, -1, 71, -11, -46, -65, 87, 17}
}
, {{-56, -66, -34, -59, 19, 84, -1, 86, 20, -24, 20, 24, -34, -44, 3, -61, 0, -25, -39, 27}
}
, {{29, 68, 86, -27, -16, -20, -44, 20, 49, -50, 40, 51, 0, 64, -76, -84, 36, -8, 90, 1}
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


const int16_t conv1d_1_bias[CONV_FILTERS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
;

const int16_t conv1d_1_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-34, -52, 52, -1, 73, -89, -51, -77}
, {-3, 54, 47, 30, -67, -56, -75, -53}
, {9, 24, 71, 24, 82, -39, -51, -39}
, {13, 25, 14, -79, 2, 38, 84, -87}
, {47, 49, 70, -60, -52, 3, 59, -85}
, {-60, 16, -79, -5, 42, 4, -69, 11}
, {39, 39, -16, -77, -1, -25, 33, -73}
, {-66, -29, -29, -24, -33, -39, 38, -65}
}
, {{59, -77, -69, 89, -73, 42, 30, -57}
, {-62, 53, -81, 44, -69, -64, 47, 79}
, {-70, -62, -13, -20, 62, -73, 76, -59}
, {53, 26, -61, 36, -74, -46, 58, -4}
, {60, 21, -21, 52, -34, -56, 62, 68}
, {-13, -51, 8, 14, 29, 79, -27, -80}
, {-3, 62, 71, 65, -74, 24, -24, 55}
, {82, 43, 71, -30, 29, 69, 2, -65}
}
, {{33, 69, -26, -58, -44, 26, 51, -41}
, {-23, 78, 2, 68, -39, -84, -60, 21}
, {26, 31, 33, -9, -89, -46, 55, 32}
, {-12, 83, 23, 36, 74, -35, -44, 30}
, {0, -45, -6, -49, -88, -37, -61, 21}
, {31, 5, -79, 11, 24, 27, -71, 15}
, {-51, 82, 78, 82, -87, -21, -69, -10}
, {26, -60, 33, 82, -68, -23, -40, -45}
}
, {{-89, -59, 1, -56, -51, -7, 6, 38}
, {-1, 78, 85, 62, 9, -53, -65, -73}
, {88, 73, -58, -44, -63, 76, 62, -30}
, {-55, -16, 40, 82, 16, -22, 85, -42}
, {33, 71, -42, 82, 47, 40, -2, 72}
, {-38, -81, -89, 45, -63, -56, 80, -3}
, {51, 72, -88, -75, 34, -8, 63, -16}
, {9, -15, 63, -1, 70, -56, -60, -44}
}
, {{52, -13, -60, -40, 12, -77, 21, -33}
, {12, 73, -30, -49, 16, -23, 46, 31}
, {57, -29, -15, 44, 47, -66, -60, -89}
, {44, 66, 61, -55, -26, -34, -33, -74}
, {-36, -39, 13, 23, -61, -79, 21, 24}
, {-15, -83, -80, 26, -1, 53, -63, 50}
, {78, 23, 44, -78, -33, 68, 14, 35}
, {-86, 51, -71, 79, -53, -1, 90, -57}
}
, {{-19, 87, 7, 21, 10, 39, 27, -62}
, {-52, -56, -84, -1, -65, 76, 65, 9}
, {37, -6, 60, 7, -44, -56, -29, -77}
, {-14, 69, 72, 22, -16, -79, 20, -23}
, {-42, 36, 77, 24, 69, 19, -50, 4}
, {44, -14, -34, 35, 0, -24, -8, 29}
, {11, -74, -30, 48, -39, 0, -14, -86}
, {43, -7, -90, 50, -75, -36, -77, -10}
}
, {{-37, -11, 13, 87, 82, -75, -69, 16}
, {-57, 0, -38, 48, 28, 75, 1, -16}
, {82, -35, -33, 13, 43, -66, -8, -28}
, {57, -40, 27, -72, 32, -74, 17, -20}
, {60, -43, 85, 62, -58, -76, 78, -83}
, {72, 52, -75, -82, 20, -74, -5, 14}
, {-4, 12, 72, 64, -47, -49, 70, -62}
, {-26, -13, 27, 46, -48, 16, 32, -56}
}
, {{89, -66, -60, -13, -83, 59, 83, -52}
, {90, 56, 15, 54, -73, -10, 65, -19}
, {-86, -11, -63, 21, -44, -36, -57, 45}
, {-54, 69, -73, -55, 57, -5, 83, 37}
, {23, -10, -42, -12, -58, 71, 5, 82}
, {81, -82, -79, 42, 53, -5, 70, -79}
, {-35, -38, 55, -39, -85, 49, -80, 86}
, {-36, 11, 31, 20, 14, 11, 61, -17}
}
, {{82, -69, 54, 62, 30, 87, 8, -78}
, {-23, 60, 46, 50, 5, 86, 22, 13}
, {78, 45, -47, 60, 74, -49, 37, -52}
, {26, -51, -78, 89, -79, 69, 55, -23}
, {-63, -6, -2, -44, 79, 41, 45, -11}
, {-73, 84, 18, 20, -53, 13, 45, -20}
, {7, -56, 46, 13, -1, -30, -54, 54}
, {34, -83, 16, 8, -15, -60, -72, -62}
}
, {{10, 66, -28, 8, 80, -58, 77, 30}
, {-21, -80, 37, 40, 85, 71, 12, -48}
, {27, 42, 11, 77, -16, -86, -41, -55}
, {-60, -37, 18, 19, -40, -15, -39, -47}
, {-5, -79, 27, -1, 30, -21, -7, -83}
, {80, -1, -13, 75, 8, 4, -87, 24}
, {-78, 30, -53, -64, 30, -7, -25, -12}
, {-52, 45, 0, 32, -8, 11, 14, 26}
}
, {{-60, -56, -34, 82, -34, 57, 72, -85}
, {49, -85, 10, 71, -84, 85, -74, 54}
, {-33, -8, -9, 68, 48, -35, 76, -81}
, {-9, -83, 4, -59, -19, -5, -19, -72}
, {35, -17, -75, 86, 88, 3, -64, 44}
, {34, -23, -35, -14, -71, -4, -7, -23}
, {24, 21, 55, -69, 69, 31, -80, 20}
, {-88, -57, 37, -74, 87, 72, 53, 26}
}
, {{-66, -29, 85, 29, -54, 73, -86, -76}
, {32, 29, -17, 53, -81, 36, 59, 38}
, {-90, -1, -16, 75, -79, 51, 5, 42}
, {19, 56, -37, 86, 9, -53, 31, -52}
, {83, 34, -84, -37, 54, 74, -71, 1}
, {88, -29, 48, 31, -11, -10, -38, -50}
, {25, -82, 11, -1, -56, 59, 34, 51}
, {-13, -34, -55, 18, -67, 72, 4, 16}
}
, {{-66, 39, -12, -51, 48, -58, -88, -7}
, {-18, -33, 61, 21, 52, 42, -37, -24}
, {76, -56, 19, -54, 50, -51, -60, -60}
, {57, 56, -23, 26, -69, -89, 35, 68}
, {79, -67, 29, 82, 72, 10, 41, -19}
, {89, -85, 11, -70, -55, 78, 15, -29}
, {-28, 0, -53, 28, -86, 48, 31, 38}
, {-75, -35, -45, -47, -22, 87, 57, -67}
}
, {{6, -18, 3, 73, 41, -16, -24, 61}
, {51, -74, 47, -86, 81, -11, 77, 72}
, {-21, 3, 1, 29, -64, -67, 78, -68}
, {-15, 3, 74, 67, -11, 3, 66, 27}
, {5, -49, 16, 8, 23, 40, 71, -73}
, {-53, 90, -90, 56, -16, -5, -5, 9}
, {62, 26, 45, -35, 24, -67, -33, 17}
, {-16, -74, -34, 53, -6, -81, -54, -71}
}
, {{12, -89, 64, -19, 88, 28, 57, -46}
, {32, -35, 7, 23, 69, 45, 24, -84}
, {49, 67, -34, -56, -65, 44, -73, 28}
, {-42, -67, 78, -55, 12, -57, 86, -24}
, {-52, -29, 72, -40, -10, 28, 63, 34}
, {-42, 3, 79, -21, 24, 42, 43, 71}
, {-51, -26, -74, -56, -59, -12, 47, -3}
, {13, 45, -10, -60, 55, 1, -68, -36}
}
, {{1, 73, -50, -62, -31, 2, 36, 66}
, {42, -90, 4, 10, 72, -88, 67, 7}
, {75, -5, -76, -30, 74, 68, -16, 32}
, {-55, 8, 56, -28, -66, -78, 54, -69}
, {-19, 5, -8, 63, 25, -6, 48, 82}
, {53, -53, -35, 37, 81, -36, 55, -20}
, {-71, -32, -2, 15, 78, 32, 61, -43}
, {0, -13, -56, 48, -59, -42, 17, 84}
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


const int16_t conv1d_2_bias[CONV_FILTERS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
;

const int16_t conv1d_2_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-34, 73, -57, -41}
, {9, 82, -81, -11}
, {47, -52, 54, -28}
, {39, -1, -65, 43}
, {-52, -89, -24, 54}
, {24, -39, 53, -66}
, {49, 3, 87, -2}
, {39, -25, -47, -8}
, {52, -51, -60, 82}
, {71, -51, 39, 52}
, {70, 59, -15, 46}
, {-16, 33, 27, -85}
, {-1, -77, 11, -11}
, {24, -39, -60, 51}
, {-60, -85, -8, 74}
, {-77, -73, 41, -17}
}
, {{59, -73, 73, 82}
, {-70, 62, 0, 78}
, {60, -34, 32, -44}
, {-3, -74, 42, -8}
, {-77, 42, 51, -53}
, {-62, -73, 3, 72}
, {21, -56, -62, 53}
, {62, 24, -80, 39}
, {-69, 30, -2, -64}
, {-13, 76, -82, -2}
, {-21, 62, -8, -58}
, {71, -24, 73, -23}
, {89, -57, 1, -71}
, {-20, -59, 29, 86}
, {52, 68, 0, 7}
, {65, 55, 46, -44}
}
, {{33, -44, 27, 80}
, {26, -89, -6, -22}
, {0, -88, 34, 28}
, {-51, -87, 0, -58}
, {69, 26, 74, 61}
, {31, -46, -39, 6}
, {-45, -37, -52, -10}
, {82, -21, 17, 73}
, {-26, 51, -49, -4}
, {33, 55, 42, -81}
, {-6, -61, 32, 90}
, {78, -69, -78, -6}
, {-58, -41, -86, 38}
, {-9, 32, -1, -37}
, {-49, 21, -22, 57}
, {82, -10, 36, 40}
}
, {{-89, -51, -17, 14}
, {88, -63, 58, -10}
, {33, 47, 14, -10}
, {51, 34, -67, -17}
, {-59, -7, -30, -54}
, {73, 76, 23, -69}
, {71, 40, 82, -74}
, {72, -8, -23, -80}
, {1, 6, -27, 23}
, {-58, 62, 83, 89}
, {-42, -2, -81, -75}
, {-88, 63, 12, 83}
, {-56, 38, -28, 64}
, {-44, -30, -13, 58}
, {82, 72, -55, 3}
, {-75, -16, -49, 3}
}
, {{52, 12, -42, 25}
, {57, 47, 19, -80}
, {-36, -61, 82, -22}
, {78, -33, 7, -8}
, {-13, -77, -18, 56}
, {-29, -66, -66, 0}
, {-39, -79, 50, -53}
, {23, 68, -63, -84}
, {-60, 21, -31, 88}
, {-15, -60, -78, -77}
, {13, 21, 32, -62}
, {44, 14, 36, 57}
, {-40, -33, 60, -9}
, {44, -89, 75, 62}
, {23, 24, -23, -70}
, {-78, 35, -46, -31}
}
, {{-19, 10, -45, 65}
, {37, -44, 40, 5}
, {-42, 69, -88, -37}
, {11, -39, 9, -82}
, {87, 39, -2, 87}
, {-6, -56, 0, 47}
, {36, 19, -76, 46}
, {-74, 0, 13, -49}
, {7, 27, -19, -88}
, {60, -29, 5, 73}
, {77, -50, -37, 12}
, {-30, -14, 10, 77}
, {21, -62, 59, -20}
, {7, -77, -65, -5}
, {24, 4, -46, -6}
, {48, -86, -19, -72}
}
, {{-37, 82, 40, -56}
, {82, 43, 3, 38}
, {60, -58, 10, 10}
, {-4, -47, -45, -83}
, {-11, -75, 15, -14}
, {-35, -66, 25, -51}
, {-43, -76, -53, -71}
, {12, -49, -41, -53}
, {13, -69, -27, -33}
, {-33, -8, 59, 52}
, {85, 78, -27, 15}
, {72, 70, 26, -53}
, {87, 16, 87, -42}
, {13, -28, -74, 7}
, {62, -83, 66, -42}
, {64, -62, -31, -31}
}
, {{89, -83, 86, 19}
, {-86, -44, -58, 33}
, {23, -58, 75, 53}
, {-35, -85, -30, 81}
, {-66, 59, 63, 86}
, {-11, -36, -3, 9}
, {-10, 71, -24, 67}
, {-38, 49, -28, 47}
, {-60, 83, 48, 21}
, {-63, -57, 41, 61}
, {-42, 5, -58, -31}
, {55, -80, 56, -82}
, {-13, -52, -75, -45}
, {21, 45, 16, 43}
, {-12, 82, 30, -29}
, {-39, 86, -19, 84}
}
, {{82, 30, -77, -82}
, {78, 74, 14, -67}
, {-63, 79, 40, 82}
, {7, -1, 70, 61}
, {-69, 87, -40, -49}
, {45, -49, -21, 22}
, {-6, 41, 85, -76}
, {-56, -30, -37, 85}
, {54, 8, -33, 39}
, {-47, 37, -3, 8}
, {-2, 45, -20, 25}
, {46, -54, 86, 56}
, {62, -78, -89, -55}
, {60, -52, -1, 40}
, {-44, -11, -64, -84}
, {13, 54, 43, -36}
}
, {{10, 80, 42, 54}
, {27, -16, -54, -86}
, {-5, 30, 76, -8}
, {-78, 30, 30, -64}
, {66, -58, 39, 36}
, {42, -86, -15, 45}
, {-79, -21, -31, -86}
, {30, -7, 29, 34}
, {-28, 77, -80, -83}
, {11, -41, 43, -72}
, {27, -7, -30, -42}
, {-53, -25, 61, 42}
, {8, 30, 87, 85}
, {77, -55, -12, 26}
, {-1, -83, -38, 30}
, {-64, -12, 8, -17}
}
, {{-60, -34, -10, -10}
, {-33, 48, -50, -80}
, {35, 88, 8, 62}
, {24, 69, 53, 66}
, {-56, 57, -4, -13}
, {-8, -35, -71, -11}
, {-17, 3, 70, 8}
, {21, 31, 22, 3}
, {-34, 72, -76, -37}
, {-9, 76, 3, 12}
, {-75, -64, 43, -31}
, {55, -80, 24, -32}
, {82, -85, 6, 33}
, {68, -81, -19, 0}
, {86, 44, -87, -57}
, {-69, 20, -32, -49}
}
, {{-66, -54, -31, 34}
, {-90, -79, -22, -70}
, {83, 54, -53, -56}
, {25, -56, 32, 46}
, {-29, 73, -22, -34}
, {-1, 51, 25, 4}
, {34, 74, 60, 12}
, {-82, 59, -22, -4}
, {85, -86, 78, 60}
, {-16, 5, -1, -85}
, {-84, -71, -7, -7}
, {11, 34, -79, -19}
, {29, -76, 32, -74}
, {75, 42, -58, 68}
, {-37, 1, 50, 0}
, {-1, 51, 30, -28}
}
, {{-66, 48, 88, 25}
, {76, 50, 32, 54}
, {79, 72, 22, -15}
, {-28, -86, 29, 38}
, {39, -58, -67, 18}
, {-56, -51, 7, 41}
, {-67, 10, -42, -72}
, {0, 48, 60, 72}
, {-12, -88, 63, -42}
, {19, -60, -88, 85}
, {29, 41, 57, 56}
, {-53, 31, 56, -9}
, {-51, -7, 45, -40}
, {-54, -60, 31, 40}
, {82, -19, -74, -25}
, {28, 38, 70, 70}
}
, {{6, 41, -52, 23}
, {-21, -64, -37, 51}
, {5, 23, -64, -80}
, {62, 24, -84, -10}
, {-18, -16, -24, -1}
, {3, -67, -51, -17}
, {-49, 40, 16, -25}
, {26, -67, 65, -10}
, {3, -24, 10, 52}
, {1, 78, -75, 24}
, {16, 71, -63, 43}
, {45, -33, 27, -85}
, {73, 61, 15, 3}
, {29, -68, 80, -67}
, {8, -73, -62, -83}
, {-35, 17, 25, -5}
}
, {{12, 88, 9, -75}
, {49, -65, 7, 62}
, {-52, -10, -22, 78}
, {-51, -59, -37, 64}
, {-89, 28, 30, -8}
, {67, 44, 84, -68}
, {-29, 28, -11, 5}
, {-26, -12, 35, 75}
, {64, 57, 58, -9}
, {-34, -73, -12, -61}
, {72, 63, 19, 77}
, {-74, 47, 76, 73}
, {-19, -46, 85, 10}
, {-56, 28, -10, -10}
, {-40, 34, 18, 7}
, {-56, -3, 5, 49}
}
, {{1, -31, -54, 52}
, {75, 74, -11, 18}
, {-19, 25, 17, -71}
, {-71, 78, 37, 4}
, {73, 2, 76, -40}
, {-5, 68, -66, -45}
, {5, -6, 25, 62}
, {-32, 32, -87, 65}
, {-50, 36, 29, 8}
, {-76, -16, 47, 8}
, {-8, 48, 78, 81}
, {-2, 61, 32, 29}
, {-62, 66, 58, -49}
, {-30, 32, -36, 51}
, {63, 82, 1, 3}
, {15, -43, -51, -52}
}
, {{-3, -67, 26, -75}
, {13, 2, -26, -66}
, {-60, 42, -78, -9}
, {-66, -33, -74, -63}
, {54, -56, -55, 56}
, {25, 38, -89, 2}
, {16, 4, -77, -45}
, {-29, -39, 64, 54}
, {47, -75, 18, -26}
, {14, 84, 58, -78}
, {-79, -69, -70, -77}
, {-29, 38, -5, -35}
, {30, -53, -28, -14}
, {-79, -87, -3, -31}
, {-5, 11, 89, 23}
, {-24, -65, 38, 70}
}
, {{-62, -69, -77, 62}
, {53, -74, 49, 39}
, {-13, 29, 4, -78}
, {82, 29, -89, -18}
, {53, -64, -23, -41}
, {26, -46, -79, 6}
, {-51, 79, 58, 69}
, {43, 69, 61, -4}
, {-81, 47, 13, -11}
, {-61, 58, 33, -73}
, {8, -27, 15, 18}
, {71, 2, 90, 21}
, {44, 79, -67, -69}
, {36, -4, 66, -85}
, {14, -80, 0, -21}
, {-30, -65, -52, 4}
}
, {{-23, -39, 80, 68}
, {-12, 74, 32, 37}
, {31, 24, -3, 45}
, {26, -68, 40, 33}
, {78, -84, 3, -16}
, {83, -35, 74, 67}
, {5, 27, -89, 0}
, {-60, -23, -22, -34}
, {2, -60, -45, -25}
, {23, -44, -20, 86}
, {-79, -71, 18, 83}
, {33, -40, -41, 35}
, {68, 21, 36, -32}
, {36, 30, 20, 46}
, {11, 15, -81, 67}
, {82, -45, 48, 49}
}
, {{-1, 9, -6, -22}
, {-55, 16, 86, 90}
, {-38, -63, -16, -44}
, {9, 70, 33, -66}
, {78, -53, 3, 45}
, {-16, -22, -7, -44}
, {-81, -56, 63, 30}
, {-15, -56, -37, 64}
, {85, -65, -62, -61}
, {40, 85, -68, -16}
, {-89, 80, 58, 45}
, {63, -60, -34, -5}
, {62, -73, -58, 75}
, {82, -42, -27, 81}
, {45, -3, 43, 3}
, {-1, -44, -77, 49}
}
, {{12, 16, -27, -6}
, {44, -26, 44, 17}
, {-15, -1, 47, 8}
, {-86, -53, -40, 3}
, {73, -23, 16, -14}
, {66, -34, -59, 73}
, {-83, 53, 70, -47}
, {51, -1, 47, -57}
, {-30, 46, 69, 0}
, {61, -33, -76, -27}
, {-80, -63, -32, -1}
, {-71, 90, 74, 29}
, {-49, 31, -39, -78}
, {-55, -74, -25, -82}
, {26, 50, 57, 36}
, {79, -57, -51, -5}
}
, {{-52, -65, 33, 28}
, {-14, -16, -89, -66}
, {44, 0, 76, 28}
, {43, -75, -78, -55}
, {-56, 76, 75, 60}
, {69, -79, 72, -65}
, {-14, -24, 83, -35}
, {-7, -36, -10, 10}
, {-84, 65, 81, 49}
, {72, 20, -31, 0}
, {-34, -8, -23, -46}
, {-90, -77, -4, -51}
, {-1, 9, 0, 52}
, {22, -23, 16, 43}
, {35, 29, -46, 0}
, {50, -10, -27, 87}
}
, {{-57, 28, -16, 83}
, {57, 32, -13, 73}
, {72, 20, 84, -37}
, {-26, -48, 46, 79}
, {0, 75, -57, 45}
, {-40, -74, 12, -76}
, {52, -74, -22, 26}
, {-13, 16, 67, -18}
, {-38, 1, 47, -90}
, {27, 17, -13, 55}
, {-75, -5, 47, 7}
, {27, 32, -21, -50}
, {48, -16, -76, -42}
, {-72, -20, 72, 0}
, {-82, 14, 23, -11}
, {46, -56, 9, -21}
}
, {{90, -73, -7, -12}
, {-54, 57, -55, -44}
, {81, 53, -32, -79}
, {-36, 14, -8, 10}
, {56, -10, -52, 21}
, {69, -5, -39, 45}
, {-82, -5, -23, -78}
, {11, 11, -1, -86}
, {15, 65, 61, 31}
, {-73, 83, -71, 90}
, {-79, 70, 45, 65}
, {31, 61, 0, -50}
, {54, -19, -75, 56}
, {-55, 37, 34, -40}
, {42, -79, -80, 66}
, {20, -17, 48, 53}
}
, {{-23, 5, 84, -5}
, {26, -79, 48, -56}
, {-73, -53, -52, -6}
, {34, -15, 43, -38}
, {60, 86, -4, 0}
, {-51, 69, -70, -37}
, {84, 13, -59, -57}
, {-83, -60, -11, -28}
, {46, 22, -58, 46}
, {-78, 55, -22, 35}
, {18, 45, 54, 81}
, {16, -72, -10, 6}
, {50, 13, -29, -10}
, {89, -23, -38, 58}
, {20, -20, -42, -61}
, {8, -62, -90, -76}
}
, {{-21, 85, 62, 15}
, {-60, -40, -81, 88}
, {80, 8, 6, 50}
, {-52, -8, 55, -47}
, {-80, 71, -83, -26}
, {-37, -15, 68, 42}
, {-1, 4, 23, 87}
, {45, 11, 57, 31}
, {37, 12, -40, -5}
, {18, -39, 43, 34}
, {-13, -87, -2, 6}
, {0, 14, 45, 35}
, {40, -48, -71, 1}
, {19, -47, 32, 84}
, {75, 24, 37, 31}
, {32, 26, -13, -28}
}
, {{49, -84, 45, -4}
, {-9, -19, 28, 42}
, {34, -71, 14, 81}
, {-88, 87, 16, 36}
, {-85, 85, -87, 9}
, {-83, -5, -4, 69}
, {-23, -4, -90, 11}
, {-57, 72, 67, 60}
, {10, -74, 41, -30}
, {4, -19, 26, -47}
, {-35, -7, -65, 19}
, {37, 53, -63, 58}
, {71, 54, 26, -14}
, {-59, -72, -83, -3}
, {-14, -23, 78, 64}
, {-74, 26, 30, 70}
}
, {{32, -81, 79, 4}
, {19, 9, -52, -32}
, {88, -11, 23, 4}
, {-13, -67, 8, 90}
, {29, 36, 62, -74}
, {56, -53, -64, 72}
, {-29, -10, 38, 16}
, {-34, 72, 34, 90}
, {-17, 59, -62, 57}
, {-37, 31, 34, 85}
, {48, -38, 58, -80}
, {-55, 4, -85, 17}
, {53, 38, -17, 6}
, {86, -52, 76, 67}
, {31, -50, 44, -1}
, {18, 16, -1, -32}
}
, {{-18, 52, 22, -69}
, {57, -69, -89, 71}
, {89, -55, -54, -25}
, {-75, -22, -75, -5}
, {-33, 42, -35, 87}
, {56, -89, -59, 23}
, {-85, 78, -16, 29}
, {-35, 87, 27, -66}
, {61, -37, 33, -56}
, {-23, 35, 31, -79}
, {11, 15, 65, -66}
, {-45, 57, -45, 72}
, {21, -24, -21, 20}
, {26, 68, -37, -19}
, {-70, -29, -49, 86}
, {-47, -67, 8, 85}
}
, {{51, 81, 28, 87}
, {-15, -11, 64, -82}
, {-53, -16, 73, 57}
, {-16, -6, 59, -29}
, {-74, -11, 25, 43}
, {3, 3, 78, -29}
, {90, -5, 84, -82}
, {-74, -81, 58, 45}
, {47, 77, -53, -47}
, {74, 66, -11, 71}
, {-90, -5, -83, 0}
, {-34, -54, -33, 30}
, {-86, 72, 40, 33}
, {67, 27, -80, -84}
, {56, 9, -54, 13}
, {53, -71, 56, -20}
}
, {{32, 69, -47, -47}
, {-42, 12, 69, -23}
, {-42, 24, -81, 61}
, {13, 55, -79, 61}
, {-35, 45, -64, -60}
, {-67, -57, 2, 33}
, {3, 42, -19, -38}
, {45, 1, 79, 85}
, {7, 24, -65, 82}
, {78, 86, -3, -28}
, {79, 43, 27, 38}
, {-10, -68, -44, 70}
, {23, -84, 16, -37}
, {-55, -24, -55, -74}
, {-21, 71, -84, 54}
, {-60, -36, 63, -15}
}
, {{42, 72, -9, 13}
, {-55, -66, 2, -17}
, {53, 81, 26, -68}
, {0, -59, 32, -90}
, {-90, -88, -73, -13}
, {8, -78, 19, -1}
, {-53, -36, -1, 37}
, {-13, -42, 28, 74}
, {4, 67, 70, -79}
, {56, 54, -65, -14}
, {-35, 55, -19, -56}
, {-56, 17, 87, 75}
, {10, 7, -40, -45}
, {-28, -69, -68, 26}
, {37, -20, -21, -19}
, {48, 84, 34, 88}
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


const int16_t conv1d_3_bias[CONV_FILTERS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
;

const int16_t conv1d_3_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-34, 47}
, {47, -46}
, {-52, 40}
, {49, -1}
, {52, -26}
, {70, -66}
, {-1, 73}
, {-60, 66}
, {73, -40}
, {-52, 28}
, {-89, -57}
, {3, -40}
, {-51, -10}
, {59, 75}
, {-77, 57}
, {-85, 43}
, {-57, -65}
, {54, 29}
, {-24, 35}
, {87, 53}
, {-60, -11}
, {-15, 88}
, {11, 57}
, {-8, 54}
, {-41, -47}
, {-28, 13}
, {54, -22}
, {-2, 77}
, {82, 7}
, {46, 4}
, {-11, -42}
, {74, 75}
}
, {{59, 87}
, {60, -28}
, {-77, 52}
, {21, -88}
, {-69, 86}
, {-21, -35}
, {89, -18}
, {52, -68}
, {-73, -10}
, {-34, 45}
, {42, -35}
, {-56, -10}
, {30, -81}
, {62, -43}
, {-57, 26}
, {68, 90}
, {73, 12}
, {32, -25}
, {51, 77}
, {-62, 11}
, {-2, -1}
, {-8, 78}
, {1, -77}
, {0, -28}
, {82, -73}
, {-44, -16}
, {-53, 50}
, {53, -42}
, {-64, -48}
, {-58, 3}
, {-71, -22}
, {7, 35}
}
, {{33, 76}
, {0, 14}
, {69, 89}
, {-45, -54}
, {-26, 22}
, {-6, -82}
, {-58, 6}
, {-49, -43}
, {-44, 75}
, {-88, -10}
, {26, 49}
, {-37, -70}
, {51, 83}
, {-61, 25}
, {-41, -62}
, {21, 69}
, {27, -35}
, {34, 81}
, {74, 85}
, {-52, 60}
, {-49, 5}
, {32, 81}
, {-86, 59}
, {-22, 82}
, {80, 73}
, {28, 53}
, {61, -52}
, {-10, -16}
, {-4, -24}
, {90, 0}
, {38, 1}
, {57, 50}
}
, {{-89, 56}
, {33, -24}
, {-59, 64}
, {71, -51}
, {1, -85}
, {-42, 6}
, {-56, 64}
, {82, 34}
, {-51, -10}
, {47, 79}
, {-7, 2}
, {40, -71}
, {6, 38}
, {-2, 73}
, {38, -27}
, {72, -12}
, {-17, 35}
, {14, -42}
, {-30, 28}
, {82, -20}
, {-27, 4}
, {-81, 11}
, {-28, 33}
, {-55, 44}
, {14, -24}
, {-10, -23}
, {-54, 0}
, {-74, 51}
, {23, 24}
, {-75, -10}
, {64, -70}
, {3, 36}
}
, {{52, -42}
, {-36, 18}
, {-13, 32}
, {-39, 11}
, {-60, 46}
, {13, -2}
, {-40, -58}
, {23, -50}
, {12, -17}
, {-61, 16}
, {-77, -58}
, {-79, -57}
, {21, -8}
, {21, -48}
, {-33, 33}
, {24, 24}
, {-42, 35}
, {82, -69}
, {-18, -8}
, {50, 28}
, {-31, 48}
, {32, -88}
, {60, -54}
, {-23, 9}
, {25, -60}
, {-22, -9}
, {56, -62}
, {-53, -34}
, {88, 51}
, {-62, 59}
, {-9, -30}
, {-70, 46}
}
, {{-19, -45}
, {-42, -63}
, {87, 55}
, {36, 77}
, {7, 67}
, {77, 26}
, {21, 42}
, {24, 57}
, {10, 19}
, {69, 70}
, {39, -65}
, {19, -6}
, {27, -53}
, {-50, 47}
, {-62, -31}
, {4, -39}
, {-45, 50}
, {-88, -44}
, {-2, 1}
, {-76, 39}
, {-19, -88}
, {-37, -10}
, {59, 71}
, {-46, 16}
, {65, -53}
, {-37, 3}
, {87, 69}
, {46, 68}
, {-88, 40}
, {12, -9}
, {-20, -15}
, {-6, -57}
}
, {{-37, 43}
, {60, 60}
, {-11, -78}
, {-43, -2}
, {13, -31}
, {85, 71}
, {87, 51}
, {62, -56}
, {82, 0}
, {-58, 51}
, {-75, -20}
, {-76, 71}
, {-69, -52}
, {78, 10}
, {16, 56}
, {-83, 13}
, {40, -29}
, {10, -17}
, {15, -58}
, {-53, -73}
, {-27, -59}
, {-27, 20}
, {87, -21}
, {66, -5}
, {-56, 60}
, {10, 74}
, {-14, 44}
, {-71, 12}
, {-33, 87}
, {15, 35}
, {-42, 0}
, {-42, 11}
}
, {{89, -1}
, {23, -5}
, {-66, 22}
, {-10, -23}
, {-60, -70}
, {-42, 11}
, {-13, -54}
, {-12, -71}
, {-83, -79}
, {-58, -10}
, {59, -86}
, {71, 58}
, {83, 46}
, {5, 34}
, {-52, 22}
, {82, 43}
, {86, 17}
, {75, -3}
, {63, -34}
, {-24, 30}
, {48, 53}
, {-58, -60}
, {-75, 73}
, {30, -7}
, {19, 57}
, {53, 42}
, {86, 2}
, {67, -72}
, {21, -29}
, {-31, -87}
, {-45, 0}
, {-29, -18}
}
, {{82, -58}
, {-63, 27}
, {-69, -7}
, {-6, -79}
, {54, -41}
, {-2, -12}
, {62, 88}
, {-44, 63}
, {30, 15}
, {79, -27}
, {87, -29}
, {41, 70}
, {8, -25}
, {45, 83}
, {-78, 51}
, {-11, 28}
, {-77, 79}
, {40, 70}
, {-40, 74}
, {85, -81}
, {-33, 17}
, {-20, 54}
, {-89, 13}
, {-64, -8}
, {-82, 68}
, {82, 71}
, {-49, 0}
, {-76, -65}
, {39, 35}
, {25, -36}
, {-55, 39}
, {-84, 2}
}
, {{10, -66}
, {-5, 35}
, {66, -84}
, {-79, 54}
, {-28, 52}
, {27, -86}
, {8, -53}
, {-1, -28}
, {80, 80}
, {30, 36}
, {-58, 65}
, {-21, 71}
, {77, -38}
, {-7, 26}
, {30, 10}
, {-83, -79}
, {42, 61}
, {76, -74}
, {39, -22}
, {-31, -65}
, {-80, 67}
, {-30, 60}
, {87, 39}
, {-38, -67}
, {54, -5}
, {-8, -65}
, {36, 64}
, {-86, -45}
, {-83, -45}
, {-42, 64}
, {85, 1}
, {30, 18}
}
, {{-60, -61}
, {35, 81}
, {-56, 27}
, {-17, 29}
, {-34, 81}
, {-75, -11}
, {82, -61}
, {86, -48}
, {-34, -30}
, {88, -20}
, {57, -9}
, {3, -70}
, {72, -85}
, {-64, -21}
, {-85, -2}
, {44, 27}
, {-10, -68}
, {8, 10}
, {-4, -90}
, {70, 66}
, {-76, -79}
, {43, 85}
, {6, -33}
, {-87, -38}
, {-10, 20}
, {62, 0}
, {-13, 2}
, {8, 60}
, {-37, -80}
, {-31, 12}
, {33, 72}
, {-57, -77}
}
, {{-66, -63}
, {83, 14}
, {-29, 74}
, {34, -71}
, {85, 42}
, {-84, -8}
, {29, 38}
, {-37, 29}
, {-54, 55}
, {54, -35}
, {73, -9}
, {74, 62}
, {-86, -46}
, {-71, 15}
, {-76, 77}
, {1, 48}
, {-31, -67}
, {-53, -36}
, {-22, -4}
, {60, 32}
, {78, -44}
, {-7, -67}
, {32, 71}
, {50, -50}
, {34, 75}
, {-56, 69}
, {-34, -25}
, {12, 8}
, {60, 58}
, {-7, -84}
, {-74, 77}
, {0, 1}
}
, {{-66, -38}
, {79, 65}
, {39, 22}
, {-67, 0}
, {-12, -30}
, {29, 7}
, {-51, 75}
, {82, -69}
, {48, 65}
, {72, -51}
, {-58, -59}
, {10, -40}
, {-88, -70}
, {41, 35}
, {-7, 0}
, {-19, -45}
, {88, 42}
, {22, 81}
, {-67, -37}
, {-42, 19}
, {63, 89}
, {57, 41}
, {45, -28}
, {-74, -62}
, {25, -46}
, {-15, -1}
, {18, -76}
, {-72, -13}
, {-42, 64}
, {56, -61}
, {-40, 52}
, {-25, -87}
}
, {{6, 63}
, {5, -57}
, {-18, 52}
, {-49, -69}
, {3, -13}
, {16, -31}
, {73, 30}
, {8, -28}
, {41, -61}
, {23, 90}
, {-16, 81}
, {40, 46}
, {-24, 59}
, {71, -2}
, {61, -75}
, {-73, -46}
, {-52, 3}
, {-64, -16}
, {-24, 49}
, {16, -56}
, {10, 36}
, {-63, 42}
, {15, -59}
, {-62, -77}
, {23, 70}
, {-80, 31}
, {-1, 45}
, {-25, -34}
, {52, -44}
, {43, 63}
, {3, 72}
, {-83, 66}
}
, {{12, 4}
, {-52, 55}
, {-89, 84}
, {-29, -90}
, {64, -69}
, {72, 76}
, {-19, 61}
, {-40, -27}
, {88, 43}
, {-10, 22}
, {28, -81}
, {28, 44}
, {57, 17}
, {63, 13}
, {-46, -19}
, {34, 75}
, {9, -83}
, {-22, -16}
, {30, 63}
, {-11, -5}
, {58, -39}
, {19, -70}
, {85, 81}
, {18, 33}
, {-75, -57}
, {78, -78}
, {-8, -81}
, {5, -20}
, {-9, 41}
, {77, 5}
, {10, 81}
, {7, -29}
}
, {{1, 68}
, {-19, -90}
, {73, -39}
, {5, 73}
, {-50, -73}
, {-8, 79}
, {-62, -21}
, {63, -21}
, {-31, -88}
, {25, -24}
, {2, -42}
, {-6, -21}
, {36, -82}
, {48, 48}
, {66, 86}
, {82, -81}
, {-54, 41}
, {17, -37}
, {76, 35}
, {25, -61}
, {29, 0}
, {78, -34}
, {58, 38}
, {1, -13}
, {52, -37}
, {-71, 26}
, {-40, -85}
, {62, 0}
, {8, 90}
, {81, 78}
, {-49, -33}
, {3, -53}
}
, {{-3, -34}
, {-60, 19}
, {54, -87}
, {16, 65}
, {47, 22}
, {-79, 59}
, {30, 78}
, {-5, -57}
, {-67, -20}
, {42, -13}
, {-56, 80}
, {4, 21}
, {-75, -59}
, {-69, -68}
, {-53, -39}
, {11, 13}
, {26, 46}
, {-78, 5}
, {-55, -4}
, {-77, -15}
, {18, 41}
, {-70, 32}
, {-28, -44}
, {89, -14}
, {-75, 89}
, {-9, 25}
, {56, -35}
, {-45, -86}
, {-26, 33}
, {-77, 67}
, {-14, 67}
, {23, 85}
}
, {{-62, 54}
, {-13, -2}
, {53, -53}
, {-51, -5}
, {-81, -49}
, {8, -10}
, {44, -5}
, {14, -54}
, {-69, 23}
, {29, -88}
, {-64, 41}
, {79, -26}
, {47, -74}
, {-27, 23}
, {79, -57}
, {-80, 38}
, {-77, 31}
, {4, 47}
, {-23, 16}
, {58, 46}
, {13, -12}
, {15, -69}
, {-67, 15}
, {0, 13}
, {62, -11}
, {-78, -26}
, {-41, 67}
, {69, 57}
, {-11, 48}
, {18, -34}
, {-69, 62}
, {-21, 66}
}
, {{-23, 19}
, {31, -40}
, {78, -25}
, {5, -47}
, {2, -38}
, {-79, 56}
, {68, 88}
, {11, -48}
, {-39, 64}
, {24, -53}
, {-84, 5}
, {27, -11}
, {-60, -4}
, {-71, 6}
, {21, 29}
, {15, -8}
, {80, 38}
, {-3, 73}
, {3, -82}
, {-89, 51}
, {-45, -76}
, {18, 74}
, {36, 84}
, {-81, -44}
, {68, -23}
, {45, -77}
, {-16, 79}
, {0, 40}
, {-25, 11}
, {83, -57}
, {-32, 45}
, {67, -17}
}
, {{-1, 35}
, {-38, 61}
, {78, -49}
, {-81, 11}
, {85, 87}
, {-89, 64}
, {62, 70}
, {45, -16}
, {9, -43}
, {-63, 36}
, {-53, 46}
, {-56, 50}
, {-65, 55}
, {80, 19}
, {-73, -1}
, {-3, -36}
, {-6, -7}
, {-16, -24}
, {3, -36}
, {63, 59}
, {-62, -39}
, {58, -73}
, {-58, 16}
, {43, 0}
, {-22, -41}
, {-44, 58}
, {45, -61}
, {30, -48}
, {-61, 80}
, {45, 62}
, {75, -30}
, {3, -68}
}
, {{12, 59}
, {-15, -69}
, {73, -2}
, {-83, -63}
, {-30, 70}
, {-80, 9}
, {-49, -53}
, {26, 77}
, {16, 83}
, {-1, 44}
, {-23, 43}
, {53, -81}
, {46, 15}
, {-63, -73}
, {31, -34}
, {50, 53}
, {-27, -62}
, {47, 3}
, {16, -32}
, {70, 1}
, {69, -28}
, {-32, 26}
, {-39, 86}
, {57, 84}
, {-6, -7}
, {8, -29}
, {-14, 23}
, {-47, -47}
, {0, 2}
, {-1, 64}
, {-78, -37}
, {36, 7}
}
, {{-52, 49}
, {44, 24}
, {-56, 0}
, {-14, 89}
, {-84, 11}
, {-34, -2}
, {-1, -12}
, {35, -18}
, {-65, -13}
, {0, -3}
, {76, -49}
, {-24, -52}
, {65, 21}
, {-8, 55}
, {9, 87}
, {29, 88}
, {33, 60}
, {76, 42}
, {75, -71}
, {83, 72}
, {81, -17}
, {-23, 28}
, {0, 31}
, {-46, -66}
, {28, 61}
, {28, -29}
, {60, -51}
, {-35, 2}
, {49, 61}
, {-46, -48}
, {52, 28}
, {0, -89}
}
, {{-57, -10}
, {72, 3}
, {0, 72}
, {52, 54}
, {-38, 89}
, {-75, -72}
, {48, 45}
, {-82, -48}
, {28, 6}
, {20, 43}
, {75, -10}
, {-74, 67}
, {1, -89}
, {-5, 62}
, {-16, 43}
, {14, -21}
, {-16, -60}
, {84, 89}
, {-57, 2}
, {-22, 9}
, {47, 24}
, {47, 7}
, {-76, -46}
, {23, 39}
, {83, -18}
, {-37, -31}
, {45, 48}
, {26, 23}
, {-90, 89}
, {7, -14}
, {-42, 59}
, {-11, 39}
}
, {{90, 6}
, {81, 48}
, {56, 80}
, {-82, 69}
, {15, -44}
, {-79, -53}
, {54, -29}
, {42, -31}
, {-73, -20}
, {53, -11}
, {-10, 41}
, {-5, 63}
, {65, 83}
, {70, -45}
, {-19, 39}
, {-79, -32}
, {-7, -19}
, {-32, 50}
, {-52, 44}
, {-23, 5}
, {61, -56}
, {45, -24}
, {-75, -88}
, {-80, -39}
, {-12, 6}
, {-79, -13}
, {21, 37}
, {-78, -29}
, {31, 50}
, {65, 48}
, {56, -24}
, {66, -79}
}
, {{-23, 21}
, {-73, 39}
, {60, 83}
, {84, 1}
, {46, 35}
, {18, 69}
, {50, -58}
, {20, -12}
, {5, 70}
, {-53, 22}
, {86, 7}
, {13, 63}
, {22, 90}
, {45, 24}
, {13, -54}
, {-20, 29}
, {84, -79}
, {-52, 53}
, {-4, 34}
, {-59, -3}
, {-58, 28}
, {54, -73}
, {-29, -84}
, {-42, 51}
, {-5, 76}
, {-6, -49}
, {0, -70}
, {-57, 52}
, {46, 43}
, {81, -28}
, {-10, 88}
, {-61, -36}
}
, {{-21, -4}
, {80, 89}
, {-80, -57}
, {-1, 75}
, {37, -4}
, {-13, -71}
, {40, -84}
, {75, -60}
, {85, 63}
, {8, -66}
, {71, -60}
, {4, 37}
, {12, -82}
, {-87, -31}
, {-48, 33}
, {24, -45}
, {62, -85}
, {6, 63}
, {-83, 43}
, {23, -78}
, {-40, 32}
, {-2, -58}
, {-71, -35}
, {37, -81}
, {15, 44}
, {50, 83}
, {-26, -85}
, {87, 39}
, {-5, -88}
, {6, 5}
, {1, -71}
, {31, -64}
}
, {{49, 61}
, {34, 42}
, {-85, 37}
, {-23, 7}
, {10, -70}
, {-35, 67}
, {71, -32}
, {-14, -11}
, {-84, 35}
, {-71, 24}
, {85, 27}
, {-4, 7}
, {-74, 13}
, {-7, -57}
, {54, -64}
, {-23, -52}
, {45, -44}
, {14, 68}
, {-87, 74}
, {-90, 27}
, {41, -4}
, {-65, 32}
, {26, 74}
, {78, -1}
, {-4, 63}
, {81, 43}
, {9, -47}
, {11, 41}
, {-30, -66}
, {19, 73}
, {-14, -54}
, {64, -6}
}
, {{32, 22}
, {88, -22}
, {29, 0}
, {-29, -55}
, {-17, -45}
, {48, 6}
, {53, 9}
, {31, -37}
, {-81, -64}
, {-11, -83}
, {36, -63}
, {-10, 61}
, {59, 35}
, {-38, -68}
, {38, -56}
, {-50, -68}
, {79, 71}
, {23, 79}
, {62, -3}
, {38, 16}
, {-62, -28}
, {58, 69}
, {-17, -89}
, {44, -4}
, {4, -21}
, {4, -36}
, {-74, 22}
, {16, -68}
, {57, 35}
, {-80, -2}
, {6, 82}
, {-1, 62}
}
, {{-18, -32}
, {89, 60}
, {-33, -34}
, {-85, -65}
, {61, -47}
, {11, 20}
, {21, 89}
, {-70, 69}
, {52, 74}
, {-55, 30}
, {42, 21}
, {78, 66}
, {-37, 10}
, {15, -80}
, {-24, -46}
, {-29, -34}
, {22, -6}
, {-54, 54}
, {-35, 34}
, {-16, 11}
, {33, -81}
, {65, -6}
, {-21, 85}
, {-49, 60}
, {-69, 8}
, {-25, 14}
, {87, -28}
, {29, -3}
, {-56, 68}
, {-66, -27}
, {20, -57}
, {86, -26}
}
, {{51, -8}
, {-53, 13}
, {-74, 20}
, {90, 81}
, {47, -46}
, {-90, 14}
, {-86, -9}
, {56, -36}
, {81, 68}
, {-16, -22}
, {-11, 83}
, {-5, -76}
, {77, -50}
, {-5, 21}
, {72, 27}
, {9, -2}
, {28, 47}
, {73, 89}
, {25, 88}
, {84, 57}
, {-53, -20}
, {-83, 52}
, {40, -11}
, {-54, 10}
, {87, 9}
, {57, 43}
, {43, -60}
, {-82, 75}
, {-47, -18}
, {0, -60}
, {33, 78}
, {13, 53}
}
, {{32, 28}
, {-42, -30}
, {-35, -25}
, {3, -22}
, {7, -54}
, {79, 86}
, {23, -80}
, {-21, -27}
, {69, -20}
, {24, 31}
, {45, 10}
, {42, -31}
, {24, 4}
, {43, -9}
, {-84, 71}
, {71, -2}
, {-47, -33}
, {-81, -54}
, {-64, -4}
, {-19, 41}
, {-65, -7}
, {27, 6}
, {16, -3}
, {-84, -79}
, {-47, -82}
, {61, -47}
, {-60, 41}
, {-38, 89}
, {82, -79}
, {38, 78}
, {-37, -87}
, {54, 1}
}
, {{42, 83}
, {53, 22}
, {-90, 88}
, {-53, -21}
, {4, 59}
, {-35, 81}
, {10, -35}
, {37, 62}
, {72, 11}
, {81, 52}
, {-88, -66}
, {-36, 63}
, {67, 14}
, {55, 16}
, {7, 68}
, {-20, -10}
, {-9, 19}
, {26, -88}
, {-73, 44}
, {-1, -38}
, {70, 10}
, {-19, 53}
, {-40, 50}
, {-21, -46}
, {13, 50}
, {-68, -39}
, {-13, 41}
, {37, 55}
, {-79, -36}
, {-56, 72}
, {-45, -47}
, {-19, 52}
}
, {{9, -33}
, {39, 73}
, {24, -75}
, {39, 66}
, {71, -41}
, {-16, -10}
, {24, 68}
, {-77, -38}
, {82, -39}
, {-1, -74}
, {-39, 70}
, {-25, -19}
, {-51, 5}
, {33, -10}
, {-39, 44}
, {-73, -41}
, {-81, -42}
, {-65, 48}
, {53, -56}
, {-47, -66}
, {39, 19}
, {27, -88}
, {-60, -43}
, {41, -68}
, {-11, -15}
, {43, -17}
, {-66, 14}
, {-8, 8}
, {52, 12}
, {-85, 38}
, {51, -25}
, {-17, -21}
}
, {{-70, -38}
, {-3, 23}
, {-62, -76}
, {62, 11}
, {-13, 36}
, {71, 46}
, {-20, 54}
, {65, 62}
, {62, 46}
, {-74, -46}
, {-73, 33}
, {24, 2}
, {76, -37}
, {-24, -49}
, {-59, 17}
, {55, 38}
, {0, 52}
, {42, 24}
, {3, 53}
, {-80, 47}
, {-82, 46}
, {73, -19}
, {29, 20}
, {46, 63}
, {78, 3}
, {-8, 88}
, {72, -35}
, {39, -78}
, {-2, -55}
, {-23, 18}
, {86, -43}
, {-44, -38}
}
, {{26, 11}
, {-51, -9}
, {31, 40}
, {82, -26}
, {33, 36}
, {78, -31}
, {-9, -6}
, {82, 6}
, {-89, -43}
, {-87, 86}
, {-46, 30}
, {-21, -29}
, {55, 71}
, {-69, 2}
, {32, 49}
, {-10, 68}
, {-6, 13}
, {0, 52}
, {-39, -21}
, {17, 20}
, {42, 65}
, {-78, 41}
, {-1, 77}
, {36, -16}
, {-22, -29}
, {-58, 19}
, {6, -18}
, {73, 64}
, {-81, 48}
, {-6, -86}
, {-37, 19}
, {40, -21}
}
, {{88, 2}
, {51, -28}
, {73, 65}
, {72, 68}
, {-58, -41}
, {-88, 80}
, {-44, 79}
, {-75, 67}
, {-63, 26}
, {34, -36}
, {76, -44}
, {-8, 20}
, {62, 13}
, {63, -36}
, {-30, -90}
, {-16, -5}
, {58, 76}
, {-67, 40}
, {23, 39}
, {-23, 88}
, {83, 69}
, {12, -90}
, {-13, -87}
, {-49, -34}
, {-10, 8}
, {-17, -18}
, {-69, 62}
, {-80, 26}
, {89, -5}
, {83, 15}
, {58, 41}
, {3, 40}
}
, {{57, -23}
, {78, 77}
, {-29, 42}
, {23, 33}
, {-15, 44}
, {44, 46}
, {44, -5}
, {-78, -59}
, {47, -2}
, {-33, -68}
, {-66, 82}
, {68, 57}
, {-60, -80}
, {14, 81}
, {-89, -14}
, {35, 65}
, {19, 51}
, {7, -3}
, {-66, 72}
, {-63, -84}
, {-78, -73}
, {36, -46}
, {75, -60}
, {-46, 30}
, {-80, 46}
, {-8, -43}
, {0, -38}
, {-84, -25}
, {-77, -40}
, {57, 19}
, {62, -67}
, {-31, -67}
}
, {{37, 5}
, {11, -73}
, {-6, 85}
, {-74, -28}
, {60, -12}
, {-30, -18}
, {7, 63}
, {48, 19}
, {-44, 69}
, {-39, 20}
, {-56, 17}
, {0, -86}
, {-29, -44}
, {-14, -10}
, {-77, 19}
, {-86, -62}
, {40, 74}
, {9, 59}
, {0, -42}
, {13, -20}
, {5, -20}
, {10, -60}
, {-65, 84}
, {-19, 21}
, {5, -10}
, {-82, -60}
, {47, 51}
, {-49, -21}
, {73, 78}
, {77, 62}
, {-5, 53}
, {-72, 13}
}
, {{82, 37}
, {-4, 88}
, {-35, -78}
, {12, 80}
, {-33, -83}
, {72, -45}
, {13, -83}
, {64, 67}
, {43, -81}
, {-47, 16}
, {-66, -19}
, {-49, -21}
, {-8, -9}
, {70, -54}
, {-28, 21}
, {-62, -5}
, {3, -57}
, {-45, -41}
, {25, -25}
, {-41, -46}
, {59, -36}
, {26, 58}
, {-74, -65}
, {-31, -39}
, {38, 52}
, {-83, -43}
, {-51, -88}
, {-53, 2}
, {52, 12}
, {-53, 77}
, {7, 12}
, {-31, 56}
}
, {{-86, -90}
, {-35, 24}
, {-11, 80}
, {-38, -51}
, {-63, -71}
, {55, 15}
, {21, 59}
, {-39, 9}
, {-44, 55}
, {-85, 77}
, {-36, -68}
, {49, -79}
, {-57, -69}
, {-80, 73}
, {45, 28}
, {86, -82}
, {-58, -10}
, {-30, 28}
, {-3, -49}
, {-28, -18}
, {41, -27}
, {56, 39}
, {16, 29}
, {-19, 84}
, {33, 78}
, {81, -76}
, {9, -52}
, {47, 74}
, {61, 44}
, {-82, -62}
, {43, -6}
, {84, -20}
}
, {{78, -37}
, {7, -29}
, {45, 28}
, {-56, -31}
, {-47, 88}
, {46, 80}
, {60, -54}
, {13, -52}
, {74, -44}
, {-1, 33}
, {-49, -78}
, {-30, 33}
, {37, 51}
, {-54, -69}
, {-52, 44}
, {54, -11}
, {14, -7}
, {70, -7}
, {-21, -22}
, {-37, 0}
, {-3, -83}
, {86, 56}
, {-1, 9}
, {43, -50}
, {-67, -28}
, {61, -25}
, {22, -53}
, {85, 11}
, {8, -47}
, {56, -32}
, {40, -21}
, {-36, 72}
}
, {{27, -23}
, {-78, 37}
, {42, -41}
, {30, -84}
, {11, -54}
, {-53, -55}
, {77, 66}
, {-64, -18}
, {-16, -78}
, {30, 29}
, {-86, 43}
, {-7, -84}
, {-41, -24}
, {-25, 16}
, {-55, 13}
, {-12, -50}
, {-54, -80}
, {30, 47}
, {-15, 58}
, {29, 30}
, {43, -78}
, {61, 1}
, {-12, 18}
, {8, 77}
, {-86, 47}
, {-64, 76}
, {45, 16}
, {34, -62}
, {-72, -39}
, {42, -61}
, {26, -86}
, {-17, 63}
}
, {{-33, 34}
, {24, 69}
, {-8, -61}
, {21, -13}
, {-9, -46}
, {55, 70}
, {68, -18}
, {-69, -86}
, {48, -63}
, {69, 1}
, {-35, 27}
, {31, -72}
, {76, 43}
, {-80, -6}
, {-81, 88}
, {20, -65}
, {-50, -56}
, {53, -52}
, {-71, -9}
, {22, -66}
, {3, -54}
, {24, -2}
, {-19, -31}
, {-32, -39}
, {-80, 70}
, {66, -59}
, {-11, 38}
, {3, -30}
, {12, 17}
, {-32, 7}
, {0, 1}
, {-49, 0}
}
, {{-90, 17}
, {25, -22}
, {-1, -36}
, {-82, -73}
, {-16, 58}
, {11, -53}
, {75, 37}
, {-1, -44}
, {-79, 28}
, {-56, -1}
, {51, -38}
, {59, 70}
, {5, 66}
, {34, -12}
, {42, -47}
, {51, 3}
, {-22, -18}
, {32, 43}
, {25, -71}
, {-22, 50}
, {-1, -12}
, {-79, 46}
, {-58, 48}
, {30, 56}
, {-70, -83}
, {46, 36}
, {4, -85}
, {-4, 66}
, {-85, 56}
, {-19, 12}
, {68, 68}
, {-28, 24}
}
, {{76, -26}
, {-28, -85}
, {-56, -69}
, {0, -6}
, {19, -16}
, {-53, -59}
, {-54, -41}
, {28, 16}
, {50, -60}
, {-86, 21}
, {-51, 76}
, {48, 14}
, {-60, 65}
, {31, -88}
, {-60, 81}
, {38, 78}
, {32, -44}
, {29, -55}
, {7, -63}
, {60, -21}
, {-88, -56}
, {56, -57}
, {31, 58}
, {70, -11}
, {54, 49}
, {38, -54}
, {41, -73}
, {72, 1}
, {85, -32}
, {-9, -35}
, {40, 84}
, {70, 45}
}
, {{-21, -61}
, {62, 36}
, {3, 62}
, {26, 6}
, {1, -65}
, {45, -10}
, {29, 18}
, {-35, -17}
, {-64, -61}
, {24, 4}
, {-67, -33}
, {-67, 81}
, {78, 5}
, {-33, -36}
, {-68, -65}
, {17, 30}
, {-37, -38}
, {-84, 69}
, {-51, -18}
, {65, 27}
, {-75, 50}
, {27, 30}
, {80, 74}
, {25, 27}
, {51, -18}
, {-10, 82}
, {-17, 26}
, {-10, 26}
, {24, -86}
, {-85, 1}
, {-67, -73}
, {-5, 32}
}
, {{49, -22}
, {-51, 49}
, {67, 8}
, {-26, 1}
, {-34, -6}
, {-74, 70}
, {-56, -37}
, {-56, 74}
, {-65, -83}
, {-59, 13}
, {44, -27}
, {-12, -82}
, {-73, 75}
, {47, -49}
, {28, -1}
, {-3, -70}
, {7, -85}
, {-37, 47}
, {84, 86}
, {35, -44}
, {-12, 11}
, {76, -21}
, {-10, 49}
, {5, 90}
, {62, 72}
, {64, 22}
, {-68, 19}
, {75, -11}
, {-61, -69}
, {73, -52}
, {-10, 0}
, {49, 32}
}
, {{75, 56}
, {-71, 56}
, {-5, 18}
, {-32, -24}
, {-76, 87}
, {-2, 81}
, {-30, -47}
, {15, 43}
, {74, 79}
, {78, -16}
, {68, -9}
, {32, -14}
, {-16, 67}
, {61, 60}
, {32, 10}
, {-43, -63}
, {-11, 48}
, {37, 89}
, {-66, 53}
, {-87, 84}
, {47, 73}
, {32, -22}
, {-36, -25}
, {-51, -18}
, {18, 19}
, {4, 4}
, {-45, -48}
, {65, -81}
, {8, -46}
, {29, 8}
, {51, 89}
, {-52, 26}
}
, {{13, -56}
, {-66, 2}
, {25, 59}
, {-29, -44}
, {14, -24}
, {-29, -15}
, {-79, 67}
, {-24, -74}
, {2, 89}
, {-33, 51}
, {38, -39}
, {-39, -46}
, {84, 6}
, {38, -55}
, {-87, -2}
, {-65, 53}
, {-26, 13}
, {-74, -45}
, {-89, -62}
, {64, 11}
, {58, -24}
, {-5, -65}
, {-3, -12}
, {38, -35}
, {-66, -16}
, {-63, 46}
, {2, 75}
, {54, -76}
, {-78, 49}
, {-35, 66}
, {-31, -62}
, {70, 84}
}
, {{53, 4}
, {82, -76}
, {26, 4}
, {43, -27}
, {-61, -56}
, {71, -85}
, {36, 3}
, {-30, 86}
, {-74, -37}
, {29, 0}
, {-46, -76}
, {69, -27}
, {58, -26}
, {2, 41}
, {-4, -48}
, {-65, 31}
, {49, -21}
, {-89, -34}
, {-79, -80}
, {61, -26}
, {33, 60}
, {90, 10}
, {66, 59}
, {-52, 48}
, {39, -88}
, {-18, 46}
, {6, -30}
, {-4, 4}
, {-73, 40}
, {21, -52}
, {-85, -49}
, {4, 59}
}
, {{-12, -55}
, {26, 68}
, {83, -79}
, {-60, -53}
, {23, 10}
, {33, -25}
, {36, 4}
, {82, 72}
, {74, -41}
, {-68, 71}
, {-35, 74}
, {-23, -7}
, {-44, 27}
, {-40, -83}
, {30, 83}
, {-45, -85}
, {32, 67}
, {40, -20}
, {74, -73}
, {-22, 12}
, {-20, 37}
, {-41, -73}
, {20, -7}
, {48, 52}
, {37, 36}
, {33, 76}
, {67, 19}
, {-34, 62}
, {86, 14}
, {35, -18}
, {46, 42}
, {49, -84}
}
, {{-55, -39}
, {9, 12}
, {-16, 3}
, {-15, 17}
, {40, 41}
, {63, 30}
, {82, 33}
, {-1, -42}
, {16, -86}
, {70, 3}
, {-22, 50}
, {-56, -62}
, {85, -67}
, {-60, -29}
, {-42, 19}
, {-44, -5}
, {86, -58}
, {33, -64}
, {-7, -5}
, {-37, 59}
, {-68, 90}
, {-34, -24}
, {-27, 83}
, {-77, -14}
, {90, -23}
, {-66, 80}
, {-44, -35}
, {64, 84}
, {-16, 80}
, {-5, 39}
, {81, 59}
, {49, -82}
}
, {{44, 34}
, {-86, -59}
, {66, -58}
, {51, -44}
, {61, -87}
, {-71, -45}
, {-55, 88}
, {79, -72}
, {-26, -60}
, {-53, 68}
, {-34, -28}
, {-1, 34}
, {-33, 64}
, {90, 67}
, {-74, -9}
, {-57, -18}
, {44, 6}
, {-40, -73}
, {-59, -25}
, {47, -34}
, {-76, -62}
, {74, -89}
, {-25, 18}
, {-51, 76}
, {17, -53}
, {3, -71}
, {73, -2}
, {-57, -76}
, {-27, -56}
, {29, 5}
, {-82, 89}
, {-5, 26}
}
, {{-14, 38}
, {43, 65}
, {69, 39}
, {-7, -23}
, {72, 23}
, {-90, 59}
, {22, 87}
, {50, 8}
, {-16, 72}
, {-75, -47}
, {-79, 3}
, {-36, -88}
, {20, 29}
, {-77, 9}
, {-23, -71}
, {-10, 59}
, {-89, 36}
, {-78, 84}
, {72, 16}
, {-10, -90}
, {-31, -67}
, {-4, -52}
, {16, -71}
, {-27, 50}
, {-66, -3}
, {-55, 83}
, {-65, -70}
, {10, 43}
, {0, 50}
, {-51, -2}
, {43, -60}
, {87, 55}
}
, {{57, 33}
, {-26, 53}
, {-40, -60}
, {-13, 53}
, {27, -56}
, {27, -32}
, {-72, 77}
, {46, -27}
, {32, 19}
, {-48, 68}
, {-74, 47}
, {16, -69}
, {17, 30}
, {32, -85}
, {-20, 80}
, {-56, 47}
, {-13, 31}
, {46, 6}
, {12, -49}
, {67, 3}
, {-13, -59}
, {-21, -1}
, {72, 37}
, {9, 40}
, {73, 42}
, {79, 26}
, {-76, -18}
, {-18, -30}
, {55, -62}
, {-50, 32}
, {0, 25}
, {-21, -69}
}
, {{-54, 25}
, {-36, 78}
, {69, 72}
, {11, 69}
, {-73, -42}
, {31, 19}
, {-55, 21}
, {20, -6}
, {57, -15}
, {14, 25}
, {-5, -1}
, {11, 4}
, {83, -86}
, {61, -27}
, {37, -66}
, {-17, 24}
, {-55, 33}
, {-8, -28}
, {-39, 1}
, {-1, -66}
, {-71, -79}
, {0, 19}
, {34, -82}
, {48, 35}
, {-44, -34}
, {10, 48}
, {45, 7}
, {-86, -71}
, {90, -46}
, {-50, 66}
, {-40, -33}
, {53, -12}
}
, {{26, -85}
, {34, -71}
, {-51, -50}
, {-83, -47}
, {-78, 24}
, {16, -37}
, {89, 60}
, {8, -15}
, {-79, 87}
, {-15, 63}
, {69, 53}
, {-60, 41}
, {55, -25}
, {-72, -90}
, {-23, 15}
, {-62, -38}
, {48, 38}
, {43, -48}
, {-70, 28}
, {-11, -58}
, {-22, -80}
, {-10, -87}
, {-38, -74}
, {-90, -34}
, {-56, -41}
, {-38, 31}
, {-37, -4}
, {-28, 67}
, {35, 89}
, {6, 23}
, {58, 80}
, {-76, 48}
}
, {{-60, -68}
, {-52, 90}
, {-37, 67}
, {45, -67}
, {18, -17}
, {0, -22}
, {19, -59}
, {32, -61}
, {-40, 64}
, {-8, -41}
, {-15, 67}
, {11, 80}
, {-39, -33}
, {14, 65}
, {-47, 87}
, {26, 73}
, {-81, 66}
, {55, -77}
, {68, 23}
, {57, 66}
, {43, 0}
, {45, 26}
, {32, -61}
, {-13, -58}
, {88, -8}
, {-47, 86}
, {42, 51}
, {31, 4}
, {34, -68}
, {35, 40}
, {84, 68}
, {-28, 48}
}
, {{-9, 9}
, {-88, 56}
, {-83, 35}
, {-57, 15}
, {4, 10}
, {37, -12}
, {-59, 46}
, {-74, 37}
, {-19, 88}
, {87, 61}
, {-5, -11}
, {72, -1}
, {-19, -81}
, {53, 85}
, {-72, -78}
, {26, 78}
, {28, 6}
, {16, -53}
, {-4, 31}
, {67, 77}
, {26, 26}
, {-63, 15}
, {-83, -15}
, {30, -1}
, {42, 54}
, {36, -89}
, {69, 70}
, {60, -52}
, {-47, 74}
, {58, 26}
, {-3, -38}
, {70, 68}
}
, {{19, -13}
, {-13, -25}
, {56, -47}
, {-34, -81}
, {-37, -9}
, {-55, 85}
, {86, -68}
, {18, -76}
, {9, 75}
, {-67, 11}
, {-53, 59}
, {72, 54}
, {31, -72}
, {4, -51}
, {-52, 1}
, {16, 30}
, {-52, 9}
, {8, -14}
, {-64, -66}
, {34, -14}
, {34, 56}
, {-85, -47}
, {76, -29}
, {-1, -42}
, {-32, 64}
, {90, -41}
, {72, 31}
, {90, 26}
, {85, -61}
, {17, 50}
, {67, -13}
, {-32, -45}
}
, {{57, 49}
, {-75, -2}
, {56, 14}
, {-35, -1}
, {-23, -25}
, {-45, 0}
, {26, -44}
, {-47, -6}
, {-69, 60}
, {-22, 34}
, {-89, -49}
, {87, -64}
, {35, -69}
, {57, -52}
, {68, 80}
, {-67, 88}
, {-89, -19}
, {-75, 32}
, {-59, -30}
, {27, -4}
, {31, 0}
, {-45, 18}
, {-37, 69}
, {8, 38}
, {71, 63}
, {-5, 62}
, {23, -21}
, {-66, 67}
, {-79, -45}
, {72, -79}
, {-19, 53}
, {85, 23}
}
, {{-15, -77}
, {-16, -29}
, {3, 12}
, {-74, -16}
, {74, 28}
, {-34, 22}
, {67, -1}
, {53, -9}
, {-11, 65}
, {-6, 78}
, {3, 60}
, {-81, 38}
, {66, 9}
, {-54, -79}
, {27, -17}
, {-71, -81}
, {64, -30}
, {59, -87}
, {78, -19}
, {58, 29}
, {-11, 58}
, {-33, 29}
, {-80, 38}
, {56, 17}
, {-82, -14}
, {-29, -80}
, {-29, 44}
, {45, 55}
, {71, -22}
, {30, -48}
, {-84, 49}
, {-20, -26}
}
, {{-42, 24}
, {13, 80}
, {-67, -64}
, {45, -23}
, {78, 42}
, {-10, 8}
, {-55, 6}
, {-60, 59}
, {12, 90}
, {55, 90}
, {-57, -10}
, {1, -47}
, {86, 74}
, {-68, 9}
, {-24, 17}
, {-36, 37}
, {69, 73}
, {-79, -71}
, {2, -11}
, {79, 83}
, {-3, 85}
, {-44, -26}
, {-55, -28}
, {63, 2}
, {-23, 84}
, {61, -64}
, {33, 0}
, {85, -37}
, {-28, 88}
, {70, 10}
, {-74, -61}
, {-15, -16}
}
, {{-55, -87}
, {0, 7}
, {8, 1}
, {-13, -55}
, {56, 55}
, {-56, -84}
, {-28, 34}
, {48, -22}
, {-66, 88}
, {-59, 37}
, {-78, -20}
, {-42, -55}
, {54, 59}
, {17, 78}
, {-69, 16}
, {84, -14}
, {2, 34}
, {32, 51}
, {19, -21}
, {28, 13}
, {-65, 68}
, {87, 63}
, {-68, -32}
, {34, -3}
, {-17, 47}
, {-90, 5}
, {-1, -78}
, {74, 28}
, {-14, -3}
, {75, 35}
, {26, -62}
, {88, 28}
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


const int16_t dense_bias[FC_UNITS] = {-56, -16, 3, 79, -12}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{-26, 54, -69, -65, -46, -15, 59, 30, 3, -14, -46, -58, 43, 34, -59, 1, -6, 1, 63, 22, 9, -61, -17, 39, 53, -16, -10, 54, 18, -30, 24, 61, 35, 54, -50, 68, -67, 58, -12, 64, 60, 36, -37, 41, -2, -16, 34, -52, 13, -57, -9, 25, -27, -35, -15, -41, 23, -3, -13, -55, 43, -35, -68, -51, 24, 46, 69, -43, 56, -6, 46, 67, -58, 31, -21, 64, 22, 57, -23, -5, -46, 23, 3, 20, -54, 34, -1, -18, 25, -17, 26, 55, -3, 34, -50, 40, 8, -28, -7, -15, 60, 34, 52, 15, 28, -60, 59, -5, -42, 47, -39, -43, -9, 5, 58, 7, 63, -43, 66, -26, 26, 68, -46, 36, -7, -12, 51, -62, 49, -25, 53, -38, 39, 26, -66, -50, -9, 64, 58, -58, 30, 63, -67, 17, 35, 40, 11, 36, 3, -51, 0, -63, 0, 64, -59, 23, 24, 45, 18, -13, 18, -59, -39, -50, -46, -4, 37, -22, 7, -50, -6, -53, 58, 41, -15, -4, 11, -13, 28, -28, 50, 60, -25, -31, -19, -29, 24, -30, 42, 25, -52, -46, 66, -61, -43, 54, -39, -1, -23, -42, -4, 23, 34, -31, -49, -37, 32, 55, -18, -47, 4, -30, -54, 2, -62, -63, -56, 54, -40, -61, -65, 40, 11, -27, -37, 47, 47, -18, 2, 35, 9, 33, 54, 19, -24, -63, -61, 26, 19, 23, -22, 1, 29, -68, 29, -5, -12, -59, 55, -21, -44, -15, 40, -3, 23, -10, -51, 13, 15, 26, -46, -47, -25, 39, 19, -21, 23, 23, -23, 41, 49, -8, -35, -9, -58, 65, -48, -56, 48, -16, 35, -40, 10, -45, -3, 36, -12, 55, 64, 67, 25, -67, 25, 13, -16, -48, -65, 34, -21, 26, 15, -66, -46, 46, 41, -46, 22, -3, 47, 30, 20, -37, -15, 52, 56, -28, -51, -40, 26, -25}
, {45, -16, 32, 52, -1, -28, 65, -58, -63, -21, -50, -64, 56, 0, 60, -2, -61, -43, 8, 14, -40, -26, 59, 22, 62, 37, -27, 66, -17, 31, 32, -26, -67, -15, 54, 44, 34, -51, 26, -60, 21, -41, -66, -9, 33, 37, -57, 34, -2, -9, -42, 48, -17, -33, -52, 20, 12, 53, -48, -17, -12, -26, 2, -54, -8, 0, -45, -28, 21, 25, 18, -10, 55, 66, -45, 60, -39, 8, 68, 44, -10, 34, 60, -59, 11, 55, 37, -57, -12, 36, 67, 41, -8, 61, 44, 30, 18, -19, -62, 21, 8, 5, 0, 31, 8, 18, 52, 23, -38, 19, -54, -23, 24, -8, 25, -66, -42, 0, 34, 57, -40, 14, 9, -62, 34, 10, -42, 1, 53, -33, -45, 63, 5, 10, -21, -53, -34, 6, 31, -68, -14, 6, -18, -49, 11, 60, 8, -46, -2, 27, 43, 32, 50, -25, -58, -25, -54, -28, -42, -16, -48, 50, 58, 32, 22, -27, 49, -6, -35, -56, -1, 0, 4, 24, -44, 19, -18, 65, -57, -2, 53, 39, 15, -60, 12, -64, -56, -15, 12, -64, 6, 36, 42, 24, -52, -30, 9, 16, -14, -17, -61, 61, -5, 30, -29, -22, 68, 49, 23, 13, -62, 7, 61, 2, 33, 65, 4, 34, -3, -32, 69, 62, -3, 19, -41, 63, -68, -53, -30, 28, -29, -34, -61, -2, -14, 0, 38, 23, 6, 54, 33, -57, 1, -60, -40, -10, 24, 24, 9, 6, -26, 7, 3, -49, -1, 40, -39, -59, 67, -32, 5, 53, -47, -59, 50, -26, 67, -65, 54, -8, -39, 19, 50, 19, 40, -23, -1, 23, 54, -5, 28, 6, -37, 18, 11, 5, 18, -64, -14, -36, -45, 26, -23, -17, -7, -36, -1, -40, -28, -51, 1, 18, -52, 50, 39, 18, -52, 23, -17, 29, -56, 11, 29, 0, -34, -18, -17, 52, 20, 55}
, {25, -4, 20, 16, -37, 68, -32, 45, 63, 37, -51, 22, -44, -15, 49, -47, 6, -49, -61, 10, -44, -58, 58, 11, 36, 25, 37, 27, -38, -47, 7, -13, -30, -56, 30, 28, -23, -43, -66, 3, -25, 42, -27, 16, 2, 57, -1, 52, -33, 36, 34, -54, -26, -44, -59, -46, 0, -11, 20, 33, -32, -8, -43, -27, -2, 25, -43, 31, -13, -62, -48, 48, 31, -59, -16, 4, 56, 31, -40, -48, 24, 52, 20, 61, 14, 62, 41, -4, -5, 35, 68, 16, 60, 17, 50, -2, -15, 18, 0, 56, -3, 10, -37, 2, 20, 19, 57, 46, -17, -61, -50, -61, -30, -20, -4, 33, 17, -27, -68, -3, -68, -45, 55, 21, -48, 0, -21, -32, 2, 67, -10, 17, 16, 63, 46, 51, -1, 59, 58, 66, -69, -31, 44, -17, 65, 60, 34, -50, -12, -45, 46, 15, 17, -40, -22, -57, 43, 59, 56, 30, 24, 63, 42, 0, -1, -8, -30, -44, -23, 12, -43, 21, -46, 22, 24, 20, -23, 45, -69, 51, -30, 35, 13, 36, 56, 43, 14, 24, 6, 58, 38, 56, 45, -18, -6, 28, 7, -38, -2, -36, -13, -26, -49, -3, -67, 4, -24, 37, 58, 1, -64, 12, -49, 12, 44, -1, 65, -67, -64, 29, 3, 53, 33, -49, -64, 55, -48, 48, 18, -38, -43, 57, -42, -16, 33, 20, -49, -25, -39, 19, -46, 57, -31, 57, 37, 8, 44, 47, -30, 37, -27, -53, 43, -45, 6, -53, -26, -44, -47, 63, 10, -44, 13, -41, -43, 65, 41, -58, 46, 26, 36, 32, -40, -59, -58, -65, 0, 7, 63, 21, 8, -54, 41, -69, -3, 3, 62, 5, 0, 10, -11, -25, -69, -48, -62, 8, 23, -42, 22, -66, -26, -45, 22, 27, 48, 31, 54, -33, -29, 69, -60, -11, -18, -9, -43, 57, -5, 21, 44, -63}
, {-68, -32, -5, 55, -20, 63, -2, 67, -8, -26, 4, 12, -12, -56, 8, -18, -60, -64, 11, -35, 69, -61, -8, -60, 47, -14, 8, 32, -22, 26, -54, 55, -56, 42, -63, 63, 55, -50, -48, 45, -69, 8, 39, 39, -1, 10, -22, 29, -50, 45, -11, -69, -61, -7, -24, -7, 28, -3, 20, 20, -43, -43, -60, 64, -49, -27, -31, -60, -32, 24, -4, 6, -16, 32, -23, -40, -14, 22, 7, 14, -29, 48, -43, -5, 44, -56, 38, 10, 64, 41, -41, -66, -4, 22, -64, -39, -7, -16, -4, -60, -27, 16, 38, -45, 43, -21, -42, 37, 25, 43, 63, 27, 53, 37, 69, -20, -56, 12, -10, -16, -10, 66, 55, -40, -65, -40, -46, -39, 42, 8, 66, 19, 20, -67, 46, -43, 66, 55, 6, 5, 56, 49, 27, 13, 45, 56, 20, 35, 36, -30, -61, 58, 9, 5, -55, -27, -16, 18, -62, 12, 56, -58, 47, -51, -10, 34, 10, 28, 53, -1, 2, -27, 60, -65, 61, 64, 63, -34, 30, 15, 53, 15, 64, -6, 26, 43, -36, 27, -57, -29, 16, -56, 48, 39, 0, -33, 63, 60, 11, 51, 26, -42, -54, -16, 38, 12, -52, -53, -43, 68, -11, -50, -6, 57, -35, -18, -64, -5, -67, 60, -41, 55, 42, -56, -16, 17, 36, 11, -50, -35, 23, -12, -19, -11, 6, -20, -50, 36, 65, 4, -11, 12, -46, -5, -59, -63, -61, -55, -53, -69, -57, -8, -41, 60, 43, -20, -68, -32, -39, 61, -46, -44, -40, -18, 15, -9, 55, -5, -32, 19, -62, 22, 61, 45, 48, -29, 15, -12, -17, 64, -13, -8, 29, 29, 3, 55, -1, -30, -36, -8, 46, -30, -59, 10, 4, -7, 53, -62, 17, -61, -58, 60, 25, -67, 14, 47, -41, -56, 36, 13, 14, -6, -36, 44, 68, 59, 42, -19, 61, -17}
, {40, 10, -59, 18, -23, 7, 20, -45, -64, -61, 9, 55, 21, 26, 44, -1, -68, -40, -3, -47, -18, 14, 66, -15, -44, 39, -69, -8, 7, -40, 20, 60, -35, -8, 32, -66, 42, -27, 66, 31, 58, -41, -39, 29, -68, 40, 55, -35, -50, 25, 44, 21, -57, -43, -10, 14, -42, -41, 12, 26, 36, -1, 2, -44, -12, -32, 16, 15, -35, -29, 26, 63, 2, -8, 33, -14, -48, -5, -41, 60, -11, -37, 41, -20, -25, 61, 31, 3, 47, -1, -32, 18, 32, -36, 20, 39, -33, -6, 45, 9, 5, 46, -23, 11, 66, 48, 22, -52, -28, 20, 20, 28, -17, 24, -31, -27, -43, 9, -42, 0, -58, 20, 67, -68, -35, -59, 40, 23, 24, 1, -8, 48, -53, 8, 67, -22, -28, -66, -41, 24, 41, -4, -57, -60, -22, -43, 26, 50, 58, 0, -65, -10, -57, 10, 20, -69, 28, 51, 20, -31, -22, -60, -46, 6, 58, 32, -49, -31, 23, -9, 51, -43, -56, -28, -8, -12, -1, 66, 25, -21, -39, 6, 42, 33, -29, 3, 41, 51, 45, -61, -34, -34, -47, 57, -17, -7, -64, 4, 48, 23, -51, 37, 31, -52, -57, -39, -53, -20, -18, 0, 40, 22, -4, -44, 18, -22, -62, -29, 48, 34, 30, 63, 25, 41, 31, -57, -34, -10, 0, -14, 16, 37, -61, -55, -24, -24, 57, 47, -51, -39, 39, -20, 69, -45, -39, 35, -31, 11, 52, -10, 35, 9, -52, 1, 48, 1, 36, 29, 63, 10, 41, 61, -60, 65, -63, 2, 17, 47, 12, 18, 1, 18, 16, -68, 52, 12, 40, -15, -17, -9, 46, -42, -18, -12, -53, -10, -56, -45, -61, 60, -25, -36, -22, -31, 29, -12, -43, 32, -17, -54, 10, -25, -67, 49, -51, 55, -57, -17, -7, -51, 3, 67, -55, 51, 32, 43, -45, -53, 21, -13}
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
