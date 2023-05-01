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


const int16_t conv1d_bias[CONV_FILTERS] = {13, 2, -4, 12, -1, 50, 2, -8}
;

const int16_t conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-84, 114, -32, 11, 14, 121, -9, 33, 65, -53, -107, -80, 21, 24, -142, 23, -107, -74, 71, 63}
}
, {{70, 8, -80, -40, -101, 27, 70, -78, 73, -16, -27, 76, -7, -99, 131, -67, -89, 58, 88, -93}
}
, {{54, -92, 32, 41, 56, -39, -37, -17, -33, 64, 65, 57, -42, 49, 36, -113, 110, -87, 117, -108}
}
, {{-126, -103, 37, 14, 139, -123, -101, 20, 47, 102, -59, 94, 60, 40, -44, -40, -91, -45, 133, 20}
}
, {{76, -94, 48, -13, 84, 85, 57, 71, -48, 88, -22, 108, 123, -19, -132, -105, -10, 21, 89, -45}
}
, {{-42, 19, -62, 65, 41, -32, 0, -23, -30, 5, 39, -79, -17, 55, 20, -30, 159, -23, -68, -91}
}
, {{-24, 38, -46, 29, 112, 47, 67, -72, 60, -73, 85, -84, -4, -100, -51, -27, -31, -138, -7, -82}
}
, {{141, 46, 144, 65, -71, 92, -34, -60, 38, -2, 112, 89, -53, -87, -32, -2, -82, 69, 84, -118}
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


const int16_t conv1d_1_bias[CONV_FILTERS] = {9, 17, -15, -16, 9, 4, -18, -1, -11, 27, 5, 2, 21, -19, 0, 12}
;

const int16_t conv1d_1_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{7, -34, 29, -23, 18, -76, -42, -91}
, {-22, 72, 48, 69, -34, -48, -125, -61}
, {-11, 25, 112, 35, 118, -23, -76, -65}
, {42, 20, 36, -54, 44, 40, 126, -106}
, {48, 106, 119, -53, -42, 0, 80, -76}
, {-74, 31, -51, -2, 49, 33, -80, 23}
, {15, 35, 8, -71, 5, -18, -36, -85}
, {-64, -31, -43, 72, -45, -33, 21, -138}
}
, {{74, -58, -72, 116, -84, 84, 33, -6}
, {-53, 62, -119, 39, -125, -95, 49, 43}
, {-90, -111, -27, -34, 40, -94, 85, -85}
, {102, 13, -62, 27, -93, -60, 57, 11}
, {69, 18, -14, 45, -70, -28, 59, 89}
, {-37, -82, 21, 24, -5, 57, 7, -110}
, {-2, 49, 54, 105, -90, 19, -4, 41}
, {94, 64, 39, -65, 40, 50, -14, -68}
}
, {{70, 77, -92, -136, -42, -8, 42, -18}
, {-58, 84, 24, 90, -25, -138, -133, 18}
, {8, 59, 38, -9, -56, -40, 10, 16}
, {-22, 86, 44, 41, 79, -44, -72, 58}
, {25, -42, -49, -66, -102, -17, -67, 30}
, {29, 28, -85, -8, 16, -10, -116, 13}
, {-47, 86, 88, 73, -68, -94, -149, -34}
, {79, -61, 3, 64, -88, -7, -19, -43}
}
, {{-122, -60, -41, -63, -53, -29, 13, 21}
, {1, 81, 107, 67, 50, -36, -86, -67}
, {92, 86, -47, -6, -78, 98, 62, -23}
, {-46, 0, 38, 111, 57, 5, 101, -32}
, {65, 66, -80, 67, 33, 28, 26, 51}
, {-34, -64, -75, 66, -58, -59, 85, 5}
, {37, 72, -74, -54, 45, 1, 57, -22}
, {43, -48, 65, -39, 92, -71, -88, -49}
}
, {{88, -10, 1, -38, -22, -57, -2, -67}
, {-9, 80, -43, -65, 19, -15, 71, 22}
, {43, -47, -8, 21, 22, -79, -42, -121}
, {42, 78, 94, -109, -72, -31, -30, -125}
, {-45, -55, 29, 29, -83, -71, 23, 49}
, {0, -86, -63, 45, -10, 40, -55, 21}
, {100, 5, 62, -44, -56, 66, 10, 22}
, {-58, 19, -77, 120, -53, -16, 74, -88}
}
, {{-5, 141, 22, 43, 41, 59, 60, -41}
, {-17, -56, -80, -24, -99, 71, 54, -10}
, {10, 49, 95, 0, -63, -68, -85, -94}
, {23, 115, 148, 65, -19, -52, 60, -13}
, {-39, 79, 118, 19, 88, 21, -27, 21}
, {3, 1, -58, 31, 4, -37, -34, 13}
, {-15, -98, -76, -18, -57, -19, -39, -118}
, {82, 13, -68, 72, -50, -2, -49, -13}
}
, {{-37, -34, 21, 67, 46, -111, -102, 29}
, {-63, 28, -53, 62, 22, 82, -10, -65}
, {98, -10, -89, 23, 55, -84, -12, -38}
, {69, 0, 74, -91, 33, -136, 0, -15}
, {67, -80, 97, 53, -73, -71, 51, -76}
, {75, 48, -133, -87, -8, -78, -6, -40}
, {15, 44, 28, 58, -57, -66, 42, -108}
, {-15, -13, -4, 63, -29, 16, 34, -61}
}
, {{62, -52, -96, -4, -103, 77, 74, -112}
, {89, 27, 26, 64, -75, -30, 41, -21}
, {-77, -23, -45, 12, -51, -18, -92, 58}
, {-68, 91, -88, -35, 67, 3, 82, 82}
, {-8, -2, -61, -28, -61, 89, 20, 105}
, {118, -85, -106, 37, 81, -31, 77, -70}
, {-29, -35, 63, -32, -132, 47, -91, 59}
, {-40, -41, 79, 14, 0, 14, 104, -12}
}
, {{101, -99, 27, 74, 19, 86, 18, -124}
, {-60, 42, 76, 67, 6, 69, 37, 14}
, {105, 57, -70, 75, 83, -67, 48, -65}
, {27, 5, -85, 114, -116, 97, 72, -4}
, {-60, 11, 4, -72, 95, 62, 50, -8}
, {-80, 108, 19, 26, -43, 20, 45, -16}
, {-17, -57, 57, 9, 19, -43, -30, 41}
, {29, -95, 34, 19, -3, -81, -61, -93}
}
, {{46, 102, -17, 52, 122, -71, 102, 60}
, {-12, -101, 42, 39, 106, 58, -23, -61}
, {34, 51, 1, 92, 2, -71, -65, -46}
, {-86, -112, 13, 21, -29, -59, -105, -55}
, {-27, -124, 17, 45, 59, -62, -9, -43}
, {78, 20, 5, 62, 25, -52, -148, 41}
, {-89, 55, -62, -77, 72, -23, -49, 3}
, {-64, 13, -66, 77, -5, 1, 1, 26}
}
, {{-44, -49, -34, 120, -51, 78, 58, -128}
, {23, -119, 20, 34, -83, 51, -86, 65}
, {-41, -42, -20, 27, 58, -49, 90, -81}
, {-54, -141, -33, -81, -5, -11, 9, -139}
, {36, -43, -88, 76, 113, 4, -60, 64}
, {10, -47, -28, -58, -95, -6, -15, -45}
, {3, 5, 55, -100, 60, 30, -105, 2}
, {-96, -68, -5, -69, 114, 97, 64, 22}
}
, {{-74, 23, 98, 25, 0, 111, -108, -122}
, {-1, -11, 25, 52, -115, 27, 72, 53}
, {-96, -5, 0, 81, -126, 60, 30, 39}
, {8, 9, -84, 85, 26, -64, -12, -23}
, {84, 22, -112, -58, 25, 75, -96, 0}
, {52, -25, 74, 51, -77, 16, -30, -77}
, {-7, -107, 42, 5, -105, 75, 56, 39}
, {-14, -30, -59, 12, -95, 129, -32, 26}
}
, {{-79, 60, -43, -55, 34, -30, -81, -1}
, {-24, -16, 51, -14, 4, 46, -38, -31}
, {108, -61, 1, -88, 32, -55, -88, -68}
, {35, 60, -61, 3, -113, -142, 39, 58}
, {106, -48, 59, 72, 121, 65, 53, 14}
, {85, -70, 14, -116, -45, 91, -12, -26}
, {-13, 19, -50, -1, -102, 88, 43, 33}
, {-63, -61, -63, -23, -11, 123, 85, -68}
}
, {{-7, -25, -14, 43, 41, -29, -29, 59}
, {68, -62, 24, -84, 62, -18, 73, 33}
, {-16, -6, -11, 19, -63, -123, 90, -81}
, {-22, 12, 109, 65, 19, 21, 120, 25}
, {-13, -59, 18, 20, 29, 73, 99, -75}
, {-53, 84, -119, 24, -32, -6, 29, 13}
, {61, 11, 51, -58, 1, -118, -61, 0}
, {14, -127, -44, 85, -10, -69, -71, -71}
}
, {{8, -83, 42, -9, 115, 35, 59, -40}
, {18, -27, 51, 48, 125, 60, 36, -127}
, {50, 72, 0, -56, -39, 45, -88, 15}
, {-66, -67, 55, -58, 32, -38, 109, -67}
, {-65, -57, 77, -54, 12, 20, 73, 24}
, {-51, -8, 75, -3, 35, 54, 29, 92}
, {-70, -20, -74, -63, -64, -18, 37, 1}
, {-10, 50, -18, -57, 106, -35, -77, -5}
}
, {{-2, 90, -23, -41, -32, -6, 32, 85}
, {1, -143, -16, 8, 89, -89, 61, -22}
, {47, -54, -116, -36, 89, 74, -30, 20}
, {-55, 6, 27, -49, -74, -46, 38, -71}
, {4, 18, -10, 45, 25, 3, 43, 98}
, {38, -96, -52, 51, 103, -27, 47, -36}
, {-111, -71, -25, 29, 103, 45, 62, -58}
, {-10, 3, -61, 56, -50, -37, 4, 109}
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


const int16_t conv1d_2_bias[CONV_FILTERS] = {0, -5, 25, -10, 21, 12, 14, -21, -6, -4, 13, -13, -5, -5, -18, -20, 30, -4, -22, -5, -2, 15, 0, -3, 3, -20, -9, -7, 10, -14, -17, 2}
;

const int16_t conv1d_2_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-24, 86, 23, -17}
, {4, 117, -72, 28}
, {-7, -23, 61, -26}
, {73, -7, -106, 29}
, {-95, -53, -68, 137}
, {25, -49, 75, -66}
, {26, 63, 116, 27}
, {62, -46, -42, -39}
, {51, -61, -50, 79}
, {54, -116, 120, 36}
, {92, 138, 0, 90}
, {-56, 37, 52, -63}
, {-9, -54, 6, 26}
, {-6, -38, -83, 24}
, {-92, -30, -34, 100}
, {-77, -99, 37, -14}
}
, {{82, -105, 88, 108}
, {-135, 17, -8, 90}
, {67, -59, 39, -26}
, {-6, -31, 63, 8}
, {-45, 11, 70, -43}
, {-77, -102, -16, 104}
, {1, -35, -60, 96}
, {50, 23, -96, 93}
, {-94, 34, -1, -74}
, {-15, 69, -132, 23}
, {-53, 85, -10, -121}
, {66, -41, 69, -56}
, {55, -54, -10, -72}
, {-19, -60, 39, 127}
, {44, 83, -45, 35}
, {58, 23, 55, -33}
}
, {{46, -60, 60, 79}
, {74, -83, 39, -17}
, {-41, -67, 43, -42}
, {-33, -37, 17, -84}
, {81, -6, 58, 113}
, {64, -30, -9, -31}
, {11, -28, -19, 18}
, {60, -13, 7, 76}
, {-8, 80, -58, 0}
, {-8, 29, 87, -52}
, {-18, -12, -6, 161}
, {97, -47, -104, 18}
, {-122, -8, -171, 31}
, {-22, 67, 8, -42}
, {-83, 51, -30, 7}
, {110, -28, 55, 61}
}
, {{-102, -38, -30, 78}
, {81, -57, 89, -29}
, {17, -6, -18, 15}
, {52, 29, -79, -5}
, {-124, -31, -95, -84}
, {112, 83, 44, -56}
, {86, 59, 108, -58}
, {48, -76, -13, -90}
, {-31, -15, -46, 8}
, {-31, 55, 85, 110}
, {-4, -12, -115, -112}
, {-70, 88, 3, 65}
, {-93, 44, -36, 41}
, {-21, -27, -6, 97}
, {56, 73, -11, 50}
, {-90, -24, -61, 18}
}
, {{162, 27, 12, 26}
, {34, 68, 39, -69}
, {-30, -14, 51, -8}
, {57, -56, -30, -41}
, {-11, -42, -91, 77}
, {0, -136, 18, 24}
, {28, -28, 72, -59}
, {26, 56, -84, -113}
, {-80, 73, -38, 117}
, {-10, -34, -63, -39}
, {19, -21, 2, -69}
, {42, 4, 23, 110}
, {-46, -36, 51, -38}
, {44, -70, 75, 72}
, {10, 66, -6, -91}
, {-94, 95, -82, -71}
}
, {{-10, 20, -75, 51}
, {69, -56, 44, 34}
, {38, 73, -91, -100}
, {13, -43, 14, -55}
, {59, 32, -25, 115}
, {4, -99, 1, 46}
, {12, -13, -124, 97}
, {-24, -70, -15, -48}
, {29, 70, -41, -137}
, {110, -107, 5, 93}
, {176, -74, -11, 72}
, {-28, -27, 12, 77}
, {16, -103, 91, -12}
, {37, -181, -51, 11}
, {95, 9, -7, -16}
, {67, -91, -20, -55}
}
, {{30, 44, 62, -67}
, {94, 33, 41, 39}
, {61, -57, 13, -15}
, {-23, -25, -25, -43}
, {17, -131, 17, -17}
, {-25, -49, 28, -63}
, {14, -50, -89, 38}
, {-14, -66, -37, -102}
, {32, -75, -13, -43}
, {-49, 37, 103, 178}
, {83, 29, 21, -18}
, {110, 105, 59, -27}
, {93, 31, 116, -64}
, {18, -17, -70, 30}
, {98, -90, 78, -42}
, {79, -75, -39, -5}
}
, {{72, -95, 63, 13}
, {-120, -65, -71, 64}
, {20, -27, 154, 125}
, {-39, -8, -54, 104}
, {-71, 65, 116, -6}
, {-44, -70, -54, 33}
, {-72, 20, 30, 77}
, {-40, 51, -32, 71}
, {-49, 85, 70, 25}
, {-114, -115, -22, -30}
, {-70, 17, -51, -9}
, {59, -72, 59, -137}
, {-46, -40, -58, -53}
, {25, 51, -17, 29}
, {-22, 86, -9, 12}
, {-35, 117, -41, 77}
}
, {{124, 4, -57, -59}
, {90, 53, -42, -81}
, {-134, 128, 44, 44}
, {1, 17, 89, 68}
, {-98, 33, -40, -21}
, {64, -60, -43, -16}
, {-27, 43, 36, -73}
, {-101, -56, -43, 127}
, {40, 17, -35, 53}
, {-90, 31, -21, 12}
, {3, 41, -65, 14}
, {72, -61, 54, 76}
, {51, -81, -146, -72}
, {15, -95, -18, 72}
, {-39, -6, -41, -56}
, {-24, 39, 56, -86}
}
, {{-15, 75, -15, 47}
, {44, 19, -4, -97}
, {98, 61, 51, 4}
, {-145, 11, 37, -32}
, {80, -20, 77, 80}
, {27, -58, 9, 19}
, {-92, -22, -49, -127}
, {-9, -4, 129, 54}
, {-46, 112, -110, -32}
, {-16, -109, 31, -120}
, {13, 39, 3, 3}
, {-26, -76, 46, 52}
, {33, 30, 108, 100}
, {57, -43, 14, 7}
, {-24, -94, -70, 51}
, {-83, -47, 12, -5}
}
, {{-52, -12, -53, -3}
, {-14, 46, -14, -95}
, {-1, 31, -5, 76}
, {55, 101, 51, 50}
, {-36, 85, -41, 8}
, {-28, 58, -59, 13}
, {0, 14, 54, 86}
, {45, 55, 45, 5}
, {-20, 104, -79, -35}
, {27, 116, -66, -7}
, {-117, -41, 20, -107}
, {75, -81, -5, -48}
, {91, -38, -32, 27}
, {130, -90, -73, 6}
, {95, 91, -166, -87}
, {-92, 42, -17, -77}
}
, {{-9, -108, -1, 79}
, {-124, -87, -4, -145}
, {41, 44, -90, -91}
, {44, -46, 43, 61}
, {-19, 53, 5, -69}
, {50, 116, 39, -28}
, {49, 45, 50, -28}
, {-28, 54, 19, -6}
, {77, -103, 59, 80}
, {15, -55, 22, -62}
, {-123, -89, 7, -37}
, {-20, 25, -72, -94}
, {19, -137, 41, -139}
, {97, 70, -65, 106}
, {-4, 38, 65, 21}
, {-19, 25, 5, -40}
}
, {{-65, 42, 45, -39}
, {40, 69, 32, 86}
, {119, 55, 79, 13}
, {-20, -130, 28, 21}
, {-24, -80, -113, -44}
, {-90, -20, 41, 46}
, {-130, 0, -67, -63}
, {18, 44, 52, 71}
, {46, -112, 46, -27}
, {-78, -36, -66, 133}
, {53, 100, 128, 85}
, {-105, -1, 33, -12}
, {-79, 16, 59, -40}
, {-63, -72, 93, 13}
, {76, -40, -91, -18}
, {-21, 27, 48, 83}
}
, {{-18, 54, -12, -1}
, {17, -70, -40, 68}
, {-35, -37, -54, -113}
, {59, 60, -56, -37}
, {-45, -29, -19, -2}
, {16, -121, -80, -19}
, {8, 43, 30, -45}
, {-10, -68, 48, 13}
, {-17, -42, -24, 70}
, {8, 127, -29, -21}
, {39, 74, -81, 89}
, {95, -69, 14, -118}
, {105, 49, -6, 45}
, {32, -11, 113, 10}
, {52, 29, -36, -125}
, {-66, 17, 60, -24}
}
, {{33, 133, -44, -131}
, {37, -50, -23, 68}
, {-78, 2, -79, 20}
, {-32, -55, -59, 35}
, {-112, 40, 43, 38}
, {91, 53, 117, -43}
, {-23, 5, -4, 2}
, {9, 43, 74, 81}
, {43, 44, 26, -55}
, {-97, -64, 12, -39}
, {24, 66, 28, 161}
, {-72, 112, 98, 96}
, {31, -77, 84, -10}
, {-37, 22, 2, 3}
, {-15, 65, 62, -16}
, {-78, -16, 15, 54}
}
, {{-27, -47, -111, 84}
, {87, 80, -12, 15}
, {-54, 52, -10, -32}
, {-126, 51, 8, 27}
, {69, -7, 40, -13}
, {6, 95, -104, -83}
, {20, 8, -1, 19}
, {-91, 37, -84, 25}
, {-61, -11, 23, 38}
, {-104, -84, 106, -82}
, {-65, 58, 17, 62}
, {2, 69, 63, 48}
, {-48, 89, 46, -46}
, {-11, 14, -37, 34}
, {62, 77, -4, 30}
, {3, -69, -34, -75}
}
, {{42, -144, 15, -58}
, {80, 35, -42, -58}
, {-103, 31, -63, -8}
, {-65, 4, -67, -118}
, {17, -47, -18, 49}
, {12, -8, -78, 42}
, {-40, -39, -87, -64}
, {-73, -38, 19, 22}
, {79, -109, 64, -30}
, {-50, 87, 160, -42}
, {-63, -148, -60, -4}
, {62, 11, -4, 10}
, {114, -58, -82, 0}
, {-95, -89, -5, -88}
, {72, 47, 124, 20}
, {-10, -81, 40, 119}
}
, {{-47, -76, -149, 41}
, {29, -94, 65, 45}
, {-56, -10, 1, -77}
, {102, 31, -98, -19}
, {41, -54, -55, -77}
, {19, -52, -76, -4}
, {-29, 189, 38, 65}
, {10, 71, 86, -50}
, {-91, 34, 39, -23}
, {-22, 83, 24, -118}
, {0, -49, 0, -8}
, {68, 33, 130, 21}
, {26, 99, -32, -48}
, {9, 17, 38, -114}
, {-16, -18, 4, -1}
, {-18, -74, -53, 37}
}
, {{-13, -51, 87, 43}
, {4, 89, 27, 13}
, {-36, -32, -23, 31}
, {36, -74, 0, 0}
, {27, -101, -23, -20}
, {120, 9, 131, 108}
, {-20, 48, -107, 38}
, {-18, -55, -12, -30}
, {-14, -72, -67, -50}
, {11, -44, -11, 93}
, {-99, -46, -23, 142}
, {73, -51, -66, 28}
, {37, 57, -16, -84}
, {10, 34, -3, 31}
, {-8, 42, -56, 85}
, {38, -64, 9, 57}
}
, {{73, -65, -7, -62}
, {-102, 42, 55, 116}
, {-49, -71, -25, -57}
, {23, 94, 22, -82}
, {133, -82, -25, 7}
, {1, 12, 8, -35}
, {-77, -106, 75, -40}
, {39, -104, -71, 71}
, {57, -24, -30, -40}
, {55, 54, -82, 16}
, {-125, 59, 89, 39}
, {49, -70, -26, 37}
, {111, -110, -99, 93}
, {51, -27, -85, 30}
, {74, -25, 80, -8}
, {10, -122, -118, 70}
}
, {{111, -10, -21, 14}
, {27, -47, 74, 25}
, {-4, -54, 10, -7}
, {-77, -77, -48, -45}
, {67, 45, 30, -57}
, {76, -35, -80, 87}
, {-78, 99, 64, -42}
, {33, 17, 42, -76}
, {-33, 43, 80, -17}
, {100, 3, -39, 31}
, {-81, -94, -78, 32}
, {-105, 74, 105, 69}
, {-49, -12, -31, -83}
, {-13, -36, -69, -49}
, {2, 45, -11, 33}
, {100, -76, -61, -29}
}
, {{-79, -132, 57, -17}
, {0, 0, -67, -53}
, {75, -20, 48, 45}
, {42, -63, -120, -78}
, {-71, 47, 89, 41}
, {91, -78, 54, -65}
, {-40, -35, 129, 0}
, {-49, -58, -32, 66}
, {-98, 64, 94, 68}
, {124, 14, 12, 25}
, {12, -11, 39, -72}
, {-102, -34, -26, -28}
, {17, 30, -46, 63}
, {12, -24, -28, 50}
, {-18, 3, -24, 7}
, {72, -20, -1, 112}
}
, {{-61, 49, -54, 105}
, {43, 58, -30, 102}
, {99, 76, 104, -11}
, {-52, -42, 78, 76}
, {0, 75, -40, 18}
, {-79, -98, -10, -84}
, {70, -107, -42, -11}
, {15, 42, 82, 0}
, {-62, -12, 37, -110}
, {28, -44, -14, 37}
, {-57, 59, 64, 21}
, {5, 17, 10, -33}
, {43, -3, -67, -52}
, {-88, -18, 74, -19}
, {-30, -53, 50, -23}
, {75, -19, 5, -28}
}
, {{171, -153, 43, -16}
, {-65, 59, -60, -55}
, {21, 25, -59, -131}
, {-8, -2, -21, -3}
, {32, -62, -62, 12}
, {91, 32, -89, 44}
, {-86, -14, 4, -62}
, {21, 10, 0, -63}
, {20, 51, 75, 23}
, {-73, 111, -42, 140}
, {-69, 99, 5, 116}
, {39, 77, 7, -68}
, {38, -27, -89, 21}
, {-65, 63, 34, -70}
, {55, -76, -100, 46}
, {27, -55, 42, 59}
}
, {{-57, 61, 71, -50}
, {27, -135, 53, -82}
, {-155, -29, -32, -72}
, {68, 0, 73, -64}
, {72, 110, 31, -1}
, {-58, 69, -6, -32}
, {76, 27, -121, -110}
, {-75, 26, -36, -16}
, {87, 13, -47, 83}
, {-26, 30, -66, -36}
, {19, 15, 22, 27}
, {19, -38, -28, 44}
, {60, 74, -44, -68}
, {101, -22, -38, 60}
, {-30, 29, -53, -40}
, {52, -74, -124, -110}
}
, {{-5, 51, 69, 66}
, {-96, -63, -85, 92}
, {119, 22, 47, 35}
, {-48, -9, 81, -63}
, {-88, 30, -111, -42}
, {11, -56, 92, 96}
, {-59, -45, -31, 65}
, {69, -43, 76, 61}
, {46, 35, -38, -20}
, {-42, -106, 30, 18}
, {-23, -179, 22, -57}
, {-61, 0, 43, 21}
, {19, -63, -45, -21}
, {-17, -118, 17, 86}
, {115, 37, 28, 22}
, {5, 26, -36, -43}
}
, {{127, -77, 95, -59}
, {1, -70, 27, 59}
, {38, -54, 51, 108}
, {-91, 95, 43, -25}
, {-53, 126, -81, -52}
, {-137, -2, 7, 70}
, {-30, -2, -122, -8}
, {-89, 128, 104, 82}
, {5, -97, 49, -46}
, {26, -33, 16, -49}
, {7, -40, -83, 31}
, {58, 54, -58, -3}
, {118, 33, 30, 18}
, {-102, -95, -77, -58}
, {-8, -65, 42, 30}
, {-58, 19, 22, 48}
}
, {{9, -102, 26, -19}
, {54, 20, -80, -39}
, {91, -26, 26, 34}
, {-19, -118, 0, 72}
, {-8, -9, 86, -62}
, {68, -56, -52, 102}
, {-53, -52, 0, 11}
, {0, 101, -6, 127}
, {0, 65, -80, 60}
, {-71, 58, 19, 61}
, {77, -27, 55, -43}
, {-57, 58, -67, -4}
, {30, 43, -35, -23}
, {114, -68, 79, 55}
, {-20, -119, 74, -28}
, {16, -15, -5, -53}
}
, {{-43, 52, -10, -35}
, {51, -101, -118, 62}
, {63, -40, -92, -128}
, {-73, -6, -52, 4}
, {-43, 36, -78, 166}
, {27, -65, -135, 20}
, {-137, 77, -67, 31}
, {-57, 122, -36, -49}
, {54, -64, 86, -24}
, {-27, 54, 15, -111}
, {-4, 14, 4, -101}
, {-45, 105, -71, 117}
, {91, -31, -75, 1}
, {-2, 79, -55, -13}
, {-40, 2, -109, 144}
, {-28, -33, 9, 68}
}
, {{21, 118, 87, -22}
, {-45, -26, 81, -88}
, {-27, 17, 99, 48}
, {-27, 0, 70, -31}
, {-63, -45, -37, 2}
, {43, 17, 83, -16}
, {54, 8, 97, -73}
, {-31, -109, 41, 90}
, {64, 102, -65, -51}
, {87, 38, -10, 53}
, {-107, 2, -51, -2}
, {-66, -100, -12, 46}
, {-133, 96, 52, 56}
, {106, 19, -94, -51}
, {70, 52, -103, 6}
, {19, -79, 45, -21}
}
, {{-54, 36, -80, -51}
, {-33, 42, 55, -25}
, {-73, 52, -101, 38}
, {1, 36, -101, 61}
, {13, 32, -58, -71}
, {-46, -25, -29, 68}
, {47, 91, -29, -90}
, {22, -4, 168, 45}
, {6, 13, -83, 105}
, {111, 114, 4, -127}
, {80, 19, 58, -34}
, {-44, -118, -25, 93}
, {30, -110, 52, -76}
, {-9, -33, -60, -37}
, {-5, 87, -106, 91}
, {-66, -66, 97, -28}
}
, {{48, 54, -7, 9}
, {-32, -61, -19, -33}
, {60, 146, 59, -47}
, {-26, -65, 33, -99}
, {-99, -55, -49, -12}
, {31, -105, 4, -24}
, {-27, -45, 14, 47}
, {5, -28, 9, 80}
, {14, 64, 81, -75}
, {91, 96, -98, -7}
, {6, 45, -24, -36}
, {-31, 12, 87, 97}
, {33, 13, -57, 5}
, {-41, -90, -42, 10}
, {35, -4, -6, -42}
, {43, 111, 32, 99}
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


const int16_t conv1d_3_bias[CONV_FILTERS] = {0, 17, -7, -12, 8, 2, -7, -4, -22, 5, 9, 20, 13, 7, -9, -16, 16, 1, 0, -2, -18, 5, 0, -14, -8, -9, -1, 7, 3, -6, 6, -5, 2, -1, 4, -7, 4, -2, 21, 13, -8, 2, 14, -3, -7, 16, -5, -6, 1, 3, -19, -9, 6, -10, 1, -4, 33, -17, -17, 1, -4, 6, -17, 11}
;

const int16_t conv1d_3_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-63, 39}
, {57, -39}
, {-53, 18}
, {53, 11}
, {40, -29}
, {95, 7}
, {0, 103}
, {-81, 28}
, {74, -43}
, {-113, 2}
, {-123, -81}
, {-17, -62}
, {-64, -14}
, {82, 93}
, {-76, 53}
, {-93, 44}
, {-53, -60}
, {34, 21}
, {-28, 35}
, {84, 44}
, {-24, -18}
, {-24, 41}
, {-2, 62}
, {-2, 48}
, {-69, -22}
, {-52, -5}
, {49, -39}
, {-2, 63}
, {113, 33}
, {33, -5}
, {-48, -45}
, {73, 74}
}
, {{42, 80}
, {53, -84}
, {-156, 47}
, {54, -75}
, {-59, 92}
, {3, -27}
, {95, -2}
, {62, -71}
, {-114, -11}
, {-15, 73}
, {12, -60}
, {-110, -24}
, {47, -70}
, {16, -59}
, {-43, 47}
, {59, 102}
, {82, 122}
, {47, -14}
, {58, 90}
, {-36, 42}
, {-15, 11}
, {5, 116}
, {-5, -93}
, {-9, -25}
, {-15, -83}
, {-20, -53}
, {-41, 78}
, {36, -47}
, {-27, -59}
, {-50, 2}
, {-75, -3}
, {-2, 34}
}
, {{23, 84}
, {-12, 13}
, {63, 117}
, {-58, -54}
, {-3, 66}
, {-53, -163}
, {-107, -27}
, {-58, -68}
, {-49, 75}
, {-86, -8}
, {25, 76}
, {-67, -82}
, {67, 87}
, {-96, 7}
, {-50, -61}
, {18, 74}
, {19, -6}
, {25, 65}
, {60, 85}
, {-58, 67}
, {-65, -11}
, {32, 83}
, {-103, 50}
, {-25, 80}
, {105, 110}
, {45, 57}
, {78, -55}
, {-16, -5}
, {-25, 25}
, {86, 2}
, {-2, -51}
, {68, 54}
}
, {{-106, 82}
, {58, -48}
, {-36, 35}
, {71, -79}
, {18, -68}
, {-36, -54}
, {-52, 48}
, {91, 37}
, {-10, -24}
, {73, 79}
, {8, -29}
, {90, -84}
, {24, 41}
, {20, 66}
, {64, -15}
, {106, -9}
, {25, 34}
, {-35, -106}
, {-7, 66}
, {113, 11}
, {-17, 73}
, {-89, 17}
, {-14, 21}
, {-85, 56}
, {-4, -31}
, {12, -25}
, {-31, 1}
, {-87, 62}
, {-3, 68}
, {-113, -53}
, {69, -90}
, {-6, 0}
}
, {{114, 45}
, {-5, 16}
, {53, 33}
, {-12, 4}
, {-13, 115}
, {-9, -54}
, {-32, -75}
, {33, -8}
, {3, 39}
, {-64, 5}
, {-83, 7}
, {-106, 115}
, {79, -28}
, {91, -18}
, {-14, -3}
, {38, 0}
, {82, 10}
, {73, -76}
, {31, -23}
, {110, 17}
, {-20, 109}
, {73, 26}
, {37, -103}
, {-42, -10}
, {0, -104}
, {9, -68}
, {59, -137}
, {-35, -14}
, {65, 38}
, {-103, 46}
, {61, -128}
, {-90, 85}
}
, {{25, -8}
, {-44, -90}
, {104, 76}
, {45, 103}
, {59, 123}
, {41, 39}
, {45, 54}
, {24, 44}
, {-10, 24}
, {49, 23}
, {-1, -38}
, {-6, -2}
, {18, -62}
, {-60, 23}
, {-63, -42}
, {7, -35}
, {44, 120}
, {-98, -60}
, {16, 15}
, {-94, 79}
, {71, -94}
, {-32, -10}
, {53, 54}
, {-31, 34}
, {2, -61}
, {-36, -12}
, {93, 48}
, {55, 84}
, {-37, 9}
, {-20, -20}
, {-26, -63}
, {-23, -81}
}
, {{-46, 23}
, {56, 63}
, {-56, -94}
, {6, 19}
, {43, -7}
, {119, 98}
, {81, 51}
, {75, -66}
, {90, -10}
, {-53, 78}
, {-106, -35}
, {-51, 77}
, {-89, -69}
, {65, -53}
, {36, 62}
, {-48, 18}
, {111, -6}
, {-38, -28}
, {20, -57}
, {-55, -112}
, {-11, -68}
, {-42, 18}
, {93, 9}
, {84, 0}
, {-12, 118}
, {67, 151}
, {-5, 53}
, {-69, 2}
, {18, 112}
, {19, 55}
, {-48, -7}
, {-31, 14}
}
, {{110, -61}
, {8, -26}
, {-59, 18}
, {-19, -71}
, {-67, -80}
, {-33, 25}
, {54, -59}
, {-37, -105}
, {-106, -107}
, {25, 76}
, {74, 13}
, {69, 135}
, {89, 57}
, {106, 68}
, {-32, 22}
, {101, 56}
, {80, 7}
, {126, -14}
, {54, -83}
, {27, 65}
, {26, 67}
, {-57, -38}
, {-75, 75}
, {22, -17}
, {68, 117}
, {-5, 36}
, {96, -16}
, {56, -84}
, {-31, -5}
, {-86, -149}
, {4, -2}
, {-17, -28}
}
, {{116, -148}
, {-96, 24}
, {-86, -14}
, {13, -46}
, {76, -81}
, {1, 0}
, {52, 122}
, {-56, 36}
, {42, -8}
, {45, -13}
, {45, -79}
, {43, 36}
, {-32, -38}
, {24, 7}
, {-77, 45}
, {-7, 41}
, {-12, 106}
, {29, 78}
, {-16, 92}
, {129, -31}
, {-25, 61}
, {3, 45}
, {-70, -14}
, {-78, -32}
, {-101, 40}
, {77, 121}
, {-73, 13}
, {-56, -86}
, {33, 11}
, {34, -34}
, {-92, 91}
, {-164, 13}
}
, {{35, -77}
, {-6, 72}
, {75, -109}
, {-91, 54}
, {-56, 23}
, {30, -72}
, {-1, -91}
, {3, -60}
, {79, 86}
, {-21, -43}
, {-49, 116}
, {-35, 59}
, {99, -37}
, {12, 36}
, {39, 1}
, {-72, -84}
, {157, 113}
, {39, -92}
, {54, -17}
, {19, -92}
, {-89, 38}
, {-35, 29}
, {108, 65}
, {-14, -66}
, {14, 14}
, {-15, -73}
, {39, 76}
, {-64, -64}
, {-73, -30}
, {-69, 68}
, {70, -16}
, {37, 18}
}
, {{-27, -51}
, {84, 134}
, {-106, 35}
, {57, 61}
, {-53, 63}
, {-96, 57}
, {34, -69}
, {118, -6}
, {-24, -15}
, {153, 67}
, {103, 34}
, {11, -68}
, {69, -145}
, {-80, -31}
, {-84, -12}
, {52, 1}
, {-99, -21}
, {75, 72}
, {22, -86}
, {92, 100}
, {-103, -66}
, {15, 131}
, {-61, -12}
, {-119, -45}
, {-8, 33}
, {81, 10}
, {-46, 32}
, {52, 130}
, {0, -52}
, {-33, 25}
, {79, 163}
, {-43, -64}
}
, {{-62, -37}
, {59, 35}
, {-11, 97}
, {21, -35}
, {118, 100}
, {-110, -23}
, {8, 56}
, {10, 10}
, {-57, 35}
, {49, -79}
, {142, 27}
, {78, 74}
, {-111, -85}
, {-19, 42}
, {-102, 82}
, {-21, 64}
, {48, -42}
, {-68, -20}
, {-35, -15}
, {61, -8}
, {93, -14}
, {-34, -120}
, {-13, 59}
, {88, -41}
, {39, 10}
, {-105, 103}
, {-25, -80}
, {21, 46}
, {6, 65}
, {1, -110}
, {12, 88}
, {-10, -1}
}
, {{-62, -52}
, {67, 51}
, {74, 15}
, {-59, 0}
, {-30, 27}
, {17, 51}
, {-90, 105}
, {110, -81}
, {66, 111}
, {88, -38}
, {-6, -57}
, {-23, -153}
, {-52, -110}
, {113, 53}
, {-16, -3}
, {-53, -70}
, {114, 5}
, {-50, 110}
, {-53, -106}
, {-39, 34}
, {6, 89}
, {119, -5}
, {65, -1}
, {-110, -61}
, {-8, -46}
, {-60, 26}
, {-34, -6}
, {-21, 55}
, {-86, 28}
, {52, -93}
, {-104, 147}
, {-2, -83}
}
, {{30, 62}
, {26, -48}
, {-37, 79}
, {-47, -85}
, {-38, -34}
, {-37, -31}
, {61, 58}
, {-14, -112}
, {48, -49}
, {18, 78}
, {4, 107}
, {25, 90}
, {6, 84}
, {120, 32}
, {81, -97}
, {-81, -40}
, {3, 92}
, {-80, 33}
, {-16, 39}
, {18, -111}
, {0, 20}
, {-61, 20}
, {76, 15}
, {-54, -144}
, {36, 106}
, {-73, 13}
, {-1, 32}
, {-41, -76}
, {117, 26}
, {58, 82}
, {46, 137}
, {-55, 74}
}
, {{25, 43}
, {-9, 96}
, {-60, 94}
, {-44, -100}
, {55, -76}
, {84, 109}
, {-1, 87}
, {-38, -18}
, {111, 48}
, {-45, -52}
, {80, -47}
, {76, 22}
, {64, 27}
, {103, 65}
, {-45, -20}
, {37, 98}
, {60, -84}
, {-46, 11}
, {41, 70}
, {-51, -43}
, {32, -62}
, {9, -105}
, {84, 108}
, {32, 27}
, {-52, -11}
, {61, -59}
, {-26, -104}
, {28, -32}
, {-59, 61}
, {35, 3}
, {-21, 37}
, {9, -38}
}
, {{16, 124}
, {-57, -98}
, {90, -34}
, {45, 85}
, {-26, -48}
, {12, 88}
, {-46, 21}
, {55, 3}
, {-29, -105}
, {9, -30}
, {-10, -92}
, {9, -51}
, {24, -73}
, {100, 80}
, {97, 90}
, {93, -57}
, {-82, 43}
, {28, -43}
, {77, 52}
, {-5, -88}
, {52, 16}
, {61, -48}
, {78, 49}
, {-26, -22}
, {27, -72}
, {-94, 64}
, {-30, -125}
, {41, -6}
, {28, 75}
, {68, 87}
, {-46, -33}
, {14, -73}
}
, {{49, -106}
, {-116, -17}
, {89, -70}
, {16, 83}
, {19, 24}
, {-3, 75}
, {10, 101}
, {-37, -20}
, {-30, 11}
, {-13, -67}
, {-62, 93}
, {51, -18}
, {-118, -33}
, {-91, -152}
, {-85, -37}
, {0, 20}
, {71, 96}
, {-30, 62}
, {-59, -10}
, {-128, 36}
, {21, 69}
, {-96, 76}
, {-6, 15}
, {108, -30}
, {-28, 62}
, {-18, 32}
, {79, -36}
, {-77, -67}
, {-56, 122}
, {-107, 77}
, {3, 109}
, {-18, 109}
}
, {{-64, 0}
, {-3, -32}
, {65, -135}
, {-25, -61}
, {-134, -49}
, {107, 29}
, {66, -2}
, {-16, -25}
, {-84, 53}
, {32, -38}
, {-48, 23}
, {100, -14}
, {7, -52}
, {-3, 7}
, {89, -45}
, {-73, 64}
, {-67, 0}
, {-52, 39}
, {-8, -14}
, {149, 85}
, {78, 24}
, {37, -38}
, {-72, 25}
, {20, 30}
, {124, 72}
, {-77, -37}
, {-59, 98}
, {85, 49}
, {-17, 78}
, {43, -108}
, {-107, 12}
, {-33, 81}
}
, {{-38, 0}
, {44, -56}
, {69, -22}
, {6, -61}
, {-7, -54}
, {-84, 83}
, {64, 103}
, {-8, -59}
, {-35, 65}
, {73, -45}
, {-116, -54}
, {38, -12}
, {-50, 9}
, {-58, -4}
, {32, 25}
, {23, 2}
, {103, 49}
, {-37, 62}
, {4, -74}
, {-100, 60}
, {-42, -76}
, {32, 74}
, {50, 77}
, {-104, -54}
, {83, 9}
, {46, -73}
, {-31, 98}
, {6, 40}
, {-3, 27}
, {64, -46}
, {-51, 35}
, {67, -10}
}
, {{-35, -27}
, {-29, 32}
, {82, -49}
, {-95, 39}
, {83, 56}
, {-90, 89}
, {57, 64}
, {82, 18}
, {16, -107}
, {-122, 69}
, {-60, 17}
, {-27, 36}
, {-99, 67}
, {87, 24}
, {-104, 14}
, {-64, -26}
, {52, -27}
, {-82, -66}
, {29, -2}
, {123, 72}
, {-31, -73}
, {130, -21}
, {-62, -36}
, {85, 41}
, {-117, -83}
, {-26, 0}
, {95, -79}
, {54, -39}
, {-85, 124}
, {83, 92}
, {38, -51}
, {9, -99}
}
, {{49, 70}
, {-26, -87}
, {19, -15}
, {-97, -68}
, {-51, 55}
, {-102, -30}
, {-60, -61}
, {41, 82}
, {20, 76}
, {-9, 64}
, {7, 62}
, {82, -54}
, {43, 8}
, {-61, -64}
, {23, -26}
, {55, 58}
, {-68, -58}
, {91, -19}
, {1, -36}
, {46, -42}
, {104, -18}
, {-66, -5}
, {-54, 111}
, {62, 79}
, {50, 91}
, {50, -10}
, {-32, -5}
, {-57, -58}
, {-62, 38}
, {0, 57}
, {-77, -11}
, {28, 13}
}
, {{-11, 81}
, {47, 19}
, {-46, 0}
, {23, 84}
, {-137, 9}
, {2, 26}
, {16, 10}
, {48, 1}
, {-110, -39}
, {-22, -58}
, {50, -89}
, {-35, -82}
, {69, 21}
, {-55, 24}
, {10, 62}
, {25, 85}
, {90, 74}
, {59, 44}
, {100, -63}
, {73, 120}
, {141, -9}
, {5, 74}
, {7, 13}
, {-12, -61}
, {-58, -7}
, {4, -59}
, {55, -79}
, {-6, 17}
, {120, 42}
, {-56, -54}
, {55, 91}
, {8, -90}
}
, {{-42, -1}
, {68, -2}
, {23, 82}
, {54, 53}
, {-9, 136}
, {-102, -91}
, {56, 46}
, {-108, -72}
, {25, 14}
, {-35, -36}
, {71, -4}
, {-82, 82}
, {-1, -104}
, {-1, 66}
, {-20, 39}
, {8, -41}
, {-12, -55}
, {125, 118}
, {-49, 6}
, {-27, 17}
, {71, 48}
, {64, 29}
, {-89, -56}
, {28, 46}
, {93, 1}
, {-63, -62}
, {53, 33}
, {22, 16}
, {-109, 84}
, {21, -18}
, {-56, 43}
, {3, 51}
}
, {{56, 45}
, {77, 93}
, {68, 61}
, {-90, 100}
, {82, -11}
, {-159, -121}
, {-25, -51}
, {56, -82}
, {-64, -28}
, {79, -17}
, {-9, 61}
, {-2, 78}
, {109, 116}
, {47, -56}
, {-14, 62}
, {-53, 14}
, {78, -2}
, {-4, 40}
, {-65, 92}
, {-36, 3}
, {41, -58}
, {42, -64}
, {-82, -96}
, {-95, -60}
, {31, -83}
, {-26, 1}
, {56, 64}
, {-101, -22}
, {116, 68}
, {37, 28}
, {77, -66}
, {72, -101}
}
, {{-19, 81}
, {-95, 2}
, {127, 115}
, {81, -10}
, {77, 43}
, {7, 53}
, {36, -33}
, {16, -9}
, {2, 57}
, {-54, -15}
, {75, -19}
, {62, 106}
, {-7, 56}
, {86, 0}
, {5, -46}
, {1, 46}
, {158, -46}
, {-37, 131}
, {11, 27}
, {-62, 0}
, {-71, 83}
, {30, -63}
, {-17, -107}
, {-62, 20}
, {-28, 56}
, {-6, -57}
, {-28, -97}
, {-59, 25}
, {4, -24}
, {76, -11}
, {9, 48}
, {-137, -66}
}
, {{-55, 3}
, {114, 141}
, {-95, -29}
, {-22, 92}
, {17, -4}
, {-58, -74}
, {29, -49}
, {69, -56}
, {102, 66}
, {40, -10}
, {118, -24}
, {17, 74}
, {29, -86}
, {-109, -7}
, {-21, 74}
, {56, -30}
, {101, -157}
, {-42, 27}
, {-55, 69}
, {42, -86}
, {-79, -10}
, {-35, -74}
, {-74, -15}
, {48, -88}
, {73, 100}
, {108, 130}
, {-64, -68}
, {100, 52}
, {-88, -85}
, {7, 34}
, {16, -81}
, {21, -84}
}
, {{115, 104}
, {23, 21}
, {-155, 59}
, {-27, -18}
, {-31, -126}
, {-29, 64}
, {88, -57}
, {18, -1}
, {-96, 19}
, {-65, 57}
, {112, 40}
, {10, 3}
, {-26, 9}
, {-1, -62}
, {47, -73}
, {-53, -64}
, {-21, 40}
, {32, 54}
, {-123, 58}
, {-131, 0}
, {-14, -24}
, {-87, 44}
, {76, 92}
, {39, 12}
, {79, 87}
, {109, 31}
, {32, -53}
, {9, 42}
, {-117, 32}
, {16, 73}
, {37, -63}
, {72, 20}
}
, {{63, 25}
, {102, -34}
, {13, -2}
, {24, -62}
, {-75, -87}
, {85, 88}
, {73, -9}
, {66, -39}
, {-120, -88}
, {-22, -4}
, {55, -29}
, {-53, 76}
, {37, 5}
, {-64, -87}
, {38, -76}
, {-57, -56}
, {132, 130}
, {49, 31}
, {75, 0}
, {58, 31}
, {-122, -55}
, {30, 126}
, {-5, -71}
, {37, -21}
, {54, -3}
, {32, -16}
, {-79, 41}
, {9, -64}
, {53, -25}
, {-85, -10}
, {58, 79}
, {-26, 44}
}
, {{-21, -46}
, {84, 56}
, {22, -22}
, {-87, -86}
, {60, -51}
, {16, 68}
, {61, 133}
, {-56, 56}
, {68, 75}
, {-122, -74}
, {77, 6}
, {98, 51}
, {-60, 5}
, {43, -76}
, {-35, -59}
, {-41, -30}
, {122, 24}
, {-44, 85}
, {-54, 25}
, {-15, -1}
, {41, -35}
, {85, 9}
, {-17, 90}
, {-53, 57}
, {-27, 13}
, {-33, 4}
, {91, -28}
, {38, 0}
, {-141, 22}
, {-82, -11}
, {1, -94}
, {75, -40}
}
, {{104, 17}
, {-86, 0}
, {-91, 39}
, {79, 100}
, {107, -10}
, {-122, -34}
, {-151, -37}
, {76, -4}
, {71, 51}
, {34, 19}
, {-10, 67}
, {-39, -105}
, {63, -63}
, {7, -6}
, {91, 31}
, {36, 11}
, {7, 57}
, {76, 89}
, {14, 119}
, {41, 70}
, {-23, -41}
, {-31, 101}
, {80, 0}
, {-57, 5}
, {73, -4}
, {75, 50}
, {58, -52}
, {-144, 39}
, {18, -28}
, {-8, -52}
, {19, 83}
, {14, 42}
}
, {{16, 65}
, {-69, -40}
, {-67, -33}
, {-24, 2}
, {30, -71}
, {174, 214}
, {28, -98}
, {-18, -43}
, {82, -28}
, {62, 102}
, {86, 45}
, {81, -82}
, {22, 11}
, {-8, -51}
, {-118, 84}
, {46, -2}
, {14, -44}
, {-128, -87}
, {-94, -9}
, {0, -10}
, {-82, 55}
, {46, 33}
, {8, 15}
, {-62, -70}
, {-2, -47}
, {117, 9}
, {-46, 48}
, {-32, 85}
, {-12, -27}
, {52, 99}
, {-7, -54}
, {88, 24}
}
, {{32, 99}
, {29, 6}
, {-93, 90}
, {-76, -43}
, {-12, 26}
, {-50, 19}
, {26, -20}
, {35, 77}
, {77, 5}
, {95, 57}
, {-78, -91}
, {-70, 74}
, {87, 37}
, {57, 9}
, {10, 70}
, {12, -11}
, {24, -31}
, {19, -138}
, {-84, 49}
, {16, -37}
, {55, -17}
, {-29, 52}
, {-12, 66}
, {-27, -63}
, {74, 74}
, {-50, -33}
, {-8, 43}
, {36, 59}
, {-149, -117}
, {-60, 57}
, {-49, -81}
, {-24, 37}
}
, {{10, -7}
, {60, 45}
, {39, -66}
, {62, 109}
, {97, -9}
, {-15, 22}
, {30, 46}
, {-49, -40}
, {109, -63}
, {-18, -76}
, {-26, 122}
, {-24, -38}
, {-19, -18}
, {99, -14}
, {-66, 59}
, {-67, -63}
, {12, -92}
, {-24, 79}
, {72, -87}
, {-84, -44}
, {14, 45}
, {64, -121}
, {-67, -53}
, {61, -108}
, {-39, -3}
, {18, -25}
, {-104, -12}
, {11, -3}
, {79, 17}
, {-118, 32}
, {78, -6}
, {0, -40}
}
, {{-71, -55}
, {-2, 25}
, {-42, -63}
, {70, 7}
, {-22, 35}
, {85, 75}
, {-41, 40}
, {99, 84}
, {53, 39}
, {-140, -120}
, {-73, 51}
, {20, -13}
, {70, -40}
, {-34, -54}
, {-75, 6}
, {46, 24}
, {18, 110}
, {53, 47}
, {6, 42}
, {-74, 36}
, {-86, 41}
, {79, -3}
, {4, 21}
, {56, 83}
, {43, -64}
, {-11, 79}
, {72, -31}
, {40, -77}
, {-1, -27}
, {-11, 19}
, {85, -59}
, {-54, -37}
}
, {{22, 108}
, {-57, -33}
, {46, 73}
, {68, -72}
, {69, 50}
, {136, -12}
, {41, 10}
, {82, -28}
, {-90, -44}
, {-112, 34}
, {-45, 80}
, {-49, -34}
, {83, 90}
, {-73, 8}
, {24, 48}
, {-61, 53}
, {-11, -4}
, {33, 16}
, {-46, -42}
, {31, -11}
, {65, 106}
, {-108, -27}
, {-43, 44}
, {25, 3}
, {-55, 0}
, {-45, 6}
, {15, -46}
, {75, 112}
, {-159, -4}
, {-42, -126}
, {-19, 64}
, {58, -34}
}
, {{69, -63}
, {56, -40}
, {62, 71}
, {56, 82}
, {16, 13}
, {-90, 89}
, {-2, 135}
, {-123, 11}
, {-70, 15}
, {-22, -72}
, {97, -41}
, {-68, 14}
, {71, 9}
, {58, -46}
, {-64, -92}
, {-15, -29}
, {127, 142}
, {-60, 85}
, {38, 42}
, {-23, 136}
, {58, 66}
, {27, -85}
, {-1, -92}
, {-32, -67}
, {-18, -70}
, {-44, -60}
, {-83, 65}
, {-87, 19}
, {85, -4}
, {84, 22}
, {65, 39}
, {-8, 64}
}
, {{89, -39}
, {50, 87}
, {-103, 58}
, {36, 35}
, {55, 2}
, {18, 57}
, {117, 31}
, {-75, -39}
, {45, -5}
, {-73, -77}
, {7, 102}
, {7, 28}
, {-101, -93}
, {74, 142}
, {-98, -42}
, {94, 54}
, {79, 28}
, {-13, -1}
, {-78, 75}
, {30, -72}
, {-60, -138}
, {76, -85}
, {96, -73}
, {-16, 44}
, {-87, 33}
, {-29, -99}
, {27, -49}
, {-106, -16}
, {-127, -24}
, {38, 46}
, {39, -33}
, {-64, -85}
}
, {{44, -48}
, {-15, -114}
, {-49, 82}
, {-33, -54}
, {23, -53}
, {-73, 0}
, {80, 63}
, {66, 11}
, {-41, 99}
, {-44, -35}
, {-21, -17}
, {46, -108}
, {-17, -57}
, {-67, -55}
, {-78, 3}
, {-95, -17}
, {23, 58}
, {-82, 27}
, {24, -63}
, {45, 6}
, {-80, -21}
, {11, 13}
, {-107, 132}
, {-26, 78}
, {-58, -69}
, {-84, -116}
, {56, 74}
, {-74, -34}
, {47, 77}
, {86, 98}
, {-20, 32}
, {-82, 57}
}
, {{73, 8}
, {56, 110}
, {-37, -54}
, {47, 78}
, {-38, -67}
, {172, -19}
, {37, -157}
, {114, 127}
, {56, -70}
, {-64, 8}
, {-79, -4}
, {-31, -70}
, {60, -5}
, {89, -114}
, {-13, 30}
, {-84, -29}
, {8, -85}
, {-84, -2}
, {62, -16}
, {-64, -5}
, {31, -97}
, {80, 95}
, {-68, -111}
, {-10, -35}
, {-25, 10}
, {-72, -66}
, {-59, -60}
, {-52, 26}
, {70, 60}
, {-44, 78}
, {45, 56}
, {53, 74}
}
, {{-107, -11}
, {-82, -9}
, {-10, 44}
, {-18, -75}
, {-63, -46}
, {102, 10}
, {30, 51}
, {-29, 22}
, {-56, 83}
, {-75, 38}
, {-80, -19}
, {32, -35}
, {-76, -91}
, {-52, 110}
, {49, 33}
, {113, -49}
, {-103, -35}
, {2, 17}
, {2, -67}
, {6, 7}
, {99, -35}
, {91, 23}
, {21, 39}
, {4, 100}
, {18, 87}
, {67, -109}
, {5, -46}
, {23, 78}
, {51, 65}
, {-41, -104}
, {40, 0}
, {94, -46}
}
, {{120, -9}
, {6, -19}
, {24, 26}
, {-35, -30}
, {-90, 89}
, {-18, -6}
, {27, -89}
, {-4, -35}
, {56, -72}
, {13, 46}
, {-89, -114}
, {-51, 70}
, {55, 49}
, {-85, -102}
, {-42, 25}
, {30, -45}
, {-14, 3}
, {89, 5}
, {-32, -51}
, {-67, 41}
, {-29, -101}
, {72, 55}
, {43, -18}
, {56, -70}
, {-59, -10}
, {67, -7}
, {10, -54}
, {77, 32}
, {138, 28}
, {96, -43}
, {112, 21}
, {-27, 65}
}
, {{9, -29}
, {-155, 12}
, {8, -84}
, {51, -32}
, {17, -42}
, {-22, -83}
, {92, 107}
, {-122, -33}
, {-46, -103}
, {2, 37}
, {-63, 48}
, {-14, -114}
, {-29, 0}
, {-3, 41}
, {-51, 31}
, {-21, -69}
, {-13, -119}
, {69, 51}
, {-37, 53}
, {41, 27}
, {23, -79}
, {60, -32}
, {-24, -15}
, {5, 81}
, {-81, -73}
, {-83, 46}
, {56, 34}
, {55, -67}
, {-77, -26}
, {80, -61}
, {109, 45}
, {4, 67}
}
, {{-82, 11}
, {10, 80}
, {44, -64}
, {30, -40}
, {-16, -115}
, {66, 107}
, {59, -46}
, {-46, -163}
, {47, -42}
, {76, -5}
, {-35, 60}
, {9, -67}
, {122, 82}
, {-102, 59}
, {-101, 108}
, {27, -108}
, {-74, 7}
, {71, -38}
, {-104, -8}
, {-10, -84}
, {-1, -67}
, {20, 32}
, {-8, -29}
, {-1, -22}
, {-96, 100}
, {90, -37}
, {-1, 9}
, {69, 22}
, {72, 93}
, {-8, 64}
, {-33, 27}
, {-60, 43}
}
, {{-58, 68}
, {42, -15}
, {12, -43}
, {-105, -143}
, {7, 91}
, {50, -43}
, {115, 34}
, {10, -28}
, {-81, 16}
, {-48, 27}
, {94, -46}
, {44, 71}
, {30, 87}
, {63, -38}
, {39, -57}
, {55, -7}
, {-26, -27}
, {29, -2}
, {21, -74}
, {-17, 48}
, {14, 13}
, {-104, -8}
, {-69, 25}
, {40, 69}
, {-162, -129}
, {17, 15}
, {1, -131}
, {8, 81}
, {-89, 70}
, {-21, -45}
, {102, 101}
, {5, 25}
}
, {{55, -70}
, {-3, -101}
, {-85, -82}
, {-2, 5}
, {10, -5}
, {-10, -33}
, {-81, -49}
, {75, 28}
, {46, -65}
, {-108, 48}
, {-35, 119}
, {76, 10}
, {-71, 60}
, {32, -73}
, {-58, 77}
, {24, 92}
, {49, 22}
, {16, -72}
, {9, -64}
, {72, -21}
, {-115, -71}
, {74, -84}
, {1, 63}
, {72, -18}
, {8, 57}
, {46, -60}
, {51, -64}
, {40, -3}
, {84, -31}
, {-45, -45}
, {14, 73}
, {78, 33}
}
, {{16, -71}
, {74, 58}
, {-43, 44}
, {-39, 4}
, {-47, -38}
, {50, -6}
, {35, 57}
, {-14, -7}
, {-64, -46}
, {31, 35}
, {-56, -44}
, {-86, 103}
, {86, 29}
, {7, 0}
, {-45, -64}
, {78, 62}
, {-5, 12}
, {-113, 48}
, {-58, -13}
, {91, 15}
, {-72, 28}
, {60, 48}
, {141, 104}
, {-6, 0}
, {85, -21}
, {34, 132}
, {-1, 78}
, {-15, 20}
, {-4, -110}
, {-147, -11}
, {-67, -95}
, {-26, 18}
}
, {{49, 2}
, {-60, 37}
, {64, -4}
, {-3, 15}
, {-43, -20}
, {-97, 42}
, {-35, -65}
, {-49, 78}
, {-39, -98}
, {-100, -51}
, {89, -10}
, {10, -71}
, {-67, 69}
, {30, -68}
, {49, -3}
, {-4, -80}
, {-30, -80}
, {-30, 65}
, {92, 97}
, {28, -63}
, {13, 11}
, {65, -41}
, {-3, 59}
, {11, 92}
, {155, 197}
, {96, 49}
, {-81, 11}
, {87, -26}
, {-119, -62}
, {70, -46}
, {0, 4}
, {71, 18}
}
, {{76, 67}
, {-71, 56}
, {-5, 37}
, {-38, -39}
, {-56, 100}
, {-12, 107}
, {-32, -60}
, {-25, 10}
, {54, 73}
, {136, 53}
, {16, -37}
, {19, -57}
, {-41, 52}
, {66, 71}
, {36, 14}
, {-78, -88}
, {-16, 36}
, {31, 94}
, {-69, 35}
, {-111, 102}
, {60, 62}
, {49, -15}
, {-46, -45}
, {-67, -30}
, {26, 23}
, {13, -9}
, {-79, -68}
, {65, -75}
, {86, 2}
, {30, -1}
, {54, 129}
, {-59, 17}
}
, {{-21, -118}
, {-61, -27}
, {51, 109}
, {-75, -57}
, {-11, -64}
, {-85, -52}
, {-89, 69}
, {19, -84}
, {19, 100}
, {2, 116}
, {61, -79}
, {-103, -82}
, {97, 20}
, {110, -39}
, {-91, -18}
, {-100, 39}
, {-87, -59}
, {-76, 2}
, {-115, -83}
, {77, 17}
, {12, -31}
, {35, -45}
, {33, -8}
, {44, -70}
, {-16, -30}
, {-27, 33}
, {18, 42}
, {66, -63}
, {-117, -15}
, {0, 95}
, {-94, -24}
, {82, 94}
}
, {{24, -74}
, {118, -40}
, {30, 53}
, {52, -35}
, {-43, -11}
, {26, -84}
, {17, -33}
, {-24, 119}
, {-79, -37}
, {47, 96}
, {-83, -76}
, {92, -19}
, {60, -24}
, {-11, 11}
, {-8, -64}
, {-77, 63}
, {29, -4}
, {-116, -41}
, {-81, -61}
, {102, 33}
, {18, 40}
, {107, 32}
, {55, 57}
, {-69, 69}
, {-14, -90}
, {43, 98}
, {30, -15}
, {30, 39}
, {-87, 106}
, {30, -39}
, {-82, -152}
, {14, 70}
}
, {{-72, -101}
, {30, 53}
, {48, -162}
, {-34, -57}
, {-10, -6}
, {65, 83}
, {10, 21}
, {129, 78}
, {81, -73}
, {-129, 37}
, {-16, 84}
, {16, -4}
, {-89, 32}
, {-88, -126}
, {11, 52}
, {-45, -99}
, {45, 88}
, {13, -11}
, {73, -98}
, {-3, 15}
, {-15, 70}
, {-55, -111}
, {-6, 0}
, {94, 80}
, {98, 43}
, {44, 71}
, {70, 37}
, {3, 77}
, {129, 64}
, {53, -19}
, {36, 61}
, {52, -70}
}
, {{-96, -20}
, {37, 17}
, {1, 37}
, {-20, -55}
, {53, 80}
, {0, 58}
, {112, -9}
, {-3, -43}
, {40, -93}
, {61, 35}
, {30, 13}
, {-37, -42}
, {108, -61}
, {-35, -61}
, {-38, 36}
, {-82, -23}
, {60, -143}
, {42, -128}
, {-16, -1}
, {-8, 62}
, {-148, 101}
, {-44, -5}
, {-32, 63}
, {-38, -5}
, {159, 92}
, {-49, 124}
, {-93, -116}
, {85, 127}
, {-86, 103}
, {-39, 50}
, {105, 64}
, {65, -64}
}
, {{104, -7}
, {-36, -71}
, {115, -157}
, {17, -25}
, {-20, -136}
, {-56, -18}
, {-12, 119}
, {131, -113}
, {-43, -37}
, {-75, 154}
, {-36, -7}
, {2, -27}
, {-35, 42}
, {58, 125}
, {-108, 39}
, {-130, 12}
, {62, 37}
, {-65, -3}
, {-43, -48}
, {105, 23}
, {-109, -45}
, {91, -137}
, {-14, -18}
, {30, 103}
, {25, 95}
, {17, -77}
, {83, 59}
, {-70, -72}
, {22, -9}
, {18, 16}
, {-112, 113}
, {2, -18}
}
, {{-28, 18}
, {82, 99}
, {70, 53}
, {13, -13}
, {30, 6}
, {-174, 39}
, {6, 1}
, {44, 34}
, {-10, 99}
, {-58, -38}
, {-60, 4}
, {-15, -29}
, {29, 37}
, {-85, -37}
, {-21, -93}
, {-33, 30}
, {-180, -47}
, {-66, 78}
, {64, -4}
, {14, -120}
, {-78, -89}
, {2, -2}
, {23, -61}
, {-36, 55}
, {-91, 13}
, {-28, 120}
, {-70, -78}
, {6, 75}
, {57, 66}
, {-10, 26}
, {50, -64}
, {104, 89}
}
, {{72, 72}
, {-35, 80}
, {-62, 23}
, {-53, 73}
, {0, -71}
, {108, 0}
, {-122, 75}
, {34, -20}
, {59, 17}
, {-61, 18}
, {-83, 51}
, {16, -125}
, {31, 13}
, {38, -58}
, {-32, 76}
, {-80, 57}
, {25, 177}
, {37, 40}
, {0, -38}
, {60, -21}
, {0, -32}
, {-91, 2}
, {111, 35}
, {27, 67}
, {35, -25}
, {101, 58}
, {-129, -34}
, {-37, -55}
, {140, 10}
, {-37, 20}
, {49, -7}
, {-16, -89}
}
, {{23, 71}
, {-50, 106}
, {78, 104}
, {48, 58}
, {-54, -70}
, {18, 35}
, {38, -1}
, {-33, -29}
, {64, -41}
, {-34, 64}
, {26, 6}
, {-18, -4}
, {117, -117}
, {100, -16}
, {72, -93}
, {-63, 68}
, {-35, -58}
, {51, -19}
, {-10, -35}
, {35, -120}
, {-64, 4}
, {-53, -25}
, {48, -63}
, {22, 12}
, {-46, 49}
, {21, 59}
, {48, -23}
, {-72, -63}
, {18, -40}
, {-42, 102}
, {-57, -29}
, {62, -62}
}
, {{20, -162}
, {63, -38}
, {-91, -72}
, {-107, -142}
, {-105, 41}
, {52, -55}
, {105, 31}
, {49, 68}
, {-11, 174}
, {-49, 140}
, {45, 88}
, {1, 17}
, {28, -40}
, {-1, -95}
, {16, 52}
, {-97, -36}
, {46, -56}
, {20, -11}
, {-34, 42}
, {-37, -131}
, {-62, -49}
, {66, -9}
, {-45, -48}
, {-98, -49}
, {-8, -93}
, {-140, 64}
, {-40, 53}
, {-106, 149}
, {38, 83}
, {-31, 2}
, {-5, 178}
, {-83, 50}
}
, {{-76, -47}
, {-62, 73}
, {-73, 31}
, {38, -74}
, {9, -44}
, {-16, -48}
, {13, -80}
, {35, -49}
, {-50, 41}
, {2, -30}
, {-7, 63}
, {58, 132}
, {-50, -35}
, {-11, 41}
, {-56, 76}
, {14, 53}
, {-160, 20}
, {76, -105}
, {53, 20}
, {48, 54}
, {27, -10}
, {38, 24}
, {11, -89}
, {-28, -68}
, {85, -36}
, {-40, 111}
, {31, 42}
, {29, 4}
, {20, -126}
, {22, 33}
, {101, 99}
, {-54, 32}
}
, {{33, 14}
, {-102, 69}
, {-99, 29}
, {-18, 24}
, {49, 39}
, {34, -7}
, {-72, 43}
, {-90, 63}
, {-44, 60}
, {134, 134}
, {-31, 38}
, {70, 7}
, {-17, -115}
, {16, 83}
, {-97, -99}
, {44, 54}
, {26, 112}
, {58, -20}
, {-37, 34}
, {21, 112}
, {12, 19}
, {-87, 47}
, {-105, -17}
, {8, -3}
, {102, 21}
, {78, -99}
, {67, 104}
, {18, -44}
, {44, 174}
, {87, 36}
, {35, 1}
, {53, 76}
}
, {{-12, 10}
, {-71, -56}
, {44, -62}
, {3, -52}
, {-38, 43}
, {-68, 55}
, {72, -168}
, {-60, -16}
, {-14, 121}
, {-65, 0}
, {-121, 54}
, {86, 117}
, {28, -135}
, {-42, -75}
, {-55, 1}
, {0, 13}
, {-19, 47}
, {25, -38}
, {-49, -80}
, {45, 28}
, {103, 59}
, {-22, 21}
, {86, -78}
, {-12, -34}
, {1, 92}
, {95, -22}
, {66, 13}
, {83, 42}
, {94, -39}
, {64, 85}
, {77, 0}
, {-23, -78}
}
, {{34, 24}
, {-66, -18}
, {73, -24}
, {-30, -11}
, {6, -22}
, {28, 16}
, {95, -17}
, {-65, -67}
, {-70, 69}
, {-41, 4}
, {-97, -83}
, {82, -53}
, {-3, -51}
, {95, -68}
, {82, 92}
, {-64, 121}
, {-106, 1}
, {-80, 19}
, {-53, -40}
, {83, -12}
, {80, 110}
, {-6, 38}
, {-92, 43}
, {-16, 38}
, {73, 123}
, {2, 29}
, {18, -29}
, {-93, 72}
, {-130, -51}
, {64, -134}
, {-61, 20}
, {116, 14}
}
, {{-25, -118}
, {-19, 2}
, {41, 35}
, {-97, -36}
, {112, 109}
, {-42, -6}
, {63, 12}
, {51, -20}
, {2, 97}
, {-23, 1}
, {15, 87}
, {-123, -1}
, {89, 11}
, {-70, -66}
, {18, -26}
, {-90, -87}
, {148, -19}
, {12, -120}
, {93, -8}
, {100, 92}
, {19, 72}
, {-30, 6}
, {-102, 8}
, {77, 33}
, {-157, -37}
, {-33, -82}
, {-24, 48}
, {63, 82}
, {41, -31}
, {55, -53}
, {-42, 35}
, {5, -43}
}
, {{-52, 44}
, {3, 62}
, {-88, -98}
, {30, -38}
, {84, 48}
, {-53, -12}
, {-42, 4}
, {-73, 39}
, {8, 96}
, {44, 50}
, {-53, 29}
, {30, 7}
, {83, 60}
, {-117, 42}
, {-36, 21}
, {-38, 37}
, {38, 55}
, {-79, -47}
, {2, -12}
, {83, 86}
, {-15, 93}
, {-66, -27}
, {-96, -52}
, {70, 7}
, {31, 154}
, {75, -79}
, {33, -16}
, {101, -33}
, {-61, 140}
, {77, 18}
, {-113, -72}
, {-22, -23}
}
, {{-43, -110}
, {7, 0}
, {37, -23}
, {-10, -31}
, {94, 112}
, {-66, -69}
, {-26, 34}
, {33, -46}
, {-57, 99}
, {-65, -12}
, {-92, -12}
, {-68, -58}
, {53, 39}
, {30, 77}
, {-75, 15}
, {93, 9}
, {92, 117}
, {45, 84}
, {25, -27}
, {31, 42}
, {-45, 102}
, {98, 51}
, {-75, -46}
, {50, -1}
, {-67, 14}
, {-112, -48}
, {12, -64}
, {72, 26}
, {31, 43}
, {88, 30}
, {44, -55}
, {84, 21}
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


const int16_t dense_bias[FC_UNITS] = {-9, 4, 5, 3, -4}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{-22, 22, -74, -81, -45, 28, -8, 0, 35, -17, -59, -71, 39, 35, -69, 39, -31, -2, 71, 64, 29, -126, -100, 50, -43, -17, -26, 44, 21, -71, 32, 56, 34, 43, -61, 106, -98, 73, 41, 36, 63, 28, -1, 46, -66, -21, 63, -60, 2, -66, -96, 67, 11, 8, -56, -25, 41, 54, 32, -27, -27, 41, -58, -83, 85, 30, 116, -15, -40, -16, 68, 71, -77, 19, -20, 86, 44, 47, -53, 12, -79, 83, 49, -56, -44, 25, -37, -44, 0, 55, 31, 46, -5, 16, -13, 31, 23, -13, -33, -45, 72, 9, 59, 8, 10, -61, 72, 10, -31, 34, -47, -76, -11, -23, 75, -19, 83, -16, 80, -55, 25, 80, -76, 53, -5, -10, 103, -33, 69, 11, 74, -50, 69, 25, -65, -131, -57, 81, 71, -156, 41, 84, -81, 11, 63, 31, 20, 45, 0, -62, 7, -134, 22, 92, -53, 46, 5, 45, 19, 0, -47, -107, -108, -120, -126, 3, 46, -25, 16, -70, -19, -83, 60, 65, -51, -41, 47, -18, 6, -62, 23, 45, -35, -129, -52, -83, 101, 20, -77, -49, -155, -77, 58, -116, -56, 52, -80, -8, -88, -2, -16, 8, 48, -28, -63, -81, -9, 115, -121, -36, 17, -56, -60, 50, -89, -62, -72, 95, -4, -44, -59, 40, 39, 3, -43, 96, 71, -55, 6, 4, 9, 19, 82, 0, -20, -74, -62, 23, 2, 62, -53, 60, 67, -85, 42, -37, -44, -110, 59, -33, -55, -10, 71, 20, 18, 4, -82, 42, 54, 25, -76, -91, 36, 4, -35, -58, 27, 20, -22, 53, 61, -8, -70, -3, -128, 42, -68, -117, 58, -43, -67, 40, 89, -122, 98, 42, -3, 66, 80, 94, 45, -139, 25, -4, -18, 22, -90, 52, 13, 39, 32, -123, -50, 28, 63, -99, 28, 1, 58, 17, 29, -57, -19, 52, 51, -45, -68, -53, 26, -23}
, {49, -2, 27, 60, -8, -54, 111, -2, -84, -40, -75, -73, 54, 17, 57, -28, -62, -112, 26, 64, -97, -45, 36, 39, 165, 49, -64, 59, -55, 67, 57, -32, -86, -5, 42, 81, 59, -55, 27, -128, 32, 9, -122, -60, 94, 21, -79, 40, 36, -18, -66, 133, -14, -71, -95, 71, 24, 85, -36, -39, 53, -6, -108, -103, -20, 1, -95, 12, 78, 9, 35, -26, 60, 82, -58, 48, -63, -28, 84, 43, -46, -31, 21, -124, 40, 78, 61, -71, -33, 7, 87, 31, -46, 24, 47, 60, 38, 6, -118, 41, -21, -16, -17, 0, -11, 17, 43, 6, -56, 32, -36, -12, 53, 27, 45, -99, -49, -14, 67, 95, -30, -1, -9, -44, 48, 11, -83, 0, 62, -98, -57, 29, 0, 24, -66, -66, 33, 11, -9, -62, 6, -2, -22, -17, -27, 49, -8, -56, 3, 42, 66, 32, 30, -27, -129, -15, -55, -27, -40, -11, -5, 69, 100, 137, -24, -48, 46, -12, -47, -57, 30, 3, 4, 11, -38, 32, -11, 103, -19, 39, 125, 29, 50, -29, 80, -112, -97, -132, 87, -44, -11, 122, -23, 53, -58, 32, 24, 9, -25, -10, -59, 77, -4, 7, -13, -2, 83, 16, -13, -29, -85, -1, 117, -8, -36, 100, 35, 46, -4, -49, 50, 63, -13, 0, -55, 35, -88, -94, -34, 24, -28, -44, -101, 21, -37, 17, 49, 28, 16, 47, 21, -79, -7, -98, -62, -83, 23, -55, -27, 29, -36, 28, -3, -71, -24, 98, -2, -5, 56, -142, 13, 70, -98, -157, 76, -22, 82, -62, 85, -38, -43, 19, 43, 27, 20, -33, -19, -14, 113, -21, 50, 29, -160, 67, -39, 4, 22, -59, -11, -36, -70, 32, 5, -34, 2, -102, -27, -8, -45, -77, 17, 5, -70, 67, 53, 22, -40, 56, -47, 42, -77, 10, 64, -10, -33, -21, -6, 88, 23, 71}
, {21, -3, 30, 5, -38, 76, -45, 79, 53, 26, -37, 41, -49, -28, 68, -28, 59, -32, -100, -23, 11, 29, 62, -36, 1, 13, 55, 37, -22, -51, 13, -42, -7, -48, 31, 13, -23, 19, -34, 30, -14, -18, 31, -22, -47, 89, 4, 65, -24, 42, 111, -132, -69, -11, -95, -104, -46, -61, -36, 4, -128, -147, -45, -48, -29, 54, -114, 10, 25, -69, -51, 73, 46, -54, 3, -5, 68, 16, -52, -70, -4, 41, 14, 0, -139, 68, -1, 49, -14, 81, 64, -6, 105, 46, 61, -27, -41, 40, -32, 73, 12, 28, -29, 0, 26, 18, 81, 84, -18, -93, -105, -88, -41, -45, -26, 64, 17, -76, -110, 4, -96, -31, 72, 20, -73, -21, -34, -63, -25, 79, 2, 7, -2, 73, 49, 95, -56, 80, 51, 23, -90, -28, 90, -59, 89, 74, 52, -59, -15, -62, 81, 3, 20, -39, 13, -64, 55, 54, 70, 40, -25, 39, 105, -60, -5, 5, -14, -17, -34, 4, -69, 54, -40, 40, 9, 6, -58, 73, -98, 37, -118, -32, -32, 45, 48, 39, -20, 49, -44, 59, 55, 70, 52, -64, -55, 3, 6, -28, -1, -35, -1, -21, -64, 38, -77, -64, 0, 60, 72, 0, -62, -23, -83, 63, 108, -5, 93, -80, -48, 13, 41, 57, 57, -71, -76, 90, -77, 90, 62, -8, -56, 63, -44, -63, 63, 21, -48, -55, -33, -4, -2, 62, -39, 107, 94, 68, 25, 57, -12, 47, -17, -51, 55, -69, 8, -46, -38, -92, -30, 81, 23, -61, 2, -39, -13, 80, 59, -59, 38, 48, 87, 50, -48, -63, -49, -59, 30, 1, 94, 41, 64, -14, 37, 2, -5, 2, 50, -14, -1, -10, 60, -43, -65, -71, -54, 30, 45, -81, 22, -96, -52, -15, 29, 22, 54, 22, 71, -36, -39, 67, -38, -2, -17, -25, -18, 62, 2, 14, 28, -47}
, {-80, -24, -9, 75, -16, 102, 13, 74, 4, 22, 25, 0, -9, -74, 12, -20, -59, -38, -20, -64, 66, -16, 0, -111, 96, -35, 58, 16, -4, 0, -96, 86, -60, 72, -78, 58, 43, -83, -131, 79, -86, 53, 36, 100, 28, 26, -39, 3, -63, 58, -18, -98, -142, -46, 24, -50, 50, -3, -16, 42, -9, -36, 4, 164, -48, -8, -24, -122, -23, 28, -25, 1, -13, 30, -20, -60, -11, 58, 28, -14, -60, 86, -86, 107, 78, -106, 69, 32, 86, 26, -44, -38, -42, 42, -80, -9, 58, -51, -4, -72, -14, 32, 66, -34, 58, -8, -44, 19, 38, 72, 88, 62, 35, 38, 68, 36, -95, 29, -63, -28, -5, 94, 59, -78, -57, -61, -60, -69, -2, 1, 68, 22, 28, -57, 48, -6, 76, 16, 61, 65, 33, 42, 23, 4, 35, 77, 5, 40, 34, -35, -123, 96, -13, -19, -89, -29, -36, 16, -77, -4, 99, -12, 5, -60, 24, 21, 13, 14, 55, 2, -11, -37, 72, -87, 91, 101, 91, -75, 59, 30, 79, 92, 72, -4, -43, 125, -47, -6, -62, -13, 58, -134, 74, 68, -28, -68, 90, 91, 69, 47, 34, -55, -90, -4, 43, 45, -91, -105, 47, 67, -37, -63, -80, 33, -61, -48, -84, -11, -105, 97, -53, 52, 37, -67, -23, -37, 27, -17, -112, -79, -10, -21, -7, -8, -6, -10, -59, 46, 71, 9, -7, -43, -83, 0, -101, -125, -87, -85, -55, -129, -71, -12, -59, 74, 48, -113, -109, -102, -1, 140, -99, -135, -45, 9, -13, -14, 23, -14, -60, -1, -107, 11, 87, 70, 66, -19, -25, 7, -83, 43, -131, -159, 75, 29, -130, 46, -1, -42, -57, -11, 49, 7, -84, 39, -20, 0, 92, -117, 15, -3, -77, 80, 43, -82, 2, 68, -51, -67, 16, 45, 2, -15, -66, 33, 76, 75, 47, -37, 52, -29}
, {49, 12, -54, 14, -20, -65, 8, -140, -79, -90, 13, 70, 28, 38, 33, -30, -109, -9, 34, -86, -46, -77, 129, 14, -145, 57, -90, 12, 14, -18, 16, 62, -33, -57, 68, -136, 60, -81, 66, 60, 45, -133, -62, 35, -101, 12, 61, -25, -80, 23, 66, -44, -2, -44, 44, 29, -66, -94, 32, 27, -28, -90, -35, -94, -117, -94, 62, -5, -89, -19, 10, 58, -3, -17, 23, 3, -60, 18, -43, 90, 82, -96, 50, 7, -11, 84, 37, -34, 78, -71, -53, 31, 67, -33, -17, 1, -119, -46, 140, 0, -1, 53, -48, 39, 80, 40, 0, -69, -29, 13, 27, 23, -19, 30, -53, -60, -23, 28, -12, -16, -48, -17, 97, -64, -44, -26, 54, 49, 59, 25, -31, 98, -63, -32, 108, -40, -25, -81, -57, 65, 51, -9, -86, -36, -23, -58, 37, 56, 61, 15, -82, 24, -39, 6, 72, -91, 53, 59, 17, -40, -31, -87, -73, -29, 138, 48, -70, -30, 36, 11, 69, -46, -79, -31, 1, -21, -51, 32, -6, -55, -85, -27, 16, 26, -28, -19, 24, 77, 82, -79, -9, -79, -20, 66, 58, -21, -82, -29, 54, -29, -61, 40, 67, -81, -56, -4, -30, -47, -22, 24, 59, 87, 19, -122, 51, -28, -92, -65, 38, 14, 21, 63, -3, 66, 66, -76, -7, 25, -12, 10, 59, 63, -57, -20, -21, -46, 55, 62, -60, -50, 17, -18, 75, -66, -74, 66, 8, 95, 46, -13, 59, -14, -65, 9, 69, -19, 60, 39, -40, 14, 38, 124, -105, 127, -126, 16, 9, 52, 9, 34, -5, 16, 32, -108, 101, 20, 84, 51, -75, 29, 106, -116, -38, -105, -48, -3, -55, -30, -57, 57, -83, -7, -36, -8, 36, -37, -51, 62, -31, -82, 19, -12, -69, 66, -79, 83, -85, -48, 41, -86, 11, 87, -59, 86, 2, 41, -55, -57, 45, -40}
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
