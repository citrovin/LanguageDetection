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
#define CONV_KERNEL_SIZE    9
#define CONV_STRIDE         8

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_31_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_31(
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
#define CONV_KERNEL_SIZE  9


const int16_t conv1d_31_bias[CONV_FILTERS] = {576, 209, 4, 187, -18, 274, -18, 53}
;

const int16_t conv1d_31_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-3, 122, 19, -58, 14, 76, 66, 113, 94}
}
, {{105, 100, -165, -56, -125, 76, 127, -172, 137}
}
, {{82, -132, -98, 169, 97, -68, -92, -14, 61}
}
, {{-188, -143, -11, 22, 69, -76, -52, 97, 118}
}
, {{103, -158, 59, -54, 100, 52, 49, 75, -171}
}
, {{-104, -49, -95, 17, 9, -41, -44, -106, -109}
}
, {{-111, 24, -122, 70, 75, 17, 72, -115, 162}
}
, {{193, 26, 148, 60, -87, 110, -58, 9, 122}
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
#define INPUT_SAMPLES   1999
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_24_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_24(
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
#define INPUT_SAMPLES       499
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    4
#define CONV_STRIDE         2

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_32_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_32(
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
#define CONV_KERNEL_SIZE  4


const int16_t conv1d_32_bias[CONV_FILTERS] = {17, -167, 30, 11, 33, -287, -155, 5, -105, -54, 54, -4, 65, -173, 58, -45}
;

const int16_t conv1d_32_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{22, 7, 33, -85}
, {-17, 211, -51, -117}
, {79, 265, 334, -80}
, {79, 119, 37, -126}
, {45, 117, -12, -169}
, {-3, -25, -11, -93}
, {0, 90, -115, -216}
, {-23, 3, -43, -2}
}
, {{51, -45, -89, 131}
, {-113, -48, -93, -34}
, {-197, -75, -124, -147}
, {118, -39, -99, 60}
, {85, 52, 39, 88}
, {0, -124, -59, 63}
, {52, 133, 144, 139}
, {74, 87, 67, 15}
}
, {{48, 94, -27, -131}
, {2, 90, -11, 172}
, {-65, 23, 94, 100}
, {-181, 77, -43, 62}
, {-22, -49, -23, -34}
, {-120, 32, -156, 199}
, {-132, 30, 32, -37}
, {68, -76, 12, 8}
}
, {{-166, -53, 11, -71}
, {176, 276, 160, 235}
, {191, 401, 241, 114}
, {-64, 57, 16, 91}
, {-4, 113, -11, 96}
, {-35, -48, -88, 19}
, {119, 177, 3, 6}
, {-18, 39, 39, 22}
}
, {{26, 89, -121, 0}
, {9, 111, -126, 36}
, {-61, -77, -90, 95}
, {10, 81, 35, -13}
, {-100, 2, -6, 31}
, {-22, -50, -123, 98}
, {16, 127, 54, -21}
, {-216, 163, -136, 76}
}
, {{-61, 59, 20, 32}
, {-164, -123, -152, -34}
, {45, -39, 99, 16}
, {-24, 124, 141, 43}
, {-217, -29, 54, 0}
, {132, -32, 35, 38}
, {-58, -41, 58, 72}
, {68, -26, -85, 91}
}
, {{-138, -80, 20, 161}
, {-4, -56, -64, 53}
, {-62, -112, -97, 12}
, {158, -73, -56, -172}
, {30, -90, 98, 136}
, {215, 42, -125, -117}
, {-95, 13, 114, 90}
, {-52, -68, 19, 117}
}
, {{122, -82, -133, 69}
, {255, 80, -128, 55}
, {-31, 47, -291, -14}
, {-82, 127, -87, -155}
, {57, 54, -192, 63}
, {-72, 66, -100, 49}
, {45, -16, -16, -101}
, {-49, 64, -25, 111}
}
, {{167, -148, -17, 94}
, {8, 69, 2, -8}
, {207, 32, 49, 4}
, {3, -111, -12, 13}
, {-52, -44, -50, -45}
, {-189, 118, 120, -3}
, {-2, -287, 6, -54}
, {107, -93, -116, 32}
}
, {{-70, 75, 34, 0}
, {-136, -205, 32, 108}
, {-36, 67, 129, 271}
, {-156, -32, -93, 60}
, {-93, -67, 58, 30}
, {74, 52, -72, 144}
, {-164, -13, -57, -111}
, {-123, 73, 58, 16}
}
, {{-93, -259, 57, 221}
, {-171, -181, -107, 152}
, {-138, -38, -4, 192}
, {-54, -103, 28, 50}
, {-37, -362, -7, 161}
, {29, -13, 7, 24}
, {-181, -347, 65, 87}
, {-140, -198, 82, 63}
}
, {{-33, -105, 59, -11}
, {208, 4, -50, 113}
, {-135, -110, -79, 34}
, {-77, 35, -37, 134}
, {205, 100, -153, -37}
, {3, -57, 61, 124}
, {92, -70, 32, 17}
, {87, -46, -178, 14}
}
, {{-39, -74, -239, -80}
, {164, -132, 163, 47}
, {124, -6, 109, -46}
, {120, 108, -6, 128}
, {-85, -88, -247, 9}
, {160, -120, 32, -29}
, {-61, -100, -45, -11}
, {-141, -113, -90, -96}
}
, {{14, -24, 12, 117}
, {71, -164, -52, 1}
, {-50, -129, -210, -43}
, {30, -9, 22, 136}
, {48, -19, 26, 73}
, {-45, 90, -212, 87}
, {81, 91, 83, -10}
, {-48, -93, -95, 78}
}
, {{38, -110, 67, 0}
, {28, -300, -17, 13}
, {-126, -249, 31, -318}
, {7, -284, 133, -15}
, {-15, -51, 58, 10}
, {-81, 7, 216, -156}
, {-18, -36, -80, 66}
, {76, 97, -45, -64}
}
, {{-114, 198, -418, -87}
, {140, 11, 131, 47}
, {77, 19, -61, 48}
, {-45, 20, 88, -16}
, {-128, 76, -316, 81}
, {59, -97, 18, 8}
, {-102, 83, -316, 71}
, {-109, 52, -254, 17}
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
#define INPUT_SAMPLES   248
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_25_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_25(
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
#define INPUT_SAMPLES       62
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         2

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_33_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_33(
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
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_33_bias[CONV_FILTERS] = {-183, 111, -47, -82, 304, 83, 33, -15, -156, 198, -78, -88, 200, 114, -32, -55, 267, 9, 52, 176, -64, -51, -138, 76, 7, -111, 175, 14, 246, -54, -80, -83}
;

const int16_t conv1d_33_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{64, 182, -35}
, {-48, 96, -85}
, {33, 4, 79}
, {-11, -26, -103}
, {-141, -216, -6}
, {-36, -2, 73}
, {132, 169, 162}
, {109, 128, 41}
, {-45, -82, -184}
, {101, 8, -4}
, {45, 12, -122}
, {-40, 98, 44}
, {18, -76, 3}
, {17, 70, -5}
, {-95, -117, 27}
, {-120, -89, -95}
}
, {{-27, 60, 143}
, {-43, 82, 14}
, {-119, -252, 7}
, {-230, 19, 147}
, {-179, 43, 93}
, {-39, -106, -6}
, {-22, -31, -73}
, {-37, 42, 100}
, {-54, -79, 5}
, {38, -89, -125}
, {22, 42, 98}
, {70, -13, 63}
, {152, -34, -201}
, {52, 13, 110}
, {3, -135, -149}
, {-144, 173, 110}
}
, {{131, -115, -21}
, {31, -86, -39}
, {-32, -50, -95}
, {35, -19, -91}
, {177, 112, 48}
, {61, -24, 32}
, {-93, -121, -168}
, {109, 38, -58}
, {-51, 28, 9}
, {103, 133, 74}
, {-40, -157, 123}
, {84, -65, -63}
, {-130, -52, -62}
, {-31, 20, -10}
, {92, -17, -202}
, {172, 142, -91}
}
, {{-5, -80, 61}
, {78, -68, 68}
, {28, 59, 30}
, {32, 30, -5}
, {-92, -102, 0}
, {-41, 67, -55}
, {58, 30, 49}
, {80, -18, 2}
, {-40, -15, 42}
, {55, 140, 108}
, {93, 17, -31}
, {-158, -2, -36}
, {-49, 244, 20}
, {-108, -52, -11}
, {30, 64, 74}
, {-73, -47, -83}
}
, {{-46, 17, -100}
, {15, -8, 14}
, {79, 56, 190}
, {117, 41, 54}
, {4, 94, 46}
, {-164, -172, 61}
, {-58, -228, 34}
, {3, -7, -66}
, {-140, -13, -106}
, {-261, -234, 51}
, {-38, -134, -97}
, {28, 0, 73}
, {-117, 117, 86}
, {-117, -127, 54}
, {208, 24, -80}
, {-375, -49, -94}
}
, {{-42, 71, 27}
, {1, -51, -30}
, {-128, 85, -18}
, {17, -72, 3}
, {118, 75, -32}
, {-12, -98, -96}
, {48, 26, -242}
, {-94, 19, 156}
, {61, 68, 81}
, {124, 71, -18}
, {150, 132, 35}
, {-78, -51, -66}
, {107, 88, 88}
, {22, -84, -170}
, {16, 82, -28}
, {-60, 7, 106}
}
, {{35, 285, 107}
, {69, -50, -35}
, {29, -69, -59}
, {-42, -208, -192}
, {102, -31, 61}
, {-40, -79, -26}
, {-40, -227, -74}
, {-127, -27, 15}
, {-21, -54, -49}
, {-74, -30, 23}
, {125, 245, 114}
, {72, 103, -25}
, {19, 104, 83}
, {27, -14, 53}
, {247, 95, 184}
, {150, 89, 12}
}
, {{-133, -334, -96}
, {-153, -222, -196}
, {-138, -188, -82}
, {87, 20, 2}
, {65, 77, 135}
, {-25, -51, -72}
, {-6, 15, 27}
, {-103, -51, 2}
, {6, 163, 96}
, {-86, 31, 61}
, {218, 279, 184}
, {44, -42, 13}
, {-23, 38, -12}
, {47, 128, 20}
, {-529, -134, -162}
, {11, -1, -42}
}
, {{16, 8, -182}
, {48, 65, -80}
, {-99, -123, -108}
, {-15, -40, -18}
, {-52, 81, -30}
, {3, -155, -82}
, {-45, 120, 76}
, {26, 185, 156}
, {43, 150, 56}
, {-113, -57, 56}
, {-100, 30, -30}
, {159, 68, 157}
, {90, -206, 36}
, {64, -61, 3}
, {-5, -11, -68}
, {-19, 60, 122}
}
, {{37, 168, 173}
, {-51, -143, -144}
, {-38, 89, 108}
, {-59, 63, 63}
, {12, -63, -46}
, {-40, -62, -12}
, {-192, -88, -37}
, {159, 57, 75}
, {-44, 108, 23}
, {-3, 14, 100}
, {108, 20, -65}
, {-120, -121, -31}
, {-4, 150, 58}
, {178, 12, -1}
, {139, 125, 53}
, {-153, -94, -42}
}
, {{-87, 8, -7}
, {-103, 56, -96}
, {-62, 94, -116}
, {30, 146, 91}
, {12, 75, 8}
, {6, -214, -14}
, {-51, 43, 65}
, {-39, 86, 16}
, {29, -85, -143}
, {-98, -100, -25}
, {43, 29, 11}
, {131, -81, 96}
, {186, -135, 123}
, {-55, -115, -84}
, {-140, -183, -417}
, {-37, -104, -84}
}
, {{21, 139, -31}
, {-123, -55, 19}
, {-9, -18, 49}
, {1, -204, -134}
, {-80, -7, -52}
, {-26, 78, 47}
, {51, 160, 107}
, {-33, 10, 99}
, {-17, -140, 15}
, {21, 54, -24}
, {-62, -53, 65}
, {-19, -14, -146}
, {8, -74, -75}
, {78, 28, -29}
, {-3, 122, 129}
, {26, 198, 132}
}
, {{-144, -63, 104}
, {79, 20, 49}
, {112, 112, 78}
, {-28, -174, -10}
, {145, -2, -73}
, {-30, -54, 72}
, {-40, -27, -1}
, {105, 169, 158}
, {10, -44, 62}
, {-57, -215, -106}
, {-56, -158, -62}
, {-18, 61, 84}
, {27, -56, -35}
, {-104, -143, -65}
, {72, -31, -5}
, {98, 123, 68}
}
, {{-20, -51, -68}
, {-17, -60, -49}
, {-18, -6, -108}
, {134, 23, -178}
, {-63, -82, -85}
, {-17, -127, -87}
, {-57, -11, 48}
, {47, 90, 231}
, {8, 40, 36}
, {54, 105, -76}
, {193, 254, 36}
, {0, -57, -26}
, {169, 88, -54}
, {11, -152, -81}
, {26, -49, -280}
, {-11, -29, 52}
}
, {{108, 97, 89}
, {44, -97, -6}
, {-44, -12, 14}
, {3, -122, -71}
, {-118, 76, 106}
, {45, -19, 76}
, {23, 5, -17}
, {-63, 49, 24}
, {110, 53, 33}
, {-146, -102, -100}
, {122, 133, 37}
, {-22, 148, 118}
, {-7, -8, 63}
, {-54, -8, -51}
, {-24, 105, 24}
, {-187, -121, -167}
}
, {{-29, -14, -92}
, {62, 20, -73}
, {-41, 55, 38}
, {-98, 118, 82}
, {81, 93, 125}
, {-71, 119, -50}
, {-103, -42, -50}
, {-26, -15, -134}
, {-60, 17, 5}
, {-179, 16, 10}
, {46, 47, -43}
, {-21, 60, 30}
, {-160, 39, 138}
, {-38, 6, -40}
, {73, 156, 28}
, {-55, -53, -143}
}
, {{-120, -58, -259}
, {35, -7, -42}
, {50, 54, -128}
, {-96, -114, -91}
, {84, 60, -18}
, {125, 101, -57}
, {103, 13, -142}
, {-104, -27, -21}
, {-96, -99, 29}
, {39, 128, 89}
, {-66, -74, -281}
, {-102, -40, -52}
, {-36, -73, -44}
, {-146, -48, -42}
, {92, 30, 79}
, {64, 36, -10}
}
, {{-174, -240, -102}
, {-21, -71, 48}
, {2, 68, 59}
, {40, 105, -13}
, {114, -36, 14}
, {14, 1, -155}
, {-87, 183, 125}
, {65, 76, 12}
, {-62, 27, 76}
, {-146, -11, -121}
, {-10, -41, 11}
, {49, 67, 86}
, {10, 122, 15}
, {-17, 5, 68}
, {-131, -225, -68}
, {6, -89, -50}
}
, {{-12, -143, -67}
, {-34, 125, 17}
, {84, 13, 85}
, {-3, -23, -24}
, {0, -1, -30}
, {135, -11, 175}
, {-38, 90, -57}
, {-95, -25, -124}
, {-21, -174, 0}
, {-61, -166, -38}
, {-129, -275, -142}
, {39, -146, 18}
, {59, -161, 66}
, {70, 108, 28}
, {-57, -11, -67}
, {101, -14, 19}
}
, {{21, -70, -47}
, {-198, 62, 168}
, {-104, -36, 19}
, {-85, 7, 37}
, {4, 34, 41}
, {-8, 49, 83}
, {-212, -157, 27}
, {40, -161, -13}
, {-19, -31, -17}
, {12, 55, 28}
, {-59, 74, 18}
, {-98, -109, -40}
, {125, -48, -223}
, {30, 17, -8}
, {-95, 15, -16}
, {-2, -262, -96}
}
, {{-46, -43, -138}
, {44, -37, 55}
, {-4, 36, 55}
, {-94, -48, 5}
, {136, 89, -9}
, {64, -45, -95}
, {-85, 115, 127}
, {-13, -31, 118}
, {-57, 154, 118}
, {37, -76, -94}
, {3, -59, -11}
, {-34, 130, 86}
, {-1, 51, 3}
, {-147, -91, -72}
, {-55, -91, -83}
, {187, 62, -11}
}
, {{-165, -190, 26}
, {40, -1, -60}
, {120, 16, 130}
, {135, 21, 7}
, {-35, 120, 134}
, {118, -45, 101}
, {-109, -96, 29}
, {-46, -158, -74}
, {-11, 86, 157}
, {76, 31, 58}
, {-36, 0, 73}
, {-156, -198, -172}
, {28, -7, -19}
, {-7, -48, -79}
, {-6, -34, -85}
, {78, 70, 82}
}
, {{-39, -84, 24}
, {73, 35, -39}
, {92, 36, 88}
, {-51, -80, 0}
, {82, 105, 78}
, {-25, -83, 56}
, {-24, -89, -40}
, {48, 27, 28}
, {-30, -11, -7}
, {59, 16, -68}
, {4, -52, -83}
, {60, -19, 28}
, {-29, -51, -48}
, {29, 37, 89}
, {70, 241, 164}
, {-3, -66, 8}
}
, {{48, -32, 92}
, {-56, 65, -69}
, {-48, -170, -78}
, {-80, -84, -24}
, {64, 33, -25}
, {66, 36, 24}
, {-120, -221, -42}
, {121, 198, 29}
, {27, 191, 148}
, {-131, -7, -39}
, {-63, -112, -40}
, {53, 119, 4}
, {23, -96, -113}
, {4, 98, 105}
, {-85, -202, -96}
, {6, 69, 114}
}
, {{-51, -138, 8}
, {66, -106, 50}
, {-108, -121, -80}
, {59, -14, 39}
, {68, 63, 42}
, {-106, 19, -155}
, {164, 68, -80}
, {-96, -121, -19}
, {29, 103, 157}
, {-86, 25, 67}
, {-65, -38, 43}
, {-2, 8, -23}
, {-6, 114, -19}
, {113, -23, 16}
, {11, -14, 2}
, {15, 41, 7}
}
, {{20, 187, 58}
, {-39, -57, -71}
, {85, 30, -37}
, {69, 102, 142}
, {-5, 51, -29}
, {3, -1, 72}
, {11, 32, 7}
, {65, 146, 120}
, {10, -24, -80}
, {-16, -4, 51}
, {-124, -19, -67}
, {-43, 0, 57}
, {38, -43, -35}
, {13, -97, -15}
, {29, -23, -59}
, {44, -2, -18}
}
, {{-22, -103, 45}
, {-64, -16, 72}
, {17, -170, 129}
, {-68, 105, 79}
, {-45, -117, 13}
, {-108, -92, -124}
, {-374, -51, -5}
, {-37, -12, 61}
, {21, -189, -36}
, {58, -5, 104}
, {6, 94, -57}
, {93, 69, -84}
, {128, 112, 44}
, {-229, -222, -168}
, {-72, -151, 30}
, {39, -82, -17}
}
, {{-193, -174, 42}
, {33, 66, -97}
, {107, 0, -169}
, {4, -21, 21}
, {-14, -5, 51}
, {88, 10, -125}
, {-77, -73, -131}
, {-123, -93, 82}
, {62, 83, -84}
, {51, 104, -50}
, {43, -6, 50}
, {-43, -6, -156}
, {74, 111, -73}
, {75, -46, 60}
, {215, 154, 18}
, {-31, -31, -22}
}
, {{135, 259, 395}
, {137, -27, -143}
, {144, -53, -22}
, {-87, -122, -183}
, {-29, -7, -48}
, {159, -75, -93}
, {-32, 80, -61}
, {-11, 106, 125}
, {-15, -72, -114}
, {4, 9, -33}
, {28, 49, 123}
, {-31, 18, -101}
, {22, 92, -76}
, {128, 71, -11}
, {-16, -8, -6}
, {13, -16, 56}
}
, {{-127, 115, 128}
, {-53, -38, 63}
, {-135, -156, -107}
, {-150, -39, 24}
, {-80, -46, -46}
, {40, -49, 104}
, {39, 80, 94}
, {30, -27, -1}
, {146, 145, 27}
, {152, 119, -70}
, {-142, -185, -183}
, {-94, -84, -54}
, {-141, 58, -80}
, {147, 60, -87}
, {96, 17, -56}
, {63, -161, -18}
}
, {{-157, 54, -98}
, {-224, 96, 162}
, {-102, -18, 0}
, {-19, 43, -83}
, {107, -18, -81}
, {-6, -58, -68}
, {50, 129, -10}
, {-186, -4, 107}
, {48, -40, -13}
, {130, 43, -47}
, {12, -67, 46}
, {135, -68, -133}
, {232, -41, -168}
, {1, -55, -105}
, {-157, -32, -132}
, {-48, 65, 107}
}
, {{-53, 89, 0}
, {31, -116, 43}
, {124, 185, -24}
, {28, -75, 74}
, {-171, -292, -114}
, {-8, 2, 111}
, {-307, -186, -61}
, {-179, -425, -52}
, {41, 95, 80}
, {86, 68, -80}
, {-279, -186, 69}
, {-112, 0, 42}
, {-6, 65, -71}
, {87, -35, -112}
, {-48, -77, -69}
, {112, 159, 123}
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
#define INPUT_SAMPLES   30
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_26_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_26(
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
#define INPUT_SAMPLES       7
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    2
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_34_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_34(
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


const int16_t conv1d_34_bias[CONV_FILTERS] = {68, -56, 101, 93, 109, 61, 86, 13, -55, -320, -277, 2, 55, -178, -54, -44, 221, -52, -77, 89, -59, 84, 55, -66, 35, 99, -33, 240, -73, -150, -34, -125, 174, 45, -139, 190, 188, 63, 124, -49, 28, -29, 158, 169, 144, -4, 89, -32, 210, -29, 125, 168, -70, 272, -67, -75, 138, 45, -53, 97, 86, 180, -67, -40}
;

const int16_t conv1d_34_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-109, 26}
, {-98, -13}
, {-131, 72}
, {12, 19}
, {153, -140}
, {78, -17}
, {64, 113}
, {11, 80}
, {155, -68}
, {-232, 36}
, {-23, -73}
, {-118, -54}
, {-98, 0}
, {-6, 105}
, {-63, 57}
, {-71, 46}
, {-196, -62}
, {49, -7}
, {-52, -20}
, {86, 23}
, {1, -3}
, {26, 135}
, {21, 37}
, {62, 27}
, {-29, -2}
, {-15, -46}
, {-110, 64}
, {-35, 75}
, {97, 38}
, {43, -82}
, {-120, 48}
, {-162, -72}
}
, {{21, -58}
, {96, 80}
, {20, 51}
, {-53, -197}
, {-203, 1}
, {-102, -82}
, {79, -135}
, {-64, 25}
, {-67, -169}
, {3, 80}
, {79, -43}
, {44, 170}
, {91, -148}
, {45, -247}
, {-69, 24}
, {61, 26}
, {84, 19}
, {86, -121}
, {11, 101}
, {-48, -49}
, {71, -96}
, {-48, 100}
, {-40, -175}
, {-49, -64}
, {69, -218}
, {70, 59}
, {-186, -71}
, {47, -24}
, {-187, -127}
, {13, 117}
, {191, -365}
, {128, 114}
}
, {{83, 102}
, {-10, 18}
, {27, 69}
, {-49, -56}
, {21, 28}
, {-16, -26}
, {-150, 12}
, {-67, -15}
, {-82, 7}
, {-74, 1}
, {24, 56}
, {14, -52}
, {90, 48}
, {-133, -107}
, {-88, -135}
, {29, 81}
, {33, -5}
, {46, 102}
, {108, 60}
, {39, 56}
, {-110, -4}
, {-2, 89}
, {-135, -2}
, {-52, 25}
, {89, 77}
, {56, 36}
, {74, 28}
, {-42, 58}
, {114, 21}
, {-17, -23}
, {64, -22}
, {64, 91}
}
, {{-144, 92}
, {111, -44}
, {-69, 94}
, {61, -44}
, {119, -33}
, {-77, -78}
, {40, 57}
, {170, 239}
, {-169, -134}
, {44, 52}
, {9, 91}
, {-142, -39}
, {24, 66}
, {194, 72}
, {25, 8}
, {110, -50}
, {75, 111}
, {-67, -103}
, {-52, 78}
, {131, -30}
, {-176, -24}
, {-83, -54}
, {-4, 46}
, {-42, 30}
, {-56, -116}
, {-1, -11}
, {64, -10}
, {-116, 75}
, {-65, 101}
, {-126, 47}
, {-118, 180}
, {-71, 16}
}
, {{148, -84}
, {85, -12}
, {-167, 116}
, {-22, -69}
, {201, 41}
, {85, -95}
, {-12, 10}
, {-73, -174}
, {-23, -92}
, {-257, 70}
, {-228, -302}
, {-83, -49}
, {46, -37}
, {57, -140}
, {81, 69}
, {-80, -47}
, {-44, -117}
, {97, -6}
, {-81, -54}
, {174, 182}
, {37, -24}
, {16, -80}
, {84, -52}
, {-202, -167}
, {28, -59}
, {-105, -124}
, {266, -131}
, {-90, -120}
, {112, -164}
, {-72, -39}
, {190, 121}
, {261, 3}
}
, {{-195, -128}
, {-86, -140}
, {-3, 41}
, {25, 77}
, {24, 132}
, {-1, 21}
, {151, 70}
, {-239, -274}
, {28, 12}
, {44, 104}
, {40, 47}
, {-57, 55}
, {-13, -33}
, {-60, -25}
, {-59, -63}
, {67, -12}
, {-75, 156}
, {-210, -272}
, {-47, 16}
, {20, 155}
, {-62, -96}
, {-84, 10}
, {91, 87}
, {-137, -103}
, {16, -45}
, {-32, -79}
, {123, 139}
, {73, 97}
, {13, 26}
, {63, 22}
, {-265, -168}
, {-208, -225}
}
, {{-55, -113}
, {113, 133}
, {-10, -151}
, {-123, -34}
, {8, -100}
, {19, 60}
, {93, 77}
, {215, 125}
, {66, -37}
, {66, 50}
, {-97, -65}
, {-163, 25}
, {159, -98}
, {34, -12}
, {44, 153}
, {-93, -27}
, {67, -21}
, {14, -6}
, {4, -110}
, {-337, 52}
, {-102, -44}
, {-315, -61}
, {44, -174}
, {166, 103}
, {-82, 141}
, {72, 53}
, {90, 88}
, {-173, -48}
, {-70, 118}
, {-121, -47}
, {66, -1}
, {-10, 114}
}
, {{-23, 59}
, {-164, 26}
, {-25, -182}
, {11, -73}
, {-15, 6}
, {-119, -85}
, {55, -21}
, {172, -129}
, {-179, -169}
, {-55, 56}
, {-64, -32}
, {18, 83}
, {86, 62}
, {135, 50}
, {-19, 64}
, {101, -8}
, {109, 15}
, {47, -48}
, {141, -88}
, {56, 0}
, {8, -37}
, {-89, -70}
, {-30, 106}
, {-92, -120}
, {61, -156}
, {68, 37}
, {158, 7}
, {-47, -136}
, {44, 2}
, {-23, -54}
, {-104, 72}
, {-9, -158}
}
, {{92, -89}
, {49, 92}
, {-16, -95}
, {-24, -66}
, {34, -48}
, {57, -77}
, {52, 74}
, {-14, 20}
, {31, 14}
, {96, -36}
, {77, -116}
, {2, -56}
, {54, 10}
, {82, -11}
, {-20, 83}
, {38, -5}
, {-69, -209}
, {89, 76}
, {-44, 9}
, {-43, -168}
, {1, 18}
, {-34, 63}
, {-102, -66}
, {-26, 40}
, {-55, 73}
, {86, 74}
, {23, -40}
, {-40, -130}
, {109, -56}
, {16, -17}
, {-51, 138}
, {-130, 90}
}
, {{60, -85}
, {145, 162}
, {173, -211}
, {-103, 23}
, {-152, -48}
, {135, -9}
, {31, 7}
, {-32, -7}
, {146, 102}
, {125, -164}
, {126, -56}
, {-35, -98}
, {87, -53}
, {-62, -92}
, {26, -46}
, {21, -118}
, {-226, -122}
, {48, -97}
, {-11, -51}
, {-28, 21}
, {-29, 48}
, {-208, 33}
, {26, -10}
, {-111, -56}
, {-31, 71}
, {11, -97}
, {124, 74}
, {-21, -29}
, {-165, -121}
, {42, 34}
, {289, -68}
, {189, 213}
}
, {{-261, -171}
, {-166, 4}
, {-286, 3}
, {-87, -34}
, {154, 221}
, {-425, -228}
, {184, 19}
, {62, 64}
, {-163, -2}
, {119, 99}
, {36, 46}
, {-162, 0}
, {-134, -144}
, {-491, -340}
, {-151, -64}
, {-21, 2}
, {138, -306}
, {-20, 16}
, {50, -119}
, {-67, -94}
, {-592, -525}
, {-109, -22}
, {41, 119}
, {138, -121}
, {-31, 0}
, {3, 3}
, {139, 167}
, {-109, -15}
, {-187, -62}
, {-285, -230}
, {-37, -281}
, {-253, 13}
}
, {{-54, -77}
, {-50, -1}
, {-25, 0}
, {65, -73}
, {-24, -24}
, {19, 57}
, {96, 90}
, {42, 67}
, {-61, 58}
, {75, -50}
, {91, -180}
, {16, 38}
, {-116, -148}
, {41, 208}
, {4, 163}
, {-29, 65}
, {-47, -18}
, {-173, -42}
, {-86, -40}
, {-4, -58}
, {96, -69}
, {-23, -86}
, {122, 120}
, {43, -28}
, {-29, 71}
, {-77, 36}
, {133, 50}
, {8, -36}
, {135, 42}
, {-21, -184}
, {-454, -311}
, {92, 135}
}
, {{-4, -19}
, {-16, 64}
, {-131, -101}
, {-61, -6}
, {-134, -246}
, {51, -53}
, {-76, 108}
, {165, 20}
, {31, 130}
, {106, 78}
, {-163, -227}
, {107, 39}
, {-10, 36}
, {67, -109}
, {-8, 15}
, {-76, -39}
, {70, 28}
, {107, 123}
, {-121, -116}
, {-124, -74}
, {61, 109}
, {85, 22}
, {14, -140}
, {-24, -11}
, {51, 39}
, {44, 20}
, {-165, -314}
, {-161, -103}
, {66, 2}
, {61, -87}
, {-84, 13}
, {42, 59}
}
, {{-4, -24}
, {18, -106}
, {36, 37}
, {-41, -71}
, {-195, -39}
, {-49, -106}
, {21, -35}
, {138, 142}
, {59, -131}
, {-152, 92}
, {34, 199}
, {-50, 126}
, {42, -173}
, {-50, 10}
, {28, -46}
, {-37, -28}
, {-3, 70}
, {-40, -82}
, {55, 40}
, {-40, 84}
, {109, 58}
, {-97, 128}
, {-69, -165}
, {66, -66}
, {122, 1}
, {-98, 12}
, {-141, 85}
, {-121, 5}
, {-71, 23}
, {81, 151}
, {-83, 168}
, {-78, 49}
}
, {{57, 17}
, {-101, -58}
, {-145, 42}
, {-78, -133}
, {93, -15}
, {-11, -42}
, {67, 23}
, {-217, -307}
, {84, -5}
, {58, 152}
, {144, -72}
, {122, 18}
, {126, 86}
, {104, -72}
, {47, 0}
, {17, 2}
, {-78, -130}
, {3, -53}
, {32, 21}
, {34, 29}
, {68, -42}
, {-35, -96}
, {91, 23}
, {73, 102}
, {-199, -209}
, {7, -40}
, {167, -83}
, {-25, 0}
, {57, -22}
, {23, -33}
, {151, 344}
, {-93, 26}
}
, {{4, 76}
, {-127, -118}
, {97, -40}
, {-30, 39}
, {-80, -92}
, {-176, -83}
, {-51, 8}
, {-65, -125}
, {-83, -90}
, {79, 2}
, {118, 117}
, {56, 55}
, {16, -84}
, {96, 169}
, {-1, 80}
, {75, -90}
, {67, 81}
, {-24, -95}
, {37, -29}
, {44, -1}
, {0, -23}
, {39, -59}
, {56, 65}
, {-35, 43}
, {-64, -53}
, {-78, 8}
, {26, -65}
, {65, -35}
, {77, 43}
, {126, 113}
, {149, 286}
, {10, 8}
}
, {{-226, -372}
, {-42, 71}
, {-33, -78}
, {-10, 29}
, {10, 1}
, {29, -120}
, {-77, -132}
, {-21, -56}
, {108, 92}
, {-19, -46}
, {-97, 56}
, {58, 0}
, {-126, 22}
, {-191, -96}
, {0, 3}
, {59, 157}
, {-55, 29}
, {-167, -7}
, {-133, 43}
, {-44, -56}
, {112, 13}
, {-68, 13}
, {-105, 0}
, {147, 98}
, {73, 66}
, {-108, 12}
, {207, 5}
, {-157, -127}
, {-293, -134}
, {27, 172}
, {-378, -73}
, {-94, 65}
}
, {{-191, -36}
, {-95, 32}
, {-97, -103}
, {-42, 39}
, {119, 4}
, {8, 174}
, {-91, -40}
, {183, 36}
, {-143, 102}
, {67, -110}
, {-3, 30}
, {-27, -136}
, {133, 33}
, {90, 226}
, {59, 27}
, {-64, 82}
, {-27, -25}
, {-114, 0}
, {65, 129}
, {153, 137}
, {-54, -97}
, {50, -110}
, {-162, -7}
, {57, 54}
, {-31, 39}
, {-45, -71}
, {-34, 7}
, {-104, 55}
, {-54, 102}
, {-156, -80}
, {8, 26}
, {-51, -4}
}
, {{-123, 15}
, {130, -14}
, {118, -63}
, {-11, -69}
, {18, -23}
, {-82, 58}
, {97, 94}
, {-125, 62}
, {-31, 13}
, {60, 40}
, {-229, -149}
, {-98, 36}
, {-89, -131}
, {-121, 180}
, {10, 59}
, {131, 0}
, {76, 80}
, {71, 42}
, {83, -18}
, {-138, -215}
, {-95, -149}
, {-14, -103}
, {71, 34}
, {-129, -143}
, {26, -3}
, {80, -83}
, {5, -71}
, {-56, 31}
, {-43, 90}
, {12, -61}
, {29, -149}
, {27, -102}
}
, {{-55, -6}
, {-51, 0}
, {-27, -97}
, {-60, -4}
, {89, 38}
, {5, 111}
, {192, 53}
, {14, -271}
, {42, -92}
, {-35, 115}
, {70, 13}
, {-491, -255}
, {-144, 63}
, {159, 82}
, {31, 27}
, {11, 20}
, {96, 140}
, {32, -103}
, {22, -206}
, {104, 45}
, {-338, -447}
, {-115, -298}
, {-247, -131}
, {-14, -114}
, {73, -73}
, {-18, 23}
, {283, -34}
, {-66, -457}
, {-48, 72}
, {-159, -113}
, {210, -28}
, {72, -68}
}
, {{-4, 17}
, {37, 21}
, {71, 37}
, {-159, -106}
, {54, 87}
, {-159, -175}
, {-97, -118}
, {116, 186}
, {32, 96}
, {-15, 38}
, {35, 42}
, {40, -65}
, {61, 99}
, {-194, -110}
, {123, 35}
, {-12, 43}
, {-15, -123}
, {47, -39}
, {5, 1}
, {32, 6}
, {54, 11}
, {-68, 29}
, {-37, 125}
, {106, 99}
, {-26, -14}
, {-10, -27}
, {149, 5}
, {-71, -106}
, {-12, -71}
, {32, 59}
, {-63, -88}
, {61, -75}
}
, {{-53, 58}
, {68, 49}
, {-46, 17}
, {-17, 92}
, {-67, 53}
, {-69, 75}
, {-15, -21}
, {2, -15}
, {-65, -12}
, {-22, -160}
, {49, -162}
, {-53, -7}
, {83, 59}
, {-86, 7}
, {24, 98}
, {47, 67}
, {-25, 33}
, {25, 57}
, {48, -54}
, {100, 168}
, {48, -21}
, {13, 56}
, {-7, 3}
, {-27, -27}
, {-8, 67}
, {35, -67}
, {29, -139}
, {-24, 27}
, {39, 136}
, {-102, -62}
, {-9, 28}
, {-67, -71}
}
, {{-21, 21}
, {58, -30}
, {-68, 70}
, {85, 63}
, {13, 113}
, {-53, -49}
, {124, 19}
, {-158, -62}
, {62, -10}
, {22, 48}
, {97, -58}
, {-37, 80}
, {11, -143}
, {-81, 12}
, {34, 16}
, {13, -24}
, {-115, -43}
, {110, 68}
, {-60, 44}
, {17, 61}
, {47, 0}
, {-12, -27}
, {-55, -102}
, {-24, 52}
, {147, 23}
, {-16, -68}
, {-1, 67}
, {36, 60}
, {-94, 102}
, {-35, -31}
, {-39, -34}
, {-36, 61}
}
, {{123, 74}
, {36, 159}
, {171, 150}
, {-64, 82}
, {-154, -281}
, {-47, -54}
, {43, 43}
, {48, -25}
, {-132, -141}
, {190, -88}
, {-85, -263}
, {-10, 157}
, {101, 34}
, {107, 6}
, {-7, -18}
, {-227, -99}
, {-4, -220}
, {-79, 23}
, {-142, -58}
, {-6, -13}
, {40, -91}
, {-31, -51}
, {-82, -160}
, {18, -27}
, {-56, -183}
, {48, 61}
, {71, -51}
, {-172, 35}
, {8, 126}
, {42, -24}
, {-30, -23}
, {111, -52}
}
, {{-29, 31}
, {-28, 77}
, {130, -156}
, {88, 41}
, {0, 158}
, {-109, -166}
, {90, -250}
, {-95, -217}
, {-62, 50}
, {39, -261}
, {161, -18}
, {-19, -26}
, {9, 107}
, {174, 81}
, {49, 20}
, {-114, 75}
, {136, -49}
, {-37, 95}
, {-19, 68}
, {-16, 21}
, {0, 116}
, {20, -132}
, {-44, 7}
, {-322, -175}
, {-79, 23}
, {-79, -86}
, {53, -219}
, {-72, 24}
, {154, -106}
, {8, -112}
, {104, 73}
, {-121, -277}
}
, {{15, -22}
, {61, 110}
, {-74, -85}
, {27, 104}
, {63, -32}
, {15, -116}
, {34, -161}
, {195, -97}
, {42, 36}
, {18, -106}
, {35, -99}
, {-58, -84}
, {50, 35}
, {-148, -15}
, {5, -24}
, {-31, 31}
, {-80, -124}
, {12, 73}
, {-337, -77}
, {16, -60}
, {15, 93}
, {-35, -84}
, {-97, 43}
, {91, -50}
, {-2, -43}
, {26, 105}
, {-76, -173}
, {53, -8}
, {-80, -90}
, {-54, 13}
, {-70, -203}
, {60, -164}
}
, {{74, 41}
, {73, 5}
, {-128, 7}
, {2, -2}
, {47, -163}
, {-15, 35}
, {133, -41}
, {-38, -134}
, {-95, 19}
, {-67, 18}
, {124, 47}
, {1, -24}
, {-30, -19}
, {75, -79}
, {29, -109}
, {-26, -44}
, {85, 34}
, {89, 44}
, {14, 69}
, {-82, -31}
, {23, -23}
, {-100, -16}
, {52, 65}
, {-38, -80}
, {47, 21}
, {88, 16}
, {12, -7}
, {-12, 66}
, {17, -73}
, {30, 93}
, {-39, -94}
, {120, -24}
}
, {{52, 98}
, {-40, 137}
, {34, 277}
, {-85, -170}
, {-70, -20}
, {34, -123}
, {68, 35}
, {49, -119}
, {117, -132}
, {30, -72}
, {-145, -160}
, {45, -55}
, {27, 97}
, {43, -192}
, {70, -53}
, {40, -451}
, {237, 191}
, {13, -13}
, {85, -49}
, {-181, -46}
, {-63, -127}
, {-144, -40}
, {-134, -182}
, {117, 85}
, {78, -167}
, {46, -265}
, {77, 43}
, {-26, -264}
, {3, -127}
, {4, 79}
, {149, -661}
, {-133, -149}
}
, {{4, 51}
, {102, 37}
, {-51, 62}
, {-68, -59}
, {-37, -200}
, {94, 96}
, {92, 119}
, {22, 141}
, {-13, -33}
, {-2, -23}
, {-39, -62}
, {238, 135}
, {-206, -109}
, {84, -179}
, {3, -48}
, {-48, -163}
, {105, 87}
, {-173, -216}
, {-64, -4}
, {-45, 52}
, {-25, -56}
, {53, -25}
, {24, 12}
, {-30, -16}
, {-228, 31}
, {-35, 4}
, {167, 47}
, {2, -81}
, {85, 144}
, {-67, -56}
, {-168, -235}
, {167, -97}
}
, {{-63, -42}
, {-9, -80}
, {-116, 20}
, {87, 31}
, {-30, -143}
, {-139, 40}
, {-35, -56}
, {51, 56}
, {74, 75}
, {22, 13}
, {-60, 14}
, {-3, -53}
, {16, -51}
, {4, 60}
, {59, -53}
, {71, -41}
, {-79, 38}
, {56, 63}
, {-43, 81}
, {42, 18}
, {-62, -14}
, {-93, 36}
, {-5, 17}
, {44, 69}
, {67, -42}
, {61, 23}
, {20, -55}
, {-153, 68}
, {-106, 59}
, {-121, -72}
, {-42, -66}
, {-56, -50}
}
, {{74, 64}
, {-3, -2}
, {-48, -9}
, {36, 4}
, {72, -49}
, {10, 2}
, {38, -216}
, {-48, -15}
, {8, -27}
, {106, 95}
, {83, 4}
, {-8, -19}
, {-2, 26}
, {41, 80}
, {-56, 86}
, {-14, -33}
, {-7, -295}
, {4, 46}
, {-181, -113}
, {-111, -174}
, {-80, 13}
, {26, -27}
, {10, 37}
, {-45, -130}
, {-126, -125}
, {109, 41}
, {-95, -116}
, {26, 127}
, {54, -127}
, {76, 108}
, {-209, -27}
, {109, 53}
}
, {{17, 34}
, {-19, -39}
, {-94, 160}
, {-93, -26}
, {46, 185}
, {-144, -45}
, {-30, -85}
, {78, 90}
, {34, -4}
, {301, 43}
, {-4, -63}
, {-17, -49}
, {43, 37}
, {126, 71}
, {-40, 41}
, {-117, 16}
, {111, 56}
, {74, -49}
, {-122, 105}
, {-255, 65}
, {127, 91}
, {-131, 45}
, {-200, 77}
, {-9, -10}
, {-47, 33}
, {-126, -14}
, {-36, -85}
, {-69, 66}
, {16, -52}
, {-41, 24}
, {60, -29}
, {-99, 129}
}
, {{31, -3}
, {61, 184}
, {41, -25}
, {11, 12}
, {-68, -88}
, {4, -83}
, {154, 88}
, {-54, -53}
, {103, 13}
, {-20, -226}
, {-217, -115}
, {-132, 11}
, {-34, 119}
, {48, -142}
, {35, 121}
, {-153, -88}
, {-52, -109}
, {-134, 2}
, {-50, -74}
, {-91, -91}
, {58, 52}
, {-112, -198}
, {-105, -27}
, {133, 74}
, {22, -98}
, {128, 27}
, {-84, -82}
, {-140, -161}
, {132, 82}
, {-143, -86}
, {-43, -55}
, {47, -108}
}
, {{-41, -59}
, {-48, -61}
, {-99, -173}
, {44, -10}
, {41, 42}
, {60, 72}
, {-8, 45}
, {-49, -34}
, {93, 28}
, {-186, -54}
, {-64, 61}
, {55, 22}
, {87, -81}
, {-35, -347}
, {-100, -64}
, {90, 77}
, {-25, 64}
, {35, 8}
, {43, 24}
, {-26, 77}
, {-94, 32}
, {44, -12}
, {20, -11}
, {19, 29}
, {139, 4}
, {0, 56}
, {21, 140}
, {22, -39}
, {58, -119}
, {-104, 20}
, {71, -159}
, {-102, 14}
}
, {{-20, -29}
, {-149, -98}
, {-45, -124}
, {49, -10}
, {86, 44}
, {140, -38}
, {44, 47}
, {29, -75}
, {-7, 39}
, {0, 191}
, {29, 148}
, {-61, -26}
, {51, -17}
, {-42, -61}
, {36, 73}
, {33, 98}
, {-181, -101}
, {-106, -41}
, {-62, 29}
, {-31, 56}
, {35, 57}
, {-119, -4}
, {39, 72}
, {-21, 19}
, {-6, 77}
, {-156, -112}
, {231, 311}
, {22, 43}
, {-58, 24}
, {-144, -120}
, {170, 226}
, {12, 0}
}
, {{66, -5}
, {32, -55}
, {-33, -21}
, {75, 94}
, {0, 17}
, {-70, 99}
, {-98, -65}
, {-239, -11}
, {-42, 67}
, {123, 88}
, {121, 28}
, {-29, -27}
, {67, 0}
, {-35, -17}
, {-162, -297}
, {-108, -42}
, {36, 78}
, {-90, 13}
, {62, 64}
, {-32, -1}
, {113, 112}
, {57, -39}
, {-57, -59}
, {-80, -72}
, {-8, -15}
, {-12, 2}
, {-38, 209}
, {-133, 34}
, {112, 26}
, {13, 10}
, {17, -153}
, {-118, -97}
}
, {{21, -22}
, {-3, 28}
, {-205, -129}
, {37, 14}
, {20, 52}
, {0, 54}
, {89, -3}
, {-218, -291}
, {47, 22}
, {-25, 68}
, {33, 86}
, {121, 91}
, {-177, -140}
, {-28, 139}
, {-19, -5}
, {51, 42}
, {-33, -28}
, {50, -2}
, {-8, 117}
, {46, -71}
, {-77, -75}
, {4, -32}
, {66, -22}
, {-28, 12}
, {-21, 95}
, {-17, -20}
, {-26, 29}
, {-42, -19}
, {58, 114}
, {61, 48}
, {143, 1}
, {-100, -303}
}
, {{61, 106}
, {-58, -72}
, {-146, 45}
, {-78, 20}
, {59, 25}
, {-94, 10}
, {19, 14}
, {-23, 14}
, {-108, 126}
, {-111, -19}
, {-11, -11}
, {-45, -49}
, {-158, -11}
, {-76, 142}
, {-108, 5}
, {-222, -41}
, {-82, 83}
, {3, 61}
, {-146, -61}
, {97, 51}
, {-132, -18}
, {-173, -25}
, {-150, 158}
, {65, 108}
, {80, 9}
, {18, 10}
, {-100, 145}
, {144, -32}
, {65, 137}
, {104, -5}
, {-17, 171}
, {-191, -25}
}
, {{199, 165}
, {-26, -61}
, {-193, -30}
, {17, 31}
, {-18, -35}
, {32, 55}
, {-97, -186}
, {85, 151}
, {87, 15}
, {94, 36}
, {-73, -2}
, {-66, 9}
, {11, -8}
, {58, 87}
, {-39, 29}
, {-72, -195}
, {-77, -223}
, {9, 83}
, {90, 33}
, {69, -67}
, {61, 17}
, {-108, -93}
, {-121, -112}
, {-102, -10}
, {50, 53}
, {-63, -6}
, {201, 28}
, {-377, -311}
, {79, 243}
, {-56, 12}
, {35, 7}
, {-210, -259}
}
, {{-72, -99}
, {26, 88}
, {54, 116}
, {-91, -68}
, {-31, -66}
, {70, 69}
, {48, 98}
, {1, 24}
, {-53, 41}
, {-56, 122}
, {-33, 3}
, {-8, -86}
, {-130, -52}
, {-48, 38}
, {128, 109}
, {91, -7}
, {-70, -100}
, {-98, -72}
, {53, 8}
, {141, 160}
, {71, 48}
, {75, 83}
, {-65, -20}
, {34, 87}
, {85, 82}
, {60, -72}
, {0, 14}
, {19, 42}
, {76, 38}
, {-82, -110}
, {-65, 151}
, {-6, -89}
}
, {{66, -89}
, {31, -208}
, {45, 49}
, {-77, -3}
, {59, -7}
, {19, 204}
, {-15, -33}
, {11, -63}
, {116, -53}
, {12, 174}
, {-31, -100}
, {-12, -4}
, {-42, -56}
, {72, 43}
, {37, 65}
, {-1, -25}
, {27, -26}
, {131, -63}
, {15, -156}
, {5, 0}
, {37, -120}
, {20, 14}
, {-72, 78}
, {98, -76}
, {-10, -98}
, {93, -45}
, {20, 19}
, {-41, 31}
, {-210, -81}
, {10, -61}
, {111, -187}
, {116, 59}
}
, {{44, 25}
, {9, 118}
, {33, -80}
, {92, -72}
, {100, -101}
, {-32, -204}
, {121, -29}
, {61, 64}
, {-232, -254}
, {53, 103}
, {-3, 3}
, {-49, -127}
, {-42, -47}
, {-50, 88}
, {55, 31}
, {-48, -277}
, {65, -226}
, {-39, 70}
, {120, -148}
, {102, 66}
, {9, -40}
, {61, -77}
, {25, 27}
, {33, 100}
, {-8, -95}
, {-42, 13}
, {152, 39}
, {-53, -427}
, {-211, -224}
, {20, -237}
, {201, -442}
, {217, 65}
}
, {{-128, 91}
, {120, 85}
, {53, -57}
, {-30, -53}
, {83, 27}
, {135, 121}
, {74, 71}
, {-228, -302}
, {36, -175}
, {28, -79}
, {48, -9}
, {96, -30}
, {52, 25}
, {-12, 153}
, {-145, 208}
, {-10, -6}
, {-55, -80}
, {69, -77}
, {-169, 73}
, {-101, 38}
, {7, -109}
, {159, -217}
, {-75, 74}
, {-22, 24}
, {-69, -100}
, {86, -99}
, {59, 118}
, {29, -42}
, {-42, 45}
, {-8, -169}
, {59, -91}
, {-92, -97}
}
, {{-86, -11}
, {110, -39}
, {97, -31}
, {-172, -62}
, {-9, 171}
, {45, -31}
, {55, 77}
, {55, -104}
, {-269, 7}
, {-26, 62}
, {-102, -102}
, {86, 70}
, {-36, -24}
, {225, -97}
, {59, -63}
, {-15, 45}
, {101, 31}
, {7, 25}
, {61, -62}
, {-106, 54}
, {42, -68}
, {-103, -22}
, {-79, 129}
, {32, 122}
, {-53, -134}
, {45, 6}
, {-58, -29}
, {-48, 61}
, {21, 58}
, {-69, 63}
, {96, -314}
, {16, -91}
}
, {{17, -23}
, {-185, -235}
, {19, -187}
, {-41, -82}
, {-28, 79}
, {16, -17}
, {-33, -17}
, {136, 73}
, {102, -113}
, {-77, 137}
, {72, 162}
, {-60, 118}
, {67, 46}
, {-12, 2}
, {-100, -58}
, {61, -72}
, {28, 113}
, {87, -70}
, {44, -109}
, {99, -21}
, {-142, -176}
, {106, -78}
, {6, -59}
, {9, -7}
, {9, 7}
, {95, -7}
, {-18, -146}
, {73, -29}
, {77, 55}
, {18, -26}
, {-191, -500}
, {-37, 119}
}
, {{-69, -112}
, {71, 77}
, {-54, 82}
, {111, 35}
, {20, -58}
, {107, -81}
, {69, 78}
, {-182, -49}
, {-68, -8}
, {146, 117}
, {-38, -150}
, {-105, -55}
, {26, 16}
, {11, 11}
, {-63, -30}
, {34, 70}
, {-146, -99}
, {-122, 1}
, {-50, 107}
, {8, -57}
, {-129, -24}
, {4, -27}
, {124, 134}
, {-24, 63}
, {-6, -29}
, {-23, 87}
, {-123, -95}
, {31, 52}
, {3, 33}
, {2, 84}
, {28, -178}
, {63, 107}
}
, {{88, -115}
, {-97, -60}
, {26, 59}
, {-16, -27}
, {2, -95}
, {-22, 89}
, {-11, -116}
, {-85, -104}
, {-49, -163}
, {-67, 125}
, {31, -137}
, {-2, -390}
, {-23, 64}
, {44, -143}
, {29, -131}
, {-20, -70}
, {-18, -54}
, {-87, -52}
, {50, -73}
, {-102, -29}
, {76, -47}
, {80, -52}
, {46, 64}
, {28, 155}
, {107, -112}
, {80, -32}
, {-53, 219}
, {81, -2}
, {-104, -6}
, {43, -80}
, {-161, -454}
, {132, 47}
}
, {{27, 40}
, {-48, 137}
, {-32, 108}
, {-32, -25}
, {-47, 84}
, {24, 111}
, {-39, -35}
, {49, 97}
, {78, 102}
, {43, -22}
, {77, 15}
, {15, -39}
, {0, 106}
, {47, 36}
, {34, 20}
, {11, -71}
, {-58, 41}
, {-14, 89}
, {-125, 100}
, {-97, 107}
, {37, 86}
, {61, 32}
, {-71, -102}
, {-78, -12}
, {-4, 20}
, {-38, 11}
, {-35, -108}
, {75, -138}
, {-84, -40}
, {-17, -103}
, {-32, 31}
, {0, 154}
}
, {{-139, -131}
, {-202, -158}
, {57, 26}
, {-39, -35}
, {149, 84}
, {18, 41}
, {-50, 121}
, {-152, -450}
, {38, 96}
, {136, 143}
, {96, -70}
, {-51, -80}
, {124, 69}
, {20, -16}
, {-40, -44}
, {-62, 90}
, {-1, 2}
, {-126, -10}
, {20, 34}
, {223, 145}
, {80, 62}
, {-4, -23}
, {-25, -28}
, {138, 9}
, {1, -31}
, {-226, -82}
, {89, 148}
, {135, -18}
, {-25, 0}
, {76, 109}
, {-199, 49}
, {-95, 76}
}
, {{47, 38}
, {216, -51}
, {-208, -204}
, {80, 60}
, {-40, -60}
, {118, -146}
, {267, -121}
, {-108, -52}
, {-99, -27}
, {86, 57}
, {-28, -79}
, {138, 38}
, {91, 15}
, {113, 35}
, {-113, -203}
, {-58, 36}
, {21, -219}
, {-91, 49}
, {-271, -180}
, {101, -185}
, {30, 95}
, {-79, -57}
, {14, 65}
, {-551, -513}
, {-13, -41}
, {3, 46}
, {-122, -210}
, {-37, -42}
, {-23, 45}
, {99, 130}
, {-259, -96}
, {-57, -269}
}
, {{37, -72}
, {49, 9}
, {-166, -149}
, {-62, -32}
, {212, -247}
, {-115, -6}
, {53, 70}
, {90, 22}
, {70, -80}
, {73, 41}
, {-213, 42}
, {-25, -24}
, {1, 94}
, {16, 69}
, {64, 97}
, {-22, -107}
, {-58, 77}
, {-72, -162}
, {116, 6}
, {168, 165}
, {-83, -30}
, {-207, -189}
, {-23, -24}
, {59, 48}
, {-25, 91}
, {-248, -86}
, {-49, -58}
, {-71, 130}
, {193, 153}
, {-65, 69}
, {-118, 21}
, {-119, -55}
}
, {{-96, -140}
, {72, 121}
, {-23, 109}
, {-34, -24}
, {90, 144}
, {128, 41}
, {18, 55}
, {237, 201}
, {-58, -98}
, {202, 5}
, {45, 77}
, {-62, -117}
, {96, -8}
, {205, 193}
, {-4, 60}
, {-67, 14}
, {128, -9}
, {-6, -97}
, {-105, -77}
, {32, -1}
, {-220, 104}
, {-68, -77}
, {-53, 35}
, {-177, -76}
, {77, -32}
, {-106, 20}
, {0, -122}
, {58, -3}
, {66, 41}
, {43, 5}
, {-207, -78}
, {-21, -153}
}
, {{-12, -57}
, {54, -141}
, {153, -54}
, {-57, -33}
, {48, -17}
, {-230, 53}
, {-162, 334}
, {31, -101}
, {-58, -218}
, {-93, 157}
, {-91, 22}
, {49, 50}
, {-387, -1}
, {119, 45}
, {-7, -4}
, {-17, 33}
, {-101, -35}
, {61, -90}
, {-8, 8}
, {178, 133}
, {-259, -9}
, {-66, 101}
, {-146, 7}
, {52, 161}
, {95, -57}
, {-52, -76}
, {23, 155}
, {-93, 3}
, {24, 40}
, {0, -160}
, {-236, 254}
, {-44, 85}
}
, {{-144, 7}
, {26, 151}
, {152, 39}
, {-10, 0}
, {-62, 12}
, {-282, -85}
, {55, 61}
, {-53, -127}
, {44, 130}
, {-62, -158}
, {-298, -38}
, {-97, -240}
, {-24, 86}
, {-377, -225}
, {12, -14}
, {34, 58}
, {-117, 17}
, {22, -47}
, {86, 80}
, {49, 146}
, {-36, -66}
, {-36, -33}
, {-92, -94}
, {-8, 108}
, {56, -59}
, {-5, 104}
, {-224, -276}
, {-10, -89}
, {19, -51}
, {16, -23}
, {-15, -290}
, {90, 99}
}
, {{78, 12}
, {59, 82}
, {21, 39}
, {-34, 43}
, {-14, -146}
, {-21, -141}
, {-28, 108}
, {43, -82}
, {17, -5}
, {-48, 18}
, {-88, 47}
, {22, -75}
, {12, 42}
, {14, -154}
, {12, 119}
, {-29, 84}
, {68, 27}
, {41, -20}
, {-47, -104}
, {-13, -38}
, {-35, -51}
, {-29, -5}
, {105, 62}
, {43, 29}
, {20, 27}
, {85, 51}
, {-6, 1}
, {13, -13}
, {30, -181}
, {-12, 58}
, {-29, 45}
, {-20, -50}
}
, {{-49, -108}
, {-103, 49}
, {67, 24}
, {-13, 7}
, {-68, 98}
, {122, 82}
, {48, -75}
, {-187, -39}
, {54, -44}
, {98, 24}
, {-112, 3}
, {-67, 35}
, {149, -104}
, {95, 104}
, {10, -15}
, {18, -67}
, {-206, -14}
, {-77, -66}
, {59, -96}
, {1, -242}
, {-27, -106}
, {-113, 29}
, {60, -113}
, {151, 94}
, {-151, -46}
, {12, 65}
, {-15, 59}
, {-202, -247}
, {198, -47}
, {-12, 68}
, {-26, -11}
, {-72, -11}
}
, {{3, -60}
, {226, -12}
, {-139, -82}
, {-181, -27}
, {-11, 118}
, {49, -132}
, {222, -109}
, {24, -21}
, {-117, -45}
, {84, 63}
, {90, 73}
, {-384, -112}
, {129, 134}
, {10, -72}
, {-133, -71}
, {-97, -18}
, {-312, -77}
, {-60, -45}
, {-394, -149}
, {61, 125}
, {-201, -345}
, {-160, -221}
, {-117, -211}
, {-79, -63}
, {-293, -50}
, {-8, 48}
, {21, 78}
, {-313, -213}
, {-33, 68}
, {-108, 113}
, {148, -144}
, {34, 39}
}
, {{-173, -109}
, {-245, 46}
, {-64, 27}
, {47, -134}
, {-86, -188}
, {21, 50}
, {20, -134}
, {18, -354}
, {-21, 74}
, {40, -61}
, {70, 32}
, {-24, 104}
, {-137, -143}
, {3, 87}
, {-32, 74}
, {-33, 72}
, {-74, 27}
, {111, -46}
, {91, 24}
, {71, -9}
, {18, 18}
, {-39, 23}
, {7, -13}
, {102, -68}
, {193, -21}
, {-62, 78}
, {96, -14}
, {-8, -128}
, {103, -166}
, {80, 27}
, {134, 109}
, {109, -68}
}
, {{-90, -1}
, {-76, 238}
, {-80, -109}
, {-96, 0}
, {-100, -118}
, {-31, -13}
, {-70, 58}
, {-139, 17}
, {71, 192}
, {94, 97}
, {-74, -66}
, {47, 31}
, {-92, -208}
, {11, 42}
, {-126, -39}
, {-3, 0}
, {55, -74}
, {-54, -46}
, {12, 48}
, {77, -27}
, {18, -26}
, {-81, -139}
, {-78, -4}
, {57, 63}
, {111, 171}
, {55, -81}
, {63, 124}
, {65, -127}
, {5, 177}
, {110, 98}
, {-260, -389}
, {71, -141}
}
, {{21, -39}
, {-50, 58}
, {19, -91}
, {-56, -56}
, {43, 65}
, {26, 230}
, {177, -52}
, {-9, -2}
, {99, 71}
, {-193, -259}
, {-17, 142}
, {-32, -80}
, {-59, -177}
, {10, -13}
, {-2, 13}
, {-21, 43}
, {-212, -168}
, {4, 15}
, {-330, -136}
, {4, 143}
, {76, 59}
, {-64, 50}
, {97, 20}
, {47, -30}
, {-18, 73}
, {23, -66}
, {-4, -57}
, {89, 68}
, {55, -123}
, {-25, -12}
, {-180, -227}
, {-141, -24}
}
, {{34, 112}
, {-84, 23}
, {-198, -67}
, {-45, 70}
, {-160, -27}
, {-192, 53}
, {-3, 82}
, {-261, -18}
, {-107, 69}
, {-12, 10}
, {-176, -165}
, {81, -67}
, {28, -17}
, {45, 0}
, {28, 156}
, {-145, 73}
, {-104, -40}
, {-99, 46}
, {-35, -60}
, {-140, -157}
, {21, 21}
, {-43, -17}
, {-20, 133}
, {-53, -129}
, {-88, 115}
, {14, 83}
, {39, -79}
, {-145, -4}
, {-133, -20}
, {51, 20}
, {82, 109}
, {197, -81}
}
, {{-18, -243}
, {155, -179}
, {28, -64}
, {-48, 76}
, {96, 188}
, {-66, 33}
, {114, 41}
, {-24, -128}
, {13, 39}
, {-106, 106}
, {-83, -49}
, {-33, -45}
, {62, -38}
, {-39, -24}
, {107, 23}
, {-101, 10}
, {131, -35}
, {129, -311}
, {72, -72}
, {11, 11}
, {1, 84}
, {-67, 47}
, {22, 35}
, {51, 2}
, {-113, -39}
, {-125, -183}
, {-61, 87}
, {-50, 101}
, {62, -50}
, {-86, -64}
, {-265, 159}
, {-17, 23}
}
, {{-62, 33}
, {-112, 18}
, {-89, -178}
, {31, -50}
, {179, 191}
, {-242, -30}
, {-57, 92}
, {18, 74}
, {15, 51}
, {87, 104}
, {50, 149}
, {30, 5}
, {67, 46}
, {-25, 44}
, {20, 106}
, {-121, -11}
, {70, 205}
, {-42, -54}
, {2, 29}
, {82, 38}
, {-32, 82}
, {-246, -144}
, {-5, 5}
, {58, 50}
, {45, 88}
, {52, -117}
, {54, 69}
, {10, -46}
, {-31, 34}
, {176, 117}
, {74, -26}
, {-77, -104}
}
, {{-188, -98}
, {-130, 32}
, {81, 24}
, {-37, -65}
, {66, 65}
, {15, -93}
, {-20, 48}
, {180, -82}
, {-58, 108}
, {-59, -59}
, {-76, -3}
, {-31, -127}
, {65, 55}
, {-206, -74}
, {-12, 6}
, {144, 38}
, {45, 24}
, {-40, -21}
, {4, 6}
, {-1, 75}
, {-17, 76}
, {103, 88}
, {0, 26}
, {115, 60}
, {-25, -43}
, {-37, 5}
, {-113, -44}
, {86, 57}
, {-100, -84}
, {42, 89}
, {-318, -41}
, {113, 83}
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
#define INPUT_SAMPLES   6
#define POOL_SIZE       6
#define POOL_STRIDE     6
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t average_pooling1d_7_output_type[INPUT_CHANNELS][POOL_LENGTH];

void average_pooling1d_7(
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

#define INPUT_DIM [1][64]
#define OUTPUT_DIM 64

//typedef number_t *flatten_11_output_type;
typedef number_t flatten_11_output_type[OUTPUT_DIM];

#define flatten_11 //noop (IN, OUT)  OUT = (number_t*)IN

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

#define INPUT_SAMPLES 64
#define FC_UNITS 5
#define ACTIVATION_LINEAR

typedef number_t dense_12_output_type[FC_UNITS];

static inline void dense_12(
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

#define INPUT_SAMPLES 64
#define FC_UNITS 5


const int16_t dense_12_bias[FC_UNITS] = {-115, 21, 2, 50, 39}
;

const int16_t dense_12_kernel[FC_UNITS][INPUT_SAMPLES] = {{-92, -322, -93, -70, 149, -94, 208, 139, 104, -245, -218, -124, 146, 272, 64, 157, -127, 84, 89, 204, 115, -75, -124, -130, 104, 2, 93, -356, 85, 103, -15, 116, 57, 13, -27, -26, 86, 108, -239, 112, -51, -229, 97, -116, -226, 141, 24, -133, -169, -58, -95, -156, -313, -131, 95, 205, 254, 46, -8, -152, 40, -255, 110, -35}
, {59, -64, -88, 133, -483, 131, 68, 179, 26, 267, -397, 82, 7, -22, 51, 123, -322, 229, 326, -156, -17, 10, -76, -331, -54, 49, -65, -211, -98, 102, -14, -42, -202, -65, -57, 84, 136, -305, 0, -115, -119, 197, -310, 272, 291, 96, -82, 20, 181, 34, -222, 220, -317, -74, -107, -381, -531, 136, -188, -24, 104, 144, -166, -63}
, {101, 168, -122, -220, -11, -105, -32, 179, -60, 99, 367, -212, -73, -31, -99, 46, 85, 161, 73, 383, 16, 21, -83, 64, -197, -16, -14, 547, -197, 70, 221, -182, 247, -39, -39, -31, -31, 89, 167, 91, 56, -150, -180, -52, -110, 152, -182, 9, -211, 101, -162, -168, 70, 171, 41, 6, -50, -46, -49, 134, -137, 24, -162, 118}
, {-151, 95, -17, -51, 199, -41, -398, 105, -96, 113, -21, -42, -225, -157, -65, -101, 223, -345, -222, -258, -118, -43, 116, -64, 8, -74, 101, -106, 354, 133, -90, -23, 110, 75, -37, 184, 116, 142, 136, -47, 59, 239, 115, -65, 3, 86, 160, -59, 61, -325, -268, -103, 274, -123, -89, -96, -63, 211, 351, -171, -170, 224, 42, 143}
, {113, 258, 44, -2, -32, 198, 38, -296, 124, 45, -640, 108, 114, -146, 146, -46, -191, 131, -139, -90, 86, 101, 43, 234, -210, -193, 23, -265, 13, -25, -162, -214, -224, 71, 181, -49, -108, -51, -146, -20, -169, -288, -70, -124, -29, -188, -2, -23, 46, 138, 289, 26, -213, 115, 28, 86, 236, -168, -195, -11, 34, 26, -15, 90}
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
  //dense_12_output_type dense_12_output);
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
#include "conv1d_31.c"
#include "weights/conv1d_31.c" // InputLayer is excluded
#include "max_pooling1d_24.c" // InputLayer is excluded
#include "conv1d_32.c"
#include "weights/conv1d_32.c" // InputLayer is excluded
#include "max_pooling1d_25.c" // InputLayer is excluded
#include "conv1d_33.c"
#include "weights/conv1d_33.c" // InputLayer is excluded
#include "max_pooling1d_26.c" // InputLayer is excluded
#include "conv1d_34.c"
#include "weights/conv1d_34.c" // InputLayer is excluded
#include "average_pooling1d_7.c" // InputLayer is excluded
#include "flatten_11.c" // InputLayer is excluded
#include "dense_12.c"
#include "weights/dense_12.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_12_output_type dense_12_output) {

  // Output array allocation
  static union {
    conv1d_31_output_type conv1d_31_output;
    conv1d_32_output_type conv1d_32_output;
    conv1d_33_output_type conv1d_33_output;
    conv1d_34_output_type conv1d_34_output;
  } activations1;

  static union {
    max_pooling1d_24_output_type max_pooling1d_24_output;
    max_pooling1d_25_output_type max_pooling1d_25_output;
    max_pooling1d_26_output_type max_pooling1d_26_output;
    average_pooling1d_7_output_type average_pooling1d_7_output;
    flatten_11_output_type flatten_11_output;
  } activations2;


  //static union {
//
//    static input_12_output_type input_12_output;
//
//    static conv1d_31_output_type conv1d_31_output;
//
//    static max_pooling1d_24_output_type max_pooling1d_24_output;
//
//    static conv1d_32_output_type conv1d_32_output;
//
//    static max_pooling1d_25_output_type max_pooling1d_25_output;
//
//    static conv1d_33_output_type conv1d_33_output;
//
//    static max_pooling1d_26_output_type max_pooling1d_26_output;
//
//    static conv1d_34_output_type conv1d_34_output;
//
//    static average_pooling1d_7_output_type average_pooling1d_7_output;
//
//    static flatten_11_output_type flatten_11_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  conv1d_31(
     // First layer uses input passed as model parameter
    input,
    conv1d_31_kernel,
    conv1d_31_bias,
    activations1.conv1d_31_output
  );
 // InputLayer is excluded 
  max_pooling1d_24(
    
    activations1.conv1d_31_output,
    activations2.max_pooling1d_24_output
  );
 // InputLayer is excluded 
  conv1d_32(
    
    activations2.max_pooling1d_24_output,
    conv1d_32_kernel,
    conv1d_32_bias,
    activations1.conv1d_32_output
  );
 // InputLayer is excluded 
  max_pooling1d_25(
    
    activations1.conv1d_32_output,
    activations2.max_pooling1d_25_output
  );
 // InputLayer is excluded 
  conv1d_33(
    
    activations2.max_pooling1d_25_output,
    conv1d_33_kernel,
    conv1d_33_bias,
    activations1.conv1d_33_output
  );
 // InputLayer is excluded 
  max_pooling1d_26(
    
    activations1.conv1d_33_output,
    activations2.max_pooling1d_26_output
  );
 // InputLayer is excluded 
  conv1d_34(
    
    activations2.max_pooling1d_26_output,
    conv1d_34_kernel,
    conv1d_34_bias,
    activations1.conv1d_34_output
  );
 // InputLayer is excluded 
  average_pooling1d_7(
    
    activations1.conv1d_34_output,
    activations2.average_pooling1d_7_output
  );
 // InputLayer is excluded 
  flatten_11(
    
    activations2.average_pooling1d_7_output,
    activations2.flatten_11_output
  );
 // InputLayer is excluded 
  dense_12(
    
    activations2.flatten_11_output,
    dense_12_kernel,
    dense_12_bias, // Last layer uses output passed as model parameter
    dense_12_output
  );

}
