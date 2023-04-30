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


const int16_t conv1d_31_bias[CONV_FILTERS] = {580, 202, 1, 181, -14, 270, -17, 45}
;

const int16_t conv1d_31_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-9, 121, 17, -57, 0, 77, 58, 115, 88}
}
, {{110, 98, -148, -53, -114, 80, 120, -169, 130}
}
, {{70, -131, -89, 168, 86, -69, -88, -16, 54}
}
, {{-179, -130, -1, 27, 67, -73, -57, 98, 106}
}
, {{95, -157, 49, -54, 91, 50, 42, 75, -180}
}
, {{-90, -48, -81, 20, 18, -39, -43, -107, -103}
}
, {{-108, 19, -121, 64, 70, 13, 69, -118, 161}
}
, {{187, 26, 137, 59, -97, 111, -62, 14, 120}
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


const int16_t conv1d_32_bias[CONV_FILTERS] = {14, -157, 32, 13, 31, -283, -158, -1, -104, -48, 50, -7, 61, -177, 58, -52}
;

const int16_t conv1d_32_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{20, 9, 34, -85}
, {-17, 210, -49, -118}
, {76, 264, 331, -78}
, {76, 117, 34, -123}
, {40, 115, -17, -169}
, {-7, -27, -16, -96}
, {0, 85, -115, -212}
, {-25, 5, -42, -3}
}
, {{54, -46, -84, 140}
, {-100, -39, -86, -24}
, {-181, -58, -112, -138}
, {125, -27, -88, 69}
, {91, 62, 56, 93}
, {12, -110, -47, 70}
, {61, 138, 153, 152}
, {74, 90, 71, 23}
}
, {{50, 99, -21, -125}
, {-2, 84, -16, 168}
, {-74, 17, 83, 96}
, {-179, 76, -43, 62}
, {-24, -50, -21, -37}
, {-117, 34, -153, 202}
, {-132, 29, 32, -40}
, {68, -73, 16, 9}
}
, {{-162, -48, 14, -65}
, {173, 268, 154, 223}
, {185, 395, 233, 103}
, {-64, 57, 15, 87}
, {-7, 107, -12, 87}
, {-32, -46, -83, 20}
, {118, 175, -1, 0}
, {-16, 44, 42, 25}
}
, {{27, 86, -124, -4}
, {6, 107, -129, 35}
, {-62, -79, -92, 95}
, {8, 76, 35, -16}
, {-102, 1, -6, 28}
, {-28, -51, -125, 96}
, {15, 123, 51, -22}
, {-216, 160, -138, 71}
}
, {{-56, 69, 27, 39}
, {-167, -124, -149, -32}
, {45, -38, 97, 19}
, {-21, 126, 140, 46}
, {-211, -28, 56, 3}
, {131, -33, 38, 38}
, {-48, -39, 62, 77}
, {73, -17, -77, 97}
}
, {{-133, -81, 20, 158}
, {-4, -54, -66, 62}
, {-61, -107, -94, 18}
, {155, -71, -63, -170}
, {29, -87, 97, 139}
, {209, 36, -136, -111}
, {-96, 16, 112, 88}
, {-49, -70, 19, 114}
}
, {{114, -90, -142, 64}
, {252, 77, -123, 52}
, {-31, 47, -286, -15}
, {-84, 124, -87, -155}
, {54, 51, -192, 62}
, {-76, 61, -101, 42}
, {41, -21, -18, -101}
, {-56, 57, -32, 108}
}
, {{167, -149, -25, 92}
, {6, 77, 8, -4}
, {205, 37, 53, 9}
, {0, -110, -9, 14}
, {-53, -38, -46, -40}
, {-191, 118, 123, -2}
, {-3, -281, 9, -54}
, {106, -94, -121, 31}
}
, {{-62, 84, 41, 11}
, {-136, -198, 33, 109}
, {-35, 71, 134, 275}
, {-150, -27, -89, 64}
, {-89, -65, 64, 35}
, {75, 53, -70, 147}
, {-158, -5, -51, -108}
, {-116, 83, 65, 25}
}
, {{-92, -261, 52, 217}
, {-174, -181, -108, 149}
, {-143, -41, -11, 181}
, {-60, -106, 23, 48}
, {-40, -366, -12, 154}
, {24, -17, 4, 19}
, {-183, -347, 65, 83}
, {-136, -197, 77, 60}
}
, {{-37, -102, 57, -16}
, {209, 8, -41, 115}
, {-129, -109, -74, 35}
, {-73, 34, -39, 131}
, {205, 100, -152, -35}
, {0, -63, 53, 120}
, {93, -67, 37, 16}
, {84, -45, -179, 11}
}
, {{-46, -72, -244, -89}
, {163, -131, 167, 48}
, {124, -5, 109, -45}
, {117, 105, -10, 122}
, {-85, -88, -248, 8}
, {155, -127, 26, -34}
, {-63, -105, -43, -11}
, {-143, -110, -89, -102}
}
, {{12, -28, 12, 114}
, {70, -167, -55, 0}
, {-51, -126, -207, -42}
, {30, -16, 19, 125}
, {46, -25, 21, 68}
, {-48, 84, -216, 77}
, {80, 84, 80, -13}
, {-52, -99, -97, 74}
}
, {{37, -112, 66, 0}
, {31, -296, -19, 12}
, {-129, -245, 32, -320}
, {5, -279, 135, -16}
, {-15, -56, 57, 11}
, {-78, 14, 216, -155}
, {-22, -37, -81, 60}
, {76, 94, -46, -64}
}
, {{-115, 193, -414, -89}
, {134, 9, 130, 45}
, {73, 12, -61, 47}
, {-52, 16, 84, -21}
, {-135, 69, -321, 77}
, {54, -106, 11, 1}
, {-108, 80, -317, 67}
, {-109, 46, -250, 13}
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


const int16_t conv1d_33_bias[CONV_FILTERS] = {-185, 104, -52, -79, 306, 87, 32, -19, -156, 187, -82, -92, 192, 112, -37, -49, 267, 0, 49, 183, -66, -49, -145, 77, 4, -119, 174, 19, 245, -62, -85, -85}
;

const int16_t conv1d_33_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{65, 182, -32}
, {-53, 93, -88}
, {33, 0, 78}
, {-8, -26, -100}
, {-141, -214, -5}
, {-42, -5, 69}
, {126, 168, 159}
, {108, 127, 36}
, {-41, -78, -183}
, {100, 4, -5}
, {35, 9, -128}
, {-39, 93, 41}
, {20, -77, -2}
, {15, 66, -9}
, {-106, -120, 19}
, {-122, -90, -96}
}
, {{-31, 55, 136}
, {-50, 75, 5}
, {-123, -253, 0}
, {-233, 15, 142}
, {-181, 36, 87}
, {-48, -116, -15}
, {-31, -35, -79}
, {-42, 34, 91}
, {-60, -87, 1}
, {31, -99, -133}
, {16, 33, 92}
, {64, -21, 56}
, {148, -39, -200}
, {46, 4, 102}
, {-3, -145, -161}
, {-147, 170, 107}
}
, {{122, -128, -31}
, {27, -96, -46}
, {-40, -57, -99}
, {23, -31, -97}
, {166, 102, 40}
, {57, -33, 27}
, {-98, -125, -177}
, {102, 32, -69}
, {-53, 26, 5}
, {102, 121, 73}
, {-44, -161, 120}
, {73, -70, -68}
, {-128, -51, -60}
, {-37, 12, -17}
, {91, -23, -207}
, {164, 136, -98}
}
, {{-6, -81, 60}
, {82, -63, 72}
, {32, 59, 29}
, {32, 30, -6}
, {-83, -100, 4}
, {-32, 73, -49}
, {54, 29, 46}
, {86, -15, 8}
, {-38, -15, 47}
, {58, 145, 109}
, {96, 15, -25}
, {-155, -2, -34}
, {-49, 235, 15}
, {-100, -50, -4}
, {30, 74, 85}
, {-64, -50, -77}
}
, {{-46, 17, -98}
, {17, -8, 14}
, {77, 56, 188}
, {118, 40, 55}
, {0, 94, 50}
, {-160, -168, 63}
, {-61, -228, 32}
, {4, -6, -66}
, {-141, -13, -106}
, {-259, -226, 54}
, {-36, -123, -89}
, {30, 1, 76}
, {-105, 126, 99}
, {-112, -124, 57}
, {215, 29, -68}
, {-373, -44, -89}
}
, {{-47, 73, 24}
, {2, -46, -31}
, {-126, 88, -10}
, {15, -72, 4}
, {117, 75, -37}
, {-12, -95, -95}
, {47, 30, -243}
, {-87, 22, 155}
, {61, 68, 87}
, {126, 74, -14}
, {148, 132, 35}
, {-76, -52, -62}
, {107, 87, 92}
, {21, -81, -168}
, {17, 85, -33}
, {-66, 9, 108}
}
, {{38, 283, 103}
, {72, -56, -34}
, {29, -71, -67}
, {-41, -209, -193}
, {107, -33, 52}
, {-41, -80, -28}
, {-41, -233, -73}
, {-128, -30, 9}
, {-24, -57, -55}
, {-74, -32, 19}
, {124, 240, 112}
, {71, 101, -27}
, {20, 105, 83}
, {31, -18, 50}
, {241, 90, 186}
, {150, 84, 7}
}
, {{-137, -336, -101}
, {-158, -222, -200}
, {-142, -190, -86}
, {83, 16, -3}
, {60, 74, 129}
, {-29, -55, -77}
, {-11, 13, 22}
, {-106, -52, 0}
, {1, 163, 88}
, {-91, 27, 58}
, {212, 276, 180}
, {39, -44, 7}
, {-26, 35, -16}
, {41, 123, 15}
, {-532, -132, -169}
, {11, -4, -44}
}
, {{16, 11, -178}
, {45, 66, -81}
, {-102, -129, -113}
, {-14, -39, -21}
, {-53, 81, -33}
, {2, -155, -78}
, {-52, 120, 74}
, {21, 181, 148}
, {42, 142, 55}
, {-107, -62, 54}
, {-97, 29, -30}
, {160, 66, 153}
, {100, -202, 36}
, {63, -62, 3}
, {-8, -15, -60}
, {-21, 60, 126}
}
, {{24, 156, 162}
, {-62, -153, -154}
, {-48, 78, 98}
, {-70, 53, 52}
, {4, -70, -52}
, {-46, -74, -22}
, {-202, -102, -50}
, {146, 47, 65}
, {-56, 98, 10}
, {-13, 2, 87}
, {99, 15, -61}
, {-129, -128, -42}
, {-12, 148, 48}
, {167, 4, -8}
, {137, 119, 49}
, {-156, -98, -45}
}
, {{-96, 2, -15}
, {-107, 52, -99}
, {-70, 84, -127}
, {23, 138, 85}
, {6, 69, 5}
, {2, -208, -11}
, {-57, 32, 54}
, {-39, 82, 12}
, {17, -97, -158}
, {-106, -102, -34}
, {48, 39, 17}
, {128, -85, 95}
, {194, -122, 126}
, {-54, -118, -86}
, {-131, -169, -416}
, {-30, -102, -83}
}
, {{22, 136, -33}
, {-123, -60, 15}
, {-11, -17, 46}
, {-1, -209, -134}
, {-78, -8, -49}
, {-32, 72, 43}
, {48, 153, 103}
, {-28, 12, 95}
, {-18, -145, 11}
, {17, 51, -27}
, {-63, -59, 62}
, {-24, -17, -149}
, {-3, -77, -75}
, {75, 26, -30}
, {0, 116, 125}
, {22, 199, 134}
}
, {{-144, -64, 99}
, {72, 14, 41}
, {110, 113, 73}
, {-30, -174, -13}
, {143, -5, -78}
, {-35, -59, 65}
, {-44, -28, -4}
, {100, 165, 151}
, {9, -46, 55}
, {-63, -224, -116}
, {-62, -160, -72}
, {-17, 61, 79}
, {26, -60, -31}
, {-108, -148, -71}
, {65, -35, -12}
, {95, 122, 68}
}
, {{-19, -51, -73}
, {-20, -62, -49}
, {-16, -7, -109}
, {136, 27, -178}
, {-64, -77, -82}
, {-22, -132, -94}
, {-61, -14, 47}
, {49, 90, 231}
, {9, 34, 31}
, {52, 98, -83}
, {186, 248, 28}
, {0, -63, -31}
, {168, 81, -63}
, {6, -156, -79}
, {24, -52, -281}
, {-11, -28, 55}
}
, {{106, 90, 80}
, {42, -102, -10}
, {-45, -13, 11}
, {0, -128, -76}
, {-116, 71, 99}
, {40, -24, 69}
, {20, 4, -21}
, {-68, 45, 18}
, {106, 49, 27}
, {-148, -112, -105}
, {121, 129, 33}
, {-26, 144, 113}
, {-13, -7, 61}
, {-55, -15, -56}
, {-28, 98, 17}
, {-187, -120, -170}
}
, {{-21, -3, -86}
, {69, 26, -66}
, {-29, 65, 48}
, {-88, 126, 90}
, {84, 95, 130}
, {-67, 121, -45}
, {-95, -34, -41}
, {-17, -9, -130}
, {-52, 20, 12}
, {-173, 24, 18}
, {52, 47, -45}
, {-11, 68, 36}
, {-158, 41, 138}
, {-30, 15, -35}
, {77, 155, 32}
, {-56, -51, -140}
}
, {{-122, -58, -266}
, {37, -6, -47}
, {47, 62, -131}
, {-100, -108, -95}
, {87, 63, -23}
, {123, 101, -53}
, {106, 5, -145}
, {-100, -20, -24}
, {-88, -96, 31}
, {31, 134, 87}
, {-68, -78, -285}
, {-104, -39, -46}
, {-40, -73, -39}
, {-145, -42, -44}
, {98, 28, 78}
, {62, 40, -13}
}
, {{-174, -242, -109}
, {-29, -81, 36}
, {-3, 60, 51}
, {32, 94, -22}
, {103, -48, 4}
, {3, -10, -163}
, {-88, 177, 117}
, {58, 66, -3}
, {-67, 27, 76}
, {-150, -17, -124}
, {-19, -41, 10}
, {38, 59, 81}
, {-2, 120, 15}
, {-23, -2, 58}
, {-139, -231, -71}
, {-2, -97, -51}
}
, {{-17, -145, -69}
, {-36, 122, 14}
, {81, 9, 81}
, {-7, -27, -30}
, {-6, -2, -26}
, {134, -10, 173}
, {-42, 84, -59}
, {-97, -28, -122}
, {-30, -171, -4}
, {-64, -167, -36}
, {-124, -277, -138}
, {37, -146, 14}
, {53, -169, 62}
, {68, 105, 25}
, {-53, -10, -67}
, {99, -16, 26}
}
, {{26, -72, -49}
, {-195, 67, 174}
, {-96, -32, 14}
, {-80, 13, 35}
, {8, 38, 43}
, {-3, 56, 85}
, {-201, -154, 34}
, {38, -164, -11}
, {-15, -24, -17}
, {18, 59, 31}
, {-61, 80, 14}
, {-93, -101, -40}
, {131, -46, -219}
, {34, 26, -11}
, {-92, 12, -6}
, {0, -257, -96}
}
, {{-40, -39, -132}
, {42, -40, 50}
, {-8, 38, 55}
, {-89, -46, 8}
, {136, 90, -7}
, {61, -50, -103}
, {-86, 114, 126}
, {-16, -27, 115}
, {-55, 158, 123}
, {35, -76, -97}
, {8, -68, -18}
, {-30, 128, 87}
, {-1, 49, 4}
, {-149, -94, -74}
, {-63, -99, -88}
, {189, 58, -10}
}
, {{-164, -191, 24}
, {42, 0, -58}
, {120, 22, 134}
, {133, 18, 4}
, {-37, 119, 133}
, {119, -40, 102}
, {-105, -92, 34}
, {-50, -157, -69}
, {-4, 86, 157}
, {77, 37, 60}
, {-26, 6, 80}
, {-155, -199, -170}
, {28, -6, -18}
, {-10, -44, -76}
, {2, -22, -79}
, {76, 65, 83}
}
, {{-47, -91, 16}
, {67, 29, -45}
, {84, 31, 82}
, {-54, -83, -3}
, {79, 98, 75}
, {-29, -91, 50}
, {-28, -94, -43}
, {41, 20, 23}
, {-39, -23, -14}
, {50, 8, -75}
, {2, -59, -79}
, {56, -24, 24}
, {-25, -48, -49}
, {25, 28, 84}
, {62, 231, 153}
, {-3, -71, 8}
}
, {{50, -36, 94}
, {-61, 65, -71}
, {-52, -167, -78}
, {-78, -84, -22}
, {62, 31, -31}
, {66, 35, 21}
, {-120, -221, -42}
, {119, 199, 29}
, {25, 192, 145}
, {-131, -4, -43}
, {-60, -112, -42}
, {53, 120, 3}
, {29, -93, -110}
, {0, 98, 100}
, {-88, -205, -99}
, {6, 64, 109}
}
, {{-48, -135, 9}
, {61, -109, 46}
, {-104, -121, -81}
, {57, -16, 36}
, {64, 59, 39}
, {-109, 17, -156}
, {162, 65, -83}
, {-98, -123, -18}
, {27, 96, 159}
, {-85, 30, 69}
, {-62, -40, 46}
, {-2, 5, -26}
, {-4, 116, -20}
, {107, -28, 9}
, {14, -19, 1}
, {14, 35, 0}
}
, {{17, 178, 52}
, {-46, -62, -76}
, {75, 26, -43}
, {62, 94, 135}
, {-11, 43, -33}
, {-2, -11, 65}
, {5, 25, 4}
, {57, 146, 115}
, {4, -28, -79}
, {-22, -15, 50}
, {-124, -24, -67}
, {-54, -7, 46}
, {28, -54, -44}
, {8, -102, -21}
, {23, -29, -68}
, {41, -8, -22}
}
, {{-21, -101, 46}
, {-73, -25, 69}
, {15, -173, 122}
, {-69, 104, 76}
, {-56, -118, 19}
, {-106, -88, -120}
, {-376, -46, -3}
, {-44, -23, 60}
, {21, -189, -39}
, {60, -2, 104}
, {8, 91, -50}
, {94, 72, -82}
, {139, 121, 54}
, {-236, -229, -165}
, {-66, -145, 25}
, {50, -76, -8}
}
, {{-192, -177, 43}
, {37, 70, -93}
, {110, 1, -173}
, {3, -21, 21}
, {-11, -4, 47}
, {94, 15, -116}
, {-72, -70, -126}
, {-127, -87, 90}
, {67, 89, -81}
, {56, 111, -44}
, {50, -3, 53}
, {-35, -6, -151}
, {84, 116, -71}
, {74, -44, 65}
, {215, 160, 18}
, {-25, -30, -26}
}
, {{135, 255, 392}
, {132, -27, -145}
, {138, -54, -23}
, {-90, -124, -187}
, {-26, -7, -47}
, {153, -78, -97}
, {-36, 78, -63}
, {-9, 106, 135}
, {-17, -66, -117}
, {-1, 11, -31}
, {21, 48, 122}
, {-36, 18, -105}
, {9, 84, -74}
, {120, 77, -10}
, {-17, -14, -10}
, {14, -23, 55}
}
, {{-133, 107, 121}
, {-59, -46, 59}
, {-143, -165, -113}
, {-152, -47, 15}
, {-85, -56, -49}
, {34, -59, 99}
, {32, 74, 90}
, {18, -32, -2}
, {142, 138, 22}
, {147, 109, -76}
, {-141, -190, -184}
, {-98, -89, -61}
, {-142, 55, -82}
, {147, 54, -86}
, {91, 4, -61}
, {65, -159, -18}
}
, {{-151, 54, -99}
, {-227, 89, 159}
, {-100, -25, -1}
, {-18, 45, -85}
, {111, -22, -85}
, {-11, -65, -74}
, {43, 122, -16}
, {-184, -8, 105}
, {45, -48, -20}
, {124, 32, -52}
, {14, -75, 47}
, {130, -65, -134}
, {228, -34, -165}
, {-1, -63, -108}
, {-158, -41, -136}
, {-45, 73, 105}
}
, {{-63, 86, -6}
, {27, -121, 39}
, {118, 179, -33}
, {20, -83, 66}
, {-174, -294, -118}
, {-9, 1, 111}
, {-304, -186, -55}
, {-185, -427, -56}
, {41, 95, 77}
, {88, 67, -76}
, {-289, -178, 76}
, {-115, -3, 40}
, {0, 62, -71}
, {82, -30, -110}
, {-45, -74, -73}
, {106, 153, 121}
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


const int16_t conv1d_34_bias[CONV_FILTERS] = {62, -59, 101, 102, 116, 59, 83, 8, -58, -321, -286, 2, 48, -182, -61, -47, 211, -60, -79, 100, -59, 87, 55, -68, 32, 98, -37, 240, -67, -151, -36, -129, 173, 42, -141, 184, 188, 60, 119, -50, 26, -29, 148, 177, 139, -4, 93, -28, 204, -35, 131, 167, -72, 268, -72, -78, 144, 42, -61, 96, 84, 184, -68, -51}
;

const int16_t conv1d_34_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-114, 18}
, {-108, -18}
, {-134, 73}
, {5, 12}
, {149, -141}
, {74, -28}
, {62, 109}
, {5, 73}
, {152, -74}
, {-229, 39}
, {-23, -80}
, {-114, -55}
, {-98, -2}
, {-7, 92}
, {-72, 45}
, {-77, 43}
, {-199, -62}
, {51, -13}
, {-47, -11}
, {78, 13}
, {0, -7}
, {15, 129}
, {14, 40}
, {57, 28}
, {-27, -10}
, {-20, -53}
, {-108, 61}
, {-47, 72}
, {98, 36}
, {48, -76}
, {-116, 39}
, {-162, -77}
}
, {{21, -58}
, {91, 78}
, {13, 45}
, {-55, -202}
, {-209, -5}
, {-97, -87}
, {82, -135}
, {-63, 26}
, {-70, -172}
, {0, 73}
, {73, -50}
, {43, 165}
, {90, -149}
, {37, -251}
, {-69, 23}
, {54, 20}
, {90, 22}
, {82, -128}
, {2, 90}
, {-49, -41}
, {68, -93}
, {-54, 94}
, {-37, -176}
, {-53, -63}
, {69, -220}
, {66, 55}
, {-195, -81}
, {46, -27}
, {-188, -128}
, {14, 125}
, {182, -369}
, {126, 101}
}
, {{86, 105}
, {-13, 18}
, {22, 65}
, {-47, -54}
, {20, 26}
, {-13, -23}
, {-151, 7}
, {-71, -17}
, {-83, 5}
, {-69, 5}
, {22, 55}
, {11, -55}
, {86, 44}
, {-132, -103}
, {-84, -131}
, {26, 77}
, {31, -5}
, {44, 100}
, {86, 40}
, {33, 48}
, {-111, -9}
, {-8, 82}
, {-133, 0}
, {-55, 18}
, {83, 72}
, {59, 39}
, {60, 12}
, {-44, 56}
, {114, 19}
, {-18, -21}
, {51, -32}
, {50, 77}
}
, {{-137, 99}
, {117, -37}
, {-77, 95}
, {69, -37}
, {136, -25}
, {-84, -77}
, {38, 56}
, {168, 238}
, {-156, -120}
, {55, 62}
, {19, 96}
, {-151, -42}
, {15, 69}
, {192, 72}
, {22, 8}
, {122, -36}
, {63, 109}
, {-54, -91}
, {-47, 86}
, {117, -33}
, {-188, -17}
, {-77, -47}
, {-1, 57}
, {-38, 33}
, {-51, -111}
, {10, -2}
, {73, -2}
, {-120, 80}
, {-66, 99}
, {-123, 60}
, {-128, 184}
, {-67, 13}
}
, {{153, -73}
, {96, -14}
, {-163, 117}
, {-17, -59}
, {199, 51}
, {94, -81}
, {-10, 18}
, {-69, -167}
, {-19, -92}
, {-260, 73}
, {-217, -291}
, {-77, -38}
, {52, -37}
, {65, -133}
, {88, 74}
, {-76, -35}
, {-34, -108}
, {95, 4}
, {-75, -57}
, {179, 195}
, {38, -18}
, {22, -75}
, {84, -53}
, {-208, -174}
, {26, -58}
, {-104, -122}
, {278, -128}
, {-78, -105}
, {114, -160}
, {-69, -31}
, {202, 125}
, {260, 17}
}
, {{-196, -124}
, {-85, -146}
, {-5, 36}
, {22, 76}
, {18, 130}
, {-4, 19}
, {151, 70}
, {-239, -277}
, {33, 16}
, {49, 100}
, {35, 39}
, {-66, 62}
, {-18, -27}
, {-61, -26}
, {-59, -68}
, {63, -14}
, {-81, 154}
, {-210, -276}
, {-40, 21}
, {22, 158}
, {-59, -99}
, {-81, 8}
, {92, 85}
, {-138, -97}
, {13, -56}
, {-34, -83}
, {127, 125}
, {74, 96}
, {2, 25}
, {59, 21}
, {-265, -162}
, {-205, -223}
}
, {{-57, -115}
, {112, 131}
, {-11, -152}
, {-126, -38}
, {7, -100}
, {18, 58}
, {92, 77}
, {213, 124}
, {64, -39}
, {64, 47}
, {-98, -66}
, {-165, 26}
, {156, -100}
, {33, -13}
, {43, 151}
, {-95, -30}
, {67, -22}
, {12, -7}
, {3, -112}
, {-338, 54}
, {-104, -46}
, {-317, -63}
, {41, -177}
, {165, 100}
, {-84, 139}
, {70, 51}
, {87, 85}
, {-175, -47}
, {-71, 115}
, {-122, -50}
, {64, -2}
, {-11, 115}
}
, {{-34, 51}
, {-164, 21}
, {-25, -182}
, {1, -81}
, {-16, 5}
, {-134, -89}
, {56, -18}
, {171, -134}
, {-189, -171}
, {-60, 50}
, {-65, -32}
, {16, 78}
, {89, 60}
, {132, 50}
, {-28, 57}
, {97, -11}
, {105, 9}
, {43, -51}
, {148, -78}
, {49, -9}
, {9, -38}
, {-82, -71}
, {-35, 100}
, {-94, -119}
, {61, -159}
, {62, 32}
, {163, 11}
, {-54, -140}
, {37, -3}
, {-22, -57}
, {-108, 73}
, {-1, -161}
}
, {{91, -89}
, {42, 89}
, {-19, -99}
, {-25, -67}
, {38, -44}
, {53, -78}
, {49, 73}
, {-16, 21}
, {30, 12}
, {102, -29}
, {85, -107}
, {-6, -63}
, {39, -6}
, {85, -7}
, {-25, 80}
, {37, -8}
, {-75, -216}
, {90, 78}
, {-52, 2}
, {-53, -177}
, {-2, 10}
, {-38, 59}
, {-116, -82}
, {-42, 22}
, {-56, 70}
, {87, 75}
, {33, -24}
, {-52, -134}
, {107, -54}
, {12, -18}
, {-60, 131}
, {-132, 83}
}
, {{58, -87}
, {142, 165}
, {170, -208}
, {-105, 18}
, {-151, -51}
, {133, -14}
, {32, 4}
, {-36, -4}
, {145, 100}
, {126, -162}
, {123, -57}
, {-37, -96}
, {86, -55}
, {-66, -95}
, {26, -45}
, {20, -119}
, {-237, -123}
, {47, -99}
, {-12, -51}
, {-36, 7}
, {-30, 46}
, {-208, 30}
, {29, -10}
, {-110, -52}
, {-35, 68}
, {12, -97}
, {119, 74}
, {-31, -34}
, {-164, -119}
, {46, 33}
, {285, -71}
, {191, 203}
}
, {{-266, -175}
, {-171, -1}
, {-285, 2}
, {-97, -42}
, {144, 213}
, {-428, -228}
, {184, 19}
, {61, 65}
, {-169, -7}
, {114, 93}
, {27, 37}
, {-162, 0}
, {-133, -150}
, {-492, -340}
, {-150, -65}
, {-31, -7}
, {138, -306}
, {-28, 8}
, {49, -124}
, {-67, -89}
, {-596, -531}
, {-117, -29}
, {38, 116}
, {136, -123}
, {-38, -7}
, {-4, -5}
, {128, 157}
, {-111, -11}
, {-188, -63}
, {-286, -227}
, {-33, -285}
, {-258, 6}
}
, {{-56, -78}
, {-54, 3}
, {-32, 0}
, {61, -75}
, {-31, -32}
, {18, 55}
, {94, 92}
, {38, 64}
, {-55, 57}
, {77, -47}
, {99, -178}
, {14, 40}
, {-113, -148}
, {41, 207}
, {3, 169}
, {-29, 63}
, {-58, -29}
, {-167, -43}
, {-91, -43}
, {-14, -57}
, {97, -70}
, {-25, -85}
, {123, 124}
, {48, -34}
, {-26, 73}
, {-71, 38}
, {130, 44}
, {-3, -44}
, {136, 42}
, {-13, -183}
, {-451, -310}
, {84, 132}
}
, {{-9, -22}
, {-15, 60}
, {-128, -101}
, {-67, -11}
, {-135, -246}
, {47, -53}
, {-80, 112}
, {165, 21}
, {28, 123}
, {103, 74}
, {-162, -228}
, {102, 36}
, {-18, 29}
, {62, -110}
, {-12, 13}
, {-83, -47}
, {67, 24}
, {103, 121}
, {-126, -126}
, {-127, -73}
, {56, 103}
, {78, 12}
, {5, -149}
, {-29, -21}
, {43, 34}
, {38, 14}
, {-163, -311}
, {-168, -109}
, {65, 5}
, {56, -95}
, {-89, 10}
, {40, 57}
}
, {{-7, -28}
, {16, -103}
, {36, 41}
, {-48, -75}
, {-191, -39}
, {-51, -106}
, {17, -35}
, {137, 140}
, {57, -134}
, {-156, 91}
, {35, 196}
, {-57, 120}
, {39, -177}
, {-53, 11}
, {23, -44}
, {-36, -31}
, {-4, 71}
, {-41, -83}
, {51, 40}
, {-43, 79}
, {107, 52}
, {-92, 126}
, {-76, -173}
, {69, -65}
, {119, -2}
, {-100, 10}
, {-139, 94}
, {-125, 0}
, {-70, 24}
, {72, 144}
, {-86, 166}
, {-77, 42}
}
, {{51, 10}
, {-103, -55}
, {-151, 37}
, {-84, -139}
, {94, -13}
, {-19, -45}
, {64, 20}
, {-216, -309}
, {81, -11}
, {61, 153}
, {145, -71}
, {114, 10}
, {118, 81}
, {101, -76}
, {44, -5}
, {12, -1}
, {-89, -138}
, {2, -53}
, {25, 14}
, {27, 25}
, {66, -49}
, {-45, -102}
, {84, 17}
, {72, 96}
, {-202, -211}
, {6, -43}
, {171, -78}
, {-35, -7}
, {51, -27}
, {18, -39}
, {141, 335}
, {-98, 18}
}
, {{-3, 71}
, {-128, -118}
, {93, -46}
, {-32, 37}
, {-75, -85}
, {-177, -85}
, {-54, 10}
, {-63, -127}
, {-89, -97}
, {75, 2}
, {117, 115}
, {49, 51}
, {14, -88}
, {91, 167}
, {0, 77}
, {72, -96}
, {66, 75}
, {-25, -91}
, {39, -28}
, {46, -2}
, {1, -22}
, {40, -57}
, {47, 56}
, {-43, 44}
, {-64, -49}
, {-83, 4}
, {32, -60}
, {64, -41}
, {74, 44}
, {121, 101}
, {143, 286}
, {14, 7}
}
, {{-234, -375}
, {-49, 71}
, {-31, -77}
, {-19, 17}
, {2, -8}
, {28, -120}
, {-77, -132}
, {-20, -56}
, {99, 87}
, {-23, -49}
, {-105, 47}
, {56, 0}
, {-124, 19}
, {-190, -97}
, {4, 0}
, {48, 147}
, {-51, 29}
, {-169, -16}
, {-139, 42}
, {-46, -50}
, {108, 6}
, {-74, 6}
, {-105, -4}
, {152, 100}
, {62, 56}
, {-118, 3}
, {200, -3}
, {-162, -125}
, {-301, -134}
, {23, 178}
, {-384, -78}
, {-94, 65}
}
, {{-201, -44}
, {-97, 26}
, {-104, -110}
, {-50, 33}
, {117, 2}
, {2, 162}
, {-93, -47}
, {181, 30}
, {-151, 100}
, {66, -110}
, {-4, 25}
, {-35, -145}
, {133, 27}
, {87, 217}
, {55, 16}
, {-73, 71}
, {-35, -37}
, {-119, -5}
, {58, 129}
, {143, 131}
, {-60, -101}
, {42, -116}
, {-166, -14}
, {53, 53}
, {-32, 34}
, {-51, -79}
, {-30, 0}
, {-119, 45}
, {-59, 95}
, {-164, -88}
, {-1, 20}
, {-60, -4}
}
, {{-123, 13}
, {130, -18}
, {115, -63}
, {-11, -71}
, {22, -29}
, {-84, 54}
, {98, 88}
, {-124, 57}
, {-33, 14}
, {61, 42}
, {-234, -145}
, {-101, 41}
, {-88, -132}
, {-124, 173}
, {13, 53}
, {132, -2}
, {72, 71}
, {67, 48}
, {86, -19}
, {-149, -219}
, {-91, -145}
, {-7, -105}
, {69, 35}
, {-133, -143}
, {22, 1}
, {78, -88}
, {-2, -66}
, {-49, 36}
, {-43, 83}
, {16, -60}
, {24, -155}
, {26, -102}
}
, {{-45, 0}
, {-52, 8}
, {-33, -93}
, {-49, 7}
, {94, 50}
, {-2, 105}
, {187, 56}
, {16, -274}
, {50, -81}
, {-30, 120}
, {80, 26}
, {-500, -258}
, {-135, 66}
, {163, 82}
, {34, 26}
, {21, 32}
, {96, 131}
, {40, -91}
, {34, -196}
, {112, 34}
, {-330, -437}
, {-109, -286}
, {-249, -131}
, {-21, -119}
, {85, -61}
, {-7, 34}
, {289, -23}
, {-55, -446}
, {-49, 75}
, {-163, -118}
, {219, -18}
, {78, -60}
}
, {{-5, 15}
, {33, 19}
, {64, 30}
, {-159, -106}
, {55, 90}
, {-163, -176}
, {-100, -121}
, {114, 184}
, {33, 95}
, {-10, 43}
, {41, 50}
, {38, -68}
, {56, 90}
, {-190, -109}
, {116, 29}
, {-12, 45}
, {-18, -124}
, {49, -38}
, {5, 3}
, {22, 10}
, {51, 4}
, {-68, 28}
, {-47, 115}
, {97, 90}
, {-22, -13}
, {-8, -25}
, {160, 22}
, {-70, -101}
, {-12, -74}
, {29, 56}
, {-58, -88}
, {64, -69}
}
, {{-46, 63}
, {60, 48}
, {-52, 14}
, {-14, 96}
, {-63, 55}
, {-74, 69}
, {-19, -30}
, {-5, -23}
, {-62, -9}
, {-15, -154}
, {52, -157}
, {-55, -7}
, {82, 58}
, {-94, -2}
, {22, 93}
, {49, 69}
, {-36, 28}
, {30, 62}
, {46, -55}
, {88, 152}
, {48, -21}
, {8, 51}
, {-4, 6}
, {-30, -35}
, {-7, 69}
, {41, -61}
, {17, -149}
, {-29, 24}
, {37, 131}
, {-97, -56}
, {-12, 21}
, {-80, -86}
}
, {{-19, 22}
, {60, -31}
, {-65, 71}
, {84, 62}
, {10, 109}
, {-47, -47}
, {123, 10}
, {-161, -64}
, {60, -12}
, {22, 48}
, {90, -67}
, {-35, 81}
, {15, -136}
, {-85, 14}
, {42, 21}
, {11, -27}
, {-117, -41}
, {107, 63}
, {-76, 30}
, {12, 54}
, {47, -2}
, {-17, -33}
, {-46, -90}
, {-16, 56}
, {139, 18}
, {-15, -68}
, {-16, 47}
, {37, 61}
, {-96, 95}
, {-38, -28}
, {-48, -40}
, {-47, 52}
}
, {{122, 71}
, {33, 158}
, {174, 150}
, {-67, 81}
, {-154, -276}
, {-44, -51}
, {43, 39}
, {47, -23}
, {-133, -147}
, {188, -89}
, {-88, -265}
, {-12, 153}
, {101, 32}
, {107, 9}
, {-8, -19}
, {-231, -98}
, {6, -210}
, {-82, 21}
, {-144, -63}
, {-6, -11}
, {40, -91}
, {-32, -52}
, {-87, -164}
, {17, -28}
, {-61, -185}
, {44, 58}
, {68, -50}
, {-171, 45}
, {6, 125}
, {37, -23}
, {-29, -25}
, {118, -52}
}
, {{-34, 23}
, {-22, 80}
, {126, -154}
, {84, 38}
, {2, 162}
, {-115, -162}
, {88, -245}
, {-89, -210}
, {-61, 46}
, {37, -258}
, {163, -16}
, {-21, -29}
, {10, 101}
, {177, 83}
, {52, 20}
, {-110, 75}
, {142, -54}
, {-35, 94}
, {-16, 66}
, {-13, 16}
, {5, 113}
, {17, -129}
, {-47, 1}
, {-319, -169}
, {-74, 21}
, {-80, -88}
, {62, -210}
, {-81, 17}
, {151, -110}
, {4, -119}
, {100, 75}
, {-119, -266}
}
, {{11, -25}
, {70, 117}
, {-63, -70}
, {25, 102}
, {65, -31}
, {12, -118}
, {36, -148}
, {199, -103}
, {41, 35}
, {14, -109}
, {34, -101}
, {-60, -78}
, {51, 38}
, {-150, -16}
, {6, -23}
, {-28, 35}
, {-83, -121}
, {14, 74}
, {-318, -59}
, {15, -54}
, {18, 99}
, {-29, -79}
, {-92, 48}
, {103, -42}
, {0, -43}
, {23, 101}
, {-60, -155}
, {52, -7}
, {-81, -85}
, {-45, 23}
, {-61, -191}
, {72, -154}
}
, {{71, 38}
, {71, 2}
, {-129, 0}
, {0, -5}
, {42, -164}
, {-10, 35}
, {137, -42}
, {-32, -127}
, {-97, 16}
, {-67, 16}
, {123, 49}
, {2, -26}
, {-34, -25}
, {79, -72}
, {29, -109}
, {-31, -48}
, {102, 35}
, {83, 39}
, {-4, 53}
, {-74, -20}
, {20, -30}
, {-101, -18}
, {45, 55}
, {-41, -84}
, {43, 18}
, {86, 16}
, {14, -5}
, {-6, 67}
, {20, -74}
, {19, 80}
, {-40, -98}
, {127, -21}
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
, {{10, 56}
, {106, 40}
, {-54, 68}
, {-64, -56}
, {-42, -200}
, {100, 100}
, {95, 115}
, {25, 138}
, {-13, -28}
, {-2, -23}
, {-32, -54}
, {236, 132}
, {-203, -112}
, {91, -180}
, {4, -46}
, {-40, -157}
, {109, 94}
, {-170, -211}
, {-62, -1}
, {-48, 57}
, {-26, -62}
, {55, -21}
, {27, 14}
, {-34, -24}
, {-230, 39}
, {-27, 9}
, {179, 51}
, {5, -89}
, {85, 140}
, {-60, -67}
, {-170, -231}
, {170, -94}
}
, {{-67, -46}
, {3, -72}
, {-109, 34}
, {84, 28}
, {-32, -146}
, {-137, 47}
, {-26, -53}
, {55, 62}
, {71, 74}
, {13, 7}
, {-67, 6}
, {-4, -53}
, {19, -44}
, {9, 65}
, {63, -47}
, {72, -39}
, {-74, 29}
, {53, 59}
, {-34, 89}
, {44, 28}
, {-60, -9}
, {-88, 42}
, {3, 28}
, {60, 86}
, {65, -44}
, {56, 19}
, {26, -55}
, {-145, 69}
, {-104, 61}
, {-118, -71}
, {-37, -58}
, {-46, -43}
}
, {{70, 60}
, {-6, -9}
, {-55, -12}
, {32, 0}
, {70, -50}
, {0, -8}
, {38, -213}
, {-51, -20}
, {8, -25}
, {100, 89}
, {85, 4}
, {-7, -18}
, {2, 29}
, {33, 70}
, {-67, 77}
, {-15, -33}
, {-10, -297}
, {5, 48}
, {-163, -98}
, {-112, -176}
, {-83, 13}
, {27, -27}
, {5, 31}
, {-44, -130}
, {-119, -118}
, {106, 38}
, {-98, -116}
, {28, 127}
, {49, -131}
, {71, 101}
, {-198, -23}
, {111, 60}
}
, {{12, 29}
, {-18, -37}
, {-93, 166}
, {-95, -28}
, {42, 179}
, {-139, -38}
, {-31, -83}
, {79, 93}
, {31, -9}
, {300, 42}
, {-5, -63}
, {-20, -53}
, {42, 35}
, {129, 74}
, {-40, 41}
, {-121, 10}
, {115, 54}
, {72, -54}
, {-121, 101}
, {-251, 64}
, {125, 89}
, {-129, 45}
, {-200, 78}
, {-4, -9}
, {-45, 29}
, {-128, -19}
, {-32, -82}
, {-67, 70}
, {16, -49}
, {-40, 19}
, {59, -27}
, {-97, 131}
}
, {{28, -5}
, {57, 175}
, {39, -29}
, {9, 10}
, {-62, -98}
, {0, -87}
, {154, 85}
, {-56, -54}
, {98, 12}
, {-22, -230}
, {-220, -121}
, {-137, 9}
, {-37, 118}
, {47, -144}
, {30, 119}
, {-154, -90}
, {-56, -109}
, {-140, -4}
, {-48, -75}
, {-89, -92}
, {54, 48}
, {-112, -198}
, {-109, -30}
, {128, 72}
, {18, -102}
, {128, 24}
, {-86, -88}
, {-136, -154}
, {127, 78}
, {-146, -86}
, {-42, -59}
, {54, -107}
}
, {{-42, -61}
, {-55, -64}
, {-106, -184}
, {42, -13}
, {37, 39}
, {61, 69}
, {-11, 33}
, {-48, -30}
, {90, 24}
, {-184, -53}
, {-66, 60}
, {54, 14}
, {84, -82}
, {-29, -347}
, {-100, -63}
, {85, 71}
, {-16, 64}
, {30, 3}
, {19, 1}
, {-22, 80}
, {-100, 23}
, {37, -18}
, {14, -16}
, {9, 22}
, {132, 0}
, {0, 56}
, {8, 127}
, {26, -43}
, {56, -124}
, {-116, 6}
, {57, -177}
, {-109, 17}
}
, {{-20, -31}
, {-151, -104}
, {-49, -126}
, {48, -12}
, {82, 39}
, {136, -36}
, {44, 43}
, {29, -81}
, {-9, 34}
, {0, 187}
, {25, 146}
, {-59, -22}
, {45, -19}
, {-45, -62}
, {37, 75}
, {32, 97}
, {-185, -95}
, {-106, -40}
, {-67, 26}
, {-34, 62}
, {36, 53}
, {-125, -11}
, {38, 71}
, {-29, 12}
, {-15, 71}
, {-158, -116}
, {228, 307}
, {20, 51}
, {-53, 22}
, {-146, -122}
, {169, 223}
, {9, -6}
}
, {{60, -11}
, {41, -52}
, {-28, -13}
, {68, 86}
, {-6, 10}
, {-67, 108}
, {-101, -61}
, {-232, -10}
, {-50, 59}
, {120, 85}
, {107, 14}
, {-26, -21}
, {69, 4}
, {-36, -12}
, {-156, -292}
, {-114, -46}
, {39, 71}
, {-100, 4}
, {57, 58}
, {-32, 0}
, {113, 114}
, {52, -43}
, {-45, -44}
, {-63, -67}
, {-22, -26}
, {-19, -4}
, {-44, 197}
, {-132, 32}
, {114, 23}
, {22, 16}
, {15, -155}
, {-120, -104}
}
, {{18, -25}
, {11, 40}
, {-196, -117}
, {35, 14}
, {21, 53}
, {2, 61}
, {90, 3}
, {-218, -287}
, {46, 19}
, {-28, 66}
, {28, 80}
, {118, 91}
, {-178, -138}
, {-29, 146}
, {-14, 2}
, {53, 45}
, {-35, -33}
, {49, -5}
, {1, 123}
, {41, -67}
, {-75, -70}
, {8, -26}
, {75, -11}
, {-10, 26}
, {-24, 91}
, {-19, -24}
, {-13, 39}
, {-49, -21}
, {58, 119}
, {69, 57}
, {139, 4}
, {-94, -301}
}
, {{56, 103}
, {-60, -76}
, {-138, 51}
, {-81, 16}
, {53, 21}
, {-87, 16}
, {19, 16}
, {-21, 18}
, {-110, 123}
, {-111, -21}
, {-12, -13}
, {-40, -44}
, {-156, -14}
, {-71, 145}
, {-110, 3}
, {-225, -45}
, {-67, 91}
, {-1, 55}
, {-156, -69}
, {102, 71}
, {-131, -22}
, {-170, -26}
, {-154, 149}
, {68, 110}
, {77, 5}
, {15, 7}
, {-103, 147}
, {152, -24}
, {65, 136}
, {106, -10}
, {-14, 173}
, {-195, -22}
}
, {{193, 157}
, {-30, -61}
, {-189, -26}
, {9, 23}
, {-27, -44}
, {29, 50}
, {-98, -192}
, {82, 149}
, {80, 11}
, {93, 35}
, {-79, -10}
, {-63, 10}
, {16, -2}
, {56, 80}
, {-44, 25}
, {-75, -200}
, {-79, -225}
, {3, 77}
, {91, 34}
, {63, -76}
, {56, 14}
, {-112, -97}
, {-117, -109}
, {-97, -3}
, {44, 49}
, {-69, -12}
, {197, 26}
, {-375, -312}
, {76, 237}
, {-56, 12}
, {39, 6}
, {-214, -261}
}
, {{-71, -96}
, {22, 89}
, {51, 112}
, {-87, -67}
, {-42, -59}
, {72, 68}
, {54, 105}
, {6, 25}
, {-59, 38}
, {-63, 109}
, {-44, 3}
, {-5, -76}
, {-121, -42}
, {-43, 44}
, {131, 110}
, {90, -10}
, {-60, -85}
, {-102, -72}
, {48, 13}
, {151, 174}
, {74, 49}
, {75, 79}
, {-69, -27}
, {27, 81}
, {85, 78}
, {48, -80}
, {5, 25}
, {28, 51}
, {78, 44}
, {-84, -119}
, {-60, 157}
, {-1, -78}
}
, {{62, -94}
, {33, -211}
, {43, 49}
, {-82, -8}
, {53, -8}
, {18, 197}
, {-12, -31}
, {12, -64}
, {112, -50}
, {3, 164}
, {-40, -102}
, {-11, 5}
, {-37, -49}
, {67, 42}
, {32, 63}
, {-4, -29}
, {40, -21}
, {127, -67}
, {21, -150}
, {18, 2}
, {34, -121}
, {20, 16}
, {-67, 77}
, {105, -67}
, {-10, -100}
, {88, -47}
, {15, 17}
, {-41, 32}
, {-212, -83}
, {6, -63}
, {122, -186}
, {115, 65}
}
, {{45, 24}
, {8, 115}
, {41, -87}
, {93, -71}
, {100, -100}
, {-28, -202}
, {121, -35}
, {59, 64}
, {-232, -259}
, {57, 102}
, {-3, 0}
, {-53, -129}
, {-47, -46}
, {-45, 86}
, {53, 34}
, {-51, -279}
, {64, -220}
, {-41, 69}
, {124, -145}
, {110, 69}
, {12, -38}
, {61, -76}
, {28, 27}
, {33, 106}
, {-7, -98}
, {-42, 9}
, {151, 34}
, {-48, -424}
, {-215, -225}
, {16, -236}
, {203, -437}
, {222, 70}
}
, {{-139, 80}
, {115, 84}
, {54, -59}
, {-41, -63}
, {72, 22}
, {135, 121}
, {73, 78}
, {-230, -298}
, {23, -182}
, {23, -83}
, {40, -12}
, {101, -29}
, {48, 25}
, {-9, 158}
, {-145, 210}
, {-20, -16}
, {-51, -77}
, {61, -82}
, {-177, 62}
, {-100, 33}
, {1, -112}
, {149, -222}
, {-78, 63}
, {-20, 26}
, {-77, -105}
, {77, -107}
, {58, 116}
, {27, -45}
, {-48, 42}
, {-8, -174}
, {59, -89}
, {-88, -92}
}
, {{-78, -5}
, {110, -38}
, {97, -31}
, {-164, -53}
, {-5, 181}
, {49, -29}
, {54, 75}
, {54, -105}
, {-265, 16}
, {-28, 66}
, {-100, -95}
, {85, 68}
, {-33, -23}
, {229, -97}
, {62, -63}
, {-6, 54}
, {105, 29}
, {13, 31}
, {68, -59}
, {-104, 60}
, {42, -71}
, {-94, -12}
, {-76, 131}
, {33, 119}
, {-46, -130}
, {52, 14}
, {-60, -29}
, {-49, 60}
, {22, 60}
, {-63, 66}
, {96, -314}
, {17, -90}
}
, {{14, -26}
, {-178, -231}
, {20, -188}
, {-46, -88}
, {-32, 72}
, {16, -11}
, {-23, -12}
, {136, 72}
, {97, -121}
, {-81, 131}
, {61, 149}
, {-45, 125}
, {70, 52}
, {-11, 3}
, {-94, -52}
, {56, -77}
, {37, 109}
, {80, -76}
, {53, -106}
, {105, -16}
, {-143, -172}
, {99, -87}
, {16, -45}
, {8, -5}
, {-4, -4}
, {87, -16}
, {-22, -152}
, {85, -27}
, {78, 61}
, {30, -17}
, {-180, -501}
, {-43, 104}
}
, {{-68, -114}
, {86, 77}
, {-51, 91}
, {111, 32}
, {13, -64}
, {101, -78}
, {70, 86}
, {-173, -45}
, {-65, -3}
, {134, 109}
, {-41, -157}
, {-104, -61}
, {38, 21}
, {9, 12}
, {-60, -31}
, {34, 72}
, {-141, -94}
, {-121, 1}
, {-31, 124}
, {17, -46}
, {-124, -14}
, {12, -24}
, {127, 136}
, {-7, 81}
, {-1, -23}
, {-27, 82}
, {-122, -93}
, {29, 59}
, {2, 35}
, {2, 84}
, {34, -171}
, {76, 120}
}
, {{89, -111}
, {-93, -67}
, {27, 53}
, {-14, -25}
, {6, -99}
, {-30, 85}
, {-4, -121}
, {-78, -101}
, {-53, -164}
, {-71, 127}
, {40, -132}
, {-7, -384}
, {-26, 64}
, {52, -140}
, {24, -127}
, {-14, -59}
, {-27, -57}
, {-87, -43}
, {52, -84}
, {-109, -36}
, {78, -47}
, {88, -51}
, {54, 69}
, {35, 157}
, {107, -105}
, {85, -26}
, {-68, 216}
, {85, 0}
, {-112, 0}
, {39, -78}
, {-162, -455}
, {128, 51}
}
, {{31, 43}
, {-46, 137}
, {-32, 113}
, {-30, -22}
, {-46, 84}
, {22, 106}
, {-40, -40}
, {41, 89}
, {81, 105}
, {43, -22}
, {75, 11}
, {15, -38}
, {4, 112}
, {40, 27}
, {32, 17}
, {15, -67}
, {-68, 35}
, {-10, 94}
, {-111, 113}
, {-104, 94}
, {38, 93}
, {60, 31}
, {-59, -90}
, {-66, -10}
, {-2, 23}
, {-36, 13}
, {-46, -123}
, {72, -143}
, {-86, -43}
, {-8, -96}
, {-28, 30}
, {-9, 146}
}
, {{-145, -135}
, {-201, -160}
, {61, 25}
, {-44, -40}
, {147, 86}
, {18, 38}
, {-49, 122}
, {-157, -450}
, {32, 96}
, {137, 143}
, {96, -72}
, {-55, -84}
, {118, 65}
, {26, -23}
, {-42, -47}
, {-66, 84}
, {-6, 0}
, {-132, -14}
, {14, 29}
, {219, 140}
, {72, 60}
, {-8, -28}
, {-31, -32}
, {138, 6}
, {-3, -36}
, {-234, -86}
, {84, 142}
, {129, -22}
, {-28, -4}
, {69, 105}
, {-204, 51}
, {-102, 75}
}
, {{42, 36}
, {206, -57}
, {-207, -196}
, {73, 54}
, {-34, -57}
, {112, -157}
, {266, -116}
, {-113, -58}
, {-102, -27}
, {86, 58}
, {-21, -77}
, {131, 33}
, {87, 16}
, {108, 32}
, {-123, -210}
, {-60, 31}
, {21, -228}
, {-89, 48}
, {-271, -190}
, {92, -195}
, {28, 89}
, {-87, -69}
, {6, 53}
, {-553, -507}
, {-16, -42}
, {0, 44}
, {-118, -210}
, {-40, -51}
, {-26, 44}
, {100, 131}
, {-258, -104}
, {-61, -275}
}
, {{42, -66}
, {50, 15}
, {-166, -150}
, {-55, -25}
, {212, -247}
, {-108, 0}
, {56, 76}
, {91, 21}
, {77, -73}
, {73, 41}
, {-213, 41}
, {-26, -19}
, {1, 97}
, {19, 74}
, {70, 103}
, {-16, -100}
, {-58, 76}
, {-71, -162}
, {123, 10}
, {167, 171}
, {-84, -31}
, {-203, -182}
, {-21, -21}
, {59, 48}
, {-18, 90}
, {-245, -82}
, {-50, -58}
, {-71, 137}
, {194, 160}
, {-65, 68}
, {-119, 27}
, {-114, -46}
}
, {{-97, -143}
, {70, 121}
, {-25, 105}
, {-34, -25}
, {90, 147}
, {126, 38}
, {15, 51}
, {235, 198}
, {-58, -99}
, {203, 7}
, {47, 79}
, {-63, -119}
, {94, -10}
, {204, 190}
, {-3, 58}
, {-67, 15}
, {120, -13}
, {-7, -97}
, {-112, -81}
, {26, -6}
, {-222, 102}
, {-74, -74}
, {-53, 34}
, {-176, -76}
, {76, -34}
, {-105, 23}
, {-3, -123}
, {56, -8}
, {65, 38}
, {40, 5}
, {-209, -83}
, {-27, -155}
}
, {{-11, -59}
, {56, -148}
, {157, -49}
, {-60, -33}
, {45, -18}
, {-233, 51}
, {-169, 334}
, {29, -96}
, {-64, -222}
, {-93, 157}
, {-86, 26}
, {57, 45}
, {-390, -3}
, {114, 43}
, {-13, -7}
, {-15, 33}
, {-95, -41}
, {63, -89}
, {-15, 5}
, {184, 128}
, {-256, -5}
, {-60, 96}
, {-145, 4}
, {52, 161}
, {91, -50}
, {-50, -77}
, {17, 153}
, {-87, 1}
, {25, 33}
, {-4, -155}
, {-233, 254}
, {-51, 75}
}
, {{-149, 4}
, {16, 137}
, {145, 31}
, {-13, -3}
, {-65, -2}
, {-291, -93}
, {53, 57}
, {-56, -128}
, {44, 129}
, {-62, -165}
, {-292, -45}
, {-92, -233}
, {-25, 83}
, {-386, -231}
, {3, -20}
, {28, 53}
, {-128, 14}
, {23, -48}
, {89, 84}
, {49, 140}
, {-40, -68}
, {-40, -41}
, {-99, -96}
, {-9, 117}
, {59, -61}
, {-8, 100}
, {-233, -293}
, {-6, -98}
, {15, -55}
, {13, -30}
, {-13, -288}
, {90, 98}
}
, {{74, 8}
, {48, 72}
, {15, 22}
, {-38, 39}
, {-16, -146}
, {-29, -153}
, {-32, 105}
, {46, -81}
, {15, -5}
, {-48, 17}
, {-81, 56}
, {21, -79}
, {7, 33}
, {15, -159}
, {2, 110}
, {-34, 78}
, {74, 31}
, {40, -20}
, {-55, -109}
, {-6, -37}
, {-42, -60}
, {-32, -10}
, {87, 43}
, {23, 16}
, {23, 31}
, {83, 50}
, {0, 13}
, {17, -16}
, {28, -185}
, {-26, 40}
, {-26, 42}
, {-18, -43}
}
, {{-48, -110}
, {-102, 49}
, {66, 25}
, {-18, 3}
, {-57, 98}
, {120, 81}
, {49, -75}
, {-188, -39}
, {52, -47}
, {93, 21}
, {-108, 4}
, {-66, 32}
, {149, -106}
, {94, 106}
, {7, -17}
, {16, -68}
, {-203, -13}
, {-76, -66}
, {66, -96}
, {6, -244}
, {-29, -115}
, {-118, 25}
, {54, -115}
, {148, 93}
, {-159, -46}
, {12, 63}
, {-13, 63}
, {-201, -242}
, {196, -50}
, {-17, 63}
, {-24, -12}
, {-77, -18}
}
, {{9, -55}
, {226, -6}
, {-139, -83}
, {-174, -21}
, {-5, 124}
, {49, -132}
, {222, -109}
, {24, -21}
, {-112, -41}
, {86, 66}
, {97, 78}
, {-385, -112}
, {131, 135}
, {9, -67}
, {-136, -70}
, {-91, -13}
, {-312, -77}
, {-54, -41}
, {-393, -146}
, {60, 127}
, {-198, -346}
, {-155, -218}
, {-118, -210}
, {-79, -60}
, {-289, -45}
, {-2, 54}
, {23, 82}
, {-314, -213}
, {-33, 69}
, {-108, 113}
, {150, -141}
, {36, 38}
}
, {{-174, -114}
, {-249, 45}
, {-60, 37}
, {42, -136}
, {-92, -197}
, {26, 55}
, {23, -137}
, {20, -356}
, {-24, 68}
, {38, -62}
, {61, 23}
, {-27, 102}
, {-133, -144}
, {5, 88}
, {-29, 79}
, {-38, 66}
, {-69, 30}
, {106, -53}
, {89, 19}
, {63, -6}
, {19, 16}
, {-44, 23}
, {8, -9}
, {112, -63}
, {187, -26}
, {-68, 73}
, {88, -19}
, {-6, -130}
, {106, -163}
, {86, 32}
, {131, 103}
, {104, -67}
}
, {{-97, -7}
, {-72, 232}
, {-77, -112}
, {-104, -7}
, {-104, -123}
, {-40, -15}
, {-71, 57}
, {-139, 19}
, {63, 187}
, {92, 96}
, {-79, -70}
, {44, 26}
, {-93, -209}
, {7, 42}
, {-130, -40}
, {-8, -10}
, {62, -80}
, {-60, -51}
, {7, 43}
, {81, -26}
, {15, -30}
, {-82, -146}
, {-80, -8}
, {57, 61}
, {101, 162}
, {48, -89}
, {69, 121}
, {63, -135}
, {4, 171}
, {110, 94}
, {-259, -390}
, {72, -139}
}
, {{22, -41}
, {-53, 47}
, {14, -101}
, {-57, -58}
, {42, 64}
, {22, 219}
, {174, -50}
, {-12, -8}
, {98, 69}
, {-190, -259}
, {-15, 143}
, {-30, -80}
, {-57, -181}
, {7, -24}
, {-9, 3}
, {-23, 40}
, {-214, -172}
, {5, 17}
, {-329, -129}
, {-4, 127}
, {69, 54}
, {-67, 45}
, {92, 14}
, {47, -39}
, {-14, 79}
, {23, -66}
, {-12, -62}
, {89, 55}
, {52, -133}
, {-25, -20}
, {-179, -225}
, {-151, -38}
}
, {{32, 111}
, {-82, 26}
, {-197, -62}
, {-47, 69}
, {-154, -16}
, {-189, 57}
, {0, 86}
, {-259, -20}
, {-110, 66}
, {-13, 16}
, {-177, -159}
, {70, -72}
, {20, -31}
, {44, 4}
, {29, 157}
, {-145, 69}
, {-115, -46}
, {-95, 49}
, {-39, -64}
, {-152, -166}
, {20, 19}
, {-46, -18}
, {-22, 130}
, {-54, -137}
, {-88, 109}
, {13, 83}
, {39, -80}
, {-160, -16}
, {-135, -15}
, {53, 25}
, {70, 100}
, {188, -82}
}
, {{-13, -241}
, {152, -179}
, {23, -60}
, {-44, 80}
, {96, 188}
, {-61, 34}
, {110, 42}
, {-22, -128}
, {18, 38}
, {-106, 107}
, {-83, -48}
, {-30, -39}
, {56, -34}
, {-44, -28}
, {111, 26}
, {-104, 15}
, {126, -37}
, {128, -310}
, {71, -66}
, {15, 14}
, {3, 85}
, {-59, 50}
, {22, 39}
, {55, 2}
, {-115, -30}
, {-127, -176}
, {-66, 90}
, {-40, 105}
, {66, -54}
, {-88, -67}
, {-262, 165}
, {-12, 22}
}
, {{-66, 30}
, {-118, 19}
, {-89, -182}
, {28, -51}
, {178, 194}
, {-245, -34}
, {-61, 90}
, {17, 74}
, {16, 49}
, {87, 107}
, {55, 155}
, {27, 0}
, {61, 42}
, {-25, 42}
, {16, 101}
, {-121, -9}
, {73, 203}
, {-42, -55}
, {-1, 26}
, {72, 41}
, {-32, 76}
, {-244, -140}
, {-15, -2}
, {58, 53}
, {47, 86}
, {53, -115}
, {67, 84}
, {3, -52}
, {-34, 29}
, {166, 111}
, {68, -31}
, {-68, -97}
}
, {{-198, -107}
, {-131, 32}
, {76, 14}
, {-50, -80}
, {55, 52}
, {8, -95}
, {-16, 43}
, {175, -89}
, {-66, 101}
, {-66, -61}
, {-84, -16}
, {-34, -137}
, {60, 55}
, {-203, -84}
, {-20, -1}
, {129, 23}
, {41, 23}
, {-46, -31}
, {-8, -7}
, {0, 75}
, {-23, 70}
, {92, 78}
, {-10, 24}
, {113, 54}
, {-27, -49}
, {-47, -4}
, {-117, -59}
, {75, 37}
, {-111, -89}
, {33, 77}
, {-328, -48}
, {99, 68}
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


const int16_t dense_12_bias[FC_UNITS] = {-120, 25, 3, 51, 39}
;

const int16_t dense_12_kernel[FC_UNITS][INPUT_SAMPLES] = {{-87, -318, -94, -56, 158, -103, 207, 131, 102, -240, -207, -116, 144, 269, 60, 156, -124, 93, 79, 216, 120, -79, -120, -127, 105, -3, 89, -356, 91, 97, -19, 112, 61, 13, -17, -19, 87, 105, -233, 121, -56, -227, 99, -114, -215, 123, 3, -130, -164, -63, -89, -153, -306, -141, 86, 204, 256, 47, -12, -146, 33, -257, 117, -29}
, {75, -60, -87, 139, -483, 135, 69, 182, 27, 261, -396, 83, 8, -16, 50, 114, -314, 226, 331, -158, -14, 17, -74, -336, -58, 49, -64, -211, -95, 102, -13, -42, -199, -64, -69, 81, 139, -301, 1, -112, -105, 204, -296, 277, 276, 108, -68, 19, 182, 34, -222, 219, -324, -66, -106, -374, -535, 136, -185, -28, 103, 132, -163, -48}
, {86, 166, -118, -237, -15, -100, -33, 173, -59, 98, 357, -214, -72, -34, -92, 55, 82, 156, 70, 371, 12, 20, -81, 65, -185, -18, -13, 547, -209, 72, 215, -176, 242, -34, -24, -27, -28, 87, 160, 86, 44, -162, -191, -59, -105, 144, -196, 12, -219, 88, -169, -173, 61, 171, 40, -1, -54, -39, -54, 130, -137, 32, -166, 103}
, {-150, 95, -21, -55, 200, -50, -396, 112, -98, 113, -28, -48, -219, -155, -63, -96, 216, -345, -215, -266, -118, -38, 107, -65, 0, -72, 97, -106, 354, 131, -89, -29, 111, 71, -48, 171, 110, 139, 130, -43, 61, 239, 112, -64, -1, 103, 155, -63, 55, -307, -268, -102, 282, -112, -84, -100, -65, 204, 346, -170, -153, 231, 33, 148}
, {100, 250, 45, -16, -47, 190, 38, -292, 127, 47, -641, 101, 108, -150, 143, -47, -198, 123, -141, -85, 80, 93, 45, 232, -208, -184, 29, -265, 4, -20, -151, -213, -225, 71, 176, -40, -107, -45, -139, -34, -167, -296, -79, -130, -20, -192, 3, -23, 37, 149, 287, 26, -228, 104, 33, 86, 240, -169, -188, -8, 28, 11, -14, 72}
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
