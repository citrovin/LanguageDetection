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
#define CONV_KERNEL_SIZE    5
#define CONV_STRIDE         6

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_12_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_12(
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
#define CONV_KERNEL_SIZE  5


const int16_t conv1d_12_bias[CONV_FILTERS] = {-7, -3, -46, 0, -35, 15, 16, 26}
;

const int16_t conv1d_12_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-86, 158, -21, -64, 5}
}
, {{113, 20, -121, -32, -135}
}
, {{76, -135, -60, 101, 61}
}
, {{-185, -138, -17, 42, 159}
}
, {{109, -149, 11, -47, 108}
}
, {{-56, -2, -109, 92, 68}
}
, {{-56, 38, -108, 66, 168}
}
, {{177, -10, 171, 87, -170}
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
#define INPUT_SAMPLES   2666
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_9_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_9(
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
#define INPUT_SAMPLES       666
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         4

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_13_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_13(
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
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_13_bias[CONV_FILTERS] = {-3, 26, -3, -40, -11, -20, -9, 22, -16, -13, 18, 27, 12, 7, -21, -11}
;

const int16_t conv1d_13_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-56, -82, 81}
, {0, 88, 93}
, {11, 27, 104}
, {33, 23, 28}
, {80, 83, 118}
, {-90, 9, -138}
, {62, 51, -35}
, {-130, -41, -48}
}
, {{88, -130, -109}
, {-88, 85, -147}
, {-142, -118, -27}
, {68, 31, -116}
, {96, 34, -14}
, {-46, -98, 8}
, {-25, 93, 118}
, {110, 65, 120}
}
, {{47, 105, -45}
, {-58, 100, -21}
, {39, 32, 54}
, {-49, 109, 23}
, {-5, -81, -7}
, {39, -9, -132}
, {-97, 119, 125}
, {34, -117, 53}
}
, {{-142, -80, 7}
, {-4, 125, 125}
, {158, 141, -74}
, {-93, -24, 53}
, {65, 132, -60}
, {-62, -123, -143}
, {81, 127, -142}
, {0, -25, 93}
}
, {{58, -42, -119}
, {0, 102, -48}
, {79, -62, -37}
, {56, 94, 88}
, {-66, -76, 8}
, {-38, -145, -144}
, {118, 23, 58}
, {-121, 72, -135}
}
, {{-9, 170, 49}
, {-80, -82, -138}
, {84, 10, 129}
, {-20, 116, 114}
, {-42, 84, 169}
, {78, -13, -49}
, {29, -115, -36}
, {72, -15, -121}
}
, {{-73, -49, -9}
, {-70, -25, -79}
, {130, -72, -60}
, {110, -77, 34}
, {67, -98, 108}
, {117, 61, -128}
, {-12, 2, 114}
, {-49, -26, 40}
}
, {{152, -97, -85}
, {154, 106, 17}
, {-142, -12, -108}
, {-84, 132, -108}
, {39, -9, -69}
, {127, -117, -117}
, {-67, -45, 90}
, {-66, 23, 47}
}
, {{112, -133, 70}
, {-55, 72, 64}
, {109, 49, -96}
, {20, -109, -142}
, {-123, -30, -27}
, {-140, 110, 10}
, {-8, -115, 55}
, {34, -149, 6}
}
, {{7, 100, -55}
, {-31, -135, 38}
, {40, 58, 7}
, {-102, -64, 9}
, {-17, -141, 34}
, {119, -12, -39}
, {-139, 34, -105}
, {-103, 58, -15}
}
, {{-90, -85, -74}
, {112, -138, -14}
, {-47, -21, -5}
, {12, -145, -10}
, {63, -40, -123}
, {79, -42, -58}
, {66, 37, 101}
, {-133, -59, 55}
}
, {{-126, -64, 115}
, {52, 37, -44}
, {-172, -12, -47}
, {25, 93, -74}
, {105, 36, -157}
, {124, -54, 57}
, {19, -134, -6}
, {-45, -58, -96}
}
, {{-84, 83, -10}
, {-37, -75, 88}
, {101, -86, 41}
, {82, 83, -57}
, {117, -109, 55}
, {122, -133, 15}
, {-72, 1, -93}
, {-126, -73, -76}
}
, {{-13, -56, -13}
, {69, -132, 64}
, {-63, -19, -23}
, {-40, -10, 101}
, {-19, -101, 3}
, {-112, 122, -171}
, {69, 18, 54}
, {-57, -137, -68}
}
, {{26, -145, 107}
, {14, -69, 6}
, {94, 96, -55}
, {-97, -147, 119}
, {-74, -57, 107}
, {-59, -11, 121}
, {-55, -59, -133}
, {43, 69, -28}
}
, {{28, 137, -60}
, {74, -149, 3}
, {169, 24, -116}
, {-79, 22, 90}
, {-6, 30, -13}
, {115, -63, -55}
, {-85, -23, -2}
, {15, -1, -86}
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
#define INPUT_SAMPLES   166
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_10_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_10(
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
#define INPUT_SAMPLES       41
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         2

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_14_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_14(
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
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_14_bias[CONV_FILTERS] = {1, 8, -20, 20, -4, -8, -32, -3, 38, 0, 19, 22, 9, 0, 9, -6}
;

const int16_t conv1d_14_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-63, 55, 86}
, {-4, 48, -102}
, {11, 86, 101}
, {16, 25, -10}
, {64, 99, -67}
, {-87, -113, 30}
, {54, -44, -11}
, {-105, -56, -48}
, {-79, -15, -127}
, {55, 34, -95}
, {3, -17, -74}
, {12, -137, 33}
, {49, -99, -17}
, {9, -8, -1}
, {21, -123, -79}
, {-65, -47, -89}
}
, {{73, -113, -118}
, {-76, -118, -99}
, {-92, -26, 81}
, {70, -99, -113}
, {84, -40, -59}
, {-34, -3, 28}
, {-9, 87, -117}
, {117, 95, 30}
, {-111, 126, 56}
, {60, 52, -93}
, {-72, -19, -90}
, {33, 42, -76}
, {25, 63, -93}
, {-74, 14, 110}
, {69, 76, 35}
, {48, -55, 90}
}
, {{26, -51, -78}
, {-44, -9, -60}
, {21, 26, -139}
, {-32, 11, 90}
, {-16, -22, -134}
, {39, -113, 24}
, {-78, 110, -126}
, {19, 27, -111}
, {88, -95, 29}
, {107, 99, -123}
, {57, 1, -49}
, {108, 43, -55}
, {-69, -77, -63}
, {-1, 2, 25}
, {103, 107, -35}
, {-89, 108, -40}
}
, {{-143, -15, -88}
, {0, 120, 5}
, {149, -57, -79}
, {-72, 56, 15}
, {26, -59, 48}
, {-77, -131, -99}
, {33, -126, 28}
, {37, 114, 108}
, {-82, -85, -15}
, {92, 78, -78}
, {116, -1, 148}
, {-21, 119, -33}
, {82, 117, 51}
, {-96, 55, -75}
, {110, -98, -7}
, {17, 32, -54}
}
, {{81, -86, 11}
, {-1, -51, 18}
, {87, -20, 78}
, {51, 70, -35}
, {-59, 9, -88}
, {11, -85, 3}
, {129, 78, -33}
, {-128, -99, -69}
, {-10, -59, -92}
, {119, -53, -18}
, {-3, 119, -35}
, {95, -71, -51}
, {-49, 29, -108}
, {-113, 29, 58}
, {65, -62, 130}
, {106, 140, -3}
}
, {{-40, 0, -5}
, {-84, -119, -98}
, {43, 80, -77}
, {-34, 89, -39}
, {-66, 104, 90}
, {74, -47, -10}
, {23, -44, -66}
, {40, -138, -117}
, {118, 24, 46}
, {-78, -8, 98}
, {11, 15, -81}
, {87, 34, -114}
, {49, 29, 19}
, {-17, 42, -48}
, {-94, 73, -11}
, {-3, 52, -66}
}
, {{-45, 26, 129}
, {-88, -66, 20}
, {105, -54, 50}
, {71, 33, 34}
, {97, 123, -60}
, {125, -71, 34}
, {27, 118, -52}
, {-52, 25, -73}
, {-19, 115, -109}
, {1, 69, 102}
, {-6, 35, -92}
, {-56, -108, -104}
, {-61, 80, -111}
, {63, -120, -95}
, {9, 94, -70}
, {-30, 45, 24}
}
, {{113, -110, -144}
, {141, 40, -85}
, {-115, -123, -51}
, {-83, -73, 104}
, {59, -31, -82}
, {99, -131, 64}
, {-22, 92, -124}
, {1, 72, 35}
, {-103, -29, 69}
, {70, 79, -13}
, {45, 86, 7}
, {118, -69, -1}
, {0, 6, 108}
, {-91, 86, 6}
, {-73, -31, 92}
, {17, 60, 42}
}
, {{92, 52, 14}
, {-24, 51, -9}
, {117, -79, 87}
, {39, -123, -129}
, {-99, -21, 81}
, {-112, -5, -92}
, {-21, 35, -31}
, {65, 24, -33}
, {-102, 75, 115}
, {69, 60, 109}
, {40, 61, -80}
, {-74, 111, 74}
, {-13, -75, 34}
, {109, 23, 5}
, {-95, -4, -61}
, {-105, 20, -64}
}
, {{-3, -53, 88}
, {-35, 50, 125}
, {27, 5, -24}
, {-91, 26, -48}
, {-31, 21, 25}
, {97, -29, 2}
, {-115, -82, 20}
, {-82, -13, -12}
, {84, 5, -91}
, {-112, 48, 88}
, {36, 68, -157}
, {-68, 6, -53}
, {-113, -19, -46}
, {-16, 94, -11}
, {19, -96, -24}
, {56, 36, 6}
}
, {{-91, -51, -55}
, {83, 15, -124}
, {-44, -16, 62}
, {-7, 6, -31}
, {35, -105, 111}
, {38, -56, -96}
, {11, 61, 70}
, {-125, 53, 122}
, {-69, 118, 82}
, {-109, 113, 122}
, {-12, 57, -79}
, {-133, -96, -12}
, {-42, 113, -3}
, {-40, -29, -8}
, {21, -109, 21}
, {-77, -106, 105}
}
, {{-102, 96, -86}
, {58, -23, -109}
, {-125, -43, -108}
, {20, -51, 32}
, {136, -100, 80}
, {113, 45, -40}
, {44, -2, -90}
, {18, -73, -79}
, {-39, 35, 93}
, {37, 58, 55}
, {35, 138, 89}
, {82, 134, -64}
, {43, -50, 99}
, {-21, 62, -3}
, {-130, -3, 91}
, {-42, 3, 95}
}
, {{-109, -26, 66}
, {-13, 94, 75}
, {94, 18, 66}
, {67, -33, -97}
, {89, 22, 89}
, {118, 9, -69}
, {-56, -92, -140}
, {-99, -56, -27}
, {53, -76, -80}
, {-47, 34, 55}
, {-111, -115, -115}
, {48, 4, -146}
, {-107, 109, 20}
, {-134, -110, 90}
, {-25, 32, 53}
, {-62, -62, 119}
}
, {{13, 9, 64}
, {61, 52, 108}
, {-41, -21, -105}
, {-25, 93, -17}
, {19, 32, 44}
, {-93, -135, -30}
, {94, 61, 39}
, {-27, -62, -11}
, {-32, 92, -25}
, {-110, -129, -21}
, {24, 57, -70}
, {19, 107, 18}
, {-82, 1, 42}
, {132, 86, -2}
, {28, -61, -106}
, {-124, 46, -124}
}
, {{33, 99, 134}
, {46, 14, 101}
, {53, -53, -98}
, {-66, 105, 20}
, {-78, 78, -31}
, {-24, 128, 46}
, {-67, -130, -86}
, {6, -15, 69}
, {-117, -10, 47}
, {-28, 55, 88}
, {46, -131, 22}
, {-95, -83, -89}
, {-20, -46, 50}
, {-12, -35, 38}
, {-56, -97, -21}
, {75, -78, -1}
}
, {{30, -51, -28}
, {57, -3, 79}
, {109, -95, 97}
, {-70, 79, -111}
, {-35, -27, 8}
, {101, -19, 105}
, {-100, -9, 82}
, {6, -62, -66}
, {109, -85, -9}
, {-122, 17, -127}
, {6, -55, 114}
, {18, -42, -103}
, {14, 97, -15}
, {-74, 44, -57}
, {-25, 30, 44}
, {4, 86, -37}
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
#define INPUT_SAMPLES   20
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_11_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_11(
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
#define INPUT_SAMPLES       5
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    2
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_15_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_15(
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
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  2


const int16_t conv1d_15_bias[CONV_FILTERS] = {3, -29, 5, 0, 25, 0, 5, -6, -14, -25, 5, 20, -12, -15, 7, -27, -29, -18, -21, -9, -3, -11, 29, 0, -3, 3, -26, -30, 2, 24, -9, -15, 2, 22, 11, 30, -2, -9, 5, 24, -10, -7, 25, -12, 20, -17, -17, 17, -10, -7, -6, 15, 1, -38, 3, 4, 8, -31, -9, -3, -9, 1, 8, -1}
;

const int16_t conv1d_15_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-23, -44}
, {55, 72}
, {-31, -7}
, {66, 110}
, {42, -71}
, {78, -11}
, {-2, 8}
, {-34, 13}
, {92, -27}
, {-59, -23}
, {-94, 72}
, {18, 30}
, {-59, 85}
, {62, 58}
, {-91, -20}
, {-98, 63}
}
, {{57, 79}
, {54, 22}
, {-101, 44}
, {-6, -96}
, {-81, 9}
, {-20, -9}
, {89, 13}
, {2, -11}
, {-84, 82}
, {-42, -47}
, {30, -73}
, {-85, 54}
, {24, -68}
, {50, -70}
, {-73, -82}
, {65, 10}
}
, {{33, 24}
, {1, 40}
, {77, 79}
, {-48, -55}
, {-34, -50}
, {-7, 30}
, {-61, -97}
, {-48, -17}
, {-49, 81}
, {-95, 34}
, {35, 76}
, {-34, -18}
, {57, -4}
, {-61, 99}
, {-36, 41}
, {27, 63}
}
, {{-85, -23}
, {58, 30}
, {-88, -59}
, {79, 83}
, {-5, -41}
, {-21, -65}
, {-54, -37}
, {69, -91}
, {-53, 3}
, {33, -19}
, {-9, -61}
, {36, -115}
, {10, 22}
, {-17, -104}
, {37, 60}
, {73, 4}
}
, {{57, -39}
, {-20, 116}
, {-4, -3}
, {-32, 76}
, {-63, -39}
, {19, 34}
, {-44, 68}
, {45, 2}
, {24, 47}
, {-34, 6}
, {-70, 94}
, {-63, -45}
, {17, 103}
, {49, -49}
, {-20, 0}
, {44, -67}
}
, {{-4, -51}
, {-26, -95}
, {100, 8}
, {50, -72}
, {26, -11}
, {76, -47}
, {54, 69}
, {67, -33}
, {28, 74}
, {53, -29}
, {56, 95}
, {14, 55}
, {47, -91}
, {-57, 7}
, {-72, -35}
, {23, -3}
}
, {{-41, 34}
, {60, 9}
, {-15, 9}
, {-44, -53}
, {16, -42}
, {94, -35}
, {92, 85}
, {61, 59}
, {85, -65}
, {-70, 3}
, {-101, -28}
, {-69, -95}
, {-80, -48}
, {79, 11}
, {18, -44}
, {-97, -57}
}
, {{85, 83}
, {26, 80}
, {-70, 65}
, {-18, -39}
, {-68, 38}
, {-57, -69}
, {-16, -96}
, {-12, 29}
, {-99, 12}
, {-71, 48}
, {54, 81}
, {85, 67}
, {80, 12}
, {-3, -44}
, {-75, -65}
, {82, -43}
}
, {{85, -81}
, {-72, 37}
, {-89, -55}
, {-27, 69}
, {74, -11}
, {-6, -12}
, {67, -87}
, {-59, -73}
, {18, -110}
, {73, 91}
, {88, -63}
, {35, -85}
, {-2, 53}
, {41, 35}
, {-97, -53}
, {-13, -76}
}
, {{23, 37}
, {-15, 89}
, {63, 26}
, {-91, -56}
, {-37, -87}
, {25, -46}
, {16, 90}
, {6, -65}
, {89, 50}
, {27, -2}
, {-73, 33}
, {-27, -101}
, {81, -85}
, {-10, -46}
, {37, 78}
, {-84, 13}
}
, {{-60, -18}
, {37, 8}
, {-49, -1}
, {-14, 90}
, {-36, -98}
, {-88, 41}
, {72, -3}
, {105, -94}
, {-35, -9}
, {100, 48}
, {63, -19}
, {14, 16}
, {75, -52}
, {-59, -29}
, {-87, 22}
, {37, -72}
}
, {{-63, -15}
, {90, -35}
, {22, 18}
, {68, 97}
, {74, 75}
, {-69, 0}
, {23, 38}
, {-7, 71}
, {-32, 63}
, {47, -66}
, {96, -9}
, {91, 35}
, {-96, 57}
, {-107, -33}
, {-88, -87}
, {-8, -1}
}
, {{-72, 88}
, {83, 22}
, {46, -70}
, {-80, -56}
, {-17, 46}
, {30, 58}
, {-57, 31}
, {93, -96}
, {48, 22}
, {77, -18}
, {-69, 15}
, {9, -80}
, {-98, -59}
, {43, 58}
, {-11, -65}
, {-31, -42}
}
, {{6, -59}
, {9, -56}
, {-30, -35}
, {-66, 2}
, {-4, -2}
, {26, -74}
, {80, 16}
, {-19, -109}
, {44, 28}
, {5, -98}
, {-29, -5}
, {54, -27}
, {-32, 51}
, {62, 39}
, {60, -3}
, {-95, -114}
}
, {{20, -1}
, {-51, -15}
, {-83, 40}
, {-25, 0}
, {58, 50}
, {72, 17}
, {-18, 66}
, {-33, 5}
, {93, -82}
, {-30, 93}
, {27, 7}
, {36, 26}
, {55, -20}
, {56, 68}
, {-59, 2}
, {37, 4}
}
, {{-7, -68}
, {-24, 5}
, {84, 87}
, {-2, 23}
, {-60, 34}
, {-14, 80}
, {-80, 45}
, {84, 16}
, {-41, 53}
, {-6, -93}
, {-6, -59}
, {14, 67}
, {22, -1}
, {40, 84}
, {35, -78}
, {72, -1}
}
, {{-8, 27}
, {-69, -66}
, {36, -67}
, {-14, -111}
, {73, 43}
, {-98, -79}
, {59, 4}
, {-12, 101}
, {-95, -98}
, {59, 8}
, {-54, 58}
, {14, -53}
, {-78, -24}
, {-100, -104}
, {-35, 9}
, {32, 51}
}
, {{-76, -92}
, {-22, -1}
, {54, -33}
, {-68, 50}
, {-108, 9}
, {1, 7}
, {30, -85}
, {13, -10}
, {-79, 58}
, {19, -102}
, {-88, -54}
, {58, 69}
, {34, -34}
, {-42, 7}
, {70, -89}
, {-103, -36}
}
, {{-24, 90}
, {48, 6}
, {83, 1}
, {-11, -102}
, {3, -35}
, {-78, 25}
, {68, 48}
, {40, -77}
, {-46, 81}
, {12, 42}
, {-98, -19}
, {64, 12}
, {-78, -23}
, {-88, 91}
, {0, -36}
, {0, 64}
}
, {{-1, 1}
, {-39, -20}
, {97, 5}
, {-90, 62}
, {99, -68}
, {-97, 61}
, {68, -62}
, {62, 61}
, {11, -9}
, {-69, -44}
, {-58, 42}
, {-49, 50}
, {-70, -64}
, {76, 52}
, {-89, 82}
, {-3, 6}
}
, {{9, -33}
, {0, 44}
, {77, 15}
, {-77, 76}
, {-3, 97}
, {-40, 20}
, {-38, -38}
, {47, 77}
, {12, -26}
, {-9, -9}
, {-31, -27}
, {90, -19}
, {56, 7}
, {-60, 2}
, {26, -97}
, {71, 44}
}
, {{-43, 36}
, {36, 76}
, {-53, 80}
, {-16, 86}
, {-91, 90}
, {-50, -32}
, {-4, 2}
, {52, -30}
, {-65, 27}
, {3, 34}
, {80, 59}
, {-34, -23}
, {72, 57}
, {-6, -45}
, {14, 50}
, {30, -4}
}
, {{-58, -15}
, {74, 96}
, {6, -59}
, {78, -21}
, {-54, 35}
, {-79, 59}
, {54, -78}
, {-102, 0}
, {34, 88}
, {48, -37}
, {92, 47}
, {-87, 16}
, {5, -103}
, {-9, -18}
, {8, -43}
, {23, -17}
}
, {{95, -11}
, {85, -34}
, {49, -65}
, {-76, -21}
, {43, 85}
, {-85, 45}
, {66, -69}
, {52, -79}
, {-81, -21}
, {42, -92}
, {-22, 27}
, {-2, -65}
, {75, 41}
, {80, 72}
, {-14, 64}
, {-58, 88}
}
, {{-40, 90}
, {-96, -64}
, {49, -9}
, {73, -61}
, {48, -48}
, {27, 54}
, {45, -22}
, {-17, -83}
, {-7, -9}
, {-61, -2}
, {65, -2}
, {-4, -53}
, {21, 61}
, {34, 91}
, {22, 8}
, {-18, -54}
}
, {{-21, 63}
, {91, 17}
, {-64, -86}
, {11, 28}
, {46, -27}
, {-8, -2}
, {42, -86}
, {113, 84}
, {90, 17}
, {-5, 66}
, {75, 1}
, {20, 124}
, {7, -8}
, {-98, 15}
, {-68, -11}
, {27, 32}
}
, {{40, 69}
, {45, 10}
, {-105, -102}
, {-55, -113}
, {27, 55}
, {-47, -67}
, {89, 56}
, {-20, 66}
, {-91, -5}
, {-71, 71}
, {63, 0}
, {8, 15}
, {-80, -25}
, {-17, -4}
, {56, -11}
, {-27, 72}
}
, {{34, 89}
, {89, 18}
, {13, 51}
, {-48, 19}
, {-29, -62}
, {51, 63}
, {58, -10}
, {-5, 14}
, {-89, -7}
, {-15, -3}
, {31, -89}
, {-2, 12}
, {65, 63}
, {-61, -105}
, {27, -3}
, {-51, -3}
}
, {{-24, 15}
, {90, -67}
, {-37, -41}
, {-97, -24}
, {50, 25}
, {19, 69}
, {14, -33}
, {-80, -61}
, {57, -87}
, {-60, -22}
, {39, 88}
, {91, 60}
, {-41, -65}
, {7, -74}
, {-35, 12}
, {-35, 98}
}
, {{38, 21}
, {-59, 72}
, {-87, 13}
, {91, 76}
, {44, -75}
, {-91, -92}
, {-102, 33}
, {18, -101}
, {71, 86}
, {-15, 59}
, {-19, 42}
, {-25, -125}
, {79, -51}
, {-22, -23}
, {77, 44}
, {-1, 5}
}
, {{27, -60}
, {-49, -84}
, {-37, -68}
, {5, -24}
, {0, -74}
, {92, 24}
, {33, 10}
, {-1, -69}
, {75, -52}
, {5, 39}
, {40, -84}
, {50, -47}
, {21, 72}
, {39, 30}
, {-106, -53}
, {64, 44}
}
, {{38, -6}
, {56, 30}
, {-106, -88}
, {-78, -14}
, {4, 76}
, {-42, -22}
, {8, -45}
, {42, -12}
, {73, 12}
, {90, -68}
, {-107, -13}
, {-41, 50}
, {70, -83}
, {55, -58}
, {9, -48}
, {-25, -22}
}
, {{1, -100}
, {52, -81}
, {28, 51}
, {54, -37}
, {72, 23}
, {-25, 12}
, {27, -87}
, {-45, 84}
, {90, -16}
, {-10, 45}
, {-39, -61}
, {-30, -7}
, {-53, 54}
, {48, -80}
, {-44, 46}
, {-80, -16}
}
, {{-84, -9}
, {-19, 31}
, {-66, 1}
, {80, -77}
, {-19, -103}
, {73, 72}
, {-33, 29}
, {31, 11}
, {61, 73}
, {-62, -11}
, {-81, 72}
, {-2, 37}
, {89, -6}
, {-41, -52}
, {-41, 95}
, {72, -28}
}
, {{42, 8}
, {-34, -12}
, {58, -16}
, {112, 33}
, {44, 44}
, {89, -59}
, {-14, -8}
, {78, 28}
, {-83, -24}
, {-95, -72}
, {-28, 10}
, {-4, 78}
, {52, -79}
, {-85, -26}
, {19, -54}
, {-7, 45}
}
, {{93, 45}
, {41, -62}
, {71, 14}
, {78, -24}
, {-88, 63}
, {-94, 6}
, {-46, -24}
, {-115, -93}
, {-76, -11}
, {48, -4}
, {75, -61}
, {-49, -105}
, {80, 88}
, {57, 69}
, {-9, 70}
, {-7, 3}
}
, {{48, 33}
, {82, -3}
, {-31, -70}
, {6, -70}
, {-27, -69}
, {49, 47}
, {44, 91}
, {-74, -55}
, {47, -81}
, {-38, 8}
, {-78, 0}
, {76, -90}
, {-76, -62}
, {7, 68}
, {-99, 77}
, {30, -34}
}
, {{28, 38}
, {8, 5}
, {-13, -1}
, {-86, 7}
, {56, 0}
, {-40, 2}
, {-1, -78}
, {27, -34}
, {-53, 3}
, {-50, -86}
, {-69, 44}
, {-11, -67}
, {-40, 77}
, {-22, 82}
, {-93, -12}
, {-97, -80}
}
, {{94, 6}
, {6, -48}
, {-29, 25}
, {16, -38}
, {-37, 54}
, {80, 28}
, {21, -78}
, {93, -38}
, {52, 46}
, {-50, -90}
, {-64, -49}
, {-31, -60}
, {0, 65}
, {82, -48}
, {-35, -1}
, {-70, -40}
}
, {{-76, -40}
, {-55, -45}
, {-23, -12}
, {-33, -21}
, {-59, 30}
, {66, 69}
, {50, 27}
, {-52, -69}
, {-53, 25}
, {-73, 83}
, {-26, -9}
, {44, 21}
, {-31, 77}
, {-104, -112}
, {70, 76}
, {103, 96}
}
, {{71, 7}
, {11, 80}
, {52, -22}
, {-67, -47}
, {-56, -20}
, {48, 89}
, {45, -16}
, {15, 76}
, {74, -81}
, {-11, 53}
, {-61, 19}
, {-14, 94}
, {28, -2}
, {-64, 57}
, {-79, 22}
, {44, -54}
}
, {{30, -70}
, {-79, 36}
, {88, 19}
, {47, 43}
, {43, 52}
, {-46, 70}
, {94, -22}
, {-14, 60}
, {6, -76}
, {36, -74}
, {-62, 64}
, {32, 76}
, {-31, -85}
, {-14, 62}
, {-75, 0}
, {10, -5}
}
, {{-42, -63}
, {24, 57}
, {-4, -70}
, {43, 60}
, {-25, -26}
, {49, 26}
, {54, -26}
, {-76, -54}
, {42, -85}
, {75, 63}
, {-52, -10}
, {-2, -22}
, {73, 16}
, {-96, -64}
, {-81, 10}
, {15, -41}
}
, {{-89, -23}
, {26, 33}
, {0, 29}
, {-103, -38}
, {-11, -1}
, {4, -80}
, {82, -64}
, {3, 15}
, {-95, -89}
, {-58, 55}
, {48, -9}
, {52, -12}
, {0, -108}
, {33, -35}
, {29, 66}
, {48, -33}
}
, {{96, 45}
, {-13, 48}
, {-41, 24}
, {16, 75}
, {0, -113}
, {-54, 59}
, {-58, 18}
, {49, 118}
, {71, 66}
, {-81, 67}
, {-19, 59}
, {55, 92}
, {-68, 86}
, {28, 0}
, {-69, 39}
, {32, 47}
}
, {{-30, -38}
, {63, -93}
, {-12, -73}
, {19, 56}
, {-8, -83}
, {64, 30}
, {31, 90}
, {-64, 24}
, {-72, 51}
, {31, 1}
, {-82, -17}
, {-71, -11}
, {91, 36}
, {-63, -99}
, {-73, -69}
, {29, -18}
}
, {{47, -3}
, {-63, -54}
, {65, 82}
, {-33, 24}
, {-39, -17}
, {-87, 73}
, {-67, -23}
, {-81, -9}
, {-74, 59}
, {-76, 57}
, {40, -82}
, {-19, 63}
, {-87, -76}
, {47, 69}
, {17, -29}
, {-8, 51}
}
, {{69, -31}
, {-87, 31}
, {0, -72}
, {-29, -106}
, {-64, 58}
, {-3, 27}
, {-52, -61}
, {54, -9}
, {62, 1}
, {102, 14}
, {64, -51}
, {18, 43}
, {-13, 4}
, {88, 40}
, {40, 48}
, {-18, -34}
}
, {{10, -29}
, {-68, -77}
, {20, -91}
, {-24, 58}
, {27, 79}
, {-36, 1}
, {-91, -1}
, {-38, 55}
, {-2, -86}
, {-34, -76}
, {39, -6}
, {-24, 46}
, {94, -86}
, {57, -22}
, {-86, -39}
, {-65, 79}
}
, {{67, 60}
, {107, -76}
, {35, -83}
, {32, 66}
, {-63, 31}
, {81, 91}
, {42, 66}
, {51, 3}
, {-75, 53}
, {23, -7}
, {-40, 15}
, {93, 27}
, {62, -76}
, {17, 44}
, {-16, -100}
, {-79, -2}
}
, {{-8, 17}
, {10, 40}
, {82, 56}
, {-68, -28}
, {17, -33}
, {20, -52}
, {33, 7}
, {75, 55}
, {74, 27}
, {-60, 33}
, {-44, 55}
, {-36, -39}
, {-35, 81}
, {-47, 36}
, {45, 48}
, {-54, 54}
}
, {{-66, 87}
, {26, 49}
, {-5, -2}
, {-5, -35}
, {44, -81}
, {68, -37}
, {84, -50}
, {32, -67}
, {14, 94}
, {77, -67}
, {-17, -51}
, {-41, 61}
, {88, -25}
, {-60, -2}
, {-46, 90}
, {-53, 41}
}
, {{62, 53}
, {-97, -39}
, {94, -50}
, {64, 46}
, {78, -81}
, {-82, 76}
, {-49, -29}
, {130, -28}
, {-34, 12}
, {-26, 11}
, {-10, 76}
, {27, -35}
, {-13, -27}
, {107, 22}
, {-72, -99}
, {-45, -8}
}
, {{-38, -84}
, {27, -98}
, {53, 66}
, {-23, -36}
, {89, -20}
, {-145, -19}
, {34, 34}
, {45, -54}
, {-36, -94}
, {-86, -57}
, {-86, -79}
, {-33, -12}
, {14, -6}
, {-117, -96}
, {-17, 41}
, {-7, 92}
}
, {{52, -22}
, {-28, 40}
, {-36, 21}
, {-11, 84}
, {27, -35}
, {28, -31}
, {-79, 63}
, {46, -7}
, {22, 70}
, {-64, 60}
, {-78, -83}
, {25, -46}
, {6, 45}
, {15, -79}
, {-35, -17}
, {-62, -27}
}
, {{-58, -47}
, {-29, -9}
, {72, -35}
, {9, 11}
, {-77, -75}
, {18, 1}
, {-57, 32}
, {38, 40}
, {59, -48}
, {21, 23}
, {-4, 48}
, {19, -94}
, {83, 95}
, {63, -56}
, {43, -28}
, {-21, 66}
}
, {{36, 45}
, {53, 39}
, {-49, -70}
, {-94, -12}
, {-81, -45}
, {29, -4}
, {99, -47}
, {-2, -107}
, {-80, -66}
, {-23, -48}
, {76, -51}
, {-49, -37}
, {59, 29}
, {-89, -2}
, {-35, 64}
, {-69, -84}
}
, {{-83, -104}
, {-72, 40}
, {-52, 65}
, {33, 38}
, {47, 62}
, {-4, 48}
, {16, 32}
, {44, 2}
, {-52, 85}
, {-28, -72}
, {-34, 23}
, {7, 25}
, {-50, 32}
, {19, 36}
, {-78, 72}
, {29, -21}
}
, {{-16, 19}
, {-108, 6}
, {-91, -11}
, {-65, 67}
, {3, 14}
, {34, -78}
, {-69, -94}
, {-81, 19}
, {-27, 41}
, {93, 21}
, {-20, 73}
, {65, 58}
, {-25, -58}
, {56, 47}
, {-88, -12}
, {23, 72}
}
, {{14, -58}
, {-23, 0}
, {41, -84}
, {-43, 23}
, {-38, 28}
, {-74, -98}
, {98, 92}
, {0, -23}
, {7, -36}
, {-60, 96}
, {-57, 64}
, {52, 51}
, {45, 92}
, {15, 18}
, {-40, 85}
, {17, -31}
}
, {{53, -102}
, {-91, -95}
, {65, -68}
, {-45, 16}
, {-45, 23}
, {-54, -50}
, {10, -59}
, {-40, -2}
, {-79, 67}
, {-34, -22}
, {-97, -2}
, {92, -80}
, {29, -101}
, {59, 77}
, {56, -29}
, {-95, 68}
}
, {{-2, 62}
, {-24, 70}
, {2, 76}
, {-75, 47}
, {87, -17}
, {-31, -38}
, {89, -89}
, {72, 61}
, {-8, -82}
, {-16, -41}
, {9, -27}
, {-93, 44}
, {85, 72}
, {-47, 27}
, {43, -108}
, {-65, -29}
}
, {{-40, 84}
, {29, -77}
, {-76, 2}
, {39, 83}
, {77, -21}
, {-5, -45}
, {-48, -47}
, {-84, 45}
, {14, -21}
, {41, 59}
, {-74, 36}
, {-9, 96}
, {89, -32}
, {-103, 52}
, {-39, -91}
, {-59, -25}
}
, {{-80, -21}
, {-28, 28}
, {2, 20}
, {-8, 28}
, {55, -67}
, {-96, 82}
, {-48, -82}
, {58, 20}
, {-71, -23}
, {-67, -114}
, {-104, -26}
, {-57, 55}
, {58, -20}
, {31, 83}
, {-62, 30}
, {90, 91}
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
#define INPUT_SAMPLES   4
#define POOL_SIZE       1
#define POOL_STRIDE     1
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t average_pooling1d_3_output_type[INPUT_CHANNELS][POOL_LENGTH];

void average_pooling1d_3(
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

#define INPUT_DIM [4][64]
#define OUTPUT_DIM 256

//typedef number_t *flatten_3_output_type;
typedef number_t flatten_3_output_type[OUTPUT_DIM];

#define flatten_3 //noop (IN, OUT)  OUT = (number_t*)IN

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

#define INPUT_SAMPLES 256
#define FC_UNITS 5
#define ACTIVATION_LINEAR

typedef number_t dense_4_output_type[FC_UNITS];

static inline void dense_4(
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

#define INPUT_SAMPLES 256
#define FC_UNITS 5


const int16_t dense_4_bias[FC_UNITS] = {-10, 5, -26, 18, 25}
;

const int16_t dense_4_kernel[FC_UNITS][INPUT_SAMPLES] = {{-25, 65, -77, -77, -17, 61, 42, 3, -57, -64, 39, 31, -9, -24, -22, 45, 0, -61, -29, 29, -23, -18, 66, 12, 29, 64, 41, 64, 72, -77, 61, -18, 59, 38, -23, 45, -13, 34, -56, 9, -27, -4, -38, -61, -39, 28, 0, -22, 50, -41, -77, -61, 52, 78, -46, 61, 48, 80, -61, 37, 66, 29, 64, -25, -62, 8, 4, 19, 30, 3, -20, 39, 33, 59, -6, 35, 61, 13, -15, -30, 74, 57, 60, 35, -69, 67, -11, -54, -64, -56, -14, 3, 13, 63, -39, 72, 22, 74, -50, 35, -19, 46, -77, 45, 62, -43, 46, 37, -47, -17, 67, 56, 47, 64, -54, 22, 34, 10, 32, -10, 0, -66, 8, 74, 20, 28, 64, 21, 20, -69, -45, -56, -15, 37, -29, 0, -7, -39, 59, 52, -11, 12, -12, 25, 61, 64, -14, -23, -30, 30, -41, 40, -60, -51, 69, -65, 68, -38, -10, -16, -5, 27, 34, -34, -11, 50, 78, 7, -10, -53, -55, 2, -49, -46, 88, -14, -73, 48, 7, -39, 37, 34, -21, -14, 0, 33, 49, 7, -81, -77, 22, 14, -11, 19, 55, -42, -7, -20, -64, 52, -46, -15, 39, -8, -2, -68, 15, 10, -47, -57, -33, 28, -27, -8, 20, -38, 51, -11, -45, -24, 56, -57, -58, 50, 54, -53, -1, -47, 53, -8, 62, 70, 29, -69, 23, 10, -53, -65, 38, -24, 4, -84, -56, 42, -25, 37, 20, 70, 13, -37, -34, 44, 1, -37, -29, 47}
, {61, -5, 46, 72, -28, 60, -54, -61, -43, -67, 59, 4, -24, -71, -45, 8, -43, -29, 78, 19, 73, -9, 87, 5, 35, -8, -69, -17, 51, 33, -51, 31, 33, -41, -69, 9, 26, -64, 31, -20, -43, 49, -8, -23, 65, 66, 87, 4, -12, -16, 10, -58, -23, -63, -40, 7, 9, -15, 50, 60, 57, -39, 12, 66, -20, 9, 45, -114, 52, 50, -50, -6, 71, 47, -4, 53, 36, 25, -18, -56, 7, 5, -2, 20, 21, 56, 29, -44, -32, -9, 55, 27, -95, -60, -23, 3, -45, 14, 5, -67, 40, -22, 17, 101, -77, 45, -25, -34, -59, -37, 5, 28, -29, -10, -45, -73, 60, 11, -44, -1, 36, 28, 46, -35, -27, -53, -24, -45, -35, 73, 62, 38, -22, 45, -8, -40, 23, 18, 31, 43, 13, -27, 57, -69, 67, 59, 18, -61, -67, -52, -8, 14, 16, 38, 50, 32, -56, -15, -10, -49, -59, 63, -6, 31, 3, 83, 77, 45, -56, 4, 63, 25, 11, -36, 22, -30, 80, 77, 4, 29, 69, -70, -35, -36, -23, -30, -66, 7, -2, 47, 29, 9, 33, -64, -13, -85, 0, 37, 45, 21, -29, 10, 0, -64, 33, -43, -79, 57, 22, 84, -6, -33, -76, 58, -78, 23, -40, 28, 58, 20, -43, -2, 19, 41, 31, -5, -53, -2, 12, 25, -64, -17, -49, 28, -35, -18, -51, -16, -56, -50, -1, 29, -46, 56, 25, -48, 26, -18, -28, 35, 49, 31, -49, -33, 33, 0}
, {23, -5, 20, 29, 84, -14, 52, 72, -71, 25, -42, -15, -68, -24, -75, -87, -50, -75, 66, 27, 39, 38, 35, -43, 31, 1, -30, -41, 38, -19, -44, -69, -32, 47, -39, 14, 74, 8, 69, -13, 35, -68, -44, -66, -46, -18, -29, 19, -25, -6, -41, -14, 46, -28, 48, 9, -41, 56, 42, -47, 21, 56, 43, -27, 68, 103, 51, 112, 68, 38, -12, -7, 89, 31, 77, 48, -10, -2, 15, 3, 22, 12, -4, 23, 9, 49, 44, -16, -59, -79, -49, -51, 36, 37, -23, -63, -73, -45, 59, 31, -12, -22, -30, -5, 12, 49, 35, 96, 54, 17, 75, 82, -56, -2, 65, 2, 59, 26, -61, -10, 54, 22, 19, -40, -57, 46, 72, 66, 26, 64, 42, -3, -5, -38, -44, -17, -60, -11, -76, -16, 19, -29, 49, -67, -8, 47, 17, 58, 50, 14, 29, 13, 51, 70, 63, -16, 8, 4, -33, 1, -10, -20, -41, 1, 0, -33, 32, 44, -73, -5, -64, -16, 1, 73, -61, -71, -5, 56, 38, -45, 64, -54, 51, 29, -41, 72, -22, -2, 21, -56, -22, -35, -38, 64, -14, 64, 13, 47, 49, -21, -31, -53, 56, -36, -55, -17, -36, -41, 10, -37, 19, -28, 101, 107, -28, 85, 40, 36, -37, -49, -75, -11, -1, 71, 24, -37, 82, -49, 12, 68, 11, 15, -6, -18, -57, -39, 9, 28, -41, 39, -10, -27, 44, 56, 12, 58, -33, -37, -82, -27, -20, -10, 89, 18, 46, 79}
, {-72, -47, -5, 51, 63, 3, 58, -10, 13, 4, -10, -63, 13, -28, -36, 54, 92, -68, -10, -64, -19, 19, 10, -20, -71, 48, -62, 39, 68, 63, -58, -49, -71, 7, 28, 29, 5, -24, 20, -55, 12, -22, -60, 5, -18, 19, -1, -3, -49, -55, -71, 61, -35, -48, -69, -46, -13, 0, -22, 24, -45, -19, 19, 3, -56, 40, -61, -36, -48, 31, -12, 60, -40, -71, -3, 17, -21, -15, -16, 11, -34, 15, 29, -37, -17, -33, 45, 30, 75, 44, 41, 25, -23, -71, 17, 4, -17, 68, 61, -51, -50, -51, -54, 15, 71, 18, 27, -67, -41, 76, 64, 15, 44, 33, 3, 3, 70, 29, 41, 47, -66, 62, 10, 4, -33, -19, -5, -73, 64, -68, 62, -56, 43, 30, 22, 64, 11, -23, 81, -23, 71, 73, -27, 36, 29, 5, 63, -26, 49, -50, 32, -65, 5, -67, 43, 36, 11, 94, 86, 44, 21, -45, -60, -26, -7, -79, -69, -62, 18, 0, 15, 74, 2, -61, -39, -86, -43, 56, 45, -69, 60, 62, 5, -9, 22, -13, -16, -13, -10, -47, 44, 70, -18, 0, -62, -3, -81, -68, -69, -59, -59, -17, -45, 61, -23, -65, -42, -34, -45, -68, -71, -40, -5, 31, -33, -33, -63, 28, 69, 50, -4, 27, 0, 6, -26, 8, 24, 24, 54, -10, -42, -48, 51, -48, -74, 5, -5, 56, -66, 19, -41, 55, 26, -62, 64, -59, -76, 32, 16, 0, -36, 36, 56, 30, -25, 64}
, {34, 7, -69, 15, 8, 10, -56, -78, 11, 65, 39, 44, 8, -67, -45, -9, -20, 23, 74, -14, 12, -97, -8, -8, 7, 51, -50, -24, -70, 44, -30, 71, 70, -45, -48, 39, 50, 56, -33, -53, 31, -2, -60, -35, -43, -88, -69, -27, 29, -8, 0, -54, -30, 19, 10, -34, 47, 73, 5, -5, -12, -52, -12, -51, -68, -95, -7, -58, 47, 21, 4, 23, -60, 0, 29, -54, 7, -57, -30, 38, -39, 27, -63, -52, 67, 37, -42, -20, 21, 25, -13, 40, -14, -30, 7, -41, -45, 30, 85, -66, -69, 24, 21, 43, -29, 30, -64, -10, -32, -44, -77, -54, 20, -22, -83, -80, -21, 41, 68, 77, -58, -10, -65, 13, -71, 27, 47, 22, -41, -76, -51, 4, 35, -50, -22, 27, 27, -66, -78, -71, 14, 13, 80, 38, -62, 5, 36, 19, -2, 32, 57, 56, -39, -37, -54, 61, -29, -75, 1, 30, -62, 35, 26, -53, -93, -59, -56, -45, 16, 3, -24, -54, -77, -90, -68, 21, 40, 71, 32, 51, -95, -50, -24, -43, 19, 19, -74, -60, -12, 69, 50, -48, -10, -53, 34, -111, 39, -32, 8, 49, 35, 6, -48, 11, -3, 34, 38, 77, 15, 52, -88, 57, -45, -34, 0, -42, 0, 14, 17, -70, 32, 55, -12, -18, 18, -80, -47, -31, -32, -61, -48, -73, -43, -45, -15, -37, -1, -36, 42, -12, -13, -38, -103, 23, 35, -74, -45, -14, -7, 51, -62, 60, 6, -74, -84, -7}
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
  //dense_4_output_type dense_4_output);
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
#include "conv1d_12.c"
#include "weights/conv1d_12.c" // InputLayer is excluded
#include "max_pooling1d_9.c" // InputLayer is excluded
#include "conv1d_13.c"
#include "weights/conv1d_13.c" // InputLayer is excluded
#include "max_pooling1d_10.c" // InputLayer is excluded
#include "conv1d_14.c"
#include "weights/conv1d_14.c" // InputLayer is excluded
#include "max_pooling1d_11.c" // InputLayer is excluded
#include "conv1d_15.c"
#include "weights/conv1d_15.c" // InputLayer is excluded
#include "average_pooling1d_3.c" // InputLayer is excluded
#include "flatten_3.c" // InputLayer is excluded
#include "dense_4.c"
#include "weights/dense_4.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_4_output_type dense_4_output) {

  // Output array allocation
  static union {
    conv1d_12_output_type conv1d_12_output;
    conv1d_13_output_type conv1d_13_output;
    conv1d_14_output_type conv1d_14_output;
    conv1d_15_output_type conv1d_15_output;
  } activations1;

  static union {
    max_pooling1d_9_output_type max_pooling1d_9_output;
    max_pooling1d_10_output_type max_pooling1d_10_output;
    max_pooling1d_11_output_type max_pooling1d_11_output;
    average_pooling1d_3_output_type average_pooling1d_3_output;
    flatten_3_output_type flatten_3_output;
  } activations2;


  //static union {
//
//    static input_4_output_type input_4_output;
//
//    static conv1d_12_output_type conv1d_12_output;
//
//    static max_pooling1d_9_output_type max_pooling1d_9_output;
//
//    static conv1d_13_output_type conv1d_13_output;
//
//    static max_pooling1d_10_output_type max_pooling1d_10_output;
//
//    static conv1d_14_output_type conv1d_14_output;
//
//    static max_pooling1d_11_output_type max_pooling1d_11_output;
//
//    static conv1d_15_output_type conv1d_15_output;
//
//    static average_pooling1d_3_output_type average_pooling1d_3_output;
//
//    static flatten_3_output_type flatten_3_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  conv1d_12(
     // First layer uses input passed as model parameter
    input,
    conv1d_12_kernel,
    conv1d_12_bias,
    activations1.conv1d_12_output
  );
 // InputLayer is excluded 
  max_pooling1d_9(
    
    activations1.conv1d_12_output,
    activations2.max_pooling1d_9_output
  );
 // InputLayer is excluded 
  conv1d_13(
    
    activations2.max_pooling1d_9_output,
    conv1d_13_kernel,
    conv1d_13_bias,
    activations1.conv1d_13_output
  );
 // InputLayer is excluded 
  max_pooling1d_10(
    
    activations1.conv1d_13_output,
    activations2.max_pooling1d_10_output
  );
 // InputLayer is excluded 
  conv1d_14(
    
    activations2.max_pooling1d_10_output,
    conv1d_14_kernel,
    conv1d_14_bias,
    activations1.conv1d_14_output
  );
 // InputLayer is excluded 
  max_pooling1d_11(
    
    activations1.conv1d_14_output,
    activations2.max_pooling1d_11_output
  );
 // InputLayer is excluded 
  conv1d_15(
    
    activations2.max_pooling1d_11_output,
    conv1d_15_kernel,
    conv1d_15_bias,
    activations1.conv1d_15_output
  );
 // InputLayer is excluded 
  average_pooling1d_3(
    
    activations1.conv1d_15_output,
    activations2.average_pooling1d_3_output
  );
 // InputLayer is excluded 
  flatten_3(
    
    activations2.average_pooling1d_3_output,
    activations2.flatten_3_output
  );
 // InputLayer is excluded 
  dense_4(
    
    activations2.flatten_3_output,
    dense_4_kernel,
    dense_4_bias, // Last layer uses output passed as model parameter
    dense_4_output
  );

}
