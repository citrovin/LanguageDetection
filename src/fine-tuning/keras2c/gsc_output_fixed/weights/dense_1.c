/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 32
#define FC_UNITS 4


const int16_t dense_1_bias[FC_UNITS] = {-38, -35, 36, 13}
;

const int16_t dense_1_kernel[FC_UNITS][INPUT_SAMPLES] = {{-33, 81, 108, -67, -40, 9, 4, 10, 21, 108, 106, 189, -7, 112, 17, 65, 76, -38, -77, 129, -66, 28, -76, 118, 39, 72, 17, -60, -43, -151, 22, -111}
, {62, 24, 36, 6, -114, -62, -40, 110, -145, 42, 10, -49, 115, -89, -104, -49, 188, -86, 31, 6, -66, 106, 206, -85, 60, 20, -140, 79, 177, 81, -96, -57}
, {54, -53, -120, -3, -20, -142, 92, 45, 121, 175, -31, 199, -11, 140, 17, -64, -49, 124, 40, -39, 36, 136, -22, -41, -137, 21, 72, -99, 10, -122, -155, 16}
, {-174, 174, -115, -14, -38, 166, 33, 84, 139, -150, -159, 86, -106, -98, 17, -94, 134, 0, 133, -51, -100, 163, 195, 129, 123, -12, -20, -128, -22, -34, -58, -38}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS