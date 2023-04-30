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