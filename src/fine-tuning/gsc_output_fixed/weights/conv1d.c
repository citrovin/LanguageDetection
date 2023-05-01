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