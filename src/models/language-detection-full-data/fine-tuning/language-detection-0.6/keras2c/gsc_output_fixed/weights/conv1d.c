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