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


const int16_t conv1d_bias[CONV_FILTERS] = {132, -142, -53, -9, -213, -11, -19, 380}
;

const int16_t conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-35, 17, 18, 1, 69, -75, 58, -106, 38, -112, 51, 2, 1, 17, -37, 65, -65, 40, -53, 6}
}
, {{-103, -16, 3, 12, -25, 16, -82, -10, -103, -15, -75, 14, -31, 36, 0, 74, 17, 45, 96, 176}
}
, {{-70, -88, -55, -34, -78, -50, 34, -116, -19, -4, -57, 12, -79, -71, 26, -11, 7, -20, 25, -16}
}
, {{17, 72, 63, 58, 58, 4, 47, -68, 116, -20, 70, -14, 62, 50, 8, 53, 33, 79, -10, 156}
}
, {{47, 78, -10, 46, -21, 25, 29, -43, 24, -66, 4, -87, 5, -105, -51, -41, -78, 64, -93, 3}
}
, {{81, -98, -68, 126, -82, -95, 66, 137, 72, -171, 2, 35, 5, -44, 115, -31, 23, -115, -91, 142}
}
, {{-64, -69, -32, -68, -41, -1, -17, 34, 25, -27, -17, 11, -2, 10, 0, -59, -91, -67, -93, -80}
}
, {{69, 127, 19, -115, -92, 12, 45, 97, 44, -21, 4, -13, -39, 65, 135, 110, -5, -3, -55, -86}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE