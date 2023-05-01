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