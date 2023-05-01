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