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


const int16_t conv1d_27_bias[CONV_FILTERS] = {-21, -8, -2, -16, -8, 0, -28, -24}
;

const int16_t conv1d_27_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-49, 98, -8, -59, 6, 100, 5, 19, 55}
}
, {{89, 35, -78, -18, -94, 54, 86, -78, 94}
}
, {{53, -80, -35, 86, 35, -40, -12, -2, -3}
}
, {{-112, -71, 11, 79, 140, -112, -77, 51, 61}
}
, {{71, -107, 14, -33, 72, 100, 50, 72, -68}
}
, {{-29, 3, -75, 71, 53, -42, -15, -27, -68}
}
, {{-64, 7, -102, 36, 94, 61, 52, -75, 57}
}
, {{109, -20, 98, 38, -139, 99, -77, -98, 34}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE