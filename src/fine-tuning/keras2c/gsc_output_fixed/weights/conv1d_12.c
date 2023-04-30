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


const int16_t conv1d_12_bias[CONV_FILTERS] = {-1, 1, -52, 11, -89, -3, -9, 14}
;

const int16_t conv1d_12_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-89, 162, 6, -73, 12}
}
, {{116, 32, -131, -44, -151}
}
, {{85, -123, -34, 111, 64}
}
, {{-212, -119, 4, 46, 168}
}
, {{96, -137, 6, -59, 137}
}
, {{-65, 3, -120, 100, 62}
}
, {{-66, 39, -101, 57, 178}
}
, {{171, -28, 155, 89, -184}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE