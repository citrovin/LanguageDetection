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


const int16_t conv1d_bias[CONV_FILTERS] = {-28, 22, 116, 578, 90, 85, -17, -12}
;

const int16_t conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{104, 31, -151, -127, 159}
}
, {{-111, -73, -50, 35, 34}
}
, {{-1, 28, -190, -44, 93}
}
, {{23, 75, 139, -6, 127}
}
, {{-116, -113, -45, 114, 154}
}
, {{-88, 37, 51, 112, -67}
}
, {{-101, 202, -197, 94, -36}
}
, {{-151, 54, -70, 9, -106}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE