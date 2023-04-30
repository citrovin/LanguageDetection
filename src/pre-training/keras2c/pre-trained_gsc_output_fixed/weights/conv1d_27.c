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


const int16_t conv1d_27_bias[CONV_FILTERS] = {-5, -2, 1, -5, -3, -6, -7, -3}
;

const int16_t conv1d_27_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-61, 118, -12, -44, 7, 113, 13, 33, 66}
}
, {{95, 19, -93, -30, -106, 45, 85, -90, 97}
}
, {{60, -88, -29, 80, 46, -47, -16, -10, 2}
}
, {{-130, -94, 5, 59, 145, -129, -76, 40, 63}
}
, {{75, -106, 14, -32, 84, 112, 63, 83, -60}
}
, {{-26, 9, -78, 80, 60, -29, -18, -18, -59}
}
, {{-68, 9, -98, 40, 118, 66, 79, -75, 83}
}
, {{131, -2, 133, 60, -138, 111, -87, -89, 31}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE