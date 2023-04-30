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


const int16_t conv1d_31_bias[CONV_FILTERS] = {580, 202, 1, 181, -14, 270, -17, 45}
;

const int16_t conv1d_31_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-9, 121, 17, -57, 0, 77, 58, 115, 88}
}
, {{110, 98, -148, -53, -114, 80, 120, -169, 130}
}
, {{70, -131, -89, 168, 86, -69, -88, -16, 54}
}
, {{-179, -130, -1, 27, 67, -73, -57, 98, 106}
}
, {{95, -157, 49, -54, 91, 50, 42, 75, -180}
}
, {{-90, -48, -81, 20, 18, -39, -43, -107, -103}
}
, {{-108, 19, -121, 64, 70, 13, 69, -118, 161}
}
, {{187, 26, 137, 59, -97, 111, -62, 14, 120}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE