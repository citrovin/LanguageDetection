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


const int16_t conv1d_12_bias[CONV_FILTERS] = {-7, -3, -46, 0, -35, 15, 16, 26}
;

const int16_t conv1d_12_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-86, 158, -21, -64, 5}
}
, {{113, 20, -121, -32, -135}
}
, {{76, -135, -60, 101, 61}
}
, {{-185, -138, -17, 42, 159}
}
, {{109, -149, 11, -47, 108}
}
, {{-56, -2, -109, 92, 68}
}
, {{-56, 38, -108, 66, 168}
}
, {{177, -10, 171, 87, -170}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE