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


const int16_t conv1d_4_bias[CONV_FILTERS] = {-31, -139, -74, 83, -77, -37, 191, 333}
;

const int16_t conv1d_4_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-93, 203, -78, -64, 37}
}
, {{-80, -1, -101, 9, -168}
}
, {{121, -122, -141, 105, 64}
}
, {{-144, -133, 30, 120, 101}
}
, {{131, -141, 31, -89, 84}
}
, {{-76, -1, -131, 100, 59}
}
, {{-53, -7, -105, 30, 180}
}
, {{208, 1, 95, 12, -119}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE