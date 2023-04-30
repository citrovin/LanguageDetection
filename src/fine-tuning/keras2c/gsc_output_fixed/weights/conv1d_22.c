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


const int16_t conv1d_22_bias[CONV_FILTERS] = {-73, -91, -53, 10, -74, -82, 42, 3}
;

const int16_t conv1d_22_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-61, 138, 2, -53, -10}
}
, {{102, 30, -47, 15, -60}
}
, {{43, -78, -41, 65, 34}
}
, {{-84, -34, 26, 11, 109}
}
, {{56, -89, 0, -3, 72}
}
, {{-13, -28, -80, 64, 46}
}
, {{-46, 26, -100, 25, 109}
}
, {{81, -61, 95, 31, -162}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE