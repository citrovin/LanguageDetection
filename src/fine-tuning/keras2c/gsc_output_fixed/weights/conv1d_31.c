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


const int16_t conv1d_31_bias[CONV_FILTERS] = {594, 192, -3, 179, -76, 273, -33, 29}
;

const int16_t conv1d_31_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-70, 162, -12, -12, -28, 111, 74, 130, 126}
}
, {{106, 111, -141, -68, -86, 59, 125, -179, 116}
}
, {{37, -116, -65, 201, 132, -75, -79, -20, 42}
}
, {{-185, -145, -59, 14, 67, -81, -96, 82, 111}
}
, {{64, -160, 40, -57, 69, 34, 11, 76, -178}
}
, {{-52, -58, -102, 18, -5, -50, -27, -129, -93}
}
, {{-140, 24, -112, 70, 56, -3, 87, -125, 165}
}
, {{141, 22, 113, 65, -129, 137, -67, 70, 143}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE