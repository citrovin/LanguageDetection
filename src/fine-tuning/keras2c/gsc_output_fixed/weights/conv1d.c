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


const int16_t conv1d_bias[CONV_FILTERS] = {193, 151, 44, -71, -43, 141, -105, -33}
;

const int16_t conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{1, -177, -30, 81, -92}
}
, {{33, 41, 71, 19, 70}
}
, {{-161, 42, 15, 86, 39}
}
, {{-57, 120, -111, -68, -112}
}
, {{51, 151, -54, -141, -76}
}
, {{-46, 171, 57, -129, 126}
}
, {{84, -2, -28, 128, -174}
}
, {{-62, -32, 22, -84, -85}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE