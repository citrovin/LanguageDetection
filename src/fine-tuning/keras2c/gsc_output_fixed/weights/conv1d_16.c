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


const int16_t conv1d_16_bias[CONV_FILTERS] = {28, 11, -24, -18, -52, 14, -5, 15}
;

const int16_t conv1d_16_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-81, 168, -21, -58, 16}
}
, {{125, 20, -120, -40, -143}
}
, {{77, -112, -56, 101, 36}
}
, {{-178, -125, 0, 74, 161}
}
, {{87, -136, 2, -33, 93}
}
, {{-56, -20, -121, 96, 72}
}
, {{-83, 19, -115, 54, 151}
}
, {{173, -24, 171, 56, -186}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE