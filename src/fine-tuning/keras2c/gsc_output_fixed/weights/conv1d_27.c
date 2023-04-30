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


const int16_t conv1d_27_bias[CONV_FILTERS] = {-12, -5, 0, -8, -2, -1, -10, -7}
;

const int16_t conv1d_27_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-57, 110, -12, -50, 1, 108, 19, 28, 58}
}
, {{91, 24, -88, -26, -104, 48, 83, -84, 95}
}
, {{56, -84, -27, 86, 34, -45, -19, -8, 0}
}
, {{-120, -82, 9, 69, 147, -120, -81, 49, 60}
}
, {{72, -109, 15, -35, 84, 110, 65, 80, -57}
}
, {{-28, 9, -78, 81, 61, -30, -11, -18, -57}
}
, {{-69, 10, -103, 40, 117, 66, 70, -76, 76}
}
, {{122, -8, 120, 52, -142, 104, -89, -93, 26}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE