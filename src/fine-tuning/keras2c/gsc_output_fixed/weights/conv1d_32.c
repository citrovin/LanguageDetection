/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    8
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  4


const int16_t conv1d_32_bias[CONV_FILTERS] = {5, -171, 14, -6, 30, -296, -155, -19, -102, -38, 57, -17, 72, -183, 51, -48}
;

const int16_t conv1d_32_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-13, -45, 32, -90}
, {19, 194, -36, -117}
, {95, 285, 356, -88}
, {75, 134, 45, -178}
, {35, 128, -12, -195}
, {1, -23, -8, -132}
, {8, 89, -129, -260}
, {-54, -30, -32, -8}
}
, {{65, -36, -91, 172}
, {-135, -50, -101, -41}
, {-167, -72, -187, -91}
, {116, -46, -106, 69}
, {86, 19, 28, 65}
, {24, -139, -54, 50}
, {92, 147, 120, 169}
, {77, 80, 60, 36}
}
, {{59, 93, 0, -181}
, {-17, 89, 5, 168}
, {-179, 19, 79, 98}
, {-199, 52, -56, 72}
, {-55, -67, -42, -43}
, {-118, 23, -181, 201}
, {-118, 30, 25, -29}
, {55, -54, 46, -28}
}
, {{-224, -53, 13, -64}
, {149, 249, 109, 206}
, {167, 390, 184, 144}
, {-52, 59, 8, 81}
, {-11, 101, -8, 71}
, {-22, -77, -96, 9}
, {96, 178, -13, -58}
, {-23, 46, 44, 43}
}
, {{19, 91, -135, -33}
, {-46, 77, -111, 31}
, {-133, -73, -145, 88}
, {-28, 97, 24, -28}
, {-136, -23, -7, 7}
, {-45, -67, -117, 88}
, {-69, 152, 52, -36}
, {-233, 159, -129, 45}
}
, {{-49, 78, 34, 21}
, {-193, -189, -140, -25}
, {-5, -52, 89, 40}
, {-48, 104, 146, 18}
, {-252, -19, 27, -23}
, {110, -41, 19, 42}
, {-38, -65, 49, 60}
, {60, -23, -68, 78}
}
, {{-159, -124, 32, 159}
, {0, -66, -39, 92}
, {-70, -101, -84, 15}
, {150, -59, -80, -213}
, {-2, -83, 89, 66}
, {205, 35, -160, -165}
, {-96, 13, 94, 61}
, {-95, -100, 44, 118}
}
, {{83, -76, -157, 72}
, {246, 53, -130, 68}
, {-32, 39, -324, 14}
, {-110, 147, -133, -152}
, {58, 21, -215, 36}
, {-63, 58, -132, 10}
, {7, -2, -47, -123}
, {-69, 74, -52, 127}
}
, {{171, -153, -63, 90}
, {1, 110, -17, -72}
, {165, 16, 29, -7}
, {-23, -97, 14, 28}
, {-53, -46, -28, -81}
, {-205, 125, 135, -22}
, {-16, -277, -2, -75}
, {118, -86, -130, 42}
}
, {{-118, 99, 43, 6}
, {-132, -228, -17, 123}
, {-3, 99, 97, 275}
, {-149, -17, -71, 37}
, {-52, -85, 63, 6}
, {54, 61, -55, 134}
, {-192, 0, -39, -86}
, {-161, 96, 71, 3}
}
, {{-97, -253, 72, 235}
, {-160, -189, -85, 201}
, {-148, -60, 0, 208}
, {-52, -123, 26, 48}
, {-14, -366, -29, 105}
, {28, -20, -7, 33}
, {-196, -422, 99, 53}
, {-125, -197, 83, 99}
}
, {{-40, -84, 49, -52}
, {219, 28, -49, 123}
, {-129, -128, -130, -43}
, {-82, 37, -53, 107}
, {211, 90, -159, -61}
, {-5, -63, 56, 115}
, {88, -88, 83, 3}
, {97, -18, -180, -7}
}
, {{-97, -46, -181, -113}
, {112, -167, 174, 58}
, {140, 14, 88, -27}
, {136, 120, 5, 130}
, {-100, -60, -250, 30}
, {162, -111, 51, -23}
, {-62, -137, 0, -40}
, {-154, -83, -89, -70}
}
, {{23, -19, -24, 121}
, {66, -230, -101, -15}
, {-112, -146, -216, -68}
, {-10, 2, 30, 120}
, {16, -29, 29, 37}
, {-75, 96, -225, 45}
, {72, 67, 60, -5}
, {-37, -104, -118, 67}
}
, {{43, -108, 51, 3}
, {19, -274, 7, -17}
, {-156, -294, 20, -252}
, {17, -344, 126, 24}
, {-20, -107, 98, 13}
, {-43, -10, 240, -162}
, {1, -54, -95, 63}
, {88, 92, -74, -58}
}
, {{-111, 245, -541, -11}
, {149, 19, 119, -5}
, {47, 5, -51, 44}
, {-47, 68, 79, -49}
, {-124, 64, -347, 61}
, {107, -91, 15, 9}
, {-125, 83, -330, -9}
, {-103, 105, -237, 62}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE