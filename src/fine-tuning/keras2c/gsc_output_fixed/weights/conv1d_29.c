/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_29_bias[CONV_FILTERS] = {-13, -16, -8, -5, -3, -6, -7, -6, -15, -18, -8, -8, -12, 0, -5, -15, -3, -7, -12, -11, -12, -7, -3, -10, -8, -8, 6, -11, -22, -17, -2, -17}
;

const int16_t conv1d_29_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-56, 59, -74}
, {-2, 80, -104}
, {41, -72, 49}
, {28, -14, -85}
, {-73, -109, -44}
, {17, -62, 49}
, {46, -4, 87}
, {46, -22, -49}
, {49, -74, -82}
, {76, -69, 30}
, {84, 82, -28}
, {-24, 26, 15}
, {-8, -101, 13}
, {14, -60, -82}
, {-63, -108, -25}
, {-99, -95, 38}
}
, {{56, -102, 69}
, {-93, 60, -17}
, {54, -54, 21}
, {-15, -93, 34}
, {-106, 35, 46}
, {-90, -101, -13}
, {9, -82, -82}
, {61, 14, -96}
, {-93, 18, -21}
, {-31, 75, -102}
, {-45, 53, -12}
, {66, -42, 68}
, {89, -77, -11}
, {-39, -87, 19}
, {50, 73, -6}
, {65, 60, 44}
}
, {{38, -54, 33}
, {26, -105, -12}
, {0, -98, 42}
, {-60, -98, 1}
, {78, 35, 87}
, {33, -53, -50}
, {-54, -40, -62}
, {95, -28, 19}
, {-27, 62, -53}
, {35, 71, 51}
, {-3, -68, 37}
, {90, -81, -89}
, {-76, -49, -98}
, {-14, 38, -6}
, {-53, 17, -28}
, {72, -5, 37}
}
, {{-111, -60, -19}
, {90, -83, 57}
, {31, 46, 10}
, {46, 30, -87}
, {-69, -14, -39}
, {78, 80, 20}
, {76, 39, 90}
, {74, -14, -31}
, {-4, 1, -34}
, {-69, 71, 98}
, {-49, 6, -76}
, {-108, 64, 9}
, {-59, 52, -31}
, {-56, -41, -21}
, {83, 80, -69}
, {-80, -6, -57}
}
, {{50, 10, -60}
, {65, 53, 21}
, {-45, -75, 90}
, {82, -48, 2}
, {-22, -96, -30}
, {-34, -79, -79}
, {-49, -93, 53}
, {17, 68, -78}
, {-74, 19, -38}
, {-21, -74, -93}
, {5, 18, 25}
, {44, 9, 34}
, {-53, -53, 59}
, {48, -107, 82}
, {24, 25, -43}
, {-88, 32, -61}
}
, {{-31, 9, -57}
, {34, -56, 39}
, {-50, 80, -106}
, {20, -43, 13}
, {97, 46, 0}
, {-14, -69, -6}
, {36, 16, -94}
, {-78, 10, 23}
, {6, 34, -21}
, {76, -29, 10}
, {99, -60, -44}
, {-33, -14, 10}
, {28, -73, 70}
, {0, -93, -82}
, {32, 9, -62}
, {48, -105, -17}
}
, {{-50, 89, 38}
, {94, 50, 4}
, {67, -66, 7}
, {-3, -60, -60}
, {-17, -91, 9}
, {-41, -76, 25}
, {-51, -86, -63}
, {16, -60, -54}
, {16, -82, -35}
, {-45, -12, 69}
, {79, 105, -33}
, {76, 77, 22}
, {100, 13, 91}
, {14, -34, -90}
, {72, -100, 62}
, {73, -73, -53}
}
, {{98, -113, 98}
, {-108, -62, -82}
, {12, -75, 80}
, {-56, -107, -33}
, {-80, 68, 74}
, {-19, -54, -9}
, {-13, 68, -37}
, {-40, 50, -45}
, {-78, 84, 48}
, {-82, -63, 42}
, {-26, 0, -67}
, {49, -100, 59}
, {-23, -64, -82}
, {21, 40, 9}
, {-7, 86, 35}
, {-51, 91, -10}
}
, {{81, 19, -94}
, {74, 71, 0}
, {-85, 77, 31}
, {-5, -20, 60}
, {-93, 81, -62}
, {36, -72, -42}
, {-20, 35, 91}
, {-84, -59, -62}
, {46, -9, -54}
, {-67, 27, -19}
, {-17, 43, -31}
, {41, -79, 83}
, {64, -96, -116}
, {51, -79, -15}
, {-61, -18, -85}
, {-1, 45, 37}
}
, {{3, 85, 41}
, {16, -31, -74}
, {-17, 27, 80}
, {-97, 31, 31}
, {64, -78, 37}
, {34, -114, -30}
, {-103, -32, -43}
, {36, -2, 32}
, {-39, 86, -97}
, {4, -51, 42}
, {25, -9, -26}
, {-76, -43, 62}
, {-1, 28, 89}
, {71, -75, -28}
, {-11, -101, -37}
, {-84, -17, 6}
}
, {{-81, -42, -15}
, {-44, 51, -67}
, {37, 98, 4}
, {24, 77, 57}
, {-67, 63, -4}
, {-17, -47, -88}
, {-27, -4, 73}
, {33, 43, 27}
, {-44, 81, -92}
, {-17, 81, 1}
, {-83, -81, 62}
, {55, -92, 29}
, {90, -95, 3}
, {72, -100, -28}
, {105, 45, -117}
, {-80, 7, -38}
}
, {{-86, -69, -49}
, {-107, -94, -26}
, {92, 58, -65}
, {29, -64, 37}
, {-39, 78, -33}
, {-8, 54, 25}
, {31, 79, 63}
, {-85, 76, -19}
, {95, -104, 87}
, {-27, -2, -10}
, {-102, -96, -11}
, {9, 31, -99}
, {26, -101, 25}
, {80, 43, -74}
, {-44, -11, 63}
, {-7, 55, 29}
}
, {{-83, 43, 91}
, {75, 46, 24}
, {80, 74, 16}
, {-42, -111, 21}
, {40, -74, -85}
, {-77, -74, -6}
, {-89, 0, -58}
, {-5, 59, 68}
, {-24, -111, 65}
, {13, -79, -113}
, {34, 38, 69}
, {-69, 26, 56}
, {-63, -20, 41}
, {-74, -81, 25}
, {85, -24, -83}
, {21, 19, 76}
}
, {{10, 57, -62}
, {-29, -68, -34}
, {1, 29, -80}
, {74, 27, -114}
, {-13, -14, -38}
, {8, -77, -52}
, {-57, 57, 17}
, {16, -82, 75}
, {10, -19, 22}
, {5, 87, -83}
, {27, 99, -95}
, {52, -42, 26}
, {86, 59, 10}
, {34, -72, 91}
, {4, -73, -79}
, {-32, 25, 41}
}
, {{12, 108, 9}
, {52, -81, 7}
, {-65, -15, -30}
, {-64, -75, -46}
, {-107, 33, 31}
, {75, 49, 94}
, {-36, 34, -11}
, {-40, -28, 22}
, {64, 58, 62}
, {-37, -83, -15}
, {79, 74, 19}
, {-94, 44, 79}
, {-22, -53, 92}
, {-71, 31, -15}
, {-43, 48, 35}
, {-67, 14, 6}
}
, {{-12, -47, -72}
, {79, 75, -20}
, {-32, 16, 8}
, {-96, 71, 25}
, {70, -8, 72}
, {-14, 64, -89}
, {-1, -14, 23}
, {-57, 15, -118}
, {-73, 27, 21}
, {-99, -37, 41}
, {-19, 34, 80}
, {-15, 53, 24}
, {-85, 64, 56}
, {-48, 25, -54}
, {63, 94, -4}
, {13, -82, -69}
}
, {{-5, -80, 29}
, {12, 0, -32}
, {-72, 46, -93}
, {-78, -40, -87}
, {59, -67, -66}
, {26, 41, -105}
, {16, 3, -92}
, {-36, -46, 74}
, {51, -89, 19}
, {16, 97, 66}
, {-92, -80, -81}
, {-36, 42, -8}
, {31, -63, -34}
, {-94, -103, -6}
, {-6, 12, 103}
, {-28, -75, 44}
}
, {{-72, -78, -87}
, {54, -93, 49}
, {-21, 27, -2}
, {86, 24, -108}
, {52, -82, -36}
, {21, -59, -99}
, {-67, 86, 63}
, {35, 58, 54}
, {-102, 45, 9}
, {-78, 62, 37}
, {-9, -32, 3}
, {69, -12, 91}
, {41, 81, -80}
, {34, -13, 69}
, {24, -98, -17}
, {-45, -78, -61}
}
, {{-40, -58, 72}
, {-25, 74, 26}
, {28, 18, -15}
, {22, -86, 34}
, {81, -106, -4}
, {82, -53, 75}
, {-3, 19, -113}
, {-66, -34, -39}
, {-6, -80, -67}
, {13, -68, -39}
, {-107, -82, 21}
, {34, -55, -57}
, {70, 18, 34}
, {32, 22, 11}
, {14, 9, -101}
, {70, -60, 43}
}
, {{-19, 0, -15}
, {-81, 12, 90}
, {-62, -81, -25}
, {-2, 76, 29}
, {72, -71, -3}
, {-35, -40, -20}
, {-105, -68, 61}
, {-42, -76, -50}
, {86, -86, -84}
, {33, 85, -105}
, {-115, 91, 37}
, {55, -83, -55}
, {60, -100, -91}
, {77, -59, -39}
, {52, -12, 23}
, {-17, -63, -95}
}
, {{3, 4, -43}
, {38, -43, 40}
, {-28, -14, 43}
, {-111, -75, -59}
, {75, -35, 7}
, {64, -48, -79}
, {-109, 49, 68}
, {56, -2, 57}
, {-42, 45, 72}
, {60, -43, -92}
, {-94, -71, -41}
, {-89, 95, 75}
, {-64, 28, -54}
, {-75, -96, -39}
, {25, 58, 50}
, {85, -62, -65}
}
, {{-54, -71, 27}
, {-22, -25, -108}
, {49, -3, 80}
, {51, -93, -100}
, {-69, 86, 76}
, {75, -98, 74}
, {-25, -28, 89}
, {-15, -49, -18}
, {-104, 73, 85}
, {72, 15, -42}
, {-36, -12, -37}
, {-116, -97, -21}
, {2, 2, -21}
, {19, -29, 11}
, {40, 41, -55}
, {60, -18, -40}
}
, {{-69, 25, -23}
, {66, 40, -15}
, {82, 21, 97}
, {-26, -52, 52}
, {-1, 81, -66}
, {-48, -87, 11}
, {57, -86, -29}
, {-19, 11, 72}
, {-51, -5, 49}
, {20, 7, -22}
, {-98, -28, 36}
, {27, 32, -27}
, {48, -26, -99}
, {-88, -27, 80}
, {-87, 16, 35}
, {37, -74, -14}
}
, {{96, -93, -17}
, {-73, 54, -76}
, {82, 47, -49}
, {-56, 0, -23}
, {54, -27, -73}
, {66, -17, -58}
, {-107, -15, -35}
, {-2, -3, -17}
, {6, 62, 56}
, {-97, 85, -93}
, {-100, 81, 31}
, {21, 57, -11}
, {52, -32, -95}
, {-76, 32, 25}
, {36, -108, -113}
, {4, -34, 46}
}
, {{-37, 1, 96}
, {19, -99, 44}
, {-89, -66, -68}
, {32, -20, 43}
, {64, 95, -6}
, {-66, 72, -90}
, {91, 8, -75}
, {-117, -85, -30}
, {42, 18, -75}
, {-99, 61, -30}
, {20, 48, 59}
, {11, -87, -14}
, {62, 17, -26}
, {93, -38, -53}
, {24, -47, -55}
, {8, -74, -109}
}
, {{-36, 90, 64}
, {-75, -50, -103}
, {86, 5, 4}
, {-67, -16, 58}
, {-94, 78, -99}
, {-49, -21, 74}
, {-8, -1, 19}
, {48, 11, 67}
, {37, 8, -51}
, {14, -54, 43}
, {-18, -102, 2}
, {-3, 13, 50}
, {39, -60, -83}
, {15, -59, 30}
, {83, 11, 26}
, {22, 17, -21}
}
, {{65, -96, 60}
, {-9, -27, 35}
, {43, -81, 18}
, {-90, 101, 26}
, {-97, 102, -95}
, {-94, -1, -7}
, {-22, -4, -108}
, {-80, 85, 79}
, {17, -80, 56}
, {15, -8, 45}
, {-39, -5, -62}
, {44, 58, -62}
, {83, 57, 35}
, {-69, -82, -94}
, {-15, -19, 101}
, {-78, 22, 38}
}
, {{24, -107, 71}
, {17, 5, -64}
, {92, -23, 17}
, {-27, -91, -6}
, {23, 30, 57}
, {58, -71, -83}
, {-39, -20, 35}
, {-40, 87, 39}
, {-27, 65, -78}
, {-56, 26, 30}
, {44, -59, 53}
, {-75, -7, -113}
, {42, 22, -44}
, {89, -68, 77}
, {34, -66, 32}
, {5, 9, -10}
}
, {{-32, 43, 17}
, {46, -94, -117}
, {85, -72, -72}
, {-101, -38, -92}
, {-53, 34, -50}
, {45, -124, -85}
, {-114, 72, -33}
, {-47, 95, 29}
, {51, -59, 23}
, {-51, 28, 13}
, {14, 0, 68}
, {-72, 45, -72}
, {16, -35, -38}
, {13, 60, -63}
, {-91, -53, -73}
, {-68, -77, 4}
}
, {{46, 66, 15}
, {-31, -23, 61}
, {-77, -32, 74}
, {-29, -21, 55}
, {-99, -24, 17}
, {-13, -10, 79}
, {87, -20, 82}
, {-90, -92, 63}
, {38, 76, -76}
, {67, 59, -31}
, {-121, -18, -102}
, {-49, -71, -46}
, {-107, 63, 27}
, {59, 14, -108}
, {55, -13, -63}
, {33, -113, 43}
}
, {{37, 80, -54}
, {-48, 13, 73}
, {-49, 29, -97}
, {18, 70, -96}
, {-38, 58, -76}
, {-81, -66, -2}
, {4, 47, -26}
, {54, 3, 88}
, {6, 29, -77}
, {82, 102, -1}
, {86, 62, 32}
, {-18, -77, -55}
, {21, -91, 15}
, {-67, -31, -67}
, {-16, 77, -100}
, {-73, -28, 75}
}
, {{39, 74, -25}
, {-74, -86, -14}
, {52, 76, 12}
, {-9, -86, 18}
, {-113, -119, -102}
, {-7, -108, 5}
, {-72, -60, -18}
, {-23, -57, 22}
, {-12, 59, 63}
, {46, 45, -93}
, {-43, 57, -24}
, {-82, 2, 83}
, {3, -9, -65}
, {-50, -98, -96}
, {27, -30, -31}
, {42, 90, 31}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE