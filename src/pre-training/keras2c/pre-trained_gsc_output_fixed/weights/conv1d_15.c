/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    16
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  2


const int16_t conv1d_15_bias[CONV_FILTERS] = {3, -29, 5, 0, 25, 0, 5, -6, -14, -25, 5, 20, -12, -15, 7, -27, -29, -18, -21, -9, -3, -11, 29, 0, -3, 3, -26, -30, 2, 24, -9, -15, 2, 22, 11, 30, -2, -9, 5, 24, -10, -7, 25, -12, 20, -17, -17, 17, -10, -7, -6, 15, 1, -38, 3, 4, 8, -31, -9, -3, -9, 1, 8, -1}
;

const int16_t conv1d_15_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-23, -44}
, {55, 72}
, {-31, -7}
, {66, 110}
, {42, -71}
, {78, -11}
, {-2, 8}
, {-34, 13}
, {92, -27}
, {-59, -23}
, {-94, 72}
, {18, 30}
, {-59, 85}
, {62, 58}
, {-91, -20}
, {-98, 63}
}
, {{57, 79}
, {54, 22}
, {-101, 44}
, {-6, -96}
, {-81, 9}
, {-20, -9}
, {89, 13}
, {2, -11}
, {-84, 82}
, {-42, -47}
, {30, -73}
, {-85, 54}
, {24, -68}
, {50, -70}
, {-73, -82}
, {65, 10}
}
, {{33, 24}
, {1, 40}
, {77, 79}
, {-48, -55}
, {-34, -50}
, {-7, 30}
, {-61, -97}
, {-48, -17}
, {-49, 81}
, {-95, 34}
, {35, 76}
, {-34, -18}
, {57, -4}
, {-61, 99}
, {-36, 41}
, {27, 63}
}
, {{-85, -23}
, {58, 30}
, {-88, -59}
, {79, 83}
, {-5, -41}
, {-21, -65}
, {-54, -37}
, {69, -91}
, {-53, 3}
, {33, -19}
, {-9, -61}
, {36, -115}
, {10, 22}
, {-17, -104}
, {37, 60}
, {73, 4}
}
, {{57, -39}
, {-20, 116}
, {-4, -3}
, {-32, 76}
, {-63, -39}
, {19, 34}
, {-44, 68}
, {45, 2}
, {24, 47}
, {-34, 6}
, {-70, 94}
, {-63, -45}
, {17, 103}
, {49, -49}
, {-20, 0}
, {44, -67}
}
, {{-4, -51}
, {-26, -95}
, {100, 8}
, {50, -72}
, {26, -11}
, {76, -47}
, {54, 69}
, {67, -33}
, {28, 74}
, {53, -29}
, {56, 95}
, {14, 55}
, {47, -91}
, {-57, 7}
, {-72, -35}
, {23, -3}
}
, {{-41, 34}
, {60, 9}
, {-15, 9}
, {-44, -53}
, {16, -42}
, {94, -35}
, {92, 85}
, {61, 59}
, {85, -65}
, {-70, 3}
, {-101, -28}
, {-69, -95}
, {-80, -48}
, {79, 11}
, {18, -44}
, {-97, -57}
}
, {{85, 83}
, {26, 80}
, {-70, 65}
, {-18, -39}
, {-68, 38}
, {-57, -69}
, {-16, -96}
, {-12, 29}
, {-99, 12}
, {-71, 48}
, {54, 81}
, {85, 67}
, {80, 12}
, {-3, -44}
, {-75, -65}
, {82, -43}
}
, {{85, -81}
, {-72, 37}
, {-89, -55}
, {-27, 69}
, {74, -11}
, {-6, -12}
, {67, -87}
, {-59, -73}
, {18, -110}
, {73, 91}
, {88, -63}
, {35, -85}
, {-2, 53}
, {41, 35}
, {-97, -53}
, {-13, -76}
}
, {{23, 37}
, {-15, 89}
, {63, 26}
, {-91, -56}
, {-37, -87}
, {25, -46}
, {16, 90}
, {6, -65}
, {89, 50}
, {27, -2}
, {-73, 33}
, {-27, -101}
, {81, -85}
, {-10, -46}
, {37, 78}
, {-84, 13}
}
, {{-60, -18}
, {37, 8}
, {-49, -1}
, {-14, 90}
, {-36, -98}
, {-88, 41}
, {72, -3}
, {105, -94}
, {-35, -9}
, {100, 48}
, {63, -19}
, {14, 16}
, {75, -52}
, {-59, -29}
, {-87, 22}
, {37, -72}
}
, {{-63, -15}
, {90, -35}
, {22, 18}
, {68, 97}
, {74, 75}
, {-69, 0}
, {23, 38}
, {-7, 71}
, {-32, 63}
, {47, -66}
, {96, -9}
, {91, 35}
, {-96, 57}
, {-107, -33}
, {-88, -87}
, {-8, -1}
}
, {{-72, 88}
, {83, 22}
, {46, -70}
, {-80, -56}
, {-17, 46}
, {30, 58}
, {-57, 31}
, {93, -96}
, {48, 22}
, {77, -18}
, {-69, 15}
, {9, -80}
, {-98, -59}
, {43, 58}
, {-11, -65}
, {-31, -42}
}
, {{6, -59}
, {9, -56}
, {-30, -35}
, {-66, 2}
, {-4, -2}
, {26, -74}
, {80, 16}
, {-19, -109}
, {44, 28}
, {5, -98}
, {-29, -5}
, {54, -27}
, {-32, 51}
, {62, 39}
, {60, -3}
, {-95, -114}
}
, {{20, -1}
, {-51, -15}
, {-83, 40}
, {-25, 0}
, {58, 50}
, {72, 17}
, {-18, 66}
, {-33, 5}
, {93, -82}
, {-30, 93}
, {27, 7}
, {36, 26}
, {55, -20}
, {56, 68}
, {-59, 2}
, {37, 4}
}
, {{-7, -68}
, {-24, 5}
, {84, 87}
, {-2, 23}
, {-60, 34}
, {-14, 80}
, {-80, 45}
, {84, 16}
, {-41, 53}
, {-6, -93}
, {-6, -59}
, {14, 67}
, {22, -1}
, {40, 84}
, {35, -78}
, {72, -1}
}
, {{-8, 27}
, {-69, -66}
, {36, -67}
, {-14, -111}
, {73, 43}
, {-98, -79}
, {59, 4}
, {-12, 101}
, {-95, -98}
, {59, 8}
, {-54, 58}
, {14, -53}
, {-78, -24}
, {-100, -104}
, {-35, 9}
, {32, 51}
}
, {{-76, -92}
, {-22, -1}
, {54, -33}
, {-68, 50}
, {-108, 9}
, {1, 7}
, {30, -85}
, {13, -10}
, {-79, 58}
, {19, -102}
, {-88, -54}
, {58, 69}
, {34, -34}
, {-42, 7}
, {70, -89}
, {-103, -36}
}
, {{-24, 90}
, {48, 6}
, {83, 1}
, {-11, -102}
, {3, -35}
, {-78, 25}
, {68, 48}
, {40, -77}
, {-46, 81}
, {12, 42}
, {-98, -19}
, {64, 12}
, {-78, -23}
, {-88, 91}
, {0, -36}
, {0, 64}
}
, {{-1, 1}
, {-39, -20}
, {97, 5}
, {-90, 62}
, {99, -68}
, {-97, 61}
, {68, -62}
, {62, 61}
, {11, -9}
, {-69, -44}
, {-58, 42}
, {-49, 50}
, {-70, -64}
, {76, 52}
, {-89, 82}
, {-3, 6}
}
, {{9, -33}
, {0, 44}
, {77, 15}
, {-77, 76}
, {-3, 97}
, {-40, 20}
, {-38, -38}
, {47, 77}
, {12, -26}
, {-9, -9}
, {-31, -27}
, {90, -19}
, {56, 7}
, {-60, 2}
, {26, -97}
, {71, 44}
}
, {{-43, 36}
, {36, 76}
, {-53, 80}
, {-16, 86}
, {-91, 90}
, {-50, -32}
, {-4, 2}
, {52, -30}
, {-65, 27}
, {3, 34}
, {80, 59}
, {-34, -23}
, {72, 57}
, {-6, -45}
, {14, 50}
, {30, -4}
}
, {{-58, -15}
, {74, 96}
, {6, -59}
, {78, -21}
, {-54, 35}
, {-79, 59}
, {54, -78}
, {-102, 0}
, {34, 88}
, {48, -37}
, {92, 47}
, {-87, 16}
, {5, -103}
, {-9, -18}
, {8, -43}
, {23, -17}
}
, {{95, -11}
, {85, -34}
, {49, -65}
, {-76, -21}
, {43, 85}
, {-85, 45}
, {66, -69}
, {52, -79}
, {-81, -21}
, {42, -92}
, {-22, 27}
, {-2, -65}
, {75, 41}
, {80, 72}
, {-14, 64}
, {-58, 88}
}
, {{-40, 90}
, {-96, -64}
, {49, -9}
, {73, -61}
, {48, -48}
, {27, 54}
, {45, -22}
, {-17, -83}
, {-7, -9}
, {-61, -2}
, {65, -2}
, {-4, -53}
, {21, 61}
, {34, 91}
, {22, 8}
, {-18, -54}
}
, {{-21, 63}
, {91, 17}
, {-64, -86}
, {11, 28}
, {46, -27}
, {-8, -2}
, {42, -86}
, {113, 84}
, {90, 17}
, {-5, 66}
, {75, 1}
, {20, 124}
, {7, -8}
, {-98, 15}
, {-68, -11}
, {27, 32}
}
, {{40, 69}
, {45, 10}
, {-105, -102}
, {-55, -113}
, {27, 55}
, {-47, -67}
, {89, 56}
, {-20, 66}
, {-91, -5}
, {-71, 71}
, {63, 0}
, {8, 15}
, {-80, -25}
, {-17, -4}
, {56, -11}
, {-27, 72}
}
, {{34, 89}
, {89, 18}
, {13, 51}
, {-48, 19}
, {-29, -62}
, {51, 63}
, {58, -10}
, {-5, 14}
, {-89, -7}
, {-15, -3}
, {31, -89}
, {-2, 12}
, {65, 63}
, {-61, -105}
, {27, -3}
, {-51, -3}
}
, {{-24, 15}
, {90, -67}
, {-37, -41}
, {-97, -24}
, {50, 25}
, {19, 69}
, {14, -33}
, {-80, -61}
, {57, -87}
, {-60, -22}
, {39, 88}
, {91, 60}
, {-41, -65}
, {7, -74}
, {-35, 12}
, {-35, 98}
}
, {{38, 21}
, {-59, 72}
, {-87, 13}
, {91, 76}
, {44, -75}
, {-91, -92}
, {-102, 33}
, {18, -101}
, {71, 86}
, {-15, 59}
, {-19, 42}
, {-25, -125}
, {79, -51}
, {-22, -23}
, {77, 44}
, {-1, 5}
}
, {{27, -60}
, {-49, -84}
, {-37, -68}
, {5, -24}
, {0, -74}
, {92, 24}
, {33, 10}
, {-1, -69}
, {75, -52}
, {5, 39}
, {40, -84}
, {50, -47}
, {21, 72}
, {39, 30}
, {-106, -53}
, {64, 44}
}
, {{38, -6}
, {56, 30}
, {-106, -88}
, {-78, -14}
, {4, 76}
, {-42, -22}
, {8, -45}
, {42, -12}
, {73, 12}
, {90, -68}
, {-107, -13}
, {-41, 50}
, {70, -83}
, {55, -58}
, {9, -48}
, {-25, -22}
}
, {{1, -100}
, {52, -81}
, {28, 51}
, {54, -37}
, {72, 23}
, {-25, 12}
, {27, -87}
, {-45, 84}
, {90, -16}
, {-10, 45}
, {-39, -61}
, {-30, -7}
, {-53, 54}
, {48, -80}
, {-44, 46}
, {-80, -16}
}
, {{-84, -9}
, {-19, 31}
, {-66, 1}
, {80, -77}
, {-19, -103}
, {73, 72}
, {-33, 29}
, {31, 11}
, {61, 73}
, {-62, -11}
, {-81, 72}
, {-2, 37}
, {89, -6}
, {-41, -52}
, {-41, 95}
, {72, -28}
}
, {{42, 8}
, {-34, -12}
, {58, -16}
, {112, 33}
, {44, 44}
, {89, -59}
, {-14, -8}
, {78, 28}
, {-83, -24}
, {-95, -72}
, {-28, 10}
, {-4, 78}
, {52, -79}
, {-85, -26}
, {19, -54}
, {-7, 45}
}
, {{93, 45}
, {41, -62}
, {71, 14}
, {78, -24}
, {-88, 63}
, {-94, 6}
, {-46, -24}
, {-115, -93}
, {-76, -11}
, {48, -4}
, {75, -61}
, {-49, -105}
, {80, 88}
, {57, 69}
, {-9, 70}
, {-7, 3}
}
, {{48, 33}
, {82, -3}
, {-31, -70}
, {6, -70}
, {-27, -69}
, {49, 47}
, {44, 91}
, {-74, -55}
, {47, -81}
, {-38, 8}
, {-78, 0}
, {76, -90}
, {-76, -62}
, {7, 68}
, {-99, 77}
, {30, -34}
}
, {{28, 38}
, {8, 5}
, {-13, -1}
, {-86, 7}
, {56, 0}
, {-40, 2}
, {-1, -78}
, {27, -34}
, {-53, 3}
, {-50, -86}
, {-69, 44}
, {-11, -67}
, {-40, 77}
, {-22, 82}
, {-93, -12}
, {-97, -80}
}
, {{94, 6}
, {6, -48}
, {-29, 25}
, {16, -38}
, {-37, 54}
, {80, 28}
, {21, -78}
, {93, -38}
, {52, 46}
, {-50, -90}
, {-64, -49}
, {-31, -60}
, {0, 65}
, {82, -48}
, {-35, -1}
, {-70, -40}
}
, {{-76, -40}
, {-55, -45}
, {-23, -12}
, {-33, -21}
, {-59, 30}
, {66, 69}
, {50, 27}
, {-52, -69}
, {-53, 25}
, {-73, 83}
, {-26, -9}
, {44, 21}
, {-31, 77}
, {-104, -112}
, {70, 76}
, {103, 96}
}
, {{71, 7}
, {11, 80}
, {52, -22}
, {-67, -47}
, {-56, -20}
, {48, 89}
, {45, -16}
, {15, 76}
, {74, -81}
, {-11, 53}
, {-61, 19}
, {-14, 94}
, {28, -2}
, {-64, 57}
, {-79, 22}
, {44, -54}
}
, {{30, -70}
, {-79, 36}
, {88, 19}
, {47, 43}
, {43, 52}
, {-46, 70}
, {94, -22}
, {-14, 60}
, {6, -76}
, {36, -74}
, {-62, 64}
, {32, 76}
, {-31, -85}
, {-14, 62}
, {-75, 0}
, {10, -5}
}
, {{-42, -63}
, {24, 57}
, {-4, -70}
, {43, 60}
, {-25, -26}
, {49, 26}
, {54, -26}
, {-76, -54}
, {42, -85}
, {75, 63}
, {-52, -10}
, {-2, -22}
, {73, 16}
, {-96, -64}
, {-81, 10}
, {15, -41}
}
, {{-89, -23}
, {26, 33}
, {0, 29}
, {-103, -38}
, {-11, -1}
, {4, -80}
, {82, -64}
, {3, 15}
, {-95, -89}
, {-58, 55}
, {48, -9}
, {52, -12}
, {0, -108}
, {33, -35}
, {29, 66}
, {48, -33}
}
, {{96, 45}
, {-13, 48}
, {-41, 24}
, {16, 75}
, {0, -113}
, {-54, 59}
, {-58, 18}
, {49, 118}
, {71, 66}
, {-81, 67}
, {-19, 59}
, {55, 92}
, {-68, 86}
, {28, 0}
, {-69, 39}
, {32, 47}
}
, {{-30, -38}
, {63, -93}
, {-12, -73}
, {19, 56}
, {-8, -83}
, {64, 30}
, {31, 90}
, {-64, 24}
, {-72, 51}
, {31, 1}
, {-82, -17}
, {-71, -11}
, {91, 36}
, {-63, -99}
, {-73, -69}
, {29, -18}
}
, {{47, -3}
, {-63, -54}
, {65, 82}
, {-33, 24}
, {-39, -17}
, {-87, 73}
, {-67, -23}
, {-81, -9}
, {-74, 59}
, {-76, 57}
, {40, -82}
, {-19, 63}
, {-87, -76}
, {47, 69}
, {17, -29}
, {-8, 51}
}
, {{69, -31}
, {-87, 31}
, {0, -72}
, {-29, -106}
, {-64, 58}
, {-3, 27}
, {-52, -61}
, {54, -9}
, {62, 1}
, {102, 14}
, {64, -51}
, {18, 43}
, {-13, 4}
, {88, 40}
, {40, 48}
, {-18, -34}
}
, {{10, -29}
, {-68, -77}
, {20, -91}
, {-24, 58}
, {27, 79}
, {-36, 1}
, {-91, -1}
, {-38, 55}
, {-2, -86}
, {-34, -76}
, {39, -6}
, {-24, 46}
, {94, -86}
, {57, -22}
, {-86, -39}
, {-65, 79}
}
, {{67, 60}
, {107, -76}
, {35, -83}
, {32, 66}
, {-63, 31}
, {81, 91}
, {42, 66}
, {51, 3}
, {-75, 53}
, {23, -7}
, {-40, 15}
, {93, 27}
, {62, -76}
, {17, 44}
, {-16, -100}
, {-79, -2}
}
, {{-8, 17}
, {10, 40}
, {82, 56}
, {-68, -28}
, {17, -33}
, {20, -52}
, {33, 7}
, {75, 55}
, {74, 27}
, {-60, 33}
, {-44, 55}
, {-36, -39}
, {-35, 81}
, {-47, 36}
, {45, 48}
, {-54, 54}
}
, {{-66, 87}
, {26, 49}
, {-5, -2}
, {-5, -35}
, {44, -81}
, {68, -37}
, {84, -50}
, {32, -67}
, {14, 94}
, {77, -67}
, {-17, -51}
, {-41, 61}
, {88, -25}
, {-60, -2}
, {-46, 90}
, {-53, 41}
}
, {{62, 53}
, {-97, -39}
, {94, -50}
, {64, 46}
, {78, -81}
, {-82, 76}
, {-49, -29}
, {130, -28}
, {-34, 12}
, {-26, 11}
, {-10, 76}
, {27, -35}
, {-13, -27}
, {107, 22}
, {-72, -99}
, {-45, -8}
}
, {{-38, -84}
, {27, -98}
, {53, 66}
, {-23, -36}
, {89, -20}
, {-145, -19}
, {34, 34}
, {45, -54}
, {-36, -94}
, {-86, -57}
, {-86, -79}
, {-33, -12}
, {14, -6}
, {-117, -96}
, {-17, 41}
, {-7, 92}
}
, {{52, -22}
, {-28, 40}
, {-36, 21}
, {-11, 84}
, {27, -35}
, {28, -31}
, {-79, 63}
, {46, -7}
, {22, 70}
, {-64, 60}
, {-78, -83}
, {25, -46}
, {6, 45}
, {15, -79}
, {-35, -17}
, {-62, -27}
}
, {{-58, -47}
, {-29, -9}
, {72, -35}
, {9, 11}
, {-77, -75}
, {18, 1}
, {-57, 32}
, {38, 40}
, {59, -48}
, {21, 23}
, {-4, 48}
, {19, -94}
, {83, 95}
, {63, -56}
, {43, -28}
, {-21, 66}
}
, {{36, 45}
, {53, 39}
, {-49, -70}
, {-94, -12}
, {-81, -45}
, {29, -4}
, {99, -47}
, {-2, -107}
, {-80, -66}
, {-23, -48}
, {76, -51}
, {-49, -37}
, {59, 29}
, {-89, -2}
, {-35, 64}
, {-69, -84}
}
, {{-83, -104}
, {-72, 40}
, {-52, 65}
, {33, 38}
, {47, 62}
, {-4, 48}
, {16, 32}
, {44, 2}
, {-52, 85}
, {-28, -72}
, {-34, 23}
, {7, 25}
, {-50, 32}
, {19, 36}
, {-78, 72}
, {29, -21}
}
, {{-16, 19}
, {-108, 6}
, {-91, -11}
, {-65, 67}
, {3, 14}
, {34, -78}
, {-69, -94}
, {-81, 19}
, {-27, 41}
, {93, 21}
, {-20, 73}
, {65, 58}
, {-25, -58}
, {56, 47}
, {-88, -12}
, {23, 72}
}
, {{14, -58}
, {-23, 0}
, {41, -84}
, {-43, 23}
, {-38, 28}
, {-74, -98}
, {98, 92}
, {0, -23}
, {7, -36}
, {-60, 96}
, {-57, 64}
, {52, 51}
, {45, 92}
, {15, 18}
, {-40, 85}
, {17, -31}
}
, {{53, -102}
, {-91, -95}
, {65, -68}
, {-45, 16}
, {-45, 23}
, {-54, -50}
, {10, -59}
, {-40, -2}
, {-79, 67}
, {-34, -22}
, {-97, -2}
, {92, -80}
, {29, -101}
, {59, 77}
, {56, -29}
, {-95, 68}
}
, {{-2, 62}
, {-24, 70}
, {2, 76}
, {-75, 47}
, {87, -17}
, {-31, -38}
, {89, -89}
, {72, 61}
, {-8, -82}
, {-16, -41}
, {9, -27}
, {-93, 44}
, {85, 72}
, {-47, 27}
, {43, -108}
, {-65, -29}
}
, {{-40, 84}
, {29, -77}
, {-76, 2}
, {39, 83}
, {77, -21}
, {-5, -45}
, {-48, -47}
, {-84, 45}
, {14, -21}
, {41, 59}
, {-74, 36}
, {-9, 96}
, {89, -32}
, {-103, 52}
, {-39, -91}
, {-59, -25}
}
, {{-80, -21}
, {-28, 28}
, {2, 20}
, {-8, 28}
, {55, -67}
, {-96, 82}
, {-48, -82}
, {58, 20}
, {-71, -23}
, {-67, -114}
, {-104, -26}
, {-57, 55}
, {58, -20}
, {31, 83}
, {-62, 30}
, {90, 91}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE