/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    32
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  2


const int16_t conv1d_30_bias[CONV_FILTERS] = {0, -3, -4, 5, 12, -2, -2, -21, 1, -7, 3, -4, -7, -13, 0, -1, -11, -6, -1, -11, -1, -2, -4, -14, -5, -1, 1, 0, -17, -3, -16, 1, 8, -1, -25, -14, 0, -17, -14, -6, -3, -9, -2, -6, 3, -4, -7, -4, -5, -11, -20, -13, -3, -4, -1, -10, 4, 1, -3, -2, -4, -1, -7, -3}
;

const int16_t conv1d_30_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-30, 49}
, {45, -48}
, {-54, 34}
, {47, -3}
, {50, -31}
, {73, -66}
, {6, 76}
, {-59, 71}
, {68, -45}
, {-50, 22}
, {-90, -60}
, {5, -39}
, {-58, -19}
, {81, 87}
, {-83, 51}
, {-89, 37}
, {-55, -63}
, {50, 24}
, {-27, 33}
, {85, 54}
, {-61, -15}
, {-14, 89}
, {8, 54}
, {-17, 41}
, {-42, -48}
, {-29, 9}
, {44, -34}
, {-2, 79}
, {89, 12}
, {47, 4}
, {-13, -46}
, {59, 60}
}
, {{56, 87}
, {63, -23}
, {-69, 57}
, {18, -90}
, {-80, 100}
, {-24, -38}
, {88, -19}
, {41, -82}
, {-75, -12}
, {-34, 49}
, {37, -30}
, {-55, -13}
, {30, -97}
, {49, -55}
, {-62, 26}
, {68, 94}
, {71, 12}
, {31, -23}
, {49, 73}
, {-50, 24}
, {0, -6}
, {-9, 78}
, {-2, -77}
, {1, -18}
, {80, -82}
, {-42, -8}
, {-54, 52}
, {52, -41}
, {-57, -47}
, {-65, 1}
, {-74, -17}
, {16, 44}
}
, {{28, 69}
, {4, 21}
, {64, 85}
, {-51, -60}
, {-23, 27}
, {-12, -89}
, {-48, 20}
, {-60, -53}
, {-52, 68}
, {-88, -11}
, {21, 46}
, {-40, -73}
, {43, 76}
, {-60, 21}
, {-44, -66}
, {16, 65}
, {27, -36}
, {33, 80}
, {63, 73}
, {-47, 61}
, {-51, 0}
, {29, 81}
, {-88, 57}
, {-26, 82}
, {73, 67}
, {22, 47}
, {54, -51}
, {-16, -22}
, {-5, -21}
, {81, -8}
, {31, -8}
, {53, 47}
}
, {{-89, 67}
, {23, -24}
, {-55, 90}
, {74, -50}
, {-1, -80}
, {-48, 3}
, {-67, 59}
, {68, 34}
, {-44, -5}
, {44, 73}
, {0, -11}
, {42, -69}
, {-4, 40}
, {4, 80}
, {48, -16}
, {79, -4}
, {-17, 35}
, {13, -49}
, {-23, 40}
, {114, 0}
, {-28, 5}
, {-71, 1}
, {-21, 41}
, {-45, 65}
, {34, -2}
, {-4, -33}
, {-38, 20}
, {-75, 56}
, {28, 28}
, {-66, -3}
, {73, -62}
, {0, 32}
}
, {{60, -34}
, {-34, 13}
, {-15, 34}
, {-22, 29}
, {-51, 59}
, {-2, -6}
, {-50, -74}
, {23, -60}
, {29, -1}
, {-83, 7}
, {-61, -51}
, {-70, -48}
, {34, 12}
, {16, -67}
, {-24, 43}
, {38, 40}
, {-42, 35}
, {98, -50}
, {-3, 3}
, {66, 44}
, {-19, 62}
, {34, -85}
, {72, -43}
, {-17, 7}
, {35, -50}
, {-23, -9}
, {57, -56}
, {-44, -29}
, {69, 43}
, {-54, 71}
, {-3, -25}
, {-60, 51}
}
, {{-25, -51}
, {-38, -58}
, {90, 62}
, {36, 76}
, {7, 68}
, {79, 28}
, {8, 35}
, {27, 53}
, {12, 21}
, {67, 77}
, {37, -63}
, {17, -6}
, {30, -47}
, {-70, 37}
, {-55, -25}
, {7, -36}
, {-48, 48}
, {-86, -41}
, {-2, 0}
, {-84, 33}
, {-18, -86}
, {-38, -10}
, {60, 72}
, {-33, 30}
, {67, -48}
, {-33, 9}
, {96, 79}
, {45, 65}
, {-80, 37}
, {10, -10}
, {-14, -14}
, {10, -48}
}
, {{-34, 48}
, {51, 49}
, {-5, -73}
, {-41, 0}
, {5, -34}
, {84, 68}
, {74, 36}
, {65, -65}
, {85, 1}
, {-67, 49}
, {-80, -21}
, {-80, 70}
, {-64, -44}
, {73, 13}
, {14, 54}
, {-83, 12}
, {40, -29}
, {7, -21}
, {20, -53}
, {-56, -70}
, {-22, -55}
, {-29, 15}
, {85, -24}
, {62, -12}
, {-54, 61}
, {6, 75}
, {-17, 44}
, {-74, 9}
, {-46, 77}
, {16, 37}
, {-38, 8}
, {-31, 22}
}
, {{76, -14}
, {17, -14}
, {-66, 6}
, {-25, -40}
, {-70, -76}
, {-43, 5}
, {-30, -60}
, {-24, -86}
, {-97, -96}
, {-74, -19}
, {42, -101}
, {51, 35}
, {80, 36}
, {0, 27}
, {-71, 1}
, {66, 26}
, {88, 19}
, {59, -21}
, {44, -49}
, {-17, 23}
, {32, 51}
, {-74, -74}
, {-90, 57}
, {21, -23}
, {17, 48}
, {35, 17}
, {83, 3}
, {50, -91}
, {19, -35}
, {-45, -105}
, {-54, -1}
, {-33, -22}
}
, {{84, -54}
, {-69, 21}
, {-69, -4}
, {-1, -74}
, {49, -44}
, {-4, -15}
, {52, 83}
, {-43, 59}
, {36, 21}
, {68, -28}
, {87, -26}
, {38, 69}
, {17, -16}
, {39, 77}
, {-74, 54}
, {-8, 31}
, {-77, 80}
, {42, 74}
, {-33, 80}
, {90, -78}
, {-32, 21}
, {-24, 51}
, {-88, 15}
, {-61, -1}
, {-86, 69}
, {82, 73}
, {-41, 8}
, {-78, -68}
, {29, 27}
, {27, -33}
, {-43, 50}
, {-80, 16}
}
, {{9, -69}
, {-9, 27}
, {62, -84}
, {-89, 47}
, {-41, 53}
, {20, -90}
, {16, -34}
, {-11, -35}
, {67, 70}
, {23, 29}
, {-68, 60}
, {-27, 62}
, {70, -41}
, {0, 47}
, {15, -8}
, {-91, -87}
, {42, 61}
, {63, -85}
, {28, -31}
, {-29, -52}
, {-89, 64}
, {-37, 53}
, {75, 32}
, {-64, -84}
, {43, -21}
, {-17, -79}
, {30, 51}
, {-94, -50}
, {-93, -55}
, {-53, 55}
, {82, -3}
, {15, 3}
}
, {{-61, -57}
, {47, 87}
, {-61, 25}
, {-15, 30}
, {-23, 92}
, {-82, -17}
, {95, -50}
, {84, -44}
, {-33, -27}
, {90, -21}
, {56, -9}
, {7, -66}
, {80, -83}
, {-60, -13}
, {-82, 0}
, {48, 31}
, {-10, -68}
, {12, 14}
, {-5, -91}
, {65, 80}
, {-68, -75}
, {43, 86}
, {12, -29}
, {-84, -38}
, {-18, 15}
, {57, -5}
, {-20, 1}
, {11, 62}
, {-39, -87}
, {-32, 10}
, {25, 69}
, {-60, -75}
}
, {{-69, -66}
, {80, 18}
, {-24, 79}
, {35, -70}
, {81, 41}
, {-85, -13}
, {19, 36}
, {-48, 26}
, {-53, 57}
, {41, -44}
, {76, -4}
, {70, 60}
, {-76, -35}
, {-82, 7}
, {-76, 80}
, {2, 50}
, {-33, -65}
, {-52, -33}
, {-23, -6}
, {62, 24}
, {80, -41}
, {-10, -70}
, {33, 72}
, {50, -47}
, {35, 73}
, {-58, 67}
, {-27, -16}
, {9, 3}
, {52, 51}
, {-8, -85}
, {-66, 84}
, {5, 8}
}
, {{-61, -36}
, {83, 72}
, {38, 19}
, {-73, -6}
, {-26, -35}
, {31, 10}
, {-42, 92}
, {92, -65}
, {42, 55}
, {81, -57}
, {-55, -64}
, {9, -48}
, {-100, -71}
, {20, 6}
, {-15, -6}
, {-24, -48}
, {88, 42}
, {20, 77}
, {-73, -46}
, {-47, 28}
, {63, 84}
, {58, 37}
, {39, -39}
, {-80, -69}
, {30, -38}
, {-20, -2}
, {18, -75}
, {-72, -14}
, {-43, 70}
, {51, -67}
, {-33, 59}
, {-28, -91}
}
, {{-1, 56}
, {9, -58}
, {-23, 47}
, {-62, -84}
, {0, -19}
, {21, -36}
, {76, 29}
, {4, -30}
, {29, -77}
, {21, 79}
, {-22, 65}
, {28, 30}
, {-23, 55}
, {76, -5}
, {48, -91}
, {-89, -62}
, {-52, 3}
, {-76, -32}
, {-32, 33}
, {20, -59}
, {5, 30}
, {-67, 34}
, {5, -74}
, {-71, -80}
, {22, 54}
, {-90, 19}
, {-1, 45}
, {-25, -51}
, {54, -41}
, {30, 47}
, {3, 70}
, {-80, 63}
}
, {{11, 6}
, {-54, 51}
, {-90, 89}
, {-25, -86}
, {63, -65}
, {66, 72}
, {-23, 61}
, {-44, -34}
, {92, 47}
, {-20, 20}
, {28, -80}
, {23, 40}
, {65, 24}
, {57, 10}
, {-45, -17}
, {37, 79}
, {7, -83}
, {-18, -12}
, {32, 64}
, {-1, 2}
, {61, -35}
, {13, -75}
, {86, 83}
, {23, 38}
, {-81, -61}
, {76, -81}
, {3, -73}
, {1, -25}
, {-18, 36}
, {75, 3}
, {19, 88}
, {20, -15}
}
, {{-1, 67}
, {-15, -85}
, {68, -42}
, {4, 72}
, {-46, -70}
, {-13, 79}
, {-44, -2}
, {58, -25}
, {-36, -94}
, {30, -21}
, {1, -43}
, {-8, -22}
, {27, -90}
, {43, 30}
, {64, 85}
, {82, -82}
, {-54, 39}
, {20, -34}
, {66, 25}
, {33, -50}
, {27, -4}
, {75, -35}
, {57, 37}
, {-1, -16}
, {45, -41}
, {-78, 18}
, {-41, -89}
, {57, -6}
, {0, 93}
, {72, 68}
, {-54, -41}
, {-7, -65}
}
, {{-7, -38}
, {-58, 20}
, {57, -79}
, {7, 57}
, {60, 18}
, {-75, 59}
, {38, 71}
, {-14, -61}
, {-78, -27}
, {48, -18}
, {-62, 71}
, {-1, 19}
, {-83, -56}
, {-65, -62}
, {-52, -48}
, {5, 6}
, {26, 46}
, {-82, 0}
, {-57, -13}
, {-74, -6}
, {16, 41}
, {-70, 27}
, {-38, -47}
, {85, -20}
, {-74, 88}
, {-11, 28}
, {49, -33}
, {-48, -88}
, {-20, 44}
, {-85, 58}
, {-18, 69}
, {21, 84}
}
, {{-59, 51}
, {-26, -2}
, {41, -58}
, {-57, -11}
, {-85, -57}
, {3, -21}
, {43, -23}
, {8, -69}
, {-73, 20}
, {9, -90}
, {-73, 40}
, {69, -33}
, {41, -77}
, {-29, 15}
, {70, -64}
, {-85, 33}
, {-77, 31}
, {0, 43}
, {-23, 19}
, {54, 39}
, {3, -32}
, {6, -74}
, {-72, 11}
, {0, 9}
, {56, -24}
, {-81, -24}
, {-41, 73}
, {61, 57}
, {-16, 41}
, {11, -38}
, {-77, 76}
, {-25, 66}
}
, {{-22, 19}
, {24, -46}
, {79, -22}
, {6, -46}
, {-2, -44}
, {-74, 55}
, {50, 69}
, {12, -49}
, {-35, 68}
, {17, -54}
, {-85, 5}
, {25, -12}
, {-54, 4}
, {-74, 1}
, {22, 29}
, {15, -7}
, {78, 39}
, {-6, 70}
, {10, -74}
, {-98, 38}
, {-44, -73}
, {19, 73}
, {35, 82}
, {-76, -42}
, {72, -20}
, {49, -73}
, {-18, 78}
, {2, 42}
, {-31, 0}
, {89, -51}
, {-26, 51}
, {78, -4}
}
, {{-8, 24}
, {-34, 67}
, {80, -45}
, {-91, 0}
, {72, 92}
, {-89, 69}
, {64, 75}
, {38, -24}
, {0, -53}
, {-59, 30}
, {-53, 35}
, {-66, 40}
, {-65, 55}
, {81, 26}
, {-77, -13}
, {-8, -44}
, {-6, -7}
, {-25, -33}
, {-3, -47}
, {54, 50}
, {-65, -39}
, {55, -77}
, {-68, 4}
, {36, -7}
, {-23, -36}
, {-43, 56}
, {43, -62}
, {27, -50}
, {-56, 85}
, {36, 49}
, {74, -31}
, {3, -67}
}
, {{16, 66}
, {-23, -79}
, {71, -8}
, {-83, -62}
, {-39, 63}
, {-80, 6}
, {-56, -62}
, {31, 85}
, {15, 83}
, {-5, 37}
, {-22, 42}
, {51, -84}
, {49, 17}
, {-66, -72}
, {25, -39}
, {47, 49}
, {-26, -63}
, {44, -1}
, {20, -26}
, {64, 0}
, {72, -26}
, {-33, 25}
, {-41, 85}
, {51, 76}
, {-4, -9}
, {8, -30}
, {-23, 12}
, {-46, -45}
, {-6, -4}
, {0, 67}
, {-82, -38}
, {20, -9}
}
, {{-52, 49}
, {38, 15}
, {-52, 3}
, {-15, 89}
, {-91, 5}
, {-31, -3}
, {-21, -31}
, {35, -20}
, {-62, -9}
, {-6, -7}
, {75, -50}
, {-25, -54}
, {73, 28}
, {-8, 66}
, {7, 85}
, {28, 87}
, {33, 62}
, {70, 36}
, {83, -63}
, {77, 62}
, {82, -14}
, {-22, 27}
, {-1, 28}
, {-46, -67}
, {32, 64}
, {32, -25}
, {61, -52}
, {-33, 4}
, {46, 53}
, {-41, -41}
, {56, 34}
, {11, -80}
}
, {{-58, -15}
, {78, 13}
, {2, 74}
, {47, 49}
, {-40, 83}
, {-69, -69}
, {30, 24}
, {-83, -46}
, {28, 6}
, {17, 40}
, {73, -13}
, {-74, 66}
, {1, -88}
, {-4, 70}
, {-17, 41}
, {11, -23}
, {-14, -56}
, {77, 80}
, {-55, 4}
, {-43, -9}
, {45, 23}
, {49, 9}
, {-79, -49}
, {23, 39}
, {88, -16}
, {-35, -29}
, {50, 50}
, {30, 26}
, {-69, 98}
, {12, -9}
, {-41, 58}
, {-5, 44}
}
, {{75, -4}
, {88, 53}
, {64, 76}
, {-97, 53}
, {15, -35}
, {-83, -56}
, {58, -26}
, {38, -33}
, {-91, -31}
, {74, -4}
, {-26, 42}
, {-22, 45}
, {64, 82}
, {75, -52}
, {-37, 23}
, {-88, -47}
, {-7, -19}
, {-41, 37}
, {-59, 34}
, {-21, 0}
, {65, -59}
, {39, -25}
, {-88, -98}
, {-74, -42}
, {-19, -13}
, {-96, -8}
, {21, 31}
, {-97, -43}
, {34, 49}
, {49, 30}
, {54, -26}
, {72, -72}
}
, {{-32, 11}
, {-65, 47}
, {63, 89}
, {79, -3}
, {47, 37}
, {18, 72}
, {47, -59}
, {20, -15}
, {2, 67}
, {-54, 25}
, {81, 5}
, {11, 62}
, {19, 88}
, {38, 16}
, {16, -52}
, {-20, 30}
, {82, -78}
, {-51, 56}
, {-10, 27}
, {-61, -3}
, {-61, 25}
, {52, -74}
, {-32, -86}
, {-36, 61}
, {-6, 79}
, {-8, -51}
, {10, -58}
, {-60, 47}
, {50, 48}
, {76, -34}
, {-7, 89}
, {-43, -18}
}
, {{-24, -4}
, {85, 94}
, {-85, -59}
, {-3, 74}
, {38, -2}
, {-18, -72}
, {59, -64}
, {70, -63}
, {79, 57}
, {14, -65}
, {70, -61}
, {2, 36}
, {3, -90}
, {-85, -45}
, {-51, 31}
, {22, -47}
, {61, -87}
, {9, 67}
, {-93, 33}
, {34, -62}
, {-42, 29}
, {-5, -59}
, {-73, -36}
, {32, -87}
, {8, 39}
, {43, 75}
, {-26, -89}
, {82, 35}
, {-10, -82}
, {-3, -4}
, {-3, -78}
, {19, -76}
}
, {{52, 66}
, {27, 33}
, {-85, 42}
, {-16, 13}
, {5, -69}
, {-37, 63}
, {64, -36}
, {-8, -11}
, {-78, 41}
, {-84, 21}
, {86, 29}
, {-7, 5}
, {-67, 21}
, {-16, -64}
, {55, -63}
, {-19, -50}
, {45, -45}
, {16, 71}
, {-80, 81}
, {-79, 36}
, {46, 0}
, {-68, 28}
, {28, 75}
, {80, 0}
, {-2, 62}
, {81, 44}
, {20, -38}
, {9, 38}
, {-40, -75}
, {22, 75}
, {1, -40}
, {76, 6}
}
, {{29, 19}
, {87, -26}
, {23, -1}
, {-32, -59}
, {-21, -45}
, {44, 7}
, {50, 6}
, {30, -42}
, {-86, -69}
, {-11, -86}
, {33, -65}
, {-13, 60}
, {61, 34}
, {-38, -71}
, {38, -59}
, {-51, -71}
, {79, 71}
, {21, 76}
, {60, -6}
, {38, 13}
, {-62, -28}
, {56, 65}
, {-19, -94}
, {41, -7}
, {0, -21}
, {1, -40}
, {-74, 22}
, {12, -70}
, {51, 38}
, {-80, -5}
, {6, 82}
, {1, 63}
}
, {{-23, -38}
, {80, 52}
, {-34, -37}
, {-101, -81}
, {44, -63}
, {8, 16}
, {13, 80}
, {-76, 63}
, {36, 56}
, {-64, 22}
, {26, 4}
, {61, 49}
, {-44, -3}
, {7, -88}
, {-42, -62}
, {-46, -42}
, {22, -6}
, {-72, 37}
, {-57, 16}
, {-21, 8}
, {27, -87}
, {56, -14}
, {-39, 68}
, {-65, 42}
, {-73, 8}
, {-42, -2}
, {87, -31}
, {12, -27}
, {-60, 62}
, {-82, -43}
, {22, -62}
, {80, -31}
}
, {{51, -11}
, {-49, 19}
, {-73, 18}
, {84, 75}
, {45, -53}
, {-83, 17}
, {-89, -16}
, {55, -32}
, {76, 63}
, {-9, -23}
, {-13, 79}
, {-2, -75}
, {71, -56}
, {-2, 27}
, {69, 23}
, {4, -8}
, {31, 48}
, {66, 81}
, {22, 85}
, {67, 43}
, {-56, -25}
, {-78, 55}
, {36, -16}
, {-58, 6}
, {92, 12}
, {57, 44}
, {33, -70}
, {-79, 78}
, {-37, -14}
, {1, -58}
, {27, 72}
, {0, 39}
}
, {{20, 18}
, {-35, -25}
, {-35, -20}
, {-8, -34}
, {0, -57}
, {75, 81}
, {20, -82}
, {-23, -30}
, {58, -31}
, {22, 35}
, {33, -1}
, {31, -42}
, {19, 2}
, {47, -25}
, {-94, 62}
, {64, -9}
, {-49, -33}
, {-89, -63}
, {-77, -17}
, {-40, 28}
, {-71, -11}
, {21, 2}
, {5, -14}
, {-81, -78}
, {-58, -89}
, {51, -56}
, {-60, 44}
, {-49, 78}
, {82, -74}
, {25, 64}
, {-33, -88}
, {62, 7}
}
, {{43, 84}
, {46, 14}
, {-89, 89}
, {-51, -18}
, {-5, 49}
, {-34, 78}
, {-6, -51}
, {30, 53}
, {75, 15}
, {75, 47}
, {-87, -62}
, {-35, 67}
, {69, 13}
, {53, 30}
, {6, 68}
, {-20, -10}
, {-9, 21}
, {24, -89}
, {-65, 50}
, {-6, -39}
, {72, 12}
, {-18, 49}
, {-36, 49}
, {-26, -49}
, {17, 48}
, {-63, -38}
, {-13, 35}
, {40, 55}
, {-74, -42}
, {-52, 79}
, {-44, -44}
, {-10, 53}
}
, {{18, -20}
, {30, 66}
, {26, -71}
, {49, 77}
, {69, -35}
, {-21, -25}
, {22, 65}
, {-81, -56}
, {90, -30}
, {-8, -82}
, {-36, 75}
, {-24, -17}
, {-42, 12}
, {39, -16}
, {-38, 46}
, {-67, -35}
, {-81, -42}
, {-60, 52}
, {66, -41}
, {-36, -63}
, {49, 27}
, {29, -86}
, {-53, -36}
, {38, -78}
, {-14, -17}
, {43, -15}
, {-69, 18}
, {-2, 13}
, {38, -1}
, {-76, 47}
, {47, -21}
, {-26, -31}
}
, {{-72, -42}
, {1, 28}
, {-64, -80}
, {56, 5}
, {-10, 35}
, {75, 49}
, {-11, 60}
, {67, 68}
, {55, 39}
, {-62, -44}
, {-74, 30}
, {26, 3}
, {67, -47}
, {-15, -42}
, {-63, 14}
, {51, 34}
, {3, 52}
, {38, 19}
, {-3, 47}
, {-89, 39}
, {-86, 41}
, {77, -16}
, {27, 18}
, {41, 57}
, {78, 2}
, {-8, 87}
, {61, -46}
, {41, -75}
, {6, -48}
, {-25, 15}
, {75, -52}
, {-57, -51}
}
, {{14, 0}
, {-54, -16}
, {20, 32}
, {64, -43}
, {25, 12}
, {66, -37}
, {-17, -15}
, {76, -5}
, {-105, -60}
, {-100, 72}
, {-62, 8}
, {-37, -45}
, {44, 63}
, {-81, -8}
, {12, 30}
, {-28, 49}
, {-6, 13}
, {-17, 32}
, {-56, -39}
, {8, 9}
, {29, 52}
, {-90, 28}
, {-18, 57}
, {25, -43}
, {-35, -41}
, {-74, 3}
, {-1, -32}
, {55, 47}
, {-85, 45}
, {-26, -103}
, {-40, 13}
, {36, -25}
}
, {{75, -11}
, {52, -25}
, {80, 71}
, {59, 56}
, {-64, -50}
, {-88, 84}
, {-56, 71}
, {-76, 64}
, {-74, 16}
, {35, -40}
, {64, -54}
, {-19, 12}
, {58, 9}
, {65, -41}
, {-37, -97}
, {-23, -13}
, {57, 77}
, {-76, 34}
, {13, 25}
, {-38, 93}
, {76, 65}
, {6, -93}
, {-23, -96}
, {-44, -33}
, {-15, 5}
, {-23, -23}
, {-79, 73}
, {-89, 14}
, {88, -7}
, {71, 2}
, {60, 42}
, {11, 47}
}
, {{61, -18}
, {69, 69}
, {-27, 46}
, {27, 36}
, {-22, 38}
, {44, 44}
, {32, -17}
, {-75, -58}
, {52, 3}
, {-43, -70}
, {-67, 82}
, {65, 55}
, {-52, -70}
, {9, 80}
, {-89, -14}
, {35, 65}
, {18, 51}
, {6, -4}
, {-57, 81}
, {-62, -83}
, {-74, -68}
, {34, -49}
, {74, -61}
, {-44, 30}
, {-78, 48}
, {-7, -40}
, {5, -36}
, {-84, -26}
, {-77, -50}
, {61, 24}
, {70, -57}
, {-19, -55}
}
, {{19, -15}
, {16, -71}
, {-10, 84}
, {-89, -48}
, {63, -19}
, {-32, -30}
, {-3, 47}
, {46, 19}
, {-55, 55}
, {-40, 20}
, {-62, 15}
, {-17, -87}
, {-25, -58}
, {-19, -5}
, {-92, 6}
, {-98, -68}
, {40, 74}
, {-4, 45}
, {-11, -60}
, {20, -32}
, {2, -29}
, {0, -68}
, {-72, 74}
, {-37, 24}
, {0, -24}
, {-101, -71}
, {47, 52}
, {-58, -27}
, {73, 79}
, {66, 43}
, {-4, 50}
, {-67, 17}
}
, {{63, 22}
, {-10, 86}
, {-36, -77}
, {-1, 65}
, {-38, -88}
, {73, -48}
, {6, -85}
, {56, 57}
, {30, -95}
, {-47, 16}
, {-70, -24}
, {-61, -36}
, {-12, -13}
, {67, -57}
, {-43, 5}
, {-75, -22}
, {3, -57}
, {-59, -62}
, {12, -40}
, {-47, -52}
, {55, -41}
, {22, 44}
, {-86, -84}
, {-32, -39}
, {34, 45}
, {-97, -50}
, {-51, -88}
, {-57, -10}
, {55, 11}
, {-66, 62}
, {2, 12}
, {-27, 60}
}
, {{-89, -94}
, {-37, 22}
, {-5, 83}
, {-47, -62}
, {-60, -72}
, {56, 13}
, {22, 58}
, {-43, 6}
, {-53, 44}
, {-86, 75}
, {-43, -71}
, {39, -91}
, {-59, -70}
, {-84, 68}
, {36, 18}
, {77, -92}
, {-58, -10}
, {-39, 18}
, {-2, -58}
, {-33, -25}
, {37, -34}
, {46, 36}
, {6, 17}
, {-28, 81}
, {31, 75}
, {74, -77}
, {11, -49}
, {37, 63}
, {60, 40}
, {-92, -73}
, {40, -4}
, {81, -22}
}
, {{81, -31}
, {3, -31}
, {47, 34}
, {-55, -30}
, {-50, 92}
, {46, 80}
, {49, -59}
, {0, -66}
, {76, -41}
, {-10, 28}
, {-49, -79}
, {-36, 30}
, {41, 62}
, {-61, -77}
, {-54, 44}
, {55, -11}
, {12, -7}
, {70, -8}
, {-19, -19}
, {-27, 11}
, {0, -77}
, {82, 50}
, {-1, 9}
, {43, -45}
, {-70, -31}
, {61, -25}
, {22, -45}
, {83, 8}
, {6, -45}
, {55, -32}
, {50, -12}
, {-32, 85}
}
, {{23, -29}
, {-77, 41}
, {28, -32}
, {24, -91}
, {-4, -60}
, {-49, -51}
, {74, 64}
, {-63, -20}
, {-24, -95}
, {20, 23}
, {-85, 45}
, {-15, -84}
, {-53, -36}
, {-45, 12}
, {-62, 0}
, {-14, -55}
, {-53, -80}
, {30, 44}
, {-21, 47}
, {22, 34}
, {40, -82}
, {61, 1}
, {-16, 12}
, {6, 74}
, {-85, 42}
, {-68, 79}
, {45, 16}
, {31, -68}
, {-72, -37}
, {36, -66}
, {23, -81}
, {-17, 64}
}
, {{-31, 42}
, {31, 79}
, {-3, -57}
, {22, -17}
, {-3, -44}
, {64, 81}
, {83, 0}
, {-46, -82}
, {44, -64}
, {81, 11}
, {-50, 17}
, {32, -77}
, {68, 36}
, {-66, -8}
, {-78, 85}
, {21, -66}
, {-48, -58}
, {54, -50}
, {-77, -9}
, {25, -63}
, {8, -63}
, {28, 5}
, {-29, -33}
, {-38, -36}
, {-68, 79}
, {69, -56}
, {-19, 39}
, {5, -22}
, {30, 19}
, {-31, 7}
, {0, -5}
, {-44, 3}
}
, {{-97, 11}
, {13, -30}
, {-12, -44}
, {-88, -78}
, {-18, 54}
, {0, -60}
, {59, 25}
, {-14, -57}
, {-86, 22}
, {-72, -20}
, {39, -47}
, {52, 64}
, {5, 66}
, {9, -39}
, {29, -54}
, {45, -3}
, {-22, -18}
, {24, 37}
, {19, -84}
, {-30, 44}
, {-8, -13}
, {-93, 35}
, {-65, 42}
, {24, 42}
, {-81, -84}
, {35, 27}
, {-13, -86}
, {-11, 63}
, {-96, 51}
, {-26, 5}
, {58, 72}
, {-35, 19}
}
, {{86, -14}
, {-40, -94}
, {-62, -77}
, {4, -1}
, {19, -14}
, {-55, -64}
, {-51, -41}
, {29, 21}
, {52, -58}
, {-90, 17}
, {-47, 77}
, {47, 11}
, {-55, 69}
, {34, -88}
, {-66, 77}
, {37, 77}
, {31, -48}
, {28, -56}
, {13, -56}
, {62, -19}
, {-84, -52}
, {55, -59}
, {33, 60}
, {64, -21}
, {50, 44}
, {36, -55}
, {24, -84}
, {73, 4}
, {70, -46}
, {-5, -32}
, {39, 82}
, {55, 30}
}
, {{-17, -61}
, {74, 43}
, {6, 67}
, {28, 5}
, {10, -60}
, {51, -3}
, {33, 26}
, {-25, -21}
, {-63, -61}
, {22, 6}
, {-64, -31}
, {-66, 80}
, {81, 4}
, {-31, -40}
, {-66, -64}
, {21, 31}
, {-37, -38}
, {-83, 69}
, {-53, -18}
, {62, 25}
, {-72, 53}
, {32, 35}
, {82, 73}
, {34, 33}
, {57, -8}
, {-10, 84}
, {-18, 24}
, {-8, 25}
, {32, -71}
, {-81, 0}
, {-62, -65}
, {0, 39}
}
, {{44, -24}
, {-44, 57}
, {69, 13}
, {-27, 4}
, {-38, -3}
, {-68, 80}
, {-74, -51}
, {-50, 75}
, {-65, -75}
, {-62, 17}
, {43, -17}
, {-8, -76}
, {-78, 81}
, {41, -47}
, {33, 3}
, {0, -62}
, {6, -83}
, {-36, 53}
, {81, 85}
, {35, -31}
, {-8, 20}
, {83, -11}
, {-9, 57}
, {13, 100}
, {69, 80}
, {68, 27}
, {-84, 16}
, {75, -7}
, {-51, -61}
, {77, -42}
, {-10, -6}
, {58, 50}
}
, {{67, 49}
, {-64, 64}
, {-6, 21}
, {-37, -29}
, {-69, 91}
, {-5, 83}
, {-30, -36}
, {14, 37}
, {70, 74}
, {80, -15}
, {64, -13}
, {28, -17}
, {-24, 62}
, {60, 52}
, {32, 8}
, {-44, -65}
, {-10, 51}
, {37, 91}
, {-76, 42}
, {-87, 87}
, {43, 69}
, {29, -24}
, {-40, -28}
, {-48, -15}
, {12, 20}
, {0, -1}
, {-41, -41}
, {60, -88}
, {17, -38}
, {21, 0}
, {52, 87}
, {-51, 36}
}
, {{7, -58}
, {-62, 1}
, {19, 55}
, {-34, -45}
, {12, -23}
, {-30, -17}
, {-83, 65}
, {-24, -75}
, {-1, 85}
, {-32, 48}
, {41, -43}
, {-46, -46}
, {82, 3}
, {36, -51}
, {-91, -1}
, {-69, 51}
, {-26, 13}
, {-79, -46}
, {-98, -65}
, {64, 13}
, {56, -23}
, {-6, -68}
, {-6, -14}
, {38, -33}
, {-73, -18}
, {-66, 43}
, {2, 75}
, {48, -79}
, {-84, 49}
, {-37, 63}
, {-28, -66}
, {68, 88}
}
, {{40, 3}
, {78, -74}
, {24, 7}
, {33, -38}
, {-74, -58}
, {66, -84}
, {29, 0}
, {-30, 91}
, {-85, -49}
, {26, 0}
, {-47, -77}
, {60, -30}
, {53, -28}
, {3, 41}
, {-13, -58}
, {-65, 19}
, {49, -21}
, {-99, -46}
, {-92, -94}
, {58, -28}
, {31, 59}
, {80, 9}
, {53, 48}
, {-56, 45}
, {38, -90}
, {-29, 38}
, {6, -30}
, {-17, 0}
, {-77, 38}
, {11, -63}
, {-85, -47}
, {0, 59}
}
, {{-20, -51}
, {26, 66}
, {84, -83}
, {-75, -61}
, {38, -5}
, {24, -23}
, {38, -3}
, {83, 73}
, {62, -46}
, {-75, 58}
, {-47, 74}
, {-39, -15}
, {-54, 27}
, {-44, -90}
, {14, 74}
, {-59, -87}
, {29, 67}
, {25, -32}
, {59, -87}
, {-15, 4}
, {-24, 36}
, {-48, -84}
, {5, -16}
, {36, 35}
, {35, 32}
, {27, 70}
, {61, 34}
, {-45, 53}
, {76, 18}
, {18, -25}
, {49, 53}
, {54, -69}
}
, {{-62, -44}
, {4, 5}
, {-29, 8}
, {-26, 9}
, {37, 28}
, {50, 26}
, {67, 25}
, {0, -51}
, {7, -91}
, {62, -5}
, {-26, 47}
, {-70, -63}
, {79, -74}
, {-68, -32}
, {-62, 7}
, {-49, -6}
, {86, -58}
, {21, -71}
, {-12, -14}
, {-33, 59}
, {-70, 84}
, {-35, -29}
, {-41, 75}
, {-84, -8}
, {87, -19}
, {-66, 80}
, {-44, -27}
, {58, 78}
, {-21, 77}
, {-14, 35}
, {82, 62}
, {48, -77}
}
, {{40, 30}
, {-86, -59}
, {66, -58}
, {48, -47}
, {57, -89}
, {-73, -45}
, {-55, 88}
, {79, -72}
, {-30, -64}
, {-53, 68}
, {-37, -28}
, {-4, 31}
, {-36, 64}
, {90, 67}
, {-77, -12}
, {-61, -21}
, {44, 6}
, {-44, -76}
, {-62, -29}
, {43, -34}
, {-80, -65}
, {70, -91}
, {-29, 14}
, {-51, 76}
, {13, -56}
, {0, -71}
, {73, -2}
, {-60, -78}
, {-27, -56}
, {26, 1}
, {-82, 89}
, {-5, 26}
}
, {{-8, 44}
, {34, 64}
, {68, 39}
, {-7, -23}
, {70, 21}
, {-85, 62}
, {23, 89}
, {53, 8}
, {-14, 73}
, {-78, -49}
, {-81, -2}
, {-36, -89}
, {17, 33}
, {-75, 4}
, {-27, -75}
, {-10, 57}
, {-89, 36}
, {-79, 81}
, {76, 16}
, {-17, -101}
, {-30, -66}
, {0, -50}
, {16, -74}
, {-35, 45}
, {-73, -4}
, {-54, 81}
, {-67, -80}
, {11, 44}
, {-8, 48}
, {-48, 0}
, {39, -67}
, {83, 46}
}
, {{54, 33}
, {-25, 55}
, {-46, -63}
, {-14, 52}
, {29, -52}
, {22, -32}
, {-55, 97}
, {41, -30}
, {27, 14}
, {-44, 71}
, {-76, 47}
, {14, -70}
, {9, 24}
, {20, -115}
, {-21, 80}
, {-56, 46}
, {-14, 30}
, {51, 11}
, {3, -59}
, {76, 13}
, {-15, -62}
, {-25, -2}
, {71, 37}
, {6, 40}
, {66, 39}
, {71, 17}
, {-77, -17}
, {-25, -36}
, {37, -65}
, {-59, 22}
, {-3, 20}
, {-31, -78}
}
, {{-53, 24}
, {-36, 84}
, {63, 66}
, {2, 60}
, {-76, -51}
, {35, 14}
, {-63, 16}
, {12, -8}
, {49, -22}
, {14, 21}
, {-11, -8}
, {10, 3}
, {76, -95}
, {58, -27}
, {28, -74}
, {-25, 18}
, {-55, 31}
, {-16, -36}
, {-48, -4}
, {-9, -81}
, {-76, -81}
, {2, 23}
, {26, -87}
, {37, 28}
, {-41, -32}
, {6, 43}
, {37, 3}
, {-91, -71}
, {92, -47}
, {-53, 63}
, {-41, -36}
, {39, -16}
}
, {{23, -101}
, {26, -74}
, {-43, -52}
, {-82, -46}
, {-89, 18}
, {16, -41}
, {81, 47}
, {16, -17}
, {-85, 90}
, {-9, 68}
, {68, 56}
, {-57, 56}
, {45, -36}
, {-67, -85}
, {-28, 15}
, {-63, -42}
, {48, 38}
, {49, -46}
, {-59, 27}
, {-13, -56}
, {-17, -78}
, {-5, -96}
, {-38, -73}
, {-79, -48}
, {-50, -37}
, {-30, 39}
, {-37, -4}
, {-27, 64}
, {26, 89}
, {5, 20}
, {58, 81}
, {-76, 50}
}
, {{-59, -64}
, {-61, 78}
, {-40, 64}
, {50, -62}
, {18, -12}
, {-9, -30}
, {19, -57}
, {32, -67}
, {-37, 67}
, {-19, -46}
, {-13, 70}
, {6, 76}
, {-34, -28}
, {9, 60}
, {-50, 85}
, {28, 75}
, {-83, 63}
, {57, -74}
, {72, 28}
, {71, 78}
, {47, 3}
, {38, 20}
, {35, -58}
, {-16, -61}
, {82, -14}
, {-50, 83}
, {49, 56}
, {28, 1}
, {12, -83}
, {33, 38}
, {86, 70}
, {-33, 42}
}
, {{-7, 12}
, {-91, 53}
, {-88, 26}
, {-60, 13}
, {4, 1}
, {36, -16}
, {-60, 40}
, {-70, 36}
, {-22, 84}
, {82, 52}
, {-8, -18}
, {70, -2}
, {-24, -88}
, {46, 85}
, {-81, -85}
, {20, 72}
, {28, 6}
, {12, -58}
, {-8, 30}
, {57, 76}
, {24, 25}
, {-66, 10}
, {-88, -19}
, {20, -16}
, {42, 54}
, {27, -99}
, {53, 74}
, {57, -54}
, {-47, 69}
, {56, 26}
, {-15, -48}
, {64, 61}
}
, {{16, -18}
, {-5, -22}
, {54, -49}
, {-38, -84}
, {-36, -3}
, {-54, 88}
, {94, -67}
, {13, -82}
, {5, 71}
, {-61, 13}
, {-55, 56}
, {74, 56}
, {25, -80}
, {8, -45}
, {-52, 0}
, {15, 28}
, {-50, 9}
, {6, -15}
, {-71, -73}
, {33, -24}
, {30, 52}
, {-82, -43}
, {75, -30}
, {-4, -44}
, {-26, 66}
, {85, -45}
, {68, 23}
, {88, 27}
, {89, -53}
, {14, 48}
, {56, -25}
, {-36, -49}
}
, {{52, 45}
, {-66, 5}
, {63, 12}
, {-42, -7}
, {-27, -24}
, {-38, 4}
, {28, -45}
, {-42, -3}
, {-76, 54}
, {-9, 37}
, {-94, -59}
, {85, -65}
, {24, -75}
, {65, -29}
, {65, 76}
, {-72, 83}
, {-89, -20}
, {-78, 29}
, {-69, -36}
, {19, -11}
, {21, -4}
, {-45, 17}
, {-43, 63}
, {5, 36}
, {72, 64}
, {-7, 55}
, {14, -26}
, {-68, 65}
, {-74, -38}
, {68, -82}
, {-25, 39}
, {90, 28}
}
, {{-7, -74}
, {-20, -36}
, {2, 8}
, {-73, -13}
, {85, 40}
, {-29, 27}
, {80, -3}
, {55, -17}
, {-10, 65}
, {-10, 77}
, {-1, 61}
, {-82, 38}
, {74, 15}
, {-59, -78}
, {31, -16}
, {-71, -78}
, {64, -30}
, {62, -83}
, {76, -18}
, {54, 21}
, {-7, 63}
, {-30, 35}
, {-79, 40}
, {48, 13}
, {-77, -15}
, {-31, -83}
, {-38, 28}
, {43, 55}
, {68, -19}
, {31, -49}
, {-88, 49}
, {-17, -21}
}
, {{-43, 22}
, {20, 81}
, {-69, -66}
, {38, -30}
, {67, 32}
, {-5, 8}
, {-53, 4}
, {-63, 66}
, {6, 83}
, {66, 88}
, {-65, -18}
, {1, -48}
, {75, 67}
, {-55, -5}
, {-30, 13}
, {-43, 28}
, {71, 75}
, {-86, -77}
, {-4, -16}
, {57, 75}
, {-9, 81}
, {-43, -27}
, {-65, -35}
, {62, -5}
, {-21, 87}
, {58, -68}
, {14, -4}
, {78, -40}
, {-19, 101}
, {66, 5}
, {-88, -65}
, {-19, -20}
}
, {{-51, -82}
, {-6, -2}
, {3, -1}
, {-11, -53}
, {49, 54}
, {-58, -88}
, {-36, 30}
, {40, -24}
, {-66, 89}
, {-63, 32}
, {-80, -22}
, {-50, -62}
, {55, 62}
, {4, 71}
, {-75, 12}
, {82, -15}
, {0, 33}
, {29, 49}
, {21, -19}
, {35, 17}
, {-62, 73}
, {84, 57}
, {-72, -36}
, {23, -13}
, {-15, 46}
, {-95, 1}
, {2, -83}
, {69, 24}
, {-22, -13}
, {74, 35}
, {30, -58}
, {91, 30}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE