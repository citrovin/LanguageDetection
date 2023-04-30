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


const int16_t conv1d_30_bias[CONV_FILTERS] = {0, -3, -1, -6, 7, 0, 0, -7, 0, -2, 6, 2, 0, -5, 0, -1, 0, -4, 0, 2, 1, 0, -1, -4, -2, -1, 1, 0, -8, -1, -2, 1, 6, 0, -4, -3, 0, -2, -5, 1, 0, 0, 8, -5, 2, 3, 0, -2, -5, -2, -3, -2, -3, 4, -1, 0, -1, 1, -1, -2, 2, 9, -3, 1}
;

const int16_t conv1d_30_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-31, 49}
, {39, -55}
, {-60, 28}
, {49, -1}
, {47, -32}
, {67, -70}
, {-3, 70}
, {-67, 58}
, {72, -43}
, {-59, 18}
, {-92, -62}
, {3, -41}
, {-56, -15}
, {49, 64}
, {-81, 52}
, {-86, 40}
, {-55, -63}
, {53, 27}
, {-25, 33}
, {85, 49}
, {-62, -14}
, {-18, 84}
, {7, 51}
, {-16, 45}
, {-41, -45}
, {-36, 3}
, {49, -28}
, {-5, 72}
, {76, 0}
, {46, 3}
, {-17, -50}
, {67, 68}
}
, {{52, 81}
, {69, -20}
, {-70, 53}
, {16, -92}
, {-72, 88}
, {-21, -34}
, {93, -15}
, {60, -60}
, {-79, -14}
, {-26, 54}
, {43, -34}
, {-56, -9}
, {28, -80}
, {70, -33}
, {-56, 29}
, {65, 89}
, {71, 12}
, {30, -26}
, {46, 75}
, {-57, 16}
, {-3, -2}
, {-8, 80}
, {0, -76}
, {8, -20}
, {82, -76}
, {-35, -8}
, {-51, 52}
, {52, -42}
, {-57, -47}
, {-63, 0}
, {-65, -17}
, {16, 44}
}
, {{29, 71}
, {2, 17}
, {70, 90}
, {-48, -57}
, {-27, 21}
, {-9, -85}
, {-55, 9}
, {-50, -43}
, {-47, 72}
, {-85, -7}
, {24, 47}
, {-38, -72}
, {47, 80}
, {-59, 28}
, {-44, -64}
, {18, 67}
, {27, -36}
, {33, 80}
, {71, 82}
, {-56, 56}
, {-53, 1}
, {31, 79}
, {-88, 57}
, {-23, 81}
, {76, 70}
, {28, 53}
, {60, -52}
, {-11, -18}
, {-4, -23}
, {86, -2}
, {38, 0}
, {53, 47}
}
, {{-92, 51}
, {27, -31}
, {-62, 67}
, {65, -57}
, {-1, -88}
, {-42, 4}
, {-63, 58}
, {75, 29}
, {-56, -16}
, {41, 73}
, {-13, -3}
, {34, -77}
, {2, 35}
, {-8, 67}
, {32, -34}
, {65, -19}
, {-17, 35}
, {7, -48}
, {-35, 22}
, {85, -18}
, {-31, 1}
, {-87, 4}
, {-33, 26}
, {-62, 38}
, {13, -26}
, {-15, -29}
, {-54, -3}
, {-81, 45}
, {19, 20}
, {-81, -16}
, {64, -73}
, {0, 32}
}
, {{58, -37}
, {-33, 21}
, {-14, 31}
, {-31, 19}
, {-52, 57}
, {13, -2}
, {-39, -58}
, {21, -48}
, {20, -9}
, {-64, 15}
, {-70, -52}
, {-73, -52}
, {29, 3}
, {23, -47}
, {-27, 39}
, {32, 34}
, {-42, 35}
, {89, -62}
, {-10, -1}
, {54, 35}
, {-25, 57}
, {34, -82}
, {68, -44}
, {-21, 12}
, {27, -57}
, {-22, -9}
, {56, -60}
, {-46, -28}
, {91, 48}
, {-55, 66}
, {-11, -28}
, {-68, 51}
}
, {{-23, -49}
, {-33, -55}
, {96, 67}
, {35, 77}
, {8, 69}
, {82, 31}
, {19, 43}
, {34, 67}
, {10, 20}
, {79, 82}
, {42, -60}
, {21, -4}
, {30, -49}
, {-40, 57}
, {-56, -25}
, {5, -37}
, {-48, 48}
, {-88, -43}
, {-3, 2}
, {-77, 43}
, {-18, -86}
, {-34, -6}
, {63, 76}
, {-36, 27}
, {67, -52}
, {-25, 16}
, {91, 75}
, {48, 72}
, {-83, 47}
, {12, -9}
, {-11, -7}
, {1, -48}
}
, {{-34, 46}
, {57, 57}
, {-11, -78}
, {-40, 0}
, {15, -29}
, {86, 73}
, {84, 48}
, {61, -57}
, {85, 1}
, {-61, 48}
, {-74, -18}
, {-77, 70}
, {-65, -47}
, {74, 6}
, {17, 57}
, {-81, 15}
, {40, -29}
, {10, -17}
, {17, -56}
, {-50, -72}
, {-25, -57}
, {-28, 20}
, {89, -19}
, {65, -6}
, {-56, 60}
, {7, 73}
, {-16, 44}
, {-71, 12}
, {-34, 86}
, {17, 37}
, {-44, 2}
, {-39, 14}
}
, {{87, -3}
, {17, -14}
, {-66, 14}
, {-13, -28}
, {-66, -76}
, {-43, 5}
, {-22, -64}
, {-16, -78}
, {-87, -84}
, {-66, -19}
, {52, -91}
, {66, 50}
, {80, 42}
, {0, 27}
, {-57, 14}
, {76, 37}
, {88, 19}
, {69, -10}
, {58, -41}
, {-25, 27}
, {47, 48}
, {-63, -66}
, {-81, 66}
, {23, -15}
, {19, 53}
, {45, 33}
, {83, 3}
, {62, -79}
, {19, -35}
, {-34, -92}
, {-50, -1}
, {-33, -22}
}
, {{83, -57}
, {-62, 29}
, {-67, -1}
, {-4, -77}
, {57, -36}
, {-2, -11}
, {60, 87}
, {-43, 67}
, {32, 18}
, {80, -25}
, {90, -25}
, {40, 70}
, {13, -19}
, {45, 85}
, {-77, 53}
, {-9, 30}
, {-77, 80}
, {40, 71}
, {-38, 76}
, {87, -79}
, {-31, 20}
, {-21, 55}
, {-86, 17}
, {-61, -3}
, {-85, 65}
, {85, 75}
, {-45, 3}
, {-75, -64}
, {40, 36}
, {26, -34}
, {-49, 46}
, {-80, 8}
}
, {{7, -67}
, {-8, 31}
, {60, -92}
, {-83, 51}
, {-27, 54}
, {23, -95}
, {14, -43}
, {-11, -39}
, {75, 77}
, {26, 29}
, {-61, 62}
, {-27, 65}
, {74, -40}
, {-13, 16}
, {21, 1}
, {-87, -82}
, {42, 61}
, {71, -78}
, {36, -25}
, {-27, -65}
, {-85, 62}
, {-37, 53}
, {82, 35}
, {-48, -77}
, {45, -14}
, {-18, -75}
, {32, 61}
, {-90, -49}
, {-90, -51}
, {-47, 60}
, {81, -5}
, {23, 11}
}
, {{-58, -56}
, {41, 88}
, {-61, 26}
, {-13, 33}
, {-27, 87}
, {-76, -12}
, {90, -52}
, {86, -42}
, {-29, -24}
, {90, -17}
, {58, -7}
, {7, -66}
, {81, -79}
, {-60, -18}
, {-81, 0}
, {50, 33}
, {-10, -68}
, {13, 15}
, {2, -83}
, {68, 67}
, {-72, -76}
, {46, 88}
, {11, -28}
, {-85, -36}
, {-11, 19}
, {62, 0}
, {-15, 2}
, {14, 66}
, {-36, -78}
, {-27, 15}
, {29, 70}
, {-60, -75}
}
, {{-64, -61}
, {89, 23}
, {-22, 82}
, {39, -65}
, {92, 52}
, {-82, -9}
, {30, 41}
, {-35, 37}
, {-49, 62}
, {60, -29}
, {86, 4}
, {77, 67}
, {-75, -33}
, {-66, 23}
, {-71, 84}
, {7, 56}
, {-33, -65}
, {-49, -29}
, {-17, 1}
, {61, 35}
, {80, -38}
, {-4, -61}
, {40, 81}
, {56, -42}
, {34, 74}
, {-49, 78}
, {-29, -20}
, {17, 14}
, {61, 61}
, {-1, -77}
, {-65, 84}
, {5, 8}
}
, {{-62, -36}
, {83, 70}
, {37, 22}
, {-65, 0}
, {-14, -30}
, {26, 11}
, {-51, 75}
, {84, -67}
, {48, 63}
, {71, -49}
, {-55, -58}
, {14, -38}
, {-92, -79}
, {39, 36}
, {-7, -3}
, {-18, -46}
, {88, 42}
, {25, 82}
, {-67, -38}
, {-47, 19}
, {65, 87}
, {59, 39}
, {44, -32}
, {-77, -66}
, {28, -43}
, {-16, 2}
, {18, -76}
, {-72, -13}
, {-43, 61}
, {57, -62}
, {-44, 50}
, {-28, -91}
}
, {{6, 56}
, {9, -58}
, {-15, 47}
, {-54, -76}
, {0, -19}
, {21, -36}
, {76, 29}
, {4, -30}
, {37, -69}
, {21, 87}
, {-22, 74}
, {36, 38}
, {-23, 55}
, {76, -5}
, {56, -83}
, {-81, -54}
, {-52, 3}
, {-68, -24}
, {-24, 41}
, {20, -59}
, {5, 30}
, {-67, 34}
, {13, -66}
, {-63, -80}
, {22, 63}
, {-82, 27}
, {-1, 45}
, {-25, -43}
, {54, -41}
, {38, 55}
, {3, 70}
, {-80, 63}
}
, {{11, 3}
, {-46, 62}
, {-83, 92}
, {-29, -89}
, {72, -59}
, {71, 75}
, {-16, 65}
, {-38, -23}
, {90, 46}
, {-5, 28}
, {32, -76}
, {26, 43}
, {63, 23}
, {69, 21}
, {-45, -18}
, {36, 78}
, {7, -83}
, {-22, -16}
, {32, 66}
, {-10, -2}
, {59, -36}
, {18, -70}
, {90, 87}
, {23, 39}
, {-81, -64}
, {83, -73}
, {-2, -75}
, {7, -16}
, {-6, 45}
, {77, 6}
, {17, 88}
, {12, -23}
}
, {{-2, 63}
, {-13, -85}
, {74, -38}
, {2, 70}
, {-50, -73}
, {-10, 77}
, {-56, -15}
, {63, -23}
, {-34, -91}
, {30, -19}
, {0, -44}
, {-8, -23}
, {33, -85}
, {50, 51}
, {63, 83}
, {80, -83}
, {-54, 39}
, {15, -39}
, {73, 33}
, {21, -64}
, {25, -4}
, {76, -36}
, {57, 36}
, {1, -13}
, {47, -42}
, {-70, 26}
, {-40, -89}
, {61, -2}
, {8, 91}
, {77, 74}
, {-50, -38}
, {1, -57}
}
, {{-5, -36}
, {-58, 20}
, {57, -84}
, {15, 65}
, {47, 20}
, {-76, 62}
, {33, 81}
, {-6, -56}
, {-70, -21}
, {47, -11}
, {-54, 76}
, {9, 25}
, {-80, -64}
, {-62, -60}
, {-48, -38}
, {12, 14}
, {26, 46}
, {-76, 6}
, {-54, -5}
, {-71, -10}
, {15, 40}
, {-64, 35}
, {-29, -45}
, {92, -16}
, {-67, 97}
, {-5, 29}
, {55, -33}
, {-43, -87}
, {-27, 44}
, {-77, 67}
, {-7, 71}
, {21, 84}
}
, {{-64, 50}
, {-20, -8}
, {46, -54}
, {-55, -9}
, {-83, -54}
, {3, -15}
, {37, -12}
, {10, -58}
, {-73, 19}
, {22, -95}
, {-70, 37}
, {74, -31}
, {44, -75}
, {-33, 17}
, {74, -62}
, {-85, 34}
, {-77, 31}
, {0, 43}
, {-28, 12}
, {58, 41}
, {11, -15}
, {10, -74}
, {-71, 11}
, {-5, 8}
, {58, -18}
, {-84, -32}
, {-41, 64}
, {64, 53}
, {-16, 41}
, {14, -37}
, {-69, 57}
, {-25, 66}
}
, {{-20, 22}
, {26, -43}
, {77, -23}
, {8, -44}
, {2, -39}
, {-75, 58}
, {64, 84}
, {12, -46}
, {-36, 66}
, {21, -55}
, {-82, 7}
, {28, -9}
, {-57, -1}
, {-72, 4}
, {24, 32}
, {16, -6}
, {78, 39}
, {-2, 74}
, {5, -80}
, {-85, 54}
, {-42, -73}
, {20, 76}
, {37, 85}
, {-80, -43}
, {71, -20}
, {45, -76}
, {-17, 80}
, {1, 41}
, {-25, 10}
, {86, -54}
, {-28, 49}
, {70, -12}
}
, {{0, 36}
, {-34, 67}
, {80, -45}
, {-79, 11}
, {85, 92}
, {-89, 69}
, {70, 79}
, {46, -14}
, {10, -42}
, {-59, 41}
, {-47, 46}
, {-52, 51}
, {-65, 55}
, {88, 26}
, {-68, 0}
, {0, -34}
, {-6, -7}
, {-12, -23}
, {9, -33}
, {62, 61}
, {-62, -39}
, {59, -72}
, {-56, 16}
, {49, 0}
, {-17, -33}
, {-40, 63}
, {43, -62}
, {35, -45}
, {-56, 85}
, {46, 63}
, {74, -31}
, {3, -67}
}
, {{17, 65}
, {-19, -74}
, {64, -12}
, {-79, -59}
, {-28, 71}
, {-83, 7}
, {-51, -56}
, {21, 72}
, {19, 86}
, {-7, 38}
, {-21, 44}
, {54, -81}
, {50, 19}
, {-70, -80}
, {30, -34}
, {52, 55}
, {-26, -63}
, {48, 4}
, {19, -29}
, {74, 4}
, {73, -26}
, {-33, 26}
, {-38, 87}
, {54, 81}
, {-5, -6}
, {3, -35}
, {-18, 19}
, {-47, -46}
, {-3, -2}
, {1, 67}
, {-87, -43}
, {28, 0}
}
, {{-48, 53}
, {39, 18}
, {-57, 0}
, {-11, 92}
, {-85, 10}
, {-33, 0}
, {-7, -18}
, {34, -19}
, {-62, -10}
, {-5, -8}
, {78, -47}
, {-22, -50}
, {68, 24}
, {-11, 53}
, {12, 89}
, {31, 90}
, {33, 62}
, {77, 43}
, {77, -69}
, {86, 74}
, {85, -14}
, {-21, 29}
, {1, 32}
, {-46, -67}
, {32, 65}
, {27, -29}
, {60, -51}
, {-34, 3}
, {46, 60}
, {-43, -45}
, {52, 30}
, {3, -88}
}
, {{-55, -8}
, {64, -3}
, {0, 73}
, {53, 55}
, {-44, 82}
, {-73, -68}
, {39, 35}
, {-83, -49}
, {29, 6}
, {13, 36}
, {73, -12}
, {-72, 69}
, {-1, -92}
, {-7, 59}
, {-13, 45}
, {14, -22}
, {-14, -56}
, {84, 88}
, {-57, 1}
, {-22, 8}
, {48, 25}
, {48, 9}
, {-79, -48}
, {19, 35}
, {91, -11}
, {-40, -34}
, {45, 48}
, {25, 21}
, {-92, 86}
, {9, -12}
, {-40, 62}
, {-13, 36}
}
, {{86, 0}
, {88, 53}
, {56, 76}
, {-86, 63}
, {13, -43}
, {-83, -56}
, {58, -26}
, {43, -29}
, {-78, -25}
, {59, -6}
, {-14, 38}
, {-9, 58}
, {64, 82}
, {75, -40}
, {-23, 35}
, {-83, -37}
, {-7, -19}
, {-36, 46}
, {-55, 40}
, {-21, 0}
, {57, -59}
, {42, -29}
, {-78, -91}
, {-76, -33}
, {-19, 0}
, {-76, -6}
, {21, 36}
, {-81, -33}
, {34, 49}
, {61, 42}
, {54, -26}
, {72, -72}
}
, {{-29, 15}
, {-67, 45}
, {71, 97}
, {80, -1}
, {43, 34}
, {21, 72}
, {52, -56}
, {27, -4}
, {1, 67}
, {-45, 29}
, {84, 8}
, {14, 65}
, {18, 87}
, {54, 34}
, {15, -51}
, {-22, 28}
, {82, -78}
, {-54, 53}
, {-6, 32}
, {-63, -5}
, {-60, 26}
, {56, -70}
, {-30, -84}
, {-38, 58}
, {-3, 77}
, {0, -42}
, {4, -65}
, {-56, 53}
, {52, 50}
, {78, -30}
, {0, 96}
, {-52, -26}
}
, {{-25, -8}
, {86, 95}
, {-81, -57}
, {-4, 72}
, {38, -2}
, {-15, -74}
, {47, -77}
, {74, -62}
, {82, 60}
, {13, -61}
, {69, -61}
, {2, 35}
, {9, -84}
, {-86, -30}
, {-51, 30}
, {22, -47}
, {61, -87}
, {5, 62}
, {-85, 41}
, {19, -80}
, {-44, 29}
, {-4, -60}
, {-73, -37}
, {37, -82}
, {10, 39}
, {49, 81}
, {-26, -90}
, {86, 38}
, {-5, -87}
, {2, 1}
, {0, -78}
, {27, -67}
}
, {{51, 62}
, {35, 44}
, {-83, 42}
, {-20, 10}
, {16, -61}
, {-33, 67}
, {72, -31}
, {-13, -10}
, {-80, 39}
, {-71, 25}
, {89, 32}
, {-4, 6}
, {-68, 19}
, {-9, -58}
, {55, -63}
, {-19, -49}
, {45, -45}
, {15, 69}
, {-83, 78}
, {-86, 31}
, {44, -1}
, {-65, 32}
, {31, 79}
, {82, 2}
, {-6, 58}
, {83, 46}
, {13, -43}
, {14, 44}
, {-27, -66}
, {22, 75}
, {-8, -48}
, {68, -1}
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
, {-93, -72}
, {52, -55}
, {8, 16}
, {13, 80}
, {-76, 63}
, {45, 66}
, {-64, 22}
, {34, 13}
, {69, 58}
, {-44, 5}
, {7, -88}
, {-33, -54}
, {-37, -42}
, {22, -6}
, {-63, 45}
, {-44, 26}
, {-21, 8}
, {27, -87}
, {56, -14}
, {-30, 77}
, {-57, 52}
, {-73, 8}
, {-34, 5}
, {87, -31}
, {21, -11}
, {-60, 62}
, {-74, -35}
, {22, -62}
, {80, -31}
}
, {{53, -6}
, {-60, 6}
, {-78, 15}
, {89, 80}
, {38, -57}
, {-88, 16}
, {-93, -16}
, {53, -39}
, {79, 66}
, {-22, -29}
, {-15, 78}
, {-3, -74}
, {72, -56}
, {-10, 15}
, {72, 26}
, {7, -5}
, {31, 48}
, {73, 88}
, {23, 85}
, {82, 54}
, {-54, -22}
, {-82, 52}
, {34, -18}
, {-60, 4}
, {94, 18}
, {52, 38}
, {39, -64}
, {-85, 70}
, {-52, -23}
, {0, -60}
, {29, 73}
, {8, 48}
}
, {{27, 23}
, {-35, -23}
, {-31, -16}
, {0, -25}
, {6, -53}
, {79, 86}
, {26, -77}
, {-18, -24}
, {66, -23}
, {31, 38}
, {45, 8}
, {41, -33}
, {22, 2}
, {48, -5}
, {-85, 70}
, {69, -4}
, {-49, -33}
, {-82, -56}
, {-67, -7}
, {-22, 40}
, {-69, -10}
, {26, 6}
, {15, -4}
, {-81, -76}
, {-51, -86}
, {64, -44}
, {-61, 42}
, {-39, 89}
, {85, -75}
, {35, 74}
, {-30, -83}
, {62, 7}
}
, {{46, 88}
, {48, 17}
, {-93, 87}
, {-50, -17}
, {4, 59}
, {-34, 82}
, {6, -40}
, {36, 60}
, {76, 14}
, {76, 47}
, {-86, -64}
, {-34, 65}
, {70, 17}
, {52, 13}
, {9, 70}
, {-18, -8}
, {-9, 21}
, {28, -86}
, {-70, 46}
, {2, -35}
, {74, 14}
, {-18, 54}
, {-38, 52}
, {-21, -47}
, {17, 54}
, {-70, -40}
, {-10, 44}
, {39, 55}
, {-81, -38}
, {-52, 75}
, {-48, -45}
, {-19, 53}
}
, {{16, -26}
, {38, 73}
, {22, -81}
, {46, 73}
, {81, -29}
, {-18, -10}
, {27, 71}
, {-78, -45}
, {89, -32}
, {-2, -78}
, {-35, 74}
, {-24, -19}
, {-43, 14}
, {30, -17}
, {-38, 45}
, {-66, -34}
, {-81, -42}
, {-60, 52}
, {61, -48}
, {-39, -63}
, {46, 24}
, {30, -87}
, {-52, -35}
, {41, -70}
, {-14, -20}
, {38, -21}
, {-68, 15}
, {-2, 13}
, {51, 11}
, {-79, 44}
, {39, -27}
, {-17, -23}
}
, {{-71, -40}
, {-6, 19}
, {-67, -83}
, {59, 8}
, {-18, 29}
, {70, 45}
, {-20, 54}
, {62, 58}
, {59, 42}
, {-76, -49}
, {-77, 28}
, {24, 2}
, {70, -43}
, {-26, -53}
, {-61, 15}
, {52, 35}
, {3, 52}
, {41, 22}
, {0, 49}
, {-82, 43}
, {-85, 43}
, {73, -20}
, {24, 14}
, {41, 58}
, {81, 7}
, {-12, 83}
, {67, -40}
, {36, -81}
, {-5, -57}
, {-25, 15}
, {77, -51}
, {-49, -43}
}
, {{22, 6}
, {-54, -12}
, {30, 36}
, {77, -31}
, {27, 30}
, {74, -34}
, {-12, -10}
, {78, 3}
, {-93, -48}
, {-90, 83}
, {-51, 25}
, {-25, -32}
, {50, 66}
, {-69, 2}
, {28, 46}
, {-14, 63}
, {-6, 13}
, {-4, 47}
, {-43, -25}
, {13, 16}
, {38, 60}
, {-82, 38}
, {-6, 72}
, {33, -20}
, {-25, -32}
, {-62, 15}
, {6, -21}
, {69, 60}
, {-85, 45}
, {-11, -91}
, {-40, 13}
, {36, -25}
}
, {{82, -4}
, {57, -24}
, {84, 78}
, {68, 64}
, {-62, -43}
, {-85, 83}
, {-46, 79}
, {-69, 73}
, {-67, 23}
, {43, -33}
, {73, -46}
, {-9, 20}
, {59, 10}
, {73, -27}
, {-29, -88}
, {-19, -8}
, {57, 77}
, {-69, 38}
, {19, 35}
, {-28, 84}
, {80, 66}
, {12, -89}
, {-15, -89}
, {-44, -29}
, {-10, 8}
, {-10, -12}
, {-65, 67}
, {-82, 25}
, {93, 1}
, {79, 11}
, {66, 48}
, {11, 47}
}
, {{61, -20}
, {75, 74}
, {-29, 42}
, {26, 36}
, {-12, 48}
, {45, 48}
, {42, -7}
, {-79, -59}
, {50, 1}
, {-36, -71}
, {-64, 85}
, {68, 57}
, {-55, -75}
, {11, 78}
, {-88, -13}
, {37, 68}
, {18, 51}
, {8, -2}
, {-63, 75}
, {-60, -80}
, {-74, -70}
, {36, -45}
, {78, -57}
, {-45, 31}
, {-80, 45}
, {-9, -43}
, {1, -38}
, {-83, -23}
, {-77, -41}
, {59, 22}
, {62, -64}
, {-27, -64}
}
, {{34, 0}
, {16, -71}
, {-10, 84}
, {-75, -31}
, {62, -7}
, {-32, -24}
, {5, 63}
, {46, 19}
, {-42, 68}
, {-38, 18}
, {-51, 26}
, {-4, -90}
, {-25, -42}
, {-10, -5}
, {-81, 16}
, {-84, -62}
, {40, 74}
, {5, 58}
, {-2, -45}
, {12, -26}
, {2, -21}
, {5, -62}
, {-62, 85}
, {-18, 27}
, {1, -17}
, {-87, -60}
, {47, 52}
, {-49, -22}
, {73, 79}
, {77, 59}
, {-4, 50}
, {-67, 17}
}
, {{78, 31}
, {-10, 86}
, {-36, -77}
, {7, 74}
, {-38, -88}
, {73, -48}
, {14, -85}
, {64, 65}
, {38, -87}
, {-47, 16}
, {-70, -24}
, {-53, -27}
, {-12, -13}
, {67, -57}
, {-33, 14}
, {-67, -11}
, {3, -57}
, {-50, -47}
, {21, -31}
, {-47, -52}
, {55, -41}
, {22, 52}
, {-79, -71}
, {-32, -39}
, {34, 45}
, {-89, -43}
, {-51, -88}
, {-57, -3}
, {55, 11}
, {-58, 71}
, {2, 12}
, {-27, 60}
}
, {{-89, -94}
, {-37, 22}
, {-5, 83}
, {-38, -53}
, {-60, -72}
, {56, 13}
, {22, 58}
, {-43, 6}
, {-45, 52}
, {-86, 75}
, {-35, -71}
, {48, -82}
, {-59, -70}
, {-84, 68}
, {44, 26}
, {85, -84}
, {-58, -10}
, {-31, 27}
, {-2, -50}
, {-33, -25}
, {37, -34}
, {54, 36}
, {14, 26}
, {-20, 81}
, {31, 75}
, {82, -77}
, {11, -49}
, {45, 71}
, {60, 40}
, {-84, -65}
, {49, -4}
, {81, -22}
}
, {{79, -36}
, {7, -26}
, {50, 33}
, {-54, -28}
, {-42, 95}
, {47, 82}
, {59, -54}
, {15, -48}
, {77, -40}
, {0, 34}
, {-45, -75}
, {-32, 33}
, {42, 58}
, {-54, -68}
, {-51, 46}
, {56, -8}
, {12, -7}
, {70, -5}
, {-18, -18}
, {-35, 4}
, {0, -79}
, {86, 57}
, {2, 14}
, {48, -46}
, {-69, -31}
, {65, -21}
, {25, -48}
, {87, 14}
, {10, -45}
, {58, -29}
, {49, -16}
, {-32, 77}
}
, {{30, -22}
, {-77, 41}
, {36, -36}
, {32, -83}
, {6, -49}
, {-49, -51}
, {75, 67}
, {-63, -14}
, {-14, -77}
, {31, 33}
, {-88, 45}
, {-5, -81}
, {-46, -25}
, {-22, 20}
, {-53, 16}
, {-11, -48}
, {-53, -80}
, {32, 49}
, {-15, 59}
, {39, 34}
, {46, -74}
, {64, 7}
, {-11, 21}
, {7, 78}
, {-78, 47}
, {-65, 78}
, {45, 16}
, {34, -59}
, {-72, -37}
, {44, -56}
, {30, -81}
, {-17, 64}
}
, {{-27, 40}
, {31, 77}
, {-1, -54}
, {28, -6}
, {-5, -40}
, {63, 76}
, {76, -4}
, {-59, -79}
, {54, -57}
, {77, 14}
, {-37, 33}
, {40, -63}
, {76, 43}
, {-72, 4}
, {-72, 96}
, {27, -57}
, {-48, -58}
, {62, -43}
, {-65, 0}
, {30, -61}
, {9, -54}
, {32, 6}
, {-15, -24}
, {-29, -27}
, {-70, 79}
, {78, -47}
, {-11, 40}
, {8, -21}
, {16, 22}
, {-25, 13}
, {2, 3}
, {-44, 3}
}
, {{-94, 14}
, {17, -30}
, {-12, -44}
, {-87, -77}
, {-23, 54}
, {5, -60}
, {68, 30}
, {-6, -50}
, {-84, 24}
, {-63, -10}
, {42, -46}
, {53, 64}
, {0, 62}
, {25, -21}
, {35, -52}
, {45, -2}
, {-22, -18}
, {25, 39}
, {19, -76}
, {-30, 44}
, {-8, -16}
, {-86, 42}
, {-65, 43}
, {23, 50}
, {-76, -88}
, {38, 28}
, {2, -86}
, {-11, 61}
, {-91, 51}
, {-24, 8}
, {63, 64}
, {-35, 19}
}
, {{83, -19}
, {-33, -89}
, {-67, -82}
, {3, -2}
, {26, -8}
, {-56, -63}
, {-53, -41}
, {22, 10}
, {54, -56}
, {-91, 15}
, {-48, 76}
, {46, 11}
, {-54, 71}
, {23, -96}
, {-64, 78}
, {40, 80}
, {31, -48}
, {30, -54}
, {12, -59}
, {63, -19}
, {-85, -53}
, {53, -60}
, {34, 60}
, {66, -16}
, {48, 44}
, {31, -61}
, {35, -79}
, {73, 2}
, {79, -39}
, {-5, -32}
, {31, 74}
, {63, 39}
}
, {{-12, -57}
, {68, 41}
, {12, 73}
, {34, 12}
, {6, -59}
, {53, 0}
, {34, 22}
, {-28, -11}
, {-58, -56}
, {31, 11}
, {-56, -23}
, {-59, 88}
, {79, 6}
, {-26, -27}
, {-61, -60}
, {25, 35}
, {-37, -38}
, {-77, 75}
, {-45, -13}
, {73, 35}
, {-69, 55}
, {35, 38}
, {87, 80}
, {31, 33}
, {63, -10}
, {-2, 93}
, {-15, 27}
, {-3, 31}
, {30, -76}
, {-76, 7}
, {-57, -65}
, {0, 39}
}
, {{49, -15}
, {-47, 55}
, {75, 19}
, {-23, 9}
, {-35, -6}
, {-65, 79}
, {-60, -38}
, {-46, 85}
, {-64, -74}
, {-53, 22}
, {48, -14}
, {-5, -70}
, {-81, 78}
, {57, -39}
, {37, 8}
, {-1, -62}
, {6, -83}
, {-33, 55}
, {84, 90}
, {44, -35}
, {-7, 19}
, {86, -7}
, {-6, 60}
, {11, 98}
, {72, 81}
, {75, 32}
, {-68, 22}
, {76, -2}
, {-56, -58}
, {78, -40}
, {-6, 4}
, {58, 41}
}
, {{69, 51}
, {-66, 60}
, {0, 28}
, {-36, -28}
, {-78, 85}
, {-2, 81}
, {-28, -44}
, {17, 45}
, {70, 75}
, {83, -12}
, {66, -12}
, {31, -15}
, {-21, 63}
, {66, 66}
, {32, 9}
, {-45, -66}
, {-10, 51}
, {35, 88}
, {-70, 50}
, {-92, 81}
, {43, 70}
, {32, -22}
, {-39, -28}
, {-50, -17}
, {17, 18}
, {8, 8}
, {-43, -49}
, {64, -82}
, {9, -44}
, {25, 4}
, {59, 94}
, {-51, 27}
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
, {{48, 3}
, {78, -74}
, {24, 7}
, {41, -30}
, {-66, -58}
, {66, -84}
, {29, 0}
, {-30, 91}
, {-76, -40}
, {26, 0}
, {-47, -77}
, {69, -30}
, {53, -28}
, {3, 41}
, {-4, -50}
, {-65, 27}
, {49, -21}
, {-91, -37}
, {-83, -86}
, {58, -28}
, {31, 59}
, {88, 9}
, {62, 57}
, {-56, 45}
, {38, -90}
, {-21, 46}
, {6, -30}
, {-9, 0}
, {-77, 38}
, {19, -54}
, {-85, -47}
, {0, 59}
}
, {{-16, -52}
, {28, 74}
, {84, -72}
, {-64, -51}
, {22, 7}
, {29, -23}
, {33, 3}
, {85, 80}
, {71, -38}
, {-66, 76}
, {-34, 86}
, {-26, -5}
, {-44, 32}
, {-38, -79}
, {29, 84}
, {-49, -82}
, {29, 67}
, {37, -18}
, {70, -73}
, {-20, 11}
, {-20, 39}
, {-42, -71}
, {18, -1}
, {50, 57}
, {33, 37}
, {37, 82}
, {70, 24}
, {-37, 63}
, {84, 18}
, {32, -14}
, {53, 47}
, {54, -77}
}
, {{-57, -38}
, {3, 8}
, {-20, 8}
, {-17, 19}
, {35, 35}
, {56, 29}
, {74, 28}
, {-4, -40}
, {13, -85}
, {67, 6}
, {-22, 59}
, {-58, -59}
, {81, -69}
, {-65, -28}
, {-44, 20}
, {-46, -3}
, {86, -58}
, {30, -63}
, {-11, -7}
, {-42, 58}
, {-68, 90}
, {-34, -23}
, {-30, 86}
, {-79, -8}
, {86, -23}
, {-67, 87}
, {-44, -34}
, {61, 84}
, {-21, 77}
, {-6, 41}
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
, {{-6, 46}
, {37, 64}
, {69, 36}
, {0, -17}
, {75, 25}
, {-88, 60}
, {21, 88}
, {47, 4}
, {-9, 78}
, {-80, -51}
, {-78, 2}
, {-32, -87}
, {22, 33}
, {-82, 4}
, {-19, -67}
, {-4, 63}
, {-89, 36}
, {-74, 87}
, {79, 22}
, {-4, -85}
, {-27, -63}
, {0, -48}
, {20, -68}
, {-28, 49}
, {-61, 1}
, {-56, 80}
, {-67, -73}
, {14, 47}
, {-2, 48}
, {-44, 3}
, {37, -64}
, {83, 46}
}
, {{52, 29}
, {-19, 60}
, {-40, -59}
, {-16, 50}
, {27, -55}
, {25, -34}
, {-66, 84}
, {46, -26}
, {28, 16}
, {-41, 75}
, {-76, 47}
, {14, -71}
, {14, 28}
, {34, -82}
, {-23, 77}
, {-58, 45}
, {-14, 30}
, {44, 4}
, {9, -51}
, {63, 1}
, {-17, -63}
, {-23, -3}
, {71, 36}
, {9, 42}
, {67, 37}
, {80, 26}
, {-76, -18}
, {-19, -30}
, {57, -61}
, {-54, 28}
, {1, 23}
, {-23, -70}
}
, {{-51, 29}
, {-42, 72}
, {64, 66}
, {11, 70}
, {-80, -51}
, {30, 17}
, {-60, 15}
, {15, -10}
, {56, -15}
, {7, 18}
, {-6, -3}
, {14, 8}
, {78, -91}
, {52, -33}
, {37, -64}
, {-18, 24}
, {-55, 31}
, {-7, -26}
, {-40, 1}
, {-3, -66}
, {-72, -79}
, {2, 22}
, {30, -85}
, {42, 29}
, {-38, -25}
, {4, 42}
, {40, 5}
, {-88, -72}
, {83, -50}
, {-49, 68}
, {-45, -38}
, {48, -16}
}
, {{23, -92}
, {35, -65}
, {-52, -52}
, {-86, -49}
, {-79, 20}
, {16, -41}
, {89, 60}
, {8, -9}
, {-82, 84}
, {-15, 67}
, {65, 52}
, {-63, 39}
, {54, -28}
, {-70, -86}
, {-26, 14}
, {-66, -39}
, {48, 38}
, {40, -51}
, {-71, 24}
, {-13, -56}
, {-25, -78}
, {-14, -90}
, {-42, -77}
, {-90, -29}
, {-59, -46}
, {-40, 32}
, {-37, -4}
, {-31, 64}
, {35, 89}
, {1, 20}
, {58, 81}
, {-76, 50}
}
, {{-60, -68}
, {-50, 93}
, {-40, 62}
, {47, -65}
, {29, -4}
, {-4, -26}
, {23, -54}
, {30, -63}
, {-37, 67}
, {-8, -41}
, {-11, 71}
, {7, 77}
, {-33, -26}
, {10, 61}
, {-50, 84}
, {28, 76}
, {-83, 63}
, {55, -76}
, {71, 27}
, {59, 69}
, {44, 1}
, {41, 23}
, {38, -55}
, {-10, -55}
, {79, -18}
, {-48, 85}
, {42, 51}
, {34, 6}
, {34, -68}
, {35, 40}
, {81, 65}
, {-25, 51}
}
, {{-5, 13}
, {-97, 47}
, {-93, 24}
, {-57, 15}
, {0, 3}
, {32, -15}
, {-65, 40}
, {-81, 28}
, {-20, 86}
, {77, 50}
, {-9, -15}
, {71, -3}
, {-23, -85}
, {42, 74}
, {-77, -83}
, {23, 75}
, {28, 6}
, {15, -55}
, {-6, 30}
, {66, 73}
, {25, 25}
, {-66, 11}
, {-88, -21}
, {21, -10}
, {43, 56}
, {26, -101}
, {64, 64}
, {56, -56}
, {-56, 66}
, {58, 26}
, {-12, -47}
, {64, 61}
}
, {{16, -16}
, {-13, -27}
, {54, -48}
, {-37, -84}
, {-41, -14}
, {-57, 84}
, {86, -69}
, {17, -79}
, {5, 71}
, {-68, 9}
, {-57, 55}
, {72, 54}
, {26, -78}
, {5, -51}
, {-54, -1}
, {13, 27}
, {-50, 9}
, {6, -16}
, {-68, -70}
, {31, -18}
, {31, 52}
, {-86, -48}
, {72, -34}
, {-5, -46}
, {-32, 65}
, {86, -45}
, {70, 28}
, {87, 23}
, {84, -61}
, {14, 47}
, {62, -19}
, {-36, -49}
}
, {{57, 50}
, {-73, -1}
, {65, 18}
, {-34, 0}
, {-25, -29}
, {-42, 5}
, {27, -44}
, {-47, -4}
, {-70, 61}
, {-19, 37}
, {-89, -52}
, {92, -59}
, {30, -74}
, {61, -47}
, {71, 83}
, {-66, 89}
, {-89, -20}
, {-72, 35}
, {-59, -30}
, {25, -4}
, {29, 0}
, {-40, 23}
, {-38, 68}
, {7, 37}
, {78, 73}
, {-2, 63}
, {23, -17}
, {-65, 69}
, {-78, -43}
, {74, -77}
, {-12, 52}
, {81, 20}
}
, {{-6, -70}
, {-11, -22}
, {10, 14}
, {-66, -7}
, {82, 36}
, {-25, 29}
, {73, 3}
, {59, -1}
, {-3, 74}
, {-1, 86}
, {11, 71}
, {-72, 47}
, {73, 18}
, {-46, -73}
, {35, -11}
, {-63, -71}
, {64, -30}
, {69, -78}
, {85, -11}
, {62, 32}
, {-7, 65}
, {-26, 36}
, {-71, 48}
, {61, 22}
, {-73, -9}
, {-22, -70}
, {-29, 43}
, {54, 64}
, {77, -19}
, {38, -39}
, {-82, 53}
, {-17, -21}
}
, {{-44, 22}
, {10, 76}
, {-68, -63}
, {42, -26}
, {68, 30}
, {-9, 10}
, {-59, 1}
, {-61, 59}
, {8, 86}
, {53, 87}
, {-62, -15}
, {2, -46}
, {80, 67}
, {-68, 9}
, {-25, 16}
, {-40, 32}
, {71, 75}
, {-81, -73}
, {-1, -15}
, {77, 80}
, {-5, 82}
, {-43, -25}
, {-62, -35}
, {58, -2}
, {-17, 92}
, {58, -67}
, {28, -3}
, {80, -41}
, {-30, 86}
, {68, 8}
, {-75, -64}
, {-19, -20}
}
, {{-52, -83}
, {-2, 5}
, {5, -2}
, {-10, -52}
, {61, 61}
, {-58, -86}
, {-29, 33}
, {45, -24}
, {-63, 92}
, {-62, 34}
, {-75, -16}
, {-43, -56}
, {60, 65}
, {12, 73}
, {-69, 16}
, {86, -11}
, {0, 33}
, {32, 52}
, {22, -18}
, {30, 16}
, {-61, 72}
, {86, 62}
, {-65, -28}
, {34, -3}
, {-19, 44}
, {-93, 3}
, {-1, -78}
, {75, 30}
, {-15, -4}
, {78, 38}
, {21, -64}
, {91, 30}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE