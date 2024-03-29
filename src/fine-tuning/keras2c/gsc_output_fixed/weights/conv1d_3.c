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


const int16_t conv1d_3_bias[CONV_FILTERS] = {-121, 33, 69, -18, -95, 84, 46, 52, -17, 45, -125, -85, -2, -178, -11, 51, -63, 14, -113, 251, -81, 102, -39, -81, 70, 86, 5, 77, 14, 57, 8, 79, 187, 60, -29, -1, -82, 148, -117, 1, 0, -94, -72, 44, -58, 70, -16, -185, 155, -21, 35, 12, -135, -41, -36, -26, -1, 45, -69, -65, 0, 51, -10, -22}
;

const int16_t conv1d_3_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-74, 14}
, {46, 87}
, {-12, -90}
, {104, 74}
, {60, 68}
, {32, -23}
, {-9, -200}
, {-84, -97}
, {82, -76}
, {-108, -146}
, {-78, 15}
, {-19, -1}
, {-79, 118}
, {2, 3}
, {-54, -156}
, {-56, 97}
}
, {{38, 105}
, {-63, 63}
, {-82, -40}
, {27, -27}
, {-49, 57}
, {-29, 63}
, {38, 19}
, {-49, -31}
, {-81, 124}
, {-79, -85}
, {8, -53}
, {110, 118}
, {71, 35}
, {-24, -136}
, {-44, -157}
, {102, 9}
}
, {{-16, -16}
, {4, -35}
, {32, 21}
, {-66, -61}
, {31, -8}
, {-3, 24}
, {-112, -205}
, {2, 104}
, {-10, 40}
, {-53, 47}
, {8, -8}
, {-175, -195}
, {43, -50}
, {-101, 26}
, {104, 104}
, {87, 13}
}
, {{-196, -170}
, {-10, -107}
, {-199, 12}
, {105, -62}
, {-99, -119}
, {86, -69}
, {-77, -73}
, {31, -43}
, {-61, 48}
, {25, 106}
, {77, -232}
, {62, -186}
, {18, 19}
, {-54, -273}
, {-36, -13}
, {84, 12}
}
, {{89, 43}
, {-229, -29}
, {-9, -154}
, {-120, 77}
, {-131, -25}
, {-27, -42}
, {-47, -15}
, {24, -103}
, {-17, -15}
, {-93, -125}
, {-48, 37}
, {35, 31}
, {86, 60}
, {60, -49}
, {-151, -117}
, {38, 7}
}
, {{8, -38}
, {-116, -76}
, {127, 89}
, {-65, -120}
, {59, -4}
, {88, 31}
, {-125, 14}
, {-18, -53}
, {31, 33}
, {73, -16}
, {46, 28}
, {-37, 0}
, {-38, -136}
, {-82, -72}
, {-2, 20}
, {-43, -89}
}
, {{-84, 85}
, {0, -52}
, {-27, -32}
, {-126, -59}
, {40, -68}
, {86, -3}
, {-3, 8}
, {124, 58}
, {99, -123}
, {-68, 35}
, {-139, -8}
, {-124, -24}
, {-104, -24}
, {44, 0}
, {-62, -67}
, {-102, -48}
}
, {{72, 50}
, {91, 25}
, {-74, 93}
, {-13, 0}
, {-152, -96}
, {-84, -98}
, {29, -71}
, {17, 57}
, {-75, 66}
, {-28, 49}
, {62, 162}
, {139, 143}
, {46, 6}
, {-70, -55}
, {-25, -67}
, {68, 12}
}
, {{-2, 47}
, {-66, 9}
, {-98, 2}
, {-9, 43}
, {-55, -145}
, {-11, -38}
, {43, -88}
, {-178, -237}
, {62, -100}
, {71, 127}
, {109, -64}
, {80, -129}
, {-31, 35}
, {48, 0}
, {3, -22}
, {-74, -94}
}
, {{-72, 96}
, {1, 29}
, {-20, -145}
, {-147, -190}
, {-173, -130}
, {20, -23}
, {-34, 40}
, {-27, -103}
, {86, 63}
, {-1, -41}
, {-108, -35}
, {-11, -86}
, {96, -37}
, {-89, -61}
, {58, 20}
, {-82, 60}
}
, {{-67, -65}
, {48, 49}
, {-206, -247}
, {14, 45}
, {-286, -236}
, {-42, 37}
, {124, 2}
, {46, -74}
, {23, 16}
, {146, 66}
, {73, -28}
, {3, 61}
, {62, -14}
, {-31, -35}
, {6, -39}
, {28, -116}
}
, {{-113, -79}
, {-23, -84}
, {24, -45}
, {84, 4}
, {188, 39}
, {-51, -50}
, {34, 44}
, {-22, 83}
, {-27, -54}
, {89, 32}
, {114, -27}
, {36, -39}
, {0, 64}
, {-91, -74}
, {-120, -96}
, {-32, 76}
}
, {{-132, 76}
, {22, 55}
, {-7, -78}
, {-80, -80}
, {10, 80}
, {-44, 66}
, {-91, 4}
, {44, 58}
, {25, 19}
, {72, -85}
, {-70, 58}
, {50, -112}
, {-141, 0}
, {0, -22}
, {-121, 115}
, {-55, -29}
}
, {{-98, -49}
, {-24, -59}
, {-107, -239}
, {-130, -98}
, {-81, -203}
, {6, -24}
, {79, -62}
, {-65, -50}
, {46, 66}
, {-74, -153}
, {-57, 44}
, {33, -64}
, {-53, -74}
, {45, -8}
, {195, -67}
, {-89, -117}
}
, {{-28, 9}
, {-46, 11}
, {-94, 30}
, {-131, -20}
, {6, 49}
, {51, 49}
, {-33, 53}
, {17, 50}
, {66, -139}
, {-102, 13}
, {64, -21}
, {-241, -175}
, {19, -109}
, {30, 17}
, {55, -38}
, {44, -14}
}
, {{-113, -80}
, {-37, -29}
, {57, 72}
, {-73, -35}
, {-21, 15}
, {35, 38}
, {-53, 43}
, {79, -5}
, {-63, -44}
, {49, -98}
, {-37, -84}
, {18, 43}
, {39, 7}
, {23, 27}
, {-69, -85}
, {116, 13}
}
, {{-20, 55}
, {-7, 45}
, {20, -114}
, {81, 48}
, {123, 31}
, {21, -141}
, {103, 12}
, {7, 125}
, {-130, -30}
, {53, -66}
, {68, 100}
, {-162, -75}
, {-104, -43}
, {-44, -54}
, {-54, 80}
, {21, 51}
}
, {{6, -118}
, {89, 79}
, {42, 29}
, {-2, 41}
, {-71, -3}
, {-2, 26}
, {21, -151}
, {34, -31}
, {-43, 56}
, {72, -170}
, {-30, -55}
, {36, -28}
, {108, 37}
, {-31, -25}
, {52, -76}
, {-126, -111}
}
, {{50, -9}
, {-45, -61}
, {133, 23}
, {-64, -159}
, {-30, -66}
, {-136, -7}
, {36, 73}
, {0, -12}
, {-33, 84}
, {3, 52}
, {-119, -5}
, {-22, -189}
, {-73, 4}
, {-13, 45}
, {-42, 34}
, {30, 73}
}
, {{-117, -11}
, {-19, -1}
, {53, -1}
, {-112, 16}
, {31, -77}
, {-33, 75}
, {153, 34}
, {55, 10}
, {23, 10}
, {-133, -97}
, {-100, -40}
, {-139, 16}
, {-196, -67}
, {25, 49}
, {-53, -30}
, {2, 48}
}
, {{-10, 105}
, {35, 8}
, {91, 34}
, {13, 59}
, {5, 57}
, {-111, -114}
, {-14, 39}
, {18, 92}
, {-42, -88}
, {-35, -52}
, {-123, -83}
, {40, 88}
, {20, -9}
, {67, 11}
, {93, 0}
, {59, 75}
}
, {{33, -73}
, {-30, 19}
, {-95, 30}
, {76, 25}
, {-137, 58}
, {-66, 14}
, {-73, 24}
, {7, -91}
, {-127, 67}
, {28, 12}
, {38, 76}
, {-181, 7}
, {18, -53}
, {-59, -36}
, {38, 41}
, {129, -42}
}
, {{-59, -136}
, {130, 114}
, {47, -6}
, {55, 28}
, {-98, -72}
, {-110, -37}
, {60, -117}
, {-48, 8}
, {27, 83}
, {32, -46}
, {36, 38}
, {-26, 120}
, {6, -94}
, {-2, -28}
, {-17, 56}
, {9, -46}
}
, {{100, -22}
, {0, -46}
, {-31, -121}
, {-88, 58}
, {-42, 64}
, {-171, -45}
, {-13, 70}
, {-3, -97}
, {-121, -150}
, {12, -116}
, {-86, -106}
, {57, -31}
, {51, 58}
, {115, 12}
, {-168, 50}
, {24, -23}
}
, {{-104, -37}
, {-34, -25}
, {29, 9}
, {-22, -121}
, {-97, -76}
, {51, 24}
, {-90, -113}
, {-57, -113}
, {16, 23}
, {-107, -42}
, {76, -92}
, {34, 8}
, {109, 55}
, {72, 19}
, {-6, -89}
, {-112, -184}
}
, {{-26, 61}
, {90, 34}
, {8, -121}
, {35, 42}
, {-157, -2}
, {-75, -84}
, {-32, -36}
, {143, 78}
, {96, -37}
, {-106, 26}
, {135, -10}
, {-23, 120}
, {-60, -28}
, {-164, -9}
, {-26, 12}
, {4, -15}
}
, {{37, 52}
, {18, -16}
, {-94, -127}
, {-112, -277}
, {100, 112}
, {5, -38}
, {161, 87}
, {-49, 12}
, {-8, 18}
, {-70, -41}
, {86, 112}
, {-77, -149}
, {52, 11}
, {9, -20}
, {0, -33}
, {0, 14}
}
, {{46, 85}
, {19, -42}
, {23, 42}
, {-40, 42}
, {0, -14}
, {24, 67}
, {59, -2}
, {16, 47}
, {-92, 14}
, {-4, 26}
, {-12, -118}
, {-33, -39}
, {54, 100}
, {-53, -44}
, {59, 83}
, {-63, -65}
}
, {{-44, -4}
, {49, -146}
, {-74, -91}
, {-50, -10}
, {111, 21}
, {-17, 76}
, {28, -144}
, {-15, -4}
, {28, -102}
, {-7, 23}
, {56, 120}
, {119, 71}
, {-141, -84}
, {-16, -67}
, {-57, -100}
, {-22, 75}
}
, {{37, 24}
, {-14, 139}
, {-29, 31}
, {101, 92}
, {46, -241}
, {-120, -88}
, {-62, -113}
, {168, 40}
, {57, 82}
, {-81, 1}
, {11, -83}
, {-5, -86}
, {-32, -50}
, {-107, 8}
, {208, 17}
, {50, -13}
}
, {{-17, -50}
, {-114, -72}
, {-37, -89}
, {-37, 8}
, {40, -118}
, {33, 12}
, {63, 56}
, {-141, -120}
, {-23, -58}
, {-10, 32}
, {104, -117}
, {69, -148}
, {38, 78}
, {-18, 13}
, {-127, 4}
, {107, 0}
}
, {{35, -51}
, {-10, -96}
, {-214, -49}
, {-320, 66}
, {-132, 32}
, {-55, -26}
, {94, -55}
, {3, 40}
, {-17, -141}
, {64, 9}
, {-167, 100}
, {-84, 95}
, {121, -22}
, {84, -175}
, {-23, -32}
, {-48, 66}
}
, {{6, -233}
, {51, -9}
, {47, 71}
, {5, -49}
, {88, 68}
, {53, 9}
, {-6, -114}
, {45, 69}
, {52, 6}
, {-35, -45}
, {-63, -138}
, {-41, -33}
, {-96, 37}
, {49, -119}
, {-80, -101}
, {-47, -27}
}
, {{-109, -129}
, {-58, 15}
, {32, 39}
, {-34, -119}
, {-197, -238}
, {-6, 60}
, {-24, -9}
, {-8, 3}
, {31, 67}
, {-118, 36}
, {-31, 60}
, {37, 131}
, {8, 62}
, {-56, -46}
, {16, 65}
, {-65, -171}
}
, {{59, -112}
, {79, 27}
, {56, -36}
, {-6, 1}
, {30, 128}
, {39, 33}
, {-54, 82}
, {48, -46}
, {-98, -79}
, {-79, -12}
, {-222, 21}
, {-44, 80}
, {62, -83}
, {-167, -4}
, {102, -144}
, {0, 28}
}
, {{45, 95}
, {115, -11}
, {-36, 37}
, {145, -21}
, {-132, -46}
, {-116, 8}
, {-15, -15}
, {-57, -23}
, {-59, -60}
, {-85, -122}
, {54, -84}
, {31, 50}
, {-25, 110}
, {60, 50}
, {-91, 138}
, {-20, -66}
}
, {{48, 39}
, {22, 0}
, {-81, -78}
, {69, -120}
, {-25, -117}
, {90, 35}
, {81, 136}
, {-223, -119}
, {49, -94}
, {-72, -19}
, {-40, -50}
, {-38, -74}
, {-176, -57}
, {57, -25}
, {51, 70}
, {32, 16}
}
, {{60, 33}
, {31, 43}
, {-85, -85}
, {-70, 32}
, {27, -9}
, {-22, -40}
, {75, 37}
, {20, -16}
, {5, 3}
, {-75, -181}
, {4, 94}
, {208, 70}
, {18, 71}
, {-83, -26}
, {169, 3}
, {-154, -67}
}
, {{57, 54}
, {91, -65}
, {-46, -90}
, {33, -65}
, {-149, -184}
, {71, -28}
, {-74, -109}
, {37, -4}
, {35, 86}
, {-147, -160}
, {-91, -41}
, {-86, -34}
, {-66, -53}
, {-99, -294}
, {27, -45}
, {-97, -177}
}
, {{-177, -118}
, {-13, -12}
, {-11, -116}
, {-102, 3}
, {-231, -142}
, {74, 50}
, {35, -40}
, {-79, 69}
, {-21, 15}
, {-164, 47}
, {-26, 60}
, {70, -92}
, {-110, 106}
, {-24, -67}
, {95, 10}
, {19, 81}
}
, {{5, -33}
, {-60, 95}
, {-33, -39}
, {-47, -3}
, {1, -67}
, {43, 78}
, {108, -55}
, {71, 37}
, {47, -167}
, {-13, 5}
, {-118, 66}
, {-210, 118}
, {1, 112}
, {-98, 6}
, {10, 67}
, {66, -59}
}
, {{37, -188}
, {-107, 95}
, {92, -77}
, {-6, 17}
, {60, 71}
, {-27, 36}
, {103, -19}
, {-175, 21}
, {-19, -124}
, {41, -42}
, {-71, 13}
, {11, 40}
, {-84, -213}
, {-31, 6}
, {-52, 109}
, {4, 1}
}
, {{-75, -91}
, {-160, -68}
, {15, -91}
, {12, -51}
, {-109, -78}
, {-8, 26}
, {59, 26}
, {-62, -20}
, {29, -42}
, {95, 77}
, {-65, -188}
, {-29, 37}
, {121, 129}
, {-226, -247}
, {-23, -39}
, {-112, -103}
}
, {{-300, -51}
, {-30, 11}
, {-137, 24}
, {-127, -119}
, {-22, 2}
, {15, -102}
, {161, -44}
, {27, 59}
, {-2, -23}
, {-24, 58}
, {-63, -52}
, {47, -13}
, {-46, -93}
, {20, -50}
, {106, 67}
, {40, 39}
}
, {{21, -16}
, {100, 28}
, {-132, -53}
, {89, 86}
, {9, -203}
, {-52, 46}
, {-140, -47}
, {86, 68}
, {24, 23}
, {-34, -35}
, {-7, -15}
, {49, 108}
, {-59, 96}
, {-18, -62}
, {-47, 20}
, {89, 69}
}
, {{1, -60}
, {119, -13}
, {-40, 16}
, {39, 5}
, {-104, -173}
, {47, 37}
, {-15, 100}
, {16, 95}
, {-108, 62}
, {3, -31}
, {-18, -90}
, {-209, -54}
, {99, 81}
, {-184, -182}
, {14, -39}
, {-61, 7}
}
, {{2, 55}
, {-101, -59}
, {98, 38}
, {-52, 53}
, {-92, -95}
, {-111, 5}
, {-19, -71}
, {-195, -71}
, {-83, -10}
, {-47, 65}
, {-98, -106}
, {159, 62}
, {-95, 31}
, {-17, 76}
, {-39, -42}
, {5, 64}
}
, {{38, -82}
, {-277, 4}
, {-146, -28}
, {-39, -147}
, {20, 46}
, {-56, 8}
, {-59, 20}
, {-248, -436}
, {99, 32}
, {0, -6}
, {53, -146}
, {68, -10}
, {-77, -142}
, {-17, 24}
, {59, -147}
, {-25, -38}
}
, {{-182, -127}
, {-75, -100}
, {-15, -108}
, {-61, 60}
, {6, 42}
, {-69, -52}
, {-186, 53}
, {-78, 158}
, {7, -14}
, {-198, -22}
, {27, 75}
, {-1, 101}
, {90, 37}
, {-2, -34}
, {-42, -85}
, {-205, 104}
}
, {{62, 45}
, {62, -66}
, {8, -270}
, {-5, 9}
, {-73, -35}
, {19, 98}
, {-59, 83}
, {-67, -58}
, {-86, -3}
, {88, 12}
, {-136, -7}
, {-27, -8}
, {50, -29}
, {38, -67}
, {-128, -61}
, {-117, -42}
}
, {{-181, -64}
, {-37, 12}
, {38, 47}
, {-111, -91}
, {94, -44}
, {68, -61}
, {-165, 43}
, {30, 43}
, {19, 32}
, {-58, 21}
, {28, 45}
, {-84, -74}
, {29, 96}
, {-24, -32}
, {-6, 49}
, {-84, -22}
}
, {{-64, 35}
, {38, 40}
, {-22, -60}
, {-5, -36}
, {92, -25}
, {95, -7}
, {80, -153}
, {-77, -125}
, {-47, 76}
, {20, -178}
, {-123, -23}
, {-49, 119}
, {89, -22}
, {-52, 5}
, {-192, -59}
, {-150, 119}
}
, {{-38, -17}
, {-110, -66}
, {141, -120}
, {29, 11}
, {54, -46}
, {-92, 24}
, {16, -31}
, {91, 3}
, {14, 15}
, {-88, -43}
, {-73, 30}
, {-114, -91}
, {-58, 75}
, {61, -25}
, {22, -199}
, {-173, -80}
}
, {{-1, -85}
, {102, 20}
, {16, 137}
, {23, -107}
, {75, -19}
, {-66, -197}
, {45, 3}
, {81, 48}
, {-70, -63}
, {-37, -83}
, {-140, -99}
, {0, 13}
, {72, 22}
, {-40, 4}
, {42, 162}
, {37, 107}
}
, {{41, -42}
, {66, 97}
, {-81, -66}
, {-47, 24}
, {22, -32}
, {-3, -50}
, {-34, 4}
, {17, 60}
, {77, 116}
, {-27, 65}
, {-87, -71}
, {18, -12}
, {7, -23}
, {-35, -60}
, {27, -32}
, {-135, -21}
}
, {{-180, -145}
, {-21, 94}
, {-86, -101}
, {23, -5}
, {-255, 38}
, {-15, 30}
, {-98, -73}
, {29, 45}
, {55, -40}
, {26, 36}
, {-69, 9}
, {-59, -262}
, {24, -34}
, {66, -22}
, {54, 40}
, {-43, 40}
}
, {{16, -28}
, {78, 82}
, {-95, 15}
, {5, 73}
, {21, 84}
, {71, 9}
, {127, -29}
, {45, -117}
, {-141, -28}
, {-31, -41}
, {98, -92}
, {97, 27}
, {-24, 92}
, {-131, -39}
, {-109, 91}
, {-150, -197}
}
, {{-17, -202}
, {-131, 2}
, {0, 41}
, {-6, 27}
, {42, 110}
, {-18, -35}
, {151, 12}
, {25, -45}
, {-54, 73}
, {-88, -57}
, {-33, 55}
, {117, -57}
, {-153, -10}
, {-7, -7}
, {-57, 44}
, {111, -42}
}
, {{-13, 30}
, {-21, -3}
, {-52, -54}
, {111, 8}
, {-71, 4}
, {32, -75}
, {-86, -180}
, {-56, 47}
, {-56, -9}
, {22, 11}
, {26, 48}
, {66, 9}
, {23, -135}
, {74, 46}
, {-46, 102}
, {-25, 68}
}
, {{-35, -46}
, {-43, -134}
, {97, -15}
, {-99, 103}
, {-174, -62}
, {-93, -134}
, {79, 57}
, {-29, -46}
, {35, -1}
, {-157, 120}
, {69, -82}
, {68, 95}
, {5, 105}
, {-56, -94}
, {-178, 26}
, {10, -34}
}
, {{52, -146}
, {-123, -15}
, {-31, -78}
, {-41, 72}
, {-12, 4}
, {-143, -90}
, {62, -5}
, {-59, -17}
, {1, 94}
, {28, 46}
, {-166, -87}
, {96, -36}
, {-80, -106}
, {28, 33}
, {92, 52}
, {-73, 71}
}
, {{84, 14}
, {93, 69}
, {-2, 103}
, {-57, -26}
, {14, -59}
, {4, -51}
, {33, -97}
, {66, 103}
, {-19, -109}
, {-70, -61}
, {92, -81}
, {59, 47}
, {20, 64}
, {-98, 58}
, {121, 8}
, {-14, -5}
}
, {{-91, 17}
, {69, -70}
, {21, -21}
, {120, 16}
, {-49, -114}
, {-85, -44}
, {-95, -64}
, {-69, 7}
, {-23, 13}
, {116, 91}
, {-26, 1}
, {60, 98}
, {-4, 4}
, {-133, 93}
, {-39, -123}
, {-25, -8}
}
, {{9, 41}
, {-14, 83}
, {23, 29}
, {6, 49}
, {28, -40}
, {-63, 10}
, {-136, -62}
, {144, -74}
, {-66, -58}
, {-99, -117}
, {-35, -50}
, {48, 35}
, {78, -6}
, {74, 45}
, {-92, -108}
, {91, 13}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE