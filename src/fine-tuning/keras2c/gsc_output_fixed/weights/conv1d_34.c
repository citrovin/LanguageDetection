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


const int16_t conv1d_34_bias[CONV_FILTERS] = {47, -65, 72, 98, 96, 51, 52, -3, -67, -341, -307, 5, 37, -201, -82, -47, 214, -72, -106, 99, -75, 77, 46, -79, -9, 77, -48, 210, -74, -177, -41, -149, 132, 27, -175, 185, 192, 29, 58, -85, 32, -35, 121, 181, 133, 0, 74, -35, 193, -40, 132, 177, -90, 279, -75, -103, 170, 21, -81, 108, 58, 175, -99, -74}
;

const int16_t conv1d_34_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-154, 105}
, {-178, 69}
, {-97, 104}
, {-58, -20}
, {179, -114}
, {172, -48}
, {110, 24}
, {47, 154}
, {214, -82}
, {-239, 16}
, {31, -92}
, {-77, 1}
, {-27, -18}
, {112, 22}
, {-59, -31}
, {-99, 43}
, {-218, 0}
, {119, -56}
, {34, -21}
, {42, -62}
, {-43, -36}
, {-16, 112}
, {-43, -36}
, {-11, -52}
, {-70, 45}
, {-30, -31}
, {-113, 160}
, {-82, 92}
, {155, 58}
, {90, -15}
, {-200, 43}
, {-36, -41}
}
, {{94, -37}
, {70, 19}
, {53, 55}
, {-66, -203}
, {-239, -32}
, {-43, -54}
, {75, -108}
, {-24, 55}
, {-47, -249}
, {38, 91}
, {90, -46}
, {131, 154}
, {129, -163}
, {-8, -253}
, {-205, 34}
, {17, 9}
, {-14, -11}
, {143, -125}
, {-55, 66}
, {-32, 64}
, {23, -141}
, {-54, 95}
, {-52, -149}
, {54, -94}
, {29, -162}
, {85, 60}
, {-193, -116}
, {96, -83}
, {-161, -105}
, {99, 66}
, {164, -288}
, {163, 126}
}
, {{71, 85}
, {-23, 41}
, {-2, 52}
, {-74, -82}
, {3, 29}
, {-20, -28}
, {-204, -22}
, {-61, 42}
, {-110, 15}
, {-85, -22}
, {20, 79}
, {3, -38}
, {83, 28}
, {-132, -84}
, {-90, -192}
, {26, 75}
, {-9, -22}
, {35, 97}
, {48, -15}
, {21, 96}
, {-140, -24}
, {-39, 48}
, {-69, -14}
, {-45, -9}
, {74, 80}
, {44, 16}
, {12, -53}
, {-8, 92}
, {73, 41}
, {-48, -32}
, {34, 6}
, {90, 93}
}
, {{-196, 117}
, {60, -110}
, {-77, 124}
, {66, -29}
, {161, -73}
, {-83, -93}
, {25, 61}
, {158, 223}
, {-180, -37}
, {33, 38}
, {-56, 106}
, {-201, -21}
, {-37, 44}
, {245, 12}
, {-26, 55}
, {127, -64}
, {9, 185}
, {-79, -109}
, {-77, 89}
, {143, 22}
, {-267, -3}
, {-54, -30}
, {-46, 11}
, {-76, 48}
, {-52, -201}
, {24, 7}
, {-41, -48}
, {-132, 127}
, {-51, 19}
, {-73, 156}
, {-162, 177}
, {19, -48}
}
, {{165, -113}
, {172, -24}
, {-109, 156}
, {-20, -62}
, {196, 21}
, {97, -61}
, {37, 24}
, {-71, -220}
, {11, -99}
, {-255, 12}
, {-182, -354}
, {-109, -29}
, {66, -35}
, {71, -113}
, {147, 77}
, {-106, -49}
, {-38, -77}
, {-57, -12}
, {-53, -141}
, {118, 254}
, {-69, -42}
, {28, -79}
, {134, -34}
, {-121, -107}
, {18, -120}
, {-93, -172}
, {258, -162}
, {-106, -153}
, {123, -159}
, {-51, -64}
, {184, 123}
, {199, -100}
}
, {{-280, -107}
, {-110, -81}
, {-62, 83}
, {2, 62}
, {-59, 94}
, {1, -35}
, {142, 107}
, {-304, -332}
, {122, 31}
, {73, 71}
, {13, 10}
, {-30, 40}
, {-55, 115}
, {-9, -86}
, {-104, -55}
, {56, -23}
, {-159, 225}
, {-237, -248}
, {-133, 23}
, {26, 154}
, {-140, -68}
, {-143, -32}
, {23, 123}
, {-72, -24}
, {6, 33}
, {-65, -124}
, {60, 71}
, {68, 126}
, {-49, 41}
, {-29, 103}
, {-300, -189}
, {-132, -208}
}
, {{-82, -134}
, {78, 100}
, {-10, -153}
, {-152, -66}
, {-1, -96}
, {-5, 58}
, {97, 67}
, {212, 103}
, {48, -38}
, {64, 35}
, {-103, -78}
, {-192, 3}
, {116, -106}
, {30, -1}
, {43, 142}
, {-120, -62}
, {73, -16}
, {10, -10}
, {-17, -133}
, {-352, 75}
, {-126, -77}
, {-355, -87}
, {15, -190}
, {151, 82}
, {-93, 136}
, {52, 31}
, {59, 64}
, {-185, -48}
, {-69, 105}
, {-127, -42}
, {32, -4}
, {-11, 116}
}
, {{-66, 58}
, {-107, -38}
, {18, -170}
, {8, -81}
, {7, 52}
, {-96, -100}
, {54, 6}
, {265, -89}
, {-174, -196}
, {-47, 100}
, {-67, -30}
, {3, 77}
, {40, 4}
, {87, 50}
, {-24, 50}
, {72, -5}
, {87, 4}
, {74, -48}
, {132, -104}
, {27, 21}
, {-17, -21}
, {-95, -70}
, {-34, 83}
, {-33, -30}
, {58, -116}
, {59, 49}
, {246, -50}
, {-13, -158}
, {111, 22}
, {64, -105}
, {-192, 106}
, {17, -176}
}
, {{115, -31}
, {1, 140}
, {-33, -106}
, {-22, -49}
, {27, -46}
, {-29, -104}
, {-36, 99}
, {-1, 69}
, {31, 38}
, {107, -12}
, {80, -100}
, {-151, -23}
, {42, -109}
, {146, 29}
, {-2, 81}
, {31, 14}
, {-56, -219}
, {109, 102}
, {-16, -19}
, {-63, -149}
, {-12, 4}
, {-41, 59}
, {-166, -175}
, {-159, -62}
, {-60, 73}
, {96, 87}
, {26, 34}
, {-84, -128}
, {40, -79}
, {3, -22}
, {-96, 134}
, {-152, 71}
}
, {{33, -74}
, {72, 209}
, {101, -215}
, {-125, -7}
, {-190, -86}
, {149, -28}
, {27, 31}
, {-59, 36}
, {124, 50}
, {95, -172}
, {106, -100}
, {-54, 0}
, {77, -55}
, {-102, -52}
, {39, 8}
, {18, -179}
, {-274, -59}
, {91, -101}
, {14, -116}
, {-58, 21}
, {-55, 25}
, {-191, 63}
, {92, -41}
, {-145, -85}
, {-60, 85}
, {-59, -135}
, {131, -22}
, {-33, 78}
, {-198, -37}
, {113, -3}
, {258, -62}
, {230, 198}
}
, {{-266, -187}
, {-195, 5}
, {-301, -1}
, {-102, -51}
, {129, 196}
, {-421, -222}
, {195, 19}
, {66, 63}
, {-177, -6}
, {113, 93}
, {19, 28}
, {-162, 50}
, {-131, -151}
, {-494, -329}
, {-135, -65}
, {-52, -14}
, {106, -321}
, {-23, 9}
, {31, -142}
, {-107, -81}
, {-641, -548}
, {-129, -51}
, {29, 109}
, {122, -124}
, {-42, -12}
, {-10, -13}
, {101, 145}
, {-102, -4}
, {-187, -63}
, {-287, -233}
, {-24, -284}
, {-254, 12}
}
, {{-40, -84}
, {-89, 58}
, {-47, -11}
, {53, -87}
, {-80, -27}
, {-51, 28}
, {69, 83}
, {22, 20}
, {-36, 73}
, {87, -56}
, {101, -192}
, {52, 13}
, {-152, -129}
, {12, 170}
, {-25, 142}
, {-24, 63}
, {-81, -49}
, {-178, -39}
, {-135, -14}
, {-62, -60}
, {89, -87}
, {-29, -106}
, {122, 150}
, {41, -15}
, {-5, 124}
, {-38, 84}
, {156, 25}
, {-74, -103}
, {95, 13}
, {-60, -137}
, {-483, -270}
, {54, 106}
}
, {{0, 25}
, {83, 176}
, {-205, -117}
, {-78, 15}
, {-63, -251}
, {-45, -41}
, {-116, 162}
, {158, 50}
, {24, 149}
, {138, 86}
, {-215, -205}
, {51, 66}
, {-51, 124}
, {36, -66}
, {-12, 89}
, {-94, -68}
, {196, 39}
, {97, 88}
, {-20, -145}
, {-146, -59}
, {25, 98}
, {86, 26}
, {0, -106}
, {19, -61}
, {45, 68}
, {24, -4}
, {-178, -278}
, {-90, -99}
, {40, 6}
, {62, -118}
, {-91, 44}
, {48, 93}
}
, {{-15, -130}
, {-92, -28}
, {23, 23}
, {-36, -84}
, {-115, -68}
, {-98, -176}
, {70, -106}
, {128, 116}
, {-17, -133}
, {-188, 147}
, {38, 207}
, {-9, 12}
, {-27, -285}
, {-98, 8}
, {44, -27}
, {-8, -56}
, {14, 40}
, {-87, -130}
, {58, 66}
, {-87, -13}
, {52, 33}
, {-129, 150}
, {-182, -219}
, {-76, -108}
, {52, 58}
, {-137, 61}
, {-198, 126}
, {-79, -16}
, {-106, -22}
, {-114, 48}
, {-33, 117}
, {-118, 0}
}
, {{46, 22}
, {-165, -41}
, {-82, 49}
, {-117, -182}
, {120, -13}
, {-77, -91}
, {81, 71}
, {-270, -322}
, {19, -1}
, {56, 142}
, {142, -28}
, {36, -13}
, {19, 4}
, {134, -53}
, {41, -55}
, {1, -43}
, {-129, -193}
, {64, -33}
, {2, 38}
, {11, 34}
, {31, -112}
, {-93, -112}
, {20, -88}
, {-58, -19}
, {-240, -231}
, {23, -38}
, {171, -29}
, {-133, -4}
, {-40, -86}
, {-19, -25}
, {111, 336}
, {-53, 78}
}
, {{-30, 69}
, {-116, -99}
, {41, -62}
, {-31, 55}
, {-39, -131}
, {-213, -112}
, {12, 130}
, {-86, -177}
, {-88, -162}
, {130, 9}
, {103, 67}
, {1, 18}
, {13, -129}
, {67, 138}
, {23, 154}
, {91, -107}
, {99, 101}
, {-9, -72}
, {13, -51}
, {7, 4}
, {-55, 4}
, {23, -75}
, {18, -14}
, {-22, 54}
, {-78, -50}
, {-105, -14}
, {18, -90}
, {-4, -102}
, {118, 8}
, {88, 33}
, {87, 300}
, {-39, -35}
}
, {{-242, -333}
, {-3, 43}
, {-58, -35}
, {-53, 6}
, {-78, 5}
, {57, -191}
, {-102, -141}
, {-2, -62}
, {68, 70}
, {-55, -103}
, {-57, 36}
, {9, 9}
, {-188, -47}
, {-176, -117}
, {7, 38}
, {57, 139}
, {-9, -76}
, {-202, -20}
, {-223, 37}
, {-52, 21}
, {34, 33}
, {-93, 30}
, {-86, -53}
, {194, 18}
, {156, 96}
, {-87, 1}
, {232, -37}
, {-249, -219}
, {-318, -100}
, {18, 201}
, {-472, -80}
, {-102, 84}
}
, {{-248, -65}
, {-121, 24}
, {-41, -18}
, {-84, 49}
, {122, -46}
, {-20, 190}
, {-44, -28}
, {146, 46}
, {-114, 5}
, {181, -4}
, {127, -62}
, {-12, -53}
, {143, -3}
, {137, 220}
, {109, 12}
, {-92, 51}
, {-61, -92}
, {-127, 26}
, {69, 129}
, {100, 90}
, {-73, -63}
, {30, -54}
, {-283, -119}
, {-65, 70}
, {-168, 50}
, {-69, -75}
, {-55, 9}
, {-82, 14}
, {-118, 52}
, {-141, -89}
, {-21, 53}
, {120, 44}
}
, {{-174, -1}
, {129, -26}
, {166, -74}
, {-24, -111}
, {-33, -46}
, {-77, 61}
, {102, 93}
, {-92, 54}
, {-43, 14}
, {99, 42}
, {-252, -156}
, {-55, 54}
, {-129, -180}
, {-134, 193}
, {-36, 75}
, {120, -14}
, {190, 79}
, {48, 39}
, {54, -37}
, {-139, -147}
, {-94, -183}
, {-48, -81}
, {50, -22}
, {-105, -134}
, {20, -43}
, {49, -68}
, {-24, -91}
, {-70, -32}
, {-37, 85}
, {17, 0}
, {12, -81}
, {52, -97}
}
, {{-65, -82}
, {-61, 7}
, {-64, -97}
, {-48, 20}
, {108, 54}
, {-27, 119}
, {92, 25}
, {16, -215}
, {57, -74}
, {1, 120}
, {90, 36}
, {-599, -356}
, {-129, 137}
, {95, 93}
, {-36, 59}
, {69, 43}
, {145, 180}
, {53, -74}
, {88, -181}
, {117, 21}
, {-318, -354}
, {-135, -273}
, {-229, -90}
, {-102, -75}
, {56, -65}
, {-12, 30}
, {293, 20}
, {-42, -534}
, {-30, 76}
, {-191, -208}
, {225, 26}
, {59, -55}
}
, {{0, 7}
, {27, -36}
, {65, 33}
, {-175, -129}
, {46, 81}
, {-248, -236}
, {-94, -71}
, {86, 178}
, {20, 97}
, {5, 53}
, {30, 53}
, {3, -99}
, {27, -7}
, {-112, -96}
, {108, 42}
, {-25, 30}
, {8, -93}
, {56, -33}
, {13, -3}
, {65, 61}
, {16, -31}
, {-116, -4}
, {-104, 62}
, {65, 76}
, {-66, -20}
, {-4, -25}
, {177, 45}
, {-53, -159}
, {-45, -79}
, {69, 74}
, {-109, -74}
, {56, -50}
}
, {{-56, 44}
, {22, 58}
, {-38, 23}
, {-29, 89}
, {-113, 34}
, {-84, 90}
, {-70, -87}
, {-4, 91}
, {-95, -50}
, {-100, -186}
, {36, -161}
, {-43, 29}
, {77, 60}
, {-97, 20}
, {20, 93}
, {35, 64}
, {-67, 75}
, {33, 38}
, {38, -79}
, {60, 141}
, {47, -12}
, {-11, 25}
, {-52, -30}
, {-127, -116}
, {-45, 81}
, {25, -67}
, {-64, -185}
, {2, 76}
, {0, 49}
, {-25, 45}
, {17, 23}
, {-28, -21}
}
, {{-33, 30}
, {60, -56}
, {-130, 55}
, {58, 38}
, {-3, 81}
, {-55, -32}
, {95, -35}
, {-161, -48}
, {85, -47}
, {-4, 17}
, {100, -113}
, {-9, 15}
, {-72, -138}
, {-80, 29}
, {75, 74}
, {0, -32}
, {-188, -110}
, {124, 58}
, {-121, -8}
, {-10, 83}
, {43, 8}
, {-47, -55}
, {-51, -114}
, {-69, 14}
, {159, 9}
, {-18, -61}
, {-74, -15}
, {60, 103}
, {-96, 100}
, {-99, -146}
, {-89, -98}
, {-66, 29}
}
, {{111, 77}
, {11, 171}
, {177, 130}
, {-80, 84}
, {-209, -287}
, {-26, -25}
, {49, 32}
, {46, 23}
, {-86, -145}
, {199, -71}
, {-84, -232}
, {21, 199}
, {55, -28}
, {112, -1}
, {2, 11}
, {-230, -114}
, {-28, -221}
, {-62, 33}
, {-162, -55}
, {-10, -22}
, {-8, -98}
, {-78, -44}
, {-101, -146}
, {74, -50}
, {-75, -180}
, {22, 9}
, {40, -33}
, {-162, 55}
, {-15, 121}
, {37, -98}
, {-52, -18}
, {119, -46}
}
, {{-58, 3}
, {-61, 15}
, {131, -136}
, {62, 23}
, {-38, 140}
, {-165, -182}
, {52, -261}
, {-125, -225}
, {-72, 39}
, {120, -237}
, {138, -24}
, {-79, -80}
, {7, 73}
, {143, 33}
, {54, 33}
, {-125, 44}
, {142, -49}
, {-55, 103}
, {-3, 67}
, {-57, -20}
, {-28, 90}
, {-35, -166}
, {-68, -27}
, {-300, -201}
, {-78, 36}
, {-99, -112}
, {71, -226}
, {-115, -28}
, {174, -60}
, {-39, -155}
, {91, 127}
, {-132, -259}
}
, {{-25, -78}
, {104, 150}
, {-58, -126}
, {12, 76}
, {34, -27}
, {68, -125}
, {86, -112}
, {202, -84}
, {16, 41}
, {29, -111}
, {32, -129}
, {-42, -154}
, {108, 93}
, {-179, -1}
, {19, -78}
, {-22, 33}
, {-70, -119}
, {16, 46}
, {-356, -34}
, {124, -67}
, {1, 86}
, {-49, -100}
, {-59, 132}
, {95, -90}
, {-29, -54}
, {-2, 82}
, {16, -183}
, {96, -10}
, {-58, -77}
, {-54, 71}
, {8, -157}
, {110, -163}
}
, {{79, 46}
, {53, -19}
, {-101, 20}
, {-11, -16}
, {-7, -162}
, {-25, 24}
, {213, -128}
, {-10, -81}
, {-119, 43}
, {-67, -3}
, {97, 87}
, {58, 27}
, {-27, 57}
, {81, -40}
, {69, -162}
, {-59, -63}
, {193, -15}
, {75, 57}
, {-63, 62}
, {-163, 31}
, {-13, -25}
, {-100, -22}
, {65, 35}
, {-24, -152}
, {152, 36}
, {64, 19}
, {-31, 40}
, {-57, 46}
, {27, -52}
, {-44, -3}
, {-34, -25}
, {119, -3}
}
, {{22, 68}
, {-55, 122}
, {34, 261}
, {-115, -200}
, {-70, -35}
, {18, -138}
, {68, 19}
, {49, -119}
, {87, -132}
, {30, -72}
, {-175, -160}
, {14, -55}
, {12, 67}
, {27, -207}
, {70, -69}
, {10, -481}
, {222, 175}
, {13, -28}
, {55, -79}
, {-196, -61}
, {-79, -157}
, {-159, -70}
, {-149, -212}
, {102, 85}
, {63, -167}
, {46, -265}
, {77, 28}
, {-41, -279}
, {3, -143}
, {-11, 63}
, {119, -661}
, {-133, -149}
}
, {{38, 77}
, {120, 5}
, {-36, 76}
, {-65, -49}
, {0, -225}
, {97, 16}
, {94, 67}
, {16, 66}
, {-40, -18}
, {-58, -47}
, {-5, -13}
, {256, 144}
, {-196, -90}
, {36, -218}
, {32, -28}
, {-24, -177}
, {33, 103}
, {-170, -205}
, {-118, -91}
, {-19, 131}
, {-50, -72}
, {62, -42}
, {37, -39}
, {-109, -105}
, {-203, 91}
, {-42, 21}
, {174, 55}
, {42, -133}
, {43, 126}
, {-125, -103}
, {-204, -281}
, {144, -87}
}
, {{-79, -64}
, {1, -86}
, {-112, 29}
, {56, 7}
, {-39, -145}
, {-131, 54}
, {-75, -82}
, {55, 20}
, {47, 66}
, {16, 3}
, {-72, 10}
, {-32, -58}
, {15, -68}
, {11, 77}
, {46, -68}
, {57, -57}
, {-87, 20}
, {53, 59}
, {-62, 49}
, {50, 7}
, {-77, -24}
, {-121, 11}
, {5, 6}
, {63, 93}
, {56, -58}
, {41, 6}
, {44, -45}
, {-160, 85}
, {-112, 64}
, {-115, -90}
, {-31, -70}
, {-58, -51}
}
, {{71, 35}
, {-78, -57}
, {6, 2}
, {28, -21}
, {47, -79}
, {63, 22}
, {52, -137}
, {-39, -40}
, {13, -43}
, {115, 95}
, {58, -32}
, {74, -17}
, {27, 103}
, {33, -18}
, {-133, 89}
, {-44, -63}
, {-39, -276}
, {33, 20}
, {-181, -78}
, {-111, -183}
, {-88, -6}
, {34, -17}
, {72, 59}
, {62, -120}
, {-105, -139}
, {111, 35}
, {-126, -86}
, {146, 141}
, {97, -99}
, {131, 182}
, {-142, -61}
, {88, 58}
}
, {{19, -10}
, {18, -77}
, {-94, 145}
, {-85, -38}
, {-38, 180}
, {-166, -58}
, {-54, -106}
, {86, 116}
, {-36, -40}
, {290, 35}
, {-54, -72}
, {-23, -106}
, {28, 32}
, {107, 57}
, {-28, 42}
, {-153, 16}
, {76, 21}
, {44, 0}
, {-105, 112}
, {-257, 19}
, {108, 84}
, {-150, 28}
, {-151, 106}
, {-41, -81}
, {-64, 51}
, {-159, -25}
, {-40, -78}
, {-85, 85}
, {-19, -58}
, {-38, 6}
, {98, -20}
, {-82, 111}
}
, {{4, -35}
, {35, 127}
, {49, -31}
, {-24, -27}
, {-66, -89}
, {-24, -96}
, {159, 42}
, {-57, -86}
, {76, 2}
, {-22, -229}
, {-232, -138}
, {-142, -8}
, {-62, 86}
, {46, -146}
, {18, 72}
, {-185, -138}
, {-65, -120}
, {-136, -1}
, {-58, -98}
, {-91, -109}
, {24, -2}
, {-128, -228}
, {-114, -42}
, {131, 60}
, {8, -115}
, {114, 10}
, {-77, -97}
, {-138, -182}
, {121, 63}
, {-176, -95}
, {-47, -90}
, {60, -107}
}
, {{-50, -106}
, {-54, -55}
, {-155, -252}
, {18, -45}
, {38, 49}
, {-4, -12}
, {-1, 37}
, {-65, -38}
, {91, 67}
, {-143, -61}
, {-76, 77}
, {19, -65}
, {134, -139}
, {-54, -382}
, {-168, -148}
, {65, 72}
, {1, 21}
, {1, -7}
, {37, 3}
, {-8, 94}
, {-117, -10}
, {27, -43}
, {81, 3}
, {81, 56}
, {132, 27}
, {2, 28}
, {-13, 64}
, {68, 55}
, {110, -116}
, {-79, -37}
, {61, -177}
, {-84, 4}
}
, {{-6, -24}
, {-113, -56}
, {0, -80}
, {-9, -77}
, {63, 19}
, {110, -99}
, {-55, -29}
, {64, -6}
, {-146, 14}
, {-31, 113}
, {6, 162}
, {-102, 4}
, {84, 24}
, {-35, -57}
, {105, 37}
, {2, 98}
, {-197, -101}
, {-156, -36}
, {-48, 32}
, {-71, 134}
, {3, 5}
, {-135, -43}
, {34, 118}
, {-19, -27}
, {-4, 84}
, {-161, -127}
, {190, 285}
, {109, 143}
, {3, 57}
, {-203, -147}
, {148, 235}
, {105, 82}
}
, {{20, -69}
, {75, -20}
, {-82, -8}
, {54, 75}
, {44, 56}
, {-75, 106}
, {-118, -81}
, {-173, -45}
, {-38, 95}
, {114, 66}
, {136, 54}
, {-26, -29}
, {64, 0}
, {-32, -68}
, {-193, -355}
, {-103, -32}
, {12, 40}
, {-112, -8}
, {34, 38}
, {-25, -27}
, {124, 94}
, {48, -49}
, {-27, 2}
, {103, -71}
, {-41, -89}
, {-37, -32}
, {-24, 220}
, {-179, 99}
, {162, 100}
, {-7, 1}
, {54, -126}
, {-121, -74}
}
, {{4, -30}
, {31, 57}
, {-235, -119}
, {37, 20}
, {54, 92}
, {-10, 82}
, {84, 12}
, {-253, -318}
, {75, 55}
, {-12, 92}
, {24, 76}
, {108, 92}
, {-193, -139}
, {-69, 133}
, {-14, -44}
, {55, 41}
, {-56, -69}
, {35, -14}
, {-12, 112}
, {57, -123}
, {-68, -78}
, {27, -6}
, {87, 17}
, {28, 1}
, {-76, 8}
, {-39, -40}
, {36, 81}
, {-99, -23}
, {54, 132}
, {13, 29}
, {96, -1}
, {-120, -295}
}
, {{24, 67}
, {-69, -63}
, {-148, 43}
, {-104, -13}
, {32, 5}
, {-92, -8}
, {19, 14}
, {-26, 19}
, {-118, 105}
, {-117, -26}
, {-21, -14}
, {-46, -59}
, {-160, -51}
, {-82, 143}
, {-110, 7}
, {-234, -69}
, {-73, 73}
, {1, 49}
, {-177, -64}
, {100, 59}
, {-140, -52}
, {-216, -70}
, {-154, 132}
, {79, 112}
, {64, 6}
, {0, -4}
, {-94, 154}
, {157, -30}
, {43, 108}
, {91, -22}
, {-13, 160}
, {-188, -18}
}
, {{135, 98}
, {-65, -125}
, {-190, -22}
, {-41, -23}
, {-49, -53}
, {25, 59}
, {-97, -189}
, {84, 150}
, {60, -3}
, {84, 35}
, {-105, -2}
, {-80, -5}
, {-37, -44}
, {52, 67}
, {-47, 11}
, {-124, -243}
, {-95, -232}
, {2, 69}
, {21, -4}
, {75, -81}
, {22, -43}
, {-149, -150}
, {-117, -140}
, {-101, -13}
, {20, 42}
, {-103, -23}
, {141, -5}
, {-367, -306}
, {72, 204}
, {-49, -7}
, {35, 7}
, {-219, -263}
}
, {{-62, -76}
, {6, 110}
, {65, 109}
, {-104, -86}
, {-73, -114}
, {61, 31}
, {53, 130}
, {6, -1}
, {-130, -12}
, {-69, 88}
, {-41, -9}
, {-13, -77}
, {-48, -97}
, {-20, 55}
, {131, 113}
, {50, -57}
, {17, 4}
, {-119, -72}
, {47, 0}
, {152, 230}
, {23, 5}
, {38, 45}
, {-86, -109}
, {85, 179}
, {98, 129}
, {62, -104}
, {36, 85}
, {29, 32}
, {150, 131}
, {-22, -137}
, {-94, 196}
, {-37, -93}
}
, {{57, -128}
, {106, -226}
, {67, 50}
, {-70, -3}
, {66, -19}
, {2, 142}
, {32, -101}
, {22, -89}
, {108, -59}
, {12, 162}
, {-37, -117}
, {2, -89}
, {-33, -45}
, {64, 20}
, {89, 38}
, {-10, -45}
, {48, -23}
, {132, -61}
, {40, -171}
, {99, 2}
, {57, -174}
, {-5, 10}
, {-61, 120}
, {242, -22}
, {6, -81}
, {71, -59}
, {-40, 36}
, {-50, 45}
, {-161, -123}
, {100, -67}
, {108, -171}
, {99, 29}
}
, {{46, 29}
, {-67, 83}
, {77, -91}
, {88, -67}
, {201, -135}
, {-23, -200}
, {65, -52}
, {26, 39}
, {-150, -257}
, {23, 57}
, {-4, -41}
, {-14, -273}
, {-22, -94}
, {-8, 106}
, {15, 45}
, {-64, -282}
, {130, -282}
, {-40, 55}
, {99, -126}
, {64, 211}
, {53, -108}
, {71, -111}
, {-11, -5}
, {107, 132}
, {-14, -59}
, {4, 4}
, {150, 16}
, {-32, -354}
, {-254, -317}
, {-34, -136}
, {221, -453}
, {268, 36}
}
, {{-209, 62}
, {102, 101}
, {77, 27}
, {-63, -101}
, {60, 43}
, {78, 142}
, {33, 121}
, {-207, -246}
, {-27, -169}
, {45, -68}
, {47, -3}
, {95, -50}
, {0, 84}
, {-67, 268}
, {-91, 231}
, {-32, -50}
, {24, -14}
, {77, -70}
, {-200, 89}
, {10, 75}
, {-43, -177}
, {141, -214}
, {-101, 79}
, {97, 0}
, {16, -30}
, {59, -103}
, {44, 105}
, {7, -26}
, {-45, 183}
, {-72, -233}
, {-38, -84}
, {-104, -60}
}
, {{-77, -96}
, {160, -61}
, {32, -2}
, {-155, -39}
, {-60, 215}
, {6, -28}
, {-54, 98}
, {18, -124}
, {-211, -13}
, {-76, -21}
, {-179, -82}
, {21, 92}
, {17, -75}
, {266, -111}
, {60, -74}
, {-36, 64}
, {80, -14}
, {-20, 16}
, {48, -60}
, {-46, -42}
, {97, -121}
, {-85, 25}
, {-145, 57}
, {46, 109}
, {-22, -110}
, {35, 5}
, {-76, -116}
, {-152, 10}
, {-2, 137}
, {-63, 54}
, {86, -297}
, {-44, -150}
}
, {{15, -33}
, {-188, -255}
, {42, -167}
, {-62, -97}
, {-26, 52}
, {6, -34}
, {-31, -3}
, {122, 87}
, {84, -122}
, {-76, 128}
, {43, 140}
, {-86, 194}
, {104, 115}
, {41, 43}
, {-87, -70}
, {61, -78}
, {23, 119}
, {77, -75}
, {51, -120}
, {45, -27}
, {-161, -198}
, {88, -105}
, {37, -2}
, {-6, 11}
, {-8, -6}
, {79, -22}
, {-42, -193}
, {106, 16}
, {142, 45}
, {-24, 27}
, {-214, -490}
, {-44, 97}
}
, {{-54, -100}
, {123, 51}
, {11, 139}
, {93, 19}
, {42, -81}
, {101, -32}
, {111, 120}
, {-169, -73}
, {-41, 14}
, {175, 145}
, {11, -193}
, {-156, -20}
, {90, 67}
, {-69, -28}
, {-35, -100}
, {7, 57}
, {-158, -141}
, {-144, -27}
, {-57, 122}
, {40, -153}
, {-165, -38}
, {0, -31}
, {89, 152}
, {91, 147}
, {-19, -30}
, {-37, 70}
, {40, -27}
, {-38, 53}
, {24, 95}
, {37, 87}
, {58, -217}
, {89, 103}
}
, {{59, -121}
, {-109, -65}
, {36, 46}
, {-29, -52}
, {-3, -96}
, {4, 54}
, {-20, -125}
, {-85, -136}
, {-15, -174}
, {-79, 122}
, {-9, -149}
, {-1, -399}
, {-11, 76}
, {42, -157}
, {41, -192}
, {-36, -96}
, {-15, -66}
, {-104, -71}
, {32, -79}
, {-123, -50}
, {76, -73}
, {62, -63}
, {44, 51}
, {-2, 150}
, {140, -138}
, {56, -27}
, {-67, 217}
, {82, -15}
, {-79, -27}
, {42, -101}
, {-148, -481}
, {114, 13}
}
, {{16, -12}
, {-85, 149}
, {-4, 178}
, {-39, -39}
, {-65, 59}
, {93, 168}
, {-18, -47}
, {67, 117}
, {57, 50}
, {20, -37}
, {59, -22}
, {94, 14}
, {-9, 141}
, {23, -4}
, {24, -7}
, {29, -81}
, {-47, 70}
, {9, 28}
, {-134, 98}
, {-104, 70}
, {63, 83}
, {46, 11}
, {-74, -70}
, {-56, -84}
, {-29, 14}
, {-67, 13}
, {-47, -149}
, {149, -112}
, {-50, -98}
, {57, -52}
, {-10, 42}
, {54, 218}
}
, {{-207, -140}
, {-258, -183}
, {97, 58}
, {-102, -80}
, {193, 129}
, {-47, 122}
, {-44, 69}
, {-168, -441}
, {45, 56}
, {88, 138}
, {131, -75}
, {-121, -47}
, {101, 70}
, {79, -4}
, {-120, -78}
, {-84, 70}
, {-42, 24}
, {-162, 0}
, {14, 21}
, {186, 74}
, {47, 21}
, {-18, -39}
, {-110, -25}
, {147, 36}
, {0, 13}
, {-183, -41}
, {-8, 116}
, {68, 15}
, {-65, -29}
, {24, 196}
, {-264, -30}
, {-65, 70}
}
, {{11, 7}
, {259, -124}
, {-121, -161}
, {65, 47}
, {-29, 69}
, {195, -132}
, {277, -86}
, {-72, -6}
, {-135, -82}
, {85, 40}
, {-64, -89}
, {117, 73}
, {128, 61}
, {179, 47}
, {-111, -148}
, {-70, 52}
, {110, -210}
, {-118, 44}
, {-213, -112}
, {39, -273}
, {16, 68}
, {-92, -58}
, {-33, 68}
, {-590, -549}
, {-41, -29}
, {-30, 29}
, {-14, -307}
, {-108, -32}
, {17, 33}
, {122, 180}
, {-211, -156}
, {-65, -282}
}
, {{-14, -72}
, {18, -29}
, {-230, -172}
, {-20, -39}
, {182, -248}
, {-151, 19}
, {75, 39}
, {73, 44}
, {26, -23}
, {63, 66}
, {-253, 122}
, {-14, -16}
, {-33, 156}
, {-31, 87}
, {171, 82}
, {-27, -113}
, {-86, 113}
, {-35, -47}
, {82, 85}
, {115, 188}
, {-2, -56}
, {-203, -211}
, {-67, 12}
, {19, 80}
, {78, 65}
, {-284, -132}
, {-62, -16}
, {-62, 156}
, {131, 159}
, {-78, 110}
, {-162, 13}
, {-20, -118}
}
, {{-68, -149}
, {86, 200}
, {-77, 57}
, {-17, -20}
, {71, 191}
, {89, -12}
, {-7, 29}
, {236, 188}
, {0, -79}
, {173, 64}
, {29, 74}
, {-53, -117}
, {93, -9}
, {212, 150}
, {-18, 47}
, {-92, 28}
, {59, -26}
, {-24, -55}
, {-176, -27}
, {-31, -16}
, {-211, 106}
, {-101, -65}
, {-32, 82}
, {-205, -45}
, {39, -76}
, {-104, 38}
, {-14, -120}
, {15, -119}
, {46, -29}
, {106, 49}
, {-179, -61}
, {-49, -168}
}
, {{-45, -65}
, {110, -256}
, {274, -12}
, {-113, -10}
, {2, -144}
, {-173, 32}
, {-205, 330}
, {-30, -104}
, {-141, -270}
, {-139, 132}
, {-25, 17}
, {135, 49}
, {-358, -70}
, {83, 31}
, {0, 2}
, {0, 6}
, {-5, -141}
, {93, -93}
, {-79, -25}
, {232, 167}
, {-339, -9}
, {-83, 100}
, {-110, 50}
, {-7, 135}
, {106, -173}
, {-118, -141}
, {19, 39}
, {-1, 32}
, {65, 19}
, {0, -160}
, {-215, 242}
, {0, 75}
}
, {{-183, 36}
, {32, 129}
, {234, 30}
, {-14, 1}
, {-37, -54}
, {-273, -95}
, {139, -30}
, {-47, -175}
, {112, 148}
, {-93, -211}
, {-265, -158}
, {-207, -258}
, {14, 84}
, {-430, -141}
, {7, -79}
, {-35, 35}
, {-75, 15}
, {56, -26}
, {137, 77}
, {27, 107}
, {-87, -68}
, {-57, -42}
, {-102, 25}
, {-30, 169}
, {-26, 20}
, {-25, 112}
, {-248, -186}
, {49, -190}
, {33, -23}
, {156, -16}
, {3, -313}
, {57, 167}
}
, {{82, 24}
, {-7, 38}
, {-17, 7}
, {-45, 39}
, {-23, -165}
, {-80, -118}
, {-3, 138}
, {-7, -4}
, {-6, 38}
, {40, 59}
, {-85, 58}
, {-1, -77}
, {-22, -62}
, {-43, -170}
, {-22, 180}
, {-48, 78}
, {117, 49}
, {40, -14}
, {-50, -111}
, {-17, 5}
, {-29, -40}
, {-36, 0}
, {73, -33}
, {7, 114}
, {98, 158}
, {108, 80}
, {-37, 23}
, {78, -35}
, {51, -196}
, {-27, -15}
, {14, 46}
, {-40, -37}
}
, {{-70, -140}
, {-125, 61}
, {51, 28}
, {-45, -28}
, {-79, 87}
, {106, 88}
, {53, -73}
, {-193, -39}
, {27, -16}
, {97, 22}
, {-108, 1}
, {-62, 34}
, {142, -92}
, {81, 103}
, {-22, -11}
, {-3, -87}
, {-191, 0}
, {-75, -70}
, {47, -130}
, {-8, -236}
, {-10, -142}
, {-143, -13}
, {21, -116}
, {111, 137}
, {-151, -51}
, {-9, 56}
, {-27, 62}
, {-181, -261}
, {182, -68}
, {-52, 46}
, {-12, -32}
, {-86, -13}
}
, {{-9, -63}
, {262, -13}
, {-139, -82}
, {-189, -28}
, {16, 151}
, {48, -132}
, {214, -109}
, {24, -21}
, {-90, -7}
, {88, 65}
, {114, 106}
, {-384, -97}
, {125, 151}
, {-3, -54}
, {-153, -89}
, {-67, 0}
, {-327, -77}
, {-64, -29}
, {-363, -131}
, {96, 158}
, {-154, -299}
, {-138, -210}
, {-135, -223}
, {-86, -78}
, {-286, -48}
, {13, 59}
, {54, 116}
, {-314, -224}
, {-37, 55}
, {-106, 98}
, {167, -140}
, {36, 41}
}
, {{-196, -143}
, {-275, 7}
, {-64, 3}
, {32, -142}
, {-131, -218}
, {3, 61}
, {52, -171}
, {14, -357}
, {-45, 46}
, {36, -65}
, {18, 13}
, {12, 91}
, {-228, -162}
, {-8, 42}
, {-54, 49}
, {-81, 63}
, {-123, -14}
, {83, -56}
, {59, -8}
, {16, -47}
, {2, 1}
, {-60, -8}
, {-6, -13}
, {194, -67}
, {114, -42}
, {-89, 59}
, {23, -51}
, {-50, -172}
, {148, -176}
, {63, 60}
, {112, 88}
, {66, -69}
}
, {{-103, -20}
, {-86, 217}
, {-79, -137}
, {-124, -29}
, {-28, -119}
, {-43, -15}
, {-100, -8}
, {-139, 22}
, {20, 136}
, {85, 92}
, {-63, -91}
, {37, 58}
, {-95, -225}
, {-5, 39}
, {-138, -15}
, {-4, -40}
, {68, -60}
, {-57, -50}
, {60, 8}
, {69, -20}
, {5, -63}
, {-71, -204}
, {-57, -8}
, {61, 28}
, {84, 166}
, {57, -97}
, {94, 108}
, {71, -160}
, {-7, 140}
, {99, 177}
, {-266, -398}
, {88, -135}
}
, {{-9, -16}
, {-3, 60}
, {63, -182}
, {-61, -55}
, {68, 90}
, {39, 234}
, {116, -71}
, {-45, -7}
, {83, 22}
, {-195, -251}
, {31, 155}
, {-35, -17}
, {-4, -207}
, {36, -107}
, {-51, 0}
, {12, 58}
, {-212, -183}
, {39, -9}
, {-393, -155}
, {9, 88}
, {75, 77}
, {-102, 20}
, {160, 41}
, {-13, -42}
, {61, 82}
, {8, -62}
, {50, -9}
, {151, 0}
, {92, -146}
, {-28, -33}
, {-211, -252}
, {-95, -14}
}
, {{20, 109}
, {-72, 44}
, {-160, -161}
, {-72, 39}
, {-189, -13}
, {-190, 45}
, {38, 39}
, {-257, -25}
, {-142, 88}
, {12, 46}
, {-148, -145}
, {32, -165}
, {59, 43}
, {37, 56}
, {60, 155}
, {-228, 38}
, {-69, -83}
, {-108, 59}
, {-71, -80}
, {-152, -117}
, {-27, 13}
, {-66, -61}
, {-63, 111}
, {-52, -250}
, {-69, 146}
, {10, 52}
, {5, 37}
, {-191, -50}
, {-151, -30}
, {15, -42}
, {29, 164}
, {218, -60}
}
, {{-15, -259}
, {171, -226}
, {-53, -81}
, {-32, 25}
, {-14, 221}
, {-102, 16}
, {74, 20}
, {21, -136}
, {-79, 114}
, {-68, 192}
, {52, -126}
, {9, -139}
, {12, -66}
, {-86, 36}
, {112, 48}
, {-192, 47}
, {82, -71}
, {146, -317}
, {56, -80}
, {-38, -78}
, {45, 95}
, {-86, 65}
, {-1, 66}
, {29, 11}
, {-112, 69}
, {-142, -142}
, {-140, 46}
, {-126, 63}
, {58, 2}
, {-113, -96}
, {-260, 149}
, {22, 62}
}
, {{-81, 11}
, {-157, 71}
, {-115, -190}
, {0, -62}
, {139, 183}
, {-251, -31}
, {-172, 115}
, {126, 97}
, {-12, 36}
, {100, 108}
, {42, 145}
, {35, 9}
, {73, 73}
, {-28, 54}
, {-8, 88}
, {-149, -19}
, {151, 232}
, {-56, -79}
, {17, 38}
, {98, -12}
, {-90, 28}
, {-255, -153}
, {-35, -29}
, {102, -2}
, {28, 81}
, {36, -126}
, {82, 124}
, {46, -45}
, {-6, 81}
, {90, 56}
, {65, -45}
, {-60, -89}
}
, {{-221, -127}
, {-107, 16}
, {99, -31}
, {-83, -112}
, {49, -4}
, {-1, -201}
, {-38, -52}
, {195, -96}
, {-22, 138}
, {-21, -62}
, {-109, -37}
, {-68, -102}
, {21, 24}
, {-155, -139}
, {7, -100}
, {130, 7}
, {7, -40}
, {-29, -16}
, {-78, -16}
, {64, 136}
, {-46, 100}
, {62, 39}
, {2, 86}
, {132, 19}
, {28, -46}
, {-53, -31}
, {5, -196}
, {83, -43}
, {-122, -183}
, {64, 102}
, {-324, 21}
, {125, 24}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE