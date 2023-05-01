/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"

 // InputLayer is excluded
#include "conv1d_31.c"
#include "weights/conv1d_31.c" // InputLayer is excluded
#include "max_pooling1d_24.c" // InputLayer is excluded
#include "conv1d_32.c"
#include "weights/conv1d_32.c" // InputLayer is excluded
#include "max_pooling1d_25.c" // InputLayer is excluded
#include "conv1d_33.c"
#include "weights/conv1d_33.c" // InputLayer is excluded
#include "max_pooling1d_26.c" // InputLayer is excluded
#include "conv1d_34.c"
#include "weights/conv1d_34.c" // InputLayer is excluded
#include "average_pooling1d_7.c" // InputLayer is excluded
#include "flatten_11.c" // InputLayer is excluded
#include "dense_12.c"
#include "weights/dense_12.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_12_output_type dense_12_output) {

  // Output array allocation
  static union {
    conv1d_31_output_type conv1d_31_output;
    conv1d_32_output_type conv1d_32_output;
    conv1d_33_output_type conv1d_33_output;
    conv1d_34_output_type conv1d_34_output;
  } activations1;

  static union {
    max_pooling1d_24_output_type max_pooling1d_24_output;
    max_pooling1d_25_output_type max_pooling1d_25_output;
    max_pooling1d_26_output_type max_pooling1d_26_output;
    average_pooling1d_7_output_type average_pooling1d_7_output;
    flatten_11_output_type flatten_11_output;
  } activations2;


  //static union {
//
//    static input_12_output_type input_12_output;
//
//    static conv1d_31_output_type conv1d_31_output;
//
//    static max_pooling1d_24_output_type max_pooling1d_24_output;
//
//    static conv1d_32_output_type conv1d_32_output;
//
//    static max_pooling1d_25_output_type max_pooling1d_25_output;
//
//    static conv1d_33_output_type conv1d_33_output;
//
//    static max_pooling1d_26_output_type max_pooling1d_26_output;
//
//    static conv1d_34_output_type conv1d_34_output;
//
//    static average_pooling1d_7_output_type average_pooling1d_7_output;
//
//    static flatten_11_output_type flatten_11_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  conv1d_31(
     // First layer uses input passed as model parameter
    input,
    conv1d_31_kernel,
    conv1d_31_bias,
    activations1.conv1d_31_output
  );
 // InputLayer is excluded 
  max_pooling1d_24(
    
    activations1.conv1d_31_output,
    activations2.max_pooling1d_24_output
  );
 // InputLayer is excluded 
  conv1d_32(
    
    activations2.max_pooling1d_24_output,
    conv1d_32_kernel,
    conv1d_32_bias,
    activations1.conv1d_32_output
  );
 // InputLayer is excluded 
  max_pooling1d_25(
    
    activations1.conv1d_32_output,
    activations2.max_pooling1d_25_output
  );
 // InputLayer is excluded 
  conv1d_33(
    
    activations2.max_pooling1d_25_output,
    conv1d_33_kernel,
    conv1d_33_bias,
    activations1.conv1d_33_output
  );
 // InputLayer is excluded 
  max_pooling1d_26(
    
    activations1.conv1d_33_output,
    activations2.max_pooling1d_26_output
  );
 // InputLayer is excluded 
  conv1d_34(
    
    activations2.max_pooling1d_26_output,
    conv1d_34_kernel,
    conv1d_34_bias,
    activations1.conv1d_34_output
  );
 // InputLayer is excluded 
  average_pooling1d_7(
    
    activations1.conv1d_34_output,
    activations2.average_pooling1d_7_output
  );
 // InputLayer is excluded 
  flatten_11(
    
    activations2.average_pooling1d_7_output,
    activations2.flatten_11_output
  );
 // InputLayer is excluded 
  dense_12(
    
    activations2.flatten_11_output,
    dense_12_kernel,
    dense_12_bias, // Last layer uses output passed as model parameter
    dense_12_output
  );

}
