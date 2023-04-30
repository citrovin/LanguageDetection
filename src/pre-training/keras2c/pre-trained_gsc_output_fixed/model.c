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
#include "conv1d_12.c"
#include "weights/conv1d_12.c" // InputLayer is excluded
#include "max_pooling1d_9.c" // InputLayer is excluded
#include "conv1d_13.c"
#include "weights/conv1d_13.c" // InputLayer is excluded
#include "max_pooling1d_10.c" // InputLayer is excluded
#include "conv1d_14.c"
#include "weights/conv1d_14.c" // InputLayer is excluded
#include "max_pooling1d_11.c" // InputLayer is excluded
#include "conv1d_15.c"
#include "weights/conv1d_15.c" // InputLayer is excluded
#include "average_pooling1d_3.c" // InputLayer is excluded
#include "flatten_3.c" // InputLayer is excluded
#include "dense_4.c"
#include "weights/dense_4.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_4_output_type dense_4_output) {

  // Output array allocation
  static union {
    conv1d_12_output_type conv1d_12_output;
    conv1d_13_output_type conv1d_13_output;
    conv1d_14_output_type conv1d_14_output;
    conv1d_15_output_type conv1d_15_output;
  } activations1;

  static union {
    max_pooling1d_9_output_type max_pooling1d_9_output;
    max_pooling1d_10_output_type max_pooling1d_10_output;
    max_pooling1d_11_output_type max_pooling1d_11_output;
    average_pooling1d_3_output_type average_pooling1d_3_output;
    flatten_3_output_type flatten_3_output;
  } activations2;


  //static union {
//
//    static input_4_output_type input_4_output;
//
//    static conv1d_12_output_type conv1d_12_output;
//
//    static max_pooling1d_9_output_type max_pooling1d_9_output;
//
//    static conv1d_13_output_type conv1d_13_output;
//
//    static max_pooling1d_10_output_type max_pooling1d_10_output;
//
//    static conv1d_14_output_type conv1d_14_output;
//
//    static max_pooling1d_11_output_type max_pooling1d_11_output;
//
//    static conv1d_15_output_type conv1d_15_output;
//
//    static average_pooling1d_3_output_type average_pooling1d_3_output;
//
//    static flatten_3_output_type flatten_3_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  conv1d_12(
     // First layer uses input passed as model parameter
    input,
    conv1d_12_kernel,
    conv1d_12_bias,
    activations1.conv1d_12_output
  );
 // InputLayer is excluded 
  max_pooling1d_9(
    
    activations1.conv1d_12_output,
    activations2.max_pooling1d_9_output
  );
 // InputLayer is excluded 
  conv1d_13(
    
    activations2.max_pooling1d_9_output,
    conv1d_13_kernel,
    conv1d_13_bias,
    activations1.conv1d_13_output
  );
 // InputLayer is excluded 
  max_pooling1d_10(
    
    activations1.conv1d_13_output,
    activations2.max_pooling1d_10_output
  );
 // InputLayer is excluded 
  conv1d_14(
    
    activations2.max_pooling1d_10_output,
    conv1d_14_kernel,
    conv1d_14_bias,
    activations1.conv1d_14_output
  );
 // InputLayer is excluded 
  max_pooling1d_11(
    
    activations1.conv1d_14_output,
    activations2.max_pooling1d_11_output
  );
 // InputLayer is excluded 
  conv1d_15(
    
    activations2.max_pooling1d_11_output,
    conv1d_15_kernel,
    conv1d_15_bias,
    activations1.conv1d_15_output
  );
 // InputLayer is excluded 
  average_pooling1d_3(
    
    activations1.conv1d_15_output,
    activations2.average_pooling1d_3_output
  );
 // InputLayer is excluded 
  flatten_3(
    
    activations2.average_pooling1d_3_output,
    activations2.flatten_3_output
  );
 // InputLayer is excluded 
  dense_4(
    
    activations2.flatten_3_output,
    dense_4_kernel,
    dense_4_bias, // Last layer uses output passed as model parameter
    dense_4_output
  );

}
