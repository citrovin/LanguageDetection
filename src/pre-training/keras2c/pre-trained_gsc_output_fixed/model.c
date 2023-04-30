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
#include "conv1d_27.c"
#include "weights/conv1d_27.c" // InputLayer is excluded
#include "max_pooling1d_21.c" // InputLayer is excluded
#include "conv1d_28.c"
#include "weights/conv1d_28.c" // InputLayer is excluded
#include "max_pooling1d_22.c" // InputLayer is excluded
#include "conv1d_29.c"
#include "weights/conv1d_29.c" // InputLayer is excluded
#include "max_pooling1d_23.c" // InputLayer is excluded
#include "conv1d_30.c"
#include "weights/conv1d_30.c" // InputLayer is excluded
#include "average_pooling1d_6.c" // InputLayer is excluded
#include "flatten_10.c" // InputLayer is excluded
#include "dense_11.c"
#include "weights/dense_11.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_11_output_type dense_11_output) {

  // Output array allocation
  static union {
    conv1d_27_output_type conv1d_27_output;
    conv1d_28_output_type conv1d_28_output;
    conv1d_29_output_type conv1d_29_output;
    conv1d_30_output_type conv1d_30_output;
  } activations1;

  static union {
    max_pooling1d_21_output_type max_pooling1d_21_output;
    max_pooling1d_22_output_type max_pooling1d_22_output;
    max_pooling1d_23_output_type max_pooling1d_23_output;
    average_pooling1d_6_output_type average_pooling1d_6_output;
    flatten_10_output_type flatten_10_output;
  } activations2;


  //static union {
//
//    static input_11_output_type input_11_output;
//
//    static conv1d_27_output_type conv1d_27_output;
//
//    static max_pooling1d_21_output_type max_pooling1d_21_output;
//
//    static conv1d_28_output_type conv1d_28_output;
//
//    static max_pooling1d_22_output_type max_pooling1d_22_output;
//
//    static conv1d_29_output_type conv1d_29_output;
//
//    static max_pooling1d_23_output_type max_pooling1d_23_output;
//
//    static conv1d_30_output_type conv1d_30_output;
//
//    static average_pooling1d_6_output_type average_pooling1d_6_output;
//
//    static flatten_10_output_type flatten_10_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  conv1d_27(
     // First layer uses input passed as model parameter
    input,
    conv1d_27_kernel,
    conv1d_27_bias,
    activations1.conv1d_27_output
  );
 // InputLayer is excluded 
  max_pooling1d_21(
    
    activations1.conv1d_27_output,
    activations2.max_pooling1d_21_output
  );
 // InputLayer is excluded 
  conv1d_28(
    
    activations2.max_pooling1d_21_output,
    conv1d_28_kernel,
    conv1d_28_bias,
    activations1.conv1d_28_output
  );
 // InputLayer is excluded 
  max_pooling1d_22(
    
    activations1.conv1d_28_output,
    activations2.max_pooling1d_22_output
  );
 // InputLayer is excluded 
  conv1d_29(
    
    activations2.max_pooling1d_22_output,
    conv1d_29_kernel,
    conv1d_29_bias,
    activations1.conv1d_29_output
  );
 // InputLayer is excluded 
  max_pooling1d_23(
    
    activations1.conv1d_29_output,
    activations2.max_pooling1d_23_output
  );
 // InputLayer is excluded 
  conv1d_30(
    
    activations2.max_pooling1d_23_output,
    conv1d_30_kernel,
    conv1d_30_bias,
    activations1.conv1d_30_output
  );
 // InputLayer is excluded 
  average_pooling1d_6(
    
    activations1.conv1d_30_output,
    activations2.average_pooling1d_6_output
  );
 // InputLayer is excluded 
  flatten_10(
    
    activations2.average_pooling1d_6_output,
    activations2.flatten_10_output
  );
 // InputLayer is excluded 
  dense_11(
    
    activations2.flatten_10_output,
    dense_11_kernel,
    dense_11_bias, // Last layer uses output passed as model parameter
    dense_11_output
  );

}
