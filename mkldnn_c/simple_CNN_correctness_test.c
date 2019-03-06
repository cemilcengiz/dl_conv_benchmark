/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

//
// Timing utility for Unix
#include <sys/time.h>
#include <sys/resource.h>
#include <stdlib.h>
double get_time()
{
    struct timeval t;
    //struct timezone tzp;
    //gettimeofday(&t, &tzp);
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec*1e-6;
}
#define PRINT_DATA 0
#define NUM_EXPERIMENTS 512

// Required for posix_memalign
#define _POSIX_C_SOURCE 200112L

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "mkldnn.h"
//header path: "/home/cemil/MKL_DNN/mkl-dnn/include/mkldnn.h"

#ifdef _WIN32
#include <malloc.h>
#endif
/*
#define BATCH 8
#define IC 3
#define OC 96
#define CONV_IH 227
#define CONV_IW 227
#define CONV_OH 55
#define CONV_OW 55
#define CONV_STRIDE 4
#define CONV_PAD 0
#define POOL_OH 27
#define POOL_OW 27
#define POOL_STRIDE 2
#define POOL_PAD 0
*/
#define CHECK(f) do { \
    mkldnn_status_t s = f; \
    if (s != mkldnn_success) { \
        printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, s); \
        exit(2); \
    } \
} while(0)

#define CHECK_TRUE(expr) do { \
    int e_ = expr; \
    if (!e_) { \
        printf("[%s:%d] %s failed\n", __FILE__, __LINE__, #expr); \
        exit(2); \
    } \
} while(0)

void *aligned_malloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#elif defined(_SX)
    return malloc(size);
#else
    void *p;
    return !posix_memalign(&p, alignment, size) ? p : NULL;
#endif
}

#ifdef _WIN32
void _free(void *ptr) {
    _aligned_free(ptr);
}
#else
void _free(void *ptr) {
    free(ptr);
}
#endif

static size_t product(int *arr, size_t size) {
    size_t prod = 1;
    for (size_t i = 0; i < size; ++i) prod *= arr[i];
    return prod;
}

static void init_data_memory(uint32_t dim, const int *dims,
        mkldnn_memory_format_t user_fmt, mkldnn_data_type_t mkldnn_f32,
        mkldnn_engine_t engine, float *data, mkldnn_primitive_t *memory)
{
  /* init  memory descriptor prim_md. Then create memory pd user_pd using prim_md
    and engine type. Then create memory primitive using user_pd. Finally,
    associate a malloc-allocated data to this memory primitive */
    mkldnn_memory_desc_t prim_md;
    mkldnn_primitive_desc_t user_pd;
    CHECK(mkldnn_memory_desc_init(&prim_md, dim, dims, mkldnn_f32, user_fmt));
    CHECK(mkldnn_memory_primitive_desc_create(&user_pd, &prim_md, engine));
    CHECK(mkldnn_primitive_create(memory, user_pd, NULL, NULL));

    void *req = NULL;
    CHECK(mkldnn_memory_get_data_handle(*memory, &req));
    CHECK_TRUE(req == NULL);
    CHECK(mkldnn_memory_set_data_handle(*memory, data));
    CHECK(mkldnn_memory_get_data_handle(*memory, &req));
    CHECK_TRUE(req == data);
    CHECK(mkldnn_primitive_desc_destroy(user_pd));
}

mkldnn_status_t prepare_reorder(
        mkldnn_primitive_t *user_memory, /** in */
        const_mkldnn_primitive_desc_t *prim_memory_pd, /** in */
        int dir_is_user_to_prim, /** in: user -> prim or prim -> user */
        mkldnn_primitive_t *prim_memory, /** out: memory primitive created */
        mkldnn_primitive_t *reorder, /** out: reorder primitive created */
        float *buffer)
{
    const_mkldnn_primitive_desc_t user_memory_pd;
    mkldnn_primitive_get_primitive_desc(*user_memory, &user_memory_pd);

    if (!mkldnn_memory_primitive_desc_equal(user_memory_pd, *prim_memory_pd)) {
        /* memory_create(&p, m, NULL) means allocate memory */
        CHECK(mkldnn_primitive_create(prim_memory, *prim_memory_pd,
                NULL, NULL));
        mkldnn_primitive_desc_t reorder_pd;
        if (dir_is_user_to_prim) {
            /* reorder primitive descriptor doesn't need engine, because it is
             * already appeared in in- and out- memory primitive descriptors */
            CHECK(mkldnn_reorder_primitive_desc_create(&reorder_pd,
                        user_memory_pd, *prim_memory_pd));
            mkldnn_primitive_at_t inputs = { *user_memory, 0 };
            const_mkldnn_primitive_t outputs[] = { *prim_memory };
            CHECK(mkldnn_primitive_create(reorder, reorder_pd, &inputs,
                        outputs));
        } else {
            CHECK(mkldnn_reorder_primitive_desc_create(&reorder_pd,
                        *prim_memory_pd, user_memory_pd));
            mkldnn_primitive_at_t inputs = { *prim_memory, 0 };
            const_mkldnn_primitive_t outputs[] = { *user_memory };
            CHECK(mkldnn_primitive_create(reorder, reorder_pd, &inputs,
                        outputs));
        }
        CHECK(mkldnn_memory_set_data_handle(*prim_memory, buffer));
        CHECK(mkldnn_primitive_desc_destroy(reorder_pd));
    } else {
        *prim_memory = NULL;
        *reorder = NULL;
    }

    return mkldnn_success;
}

mkldnn_status_t simple_net(int input_size[], int conv1_size[], int pool1_window_size,
                            int conv2_size[], int pool2_window_size, int output_size[]) {
    /* input-output sizes of overall system */
    int inBS, inCh, inHeight, inWidth, outCh, outHeight, outWidth;
    inBS = input_size[0]; inCh = input_size[1]; inHeight = input_size[2]; inWidth = input_size[3];
    // inBS = 1; inCh = 1; inHeight = 6; inWidth = 6;
    outCh = output_size[0]; outHeight = output_size[1]; outWidth = output_size[2];
    // #outCh = 1; outHeight = 3; outWidth = 3;

    mkldnn_engine_t engine;
    CHECK(mkldnn_engine_create(&engine, mkldnn_cpu, 0 /* idx */));

    float *net_src = (float *)aligned_malloc(
            inBS * inCh * inHeight * inWidth * sizeof(float), 64);
    float *net_dst = (float *)aligned_malloc(
            inBS * outCh * outHeight * outWidth * sizeof(float), 64);

    /* check output memory before of computation */
    int outSize = inBS * outCh * outHeight * outWidth;
    if (PRINT_DATA) {
        printf("\nOutput values before calculation:\n");
        for (int i = 0; i < outSize; i++) {
            printf("%d th entry = %f\n", i ,net_dst[i]);
        }
    }


    /* fill the user input tensor with values for testing */
    int inSize = inBS * inCh * inHeight * inWidth;
    for (int i = 0; i < inSize; i++) {
        net_src[i] = i;
    }
    if (PRINT_DATA) {
        printf("Input values: \n");
        for (int i = 0; i < inSize; i++) {
            printf("%d th entry = %f\n", i ,net_src[i]);
        }
    }

    /* conv1
     * {inBS, inCh, inHeight, inWidth} (x) {conv1_outCh, conv1_inCh, conv1_kernelH, conv1_kernelW} ->
     * {inBS, conv1_outCh, conv1_outH, conv1_outW}
     * strides: {1, 1}
     */
    int conv1_outCh, conv1_inCh, conv1_kernelH, conv1_kernelW, conv1_outW, conv1_outH;
    conv1_outCh = conv1_size[0]; conv1_inCh = conv1_size[1]; conv1_kernelH = conv1_size[2]; conv1_kernelW = conv1_size[3];
    //conv1_outCh = 1; conv1_inCh = inCh; conv1_kernelH = 3; conv1_kernelW = 3;
    conv1_outH = inHeight-conv1_kernelH+1;
    conv1_outW = inWidth-conv1_kernelW+1;
    int conv1_user_src_sizes[4] = { inBS, inCh, inHeight, inWidth };
    int conv1_user_weights_sizes[4] = { conv1_outCh, conv1_inCh, conv1_kernelH, conv1_kernelW };
    int conv1_bias_sizes[4] = { conv1_outCh };
    int conv1_user_dst_sizes[4] = { inBS, conv1_outCh, conv1_outH, conv1_outW };
    int conv1_strides[2] = { 1, 1 };
    int conv1_padding[2] = { 0, 0 };

    float *conv1_src = net_src;
    float *conv1_weights = (float *)aligned_malloc(
            product(conv1_user_weights_sizes, 4) * sizeof(float), 64);
    float *conv1_bias = (float *)aligned_malloc(
            product(conv1_bias_sizes, 1) * sizeof(float), 64);

    /* fill the user conv filter tensors with values for testing */
    int conv1_weightsSize = product(conv1_user_weights_sizes, 4);
    for (int i = 0; i < conv1_weightsSize; i++) {
        conv1_weights[i] = i;
    }
    if (PRINT_DATA) {
        printf("conv1 values: \n");
        for (int i = 0; i < conv1_weightsSize; i++) {
            printf("%d th entry = %f\n", i ,conv1_weights[i]);
        }
    }

    int conv1_biasSize = product(conv1_bias_sizes, 1);
    for (int i = 0; i < conv1_biasSize; i++) {
        conv1_bias[i] = 0;
    }

    /* create memory for user data */
    mkldnn_primitive_t conv1_user_src_memory, conv1_user_weights_memory,
        conv1_user_bias_memory;
    init_data_memory(4, conv1_user_src_sizes, mkldnn_nchw, mkldnn_f32, engine,
            conv1_src, &conv1_user_src_memory);
    init_data_memory(4, conv1_user_weights_sizes, mkldnn_oihw, mkldnn_f32,
            engine, conv1_weights, &conv1_user_weights_memory);
    init_data_memory(1, conv1_bias_sizes, mkldnn_x, mkldnn_f32, engine,
            conv1_bias, &conv1_user_bias_memory);


    /* create data descriptors for convolution w/ no specified format */

    mkldnn_memory_desc_t conv1_src_md, conv1_weights_md, conv1_bias_md,
        conv1_dst_md;
    CHECK(mkldnn_memory_desc_init(&conv1_src_md, 4, conv1_user_src_sizes,
        mkldnn_f32, mkldnn_any));
    CHECK(mkldnn_memory_desc_init(&conv1_weights_md, 4, conv1_user_weights_sizes,
        mkldnn_f32, mkldnn_any));
    CHECK(mkldnn_memory_desc_init(&conv1_bias_md, 1, conv1_bias_sizes,
        mkldnn_f32, mkldnn_x));
    CHECK(mkldnn_memory_desc_init(&conv1_dst_md, 4, conv1_user_dst_sizes,
        mkldnn_f32, mkldnn_any));

    /* create a convolution */
    /* create conv operation descriptor */
    mkldnn_convolution_desc_t conv1_any_desc;
    CHECK(mkldnn_convolution_forward_desc_init(&conv1_any_desc, mkldnn_forward,
            mkldnn_convolution_direct, &conv1_src_md, &conv1_weights_md,
            &conv1_bias_md, &conv1_dst_md, conv1_strides, conv1_padding,
            conv1_padding, mkldnn_padding_zero));

    /* create conv primitive descriptor */
    mkldnn_primitive_desc_t conv1_pd;
    CHECK(mkldnn_primitive_desc_create(&conv1_pd, &conv1_any_desc,
            engine, NULL));

    mkldnn_primitive_t conv1_internal_src_memory, conv1_internal_weights_memory;

    /* create reorder primitives between user data and convolution srcs
    * if required */
    mkldnn_primitive_t conv1_reorder_src, conv1_reorder_weights;

    const_mkldnn_primitive_desc_t src1_pd = mkldnn_primitive_desc_query_pd(
            conv1_pd, mkldnn_query_src_pd, 0);
    size_t conv1_src_size = mkldnn_memory_primitive_desc_get_size(src1_pd);
    float *conv1_src_buffer = (float *)aligned_malloc(conv1_src_size, 64);
    CHECK(prepare_reorder(&conv1_user_src_memory, &src1_pd, 1,
        &conv1_internal_src_memory, &conv1_reorder_src, conv1_src_buffer));/* 1 for user->prim */

    const_mkldnn_primitive_desc_t weights1_pd = mkldnn_primitive_desc_query_pd(
            conv1_pd, mkldnn_query_weights_pd, 0);
    size_t conv1_weights_size
            = mkldnn_memory_primitive_desc_get_size(weights1_pd);
    float *conv1_weights_buffer = (float *)aligned_malloc(conv1_weights_size, 64);
    CHECK(prepare_reorder(&conv1_user_weights_memory, &weights1_pd, 1,
            &conv1_internal_weights_memory, &conv1_reorder_weights,
            conv1_weights_buffer));/* 1 for user->prim */

    mkldnn_primitive_t conv1_src_memory = conv1_internal_src_memory ?
        conv1_internal_src_memory : conv1_user_src_memory;
    mkldnn_primitive_t conv1_weights_memory = conv1_internal_weights_memory ?
        conv1_internal_weights_memory : conv1_user_weights_memory;

    mkldnn_primitive_at_t conv1_srcs[] = {
        mkldnn_primitive_at(conv1_src_memory, 0),
        mkldnn_primitive_at(conv1_weights_memory, 0),
        mkldnn_primitive_at(conv1_user_bias_memory, 0)
    };

    mkldnn_primitive_t conv1_internal_dst_memory;
    /* create memory for dst1 data, we don't need reorder it to user data */
    const_mkldnn_primitive_desc_t conv1_dst_pd
            = mkldnn_primitive_desc_query_pd(conv1_pd, mkldnn_query_dst_pd, 0);
    CHECK(mkldnn_primitive_create(
            &conv1_internal_dst_memory, conv1_dst_pd, NULL, NULL));
    size_t conv1_dst_size = mkldnn_memory_primitive_desc_get_size(conv1_dst_pd);
    float *conv1_dst_buffer = (float *)aligned_malloc(conv1_dst_size, 64);
    CHECK(mkldnn_memory_set_data_handle(
            conv1_internal_dst_memory, conv1_dst_buffer));

    const_mkldnn_primitive_t conv1_dsts[] = { conv1_internal_dst_memory };

    /* finally create a convolution primitive */
    mkldnn_primitive_t conv1;
    CHECK(mkldnn_primitive_create(&conv1, conv1_pd, conv1_srcs, conv1_dsts));


    /* pool1
     * {inBS, conv1_outCh, conv1_outH, conv1_outW} -> {inBS, conv1_outCh, pool1_outH, pool1_outW}
     * kernel: {2, 2}
     * strides: {0, 0}
     */

    int pool1_outH, pool1_outW, pool1_window;
    pool1_window = pool1_window_size;
    //pool1_window = 1;
    pool1_outH = conv1_outH / pool1_window;
    pool1_outW = conv1_outW / pool1_window;

    int32_t pool1_dst_sizes[4] = { inBS, conv1_outCh, pool1_outH, pool1_outW };
    int32_t pool1_kernel[2] = { pool1_window, pool1_window };
    int32_t pool1_strides[2] = { pool1_window, pool1_window };
    int32_t pool1_padding[2] = { 0, 0 };

    /* create a pooling */

    /* create pooling memory descriptor on dst memory descriptor
     *  from previous primitive */
    const mkldnn_memory_desc_t *pool1_src_md =
        mkldnn_primitive_desc_query_memory_d(conv1_dst_pd);

    /* create descriptors for dst pooling data */
    mkldnn_memory_desc_t pool1_dst_md;
    CHECK(mkldnn_memory_desc_init(
            &pool1_dst_md, 4, pool1_dst_sizes, mkldnn_f32, mkldnn_any));

    /* first create pool operation descriptor. Then create pool pd. */
    mkldnn_pooling_desc_t pool1_desc;
    CHECK(mkldnn_pooling_forward_desc_init(&pool1_desc, mkldnn_forward,
            mkldnn_pooling_max, pool1_src_md, &pool1_dst_md, pool1_strides,
            pool1_kernel, pool1_padding, pool1_padding, mkldnn_padding_zero));

    mkldnn_primitive_desc_t pool1_pd;
    CHECK(mkldnn_primitive_desc_create(&pool1_pd, &pool1_desc, engine, NULL));

    /* create memory for workspace */
    mkldnn_primitive_t pool1_indices_memory;
    const_mkldnn_primitive_desc_t pool1_indices_pd =
        mkldnn_primitive_desc_query_pd(pool1_pd, mkldnn_query_workspace_pd, 0);
    CHECK(mkldnn_primitive_create(&pool1_indices_memory,
            pool1_indices_pd, NULL, NULL));
    size_t pool1_indices_size =
        mkldnn_memory_primitive_desc_get_size(pool1_indices_pd);
    float *pool1_indices_buffer = (float*)aligned_malloc(pool1_indices_size, 64);
    CHECK(mkldnn_memory_set_data_handle(pool1_indices_memory,
            pool1_indices_buffer));

    mkldnn_primitive_t pool1_internal_dst_memory;
    /* create memory for dst data, we don't need reorder it to user data,
    since there are still some remaining layers */
    const_mkldnn_primitive_desc_t pool1_dst_pd =
        mkldnn_primitive_desc_query_pd(pool1_pd, mkldnn_query_dst_pd, 0);
    CHECK(mkldnn_primitive_create(
        &pool1_internal_dst_memory, pool1_dst_pd, NULL, NULL));
    size_t pool1_dst_size = mkldnn_memory_primitive_desc_get_size(pool1_dst_pd);
    float *pool1_dst_buffer = (float *)aligned_malloc(pool1_dst_size, 64);
    CHECK(mkldnn_memory_set_data_handle(
            pool1_internal_dst_memory, pool1_dst_buffer));

    mkldnn_primitive_at_t pool1_srcs = { conv1_internal_dst_memory, 0 };
    const_mkldnn_primitive_t pool1_dsts[] = { pool1_internal_dst_memory,
            pool1_indices_memory };

    /* finally create a pooling primitive */
    mkldnn_primitive_t pool1;
    CHECK(mkldnn_primitive_create(&pool1, pool1_pd, &pool1_srcs, pool1_dsts));


     /* conv2
      * {inBS, conv1_outCh, pool1_outH, pool1_outW} (x) {conv2_outCh, conv2_inCh, conv2_kernelH, conv2_kernelW} ->
      * {inBS, conv2_outCh, conv2_outH, conv2_outW}
      * strides: {1, 1}
      */

     int conv2_outCh, conv2_inCh, conv2_kernelH, conv2_kernelW, conv2_outW, conv2_outH;
     conv2_outCh = conv2_size[0]; conv2_inCh = conv2_size[1]; conv2_kernelH = conv2_size[2]; conv2_kernelW = conv2_size[3];
     //conv2_outCh = 1; conv2_inCh = conv1_outCh; conv2_kernelH = 2; conv2_kernelW = 2;
     conv2_outH = pool1_outH - conv2_kernelH + 1;
     conv2_outW = pool1_outW - conv2_kernelW + 1;
     int conv2_user_weights_sizes[4] = { conv2_outCh, conv2_inCh, conv2_kernelH, conv2_kernelW };
     int conv2_bias_sizes[4] = { conv2_outCh };
     int conv2_user_dst_sizes[4] = { inBS, conv2_outCh, conv2_outH, conv2_outW};
     //int conv2_dst_sizes[4] = { inBS, conv2_outCh, conv2_outH, conv2_outW};
     int conv2_strides[2] = { 1, 1 };
     int conv2_padding[2] = { 0, 0 };

     // conv2_src comes from pool1_dst
     float *conv2_weights = (float *)aligned_malloc(
          product(conv2_user_weights_sizes, 4) * sizeof(float), 64);
     float *conv2_bias = (float *)aligned_malloc(
          product(conv2_bias_sizes, 1) * sizeof(float), 64);

          /* fill the user conv2 filter tensors with values for testing */
          int conv2_weightsSize = product(conv2_user_weights_sizes, 4);
          for (int i = 0; i < conv2_weightsSize; i++) {
              conv2_weights[i] = i;
          }
          if (PRINT_DATA) {
              printf("conv2 values: \n");
              for (int i = 0; i < conv2_weightsSize; i++) {
                  printf("%d th entry = %f\n", i ,conv2_weights[i]);
              }
          }

          int conv2_biasSize = product(conv2_bias_sizes, 1);
          for (int i = 0; i < conv2_biasSize; i++) {
              conv2_bias[i] = 0;
          }


      /* create memory for user weight and bias data */
      mkldnn_primitive_t conv2_user_weights_memory, conv2_user_bias_memory;
      init_data_memory(4, conv2_user_weights_sizes, mkldnn_oihw, mkldnn_f32,
              engine, conv2_weights, &conv2_user_weights_memory);
      init_data_memory(1, conv2_bias_sizes, mkldnn_x, mkldnn_f32, engine,
              conv2_bias, &conv2_user_bias_memory);

      /* create conv2 src memory descriptor from dst memory descriptor
       *  of previous primitive */
      const mkldnn_memory_desc_t *conv2_src_md =
          mkldnn_primitive_desc_query_memory_d(pool1_dst_pd);
    /* create remaining conv2 data descriptors with no specified format */
      mkldnn_memory_desc_t conv2_weights_md, conv2_bias_md, conv2_dst_md;
      CHECK(mkldnn_memory_desc_init(&conv2_weights_md, 4, conv2_user_weights_sizes,
          mkldnn_f32, mkldnn_any));
      CHECK(mkldnn_memory_desc_init(&conv2_bias_md, 1, conv2_bias_sizes,
          mkldnn_f32, mkldnn_x));
      CHECK(mkldnn_memory_desc_init(
              &conv2_dst_md, 4, conv2_user_dst_sizes, mkldnn_f32, mkldnn_any));

      /* create a convolution */
      /* create conv operation descriptor */
      mkldnn_convolution_desc_t conv2_any_desc;
      CHECK(mkldnn_convolution_forward_desc_init(&conv2_any_desc, mkldnn_forward,
                      mkldnn_convolution_direct, conv2_src_md, &conv2_weights_md,
                      &conv2_bias_md, &conv2_dst_md, conv2_strides, conv2_padding,
                      conv2_padding, mkldnn_padding_zero));

      /* create conv primitive descriptor */
      mkldnn_primitive_desc_t conv2_pd;
      CHECK(mkldnn_primitive_desc_create(&conv2_pd, &conv2_any_desc,
             engine, NULL));


      mkldnn_primitive_t conv2_internal_weights_memory;
      /* create reorder primitives between user weights data and convolution weights
      * if required */
      mkldnn_primitive_t conv2_reorder_weights;

      const_mkldnn_primitive_desc_t weights2_pd = mkldnn_primitive_desc_query_pd(
              conv2_pd, mkldnn_query_weights_pd, 0);
      size_t conv2_weights_size
              = mkldnn_memory_primitive_desc_get_size(weights2_pd);
      float *conv2_weights_buffer = (float *)aligned_malloc(conv2_weights_size, 64);
      CHECK(prepare_reorder(&conv2_user_weights_memory, &weights2_pd, 1,
              &conv2_internal_weights_memory, &conv2_reorder_weights,
              conv2_weights_buffer));/* 1 for user->prim */

      mkldnn_primitive_t conv2_weights_memory = conv2_internal_weights_memory ?
          conv2_internal_weights_memory : conv2_user_weights_memory;

      mkldnn_primitive_at_t conv2_srcs[] = {
              mkldnn_primitive_at(pool1_internal_dst_memory, 0),
              mkldnn_primitive_at(conv2_weights_memory, 0),
              mkldnn_primitive_at(conv2_user_bias_memory, 0)
      };


      mkldnn_primitive_t conv2_internal_dst_memory;
      /* create memory for conv2_dst data, we don't need reorder it to user data */
      const_mkldnn_primitive_desc_t conv2_dst_pd
            = mkldnn_primitive_desc_query_pd(conv2_pd, mkldnn_query_dst_pd, 0);
      CHECK(mkldnn_primitive_create(
             &conv2_internal_dst_memory, conv2_dst_pd, NULL, NULL));
      size_t conv2_dst_size = mkldnn_memory_primitive_desc_get_size(conv2_dst_pd);
      float *conv2_dst_buffer = (float *)aligned_malloc(conv2_dst_size, 64);
      CHECK(mkldnn_memory_set_data_handle(
                conv2_internal_dst_memory, conv2_dst_buffer));

      const_mkldnn_primitive_t conv2_dsts[] = { conv2_internal_dst_memory };


      /* finally create a convolution primitive */
      mkldnn_primitive_t conv2;
      CHECK(mkldnn_primitive_create(&conv2, conv2_pd, conv2_srcs, conv2_dsts));


      /* pool2
       * {inBS, conv2_outCh, conv2_outH, conv2_outW} -> {inBS, conv2_outCh, pool2_outH, pool2_outW}
       * kernel: {2, 2}
       * strides: {0, 0}
       */

      int pool2_outH, pool2_outW, pool2_window;
      pool2_window = pool2_window_size;
      //pool2_window = 1;
      pool2_outH = conv2_outH / pool2_window;
      pool2_outW = conv2_outW / pool2_window;

      int32_t pool2_dst_sizes[4] = { inBS, conv2_outCh, pool2_outH, pool2_outW };
      int32_t pool2_kernel[2] = { pool2_window, pool2_window };
      int32_t pool2_strides[2] = { pool2_window, pool2_window };
      int32_t pool2_padding[2] = { 0, 0 };


      /* create pooling memory descriptor on dst memory descriptor
       *  from previous primitive */
      const mkldnn_memory_desc_t *pool2_src_md =
          mkldnn_primitive_desc_query_memory_d(conv2_dst_pd);

      /* create descriptors for dst pooling data */
      mkldnn_memory_desc_t pool2_dst_md;
      CHECK(mkldnn_memory_desc_init(
              &pool2_dst_md, 4, pool2_dst_sizes, mkldnn_f32, mkldnn_any));

    /* create a pooling */
    /* first create pool operation descriptor. Then create pool pd. */
     mkldnn_pooling_desc_t pool2_desc;
     CHECK(mkldnn_pooling_forward_desc_init(&pool2_desc, mkldnn_forward,
             mkldnn_pooling_max, pool2_src_md, &pool2_dst_md, pool2_strides,
             pool2_kernel, pool2_padding, pool2_padding, mkldnn_padding_zero));

      /* create pool2 primitive descriptor */
      mkldnn_primitive_desc_t pool2_pd;
      CHECK(mkldnn_primitive_desc_create(&pool2_pd, &pool2_desc, engine, NULL));


      mkldnn_primitive_t pool2_internal_dst_memory;
      /* create memory for pool2_dst data */
      const_mkldnn_primitive_desc_t pool2_dst_pd =
          mkldnn_primitive_desc_query_pd(pool2_pd, mkldnn_query_dst_pd, 0);
      CHECK(mkldnn_primitive_create(
              &pool2_internal_dst_memory, pool2_dst_pd, NULL, NULL));
      size_t pool2_dst_size = mkldnn_memory_primitive_desc_get_size(pool2_dst_pd);
      float *pool2_dst_buffer = (float *)aligned_malloc(pool2_dst_size, 64);

      /* create memory for user data in output part */
      mkldnn_primitive_t pool2_user_dst_memory;
      init_data_memory(4, pool2_dst_sizes, mkldnn_nchw, mkldnn_f32, engine,
           net_dst, &pool2_user_dst_memory);

      /* create reorder primitives between user data and pooling dsts
       * if required */
      mkldnn_primitive_t pool2_reorder_dst;
      CHECK(prepare_reorder(&pool2_user_dst_memory, &pool2_dst_pd, 0,
            &pool2_internal_dst_memory, &pool2_reorder_dst, pool2_dst_buffer));


      /* create memory for workspace */
      mkldnn_primitive_t pool2_indices_memory;
      const_mkldnn_primitive_desc_t pool2_indices_pd =
          mkldnn_primitive_desc_query_pd(pool2_pd, mkldnn_query_workspace_pd, 0);
      CHECK(mkldnn_primitive_create(&pool2_indices_memory,
              pool2_indices_pd, NULL, NULL));
      size_t pool2_indices_size =
          mkldnn_memory_primitive_desc_get_size(pool2_indices_pd);
      float *pool2_indices_buffer = (float*)aligned_malloc(pool2_indices_size, 64);
      CHECK(mkldnn_memory_set_data_handle(pool2_indices_memory,
              pool2_indices_buffer));


     mkldnn_primitive_at_t pool2_srcs = { conv2_internal_dst_memory, 0 };

     mkldnn_primitive_t pool2_dst_memory;
     pool2_dst_memory = pool2_internal_dst_memory ? pool2_internal_dst_memory
          : pool2_user_dst_memory;

      const_mkldnn_primitive_t pool2_dsts[] = { pool2_dst_memory,
              pool2_indices_memory };

      /* finally create a pooling primitive */
      mkldnn_primitive_t pool2;
      CHECK(mkldnn_primitive_create(&pool2, pool2_pd, &pool2_srcs, pool2_dsts));

    /* build a simple net */
    uint32_t n = 0;
    mkldnn_primitive_t net[10];

    if (conv1_reorder_src) net[n++] = conv1_reorder_src;
    if (conv1_reorder_weights) net[n++] = conv1_reorder_weights;
    net[n++] = conv1;
    net[n++] = pool1;
    net[n++] = conv2;
    net[n++] = pool2;
    if (pool2_reorder_dst) net[n++] = pool2_reorder_dst;

    mkldnn_stream_t stream;
    CHECK(mkldnn_stream_create(&stream, mkldnn_eager));
    CHECK(mkldnn_stream_submit(stream, n, net, NULL));
    CHECK(mkldnn_stream_wait(stream, n, NULL));



    /* check result of computation */
    /*
    int pool2_out_siz = inBS * conv2_outCh * conv2_outH * conv2_outW;
    void* _pool2_out_data_handle = NULL;
    CHECK(mkldnn_memory_get_data_handle(pool2_dst_memory, &_pool2_out_data_handle));
    float* pool2_out_data_handle = (float *)_pool2_out_data_handle;
    */

    /* since we did the conv out reordering i.e., out->prim, we don't need to
    bother with blocks. Just use the reordered output finally */
    if (PRINT_DATA) {
        printf("\nReordered outputs i.e the simplest way of outputting if you did reordering\n");
        for (int i = 0; i < outSize; i++) {
            printf("Directly %d th entry = %f\n", i ,net_dst[i]);
        }
    }



    /* clean-up */
    CHECK(mkldnn_primitive_desc_destroy(conv1_pd));
    CHECK(mkldnn_primitive_desc_destroy(pool1_pd));
    CHECK(mkldnn_primitive_desc_destroy(conv2_pd));
    CHECK(mkldnn_primitive_desc_destroy(pool2_pd));

    mkldnn_stream_destroy(stream);

    _free(net_src);
    _free(net_dst);

    /* clean conv1 resources */
    mkldnn_primitive_destroy(conv1_user_src_memory);
    mkldnn_primitive_destroy(conv1_user_weights_memory);
    mkldnn_primitive_destroy(conv1_user_bias_memory);
    mkldnn_primitive_destroy(conv1_internal_src_memory);
    mkldnn_primitive_destroy(conv1_internal_weights_memory);
    mkldnn_primitive_destroy(conv1_internal_dst_memory);
    mkldnn_primitive_destroy(conv1_reorder_src);
    mkldnn_primitive_destroy(conv1_reorder_weights);
    mkldnn_primitive_destroy(conv1);

    _free(conv1_weights);
    _free(conv1_bias);
    _free(conv1_src_buffer);
    _free(conv1_weights_buffer);
    _free(conv1_dst_buffer);


    /* clean pool1 resources */
    mkldnn_primitive_destroy(pool1_internal_dst_memory);
    mkldnn_primitive_destroy(pool1_indices_memory);
    mkldnn_primitive_destroy(pool1);

    _free(pool1_dst_buffer);
    _free(pool1_indices_buffer);


    /* clean conv2 resources */
    mkldnn_primitive_destroy(conv2_user_weights_memory);
    mkldnn_primitive_destroy(conv2_user_bias_memory);
    mkldnn_primitive_destroy(conv2_internal_weights_memory);
    mkldnn_primitive_destroy(conv2_reorder_weights);
    mkldnn_primitive_destroy(conv2_internal_dst_memory);
    mkldnn_primitive_destroy(conv2);

    _free(conv2_weights);
    _free(conv2_bias);
    _free(conv2_weights_buffer);
    _free(conv2_dst_buffer);


    /* clean pool2 resources */
    mkldnn_primitive_destroy(pool2_user_dst_memory);
    mkldnn_primitive_destroy(pool2_internal_dst_memory);
    mkldnn_primitive_destroy(pool2_indices_memory);
    mkldnn_primitive_destroy(pool2_reorder_dst);
    mkldnn_primitive_destroy(pool2);

    _free(pool2_dst_buffer);
    _free(pool2_indices_buffer);


    mkldnn_engine_destroy(engine);

    return mkldnn_success;
}

/***
* Usage:
To compile:
gcc -Wall simple_CNN_correctness_test.c -o bin/simple_CNN_correctness_test_c -I /kuacc/users/ccengiz17/MKL_DNN/mkl-dnn/include -L /kuacc/users/ccengiz17/MKL_DNN/mkl-dnn/build/src -lmkldnn -std=c99

To add the dynamic library:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/kuacc/users/ccengiz17/MKL_DNN/mkl-dnn/build/src/
***/

int main(int argc, char **argv) {
    int input_size[] = {1,1,6,6}; /* inBS, inCh, inHeight, inWidth */
    int conv1_size[] = {1,1,3,3}; /* conv1_outCh, conv1_inCh, conv1_kernelH, conv1_kernelW */
    int pool1_window_size = 1;
    int conv2_size[] = {1,1,2,2}; /* conv2_outCh, conv2_inCh, conv2_kernelH, conv2_kernelW */
    int pool2_window_size = 1;
    int output_size[] = {1, 3, 3};  /* outCh, outHeight, outWidth */

    double t1, t2;
    double t_avg = 0;
    for(int i = 0; i < NUM_EXPERIMENTS; i++) {
        t1 = get_time();
        simple_net(input_size, conv1_size, pool1_window_size, conv2_size, pool2_window_size, output_size);
        t2 = get_time();
        t_avg += t2-t1;
    }
    t_avg /= NUM_EXPERIMENTS;
    printf("\nElapsed time : %f secondsn\n", t_avg);

    mkldnn_status_t result = simple_net(input_size, conv1_size, pool1_window_size, conv2_size, pool2_window_size, output_size);
    printf("%s\n", (result == mkldnn_success) ? "passed" : "failed");
    return result;
}
