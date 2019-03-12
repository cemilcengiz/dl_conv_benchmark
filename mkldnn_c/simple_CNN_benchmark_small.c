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

mkldnn_status_t simple_net() {
    /* input-output sizes of overall system */
    int inBS, inCh, inHeight, inWidth, outCh, outHeight, outWidth;
    inBS = 1; inCh = 1; inHeight = 6; inWidth = 6;
    outCh = 1; outHeight = 4; outWidth = 4;

    mkldnn_engine_t engine;
    CHECK(mkldnn_engine_create(&engine, mkldnn_cpu, 0 /* idx */));

    float *net_src = (float *)aligned_malloc(
            inBS * inCh * inHeight * inWidth * sizeof(float), 64);
    float *net_dst = (float *)aligned_malloc(
            inBS * outCh * outHeight * outWidth * sizeof(float), 64);

            /* check output memory before of computation */
            int outSize = inBS * outCh * outHeight * outWidth;
            printf("\nOutput values before calculation:\n");
            for (int i = 0; i < outSize; i++) {
                printf("%d th entry = %f\n", i ,net_dst[i]);
            }

    /* fill the user input tensor with values for testing */
    int inSize = inBS * inCh * inHeight * inWidth;
    for (int i = 0; i < inSize; i++) {
        net_src[i] = i;
    }
    printf("Input values: \n");
    for (int i = 0; i < inSize; i++) {
        printf("%d th entry = %f\n", i ,net_src[i]);
    }

    /* conv1
     * {inBS, inCh, inHeight, inWidth} (x) {conv1_outCh, conv1_inCh, conv1_kernelH, conv1_kernelW} ->
     * {inBS, conv1_outCh, conv1_outH, conv1_outW}
     * strides: {1, 1}
     */
    int conv1_outCh, conv1_inCh, conv1_kernelH, conv1_kernelW, conv1_outW, conv1_outH;
    conv1_outCh = 1; conv1_inCh = inCh; conv1_kernelH = 3; conv1_kernelW = 3;
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
    printf("conv1 values: \n");
    for (int i = 0; i < conv1_weightsSize; i++) {
        printf("%d th entry = %f\n", i ,conv1_weights[i]);
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


    /* create memory for user data in output part */
    mkldnn_primitive_t conv1_user_dst_memory;
    init_data_memory(4, conv1_user_dst_sizes, mkldnn_nchw, mkldnn_f32, engine,
        net_dst, &conv1_user_dst_memory);


    /* create a convolution */
    /* create conv operation descrisptor */
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
    /* create memory for dst1 data */
    const_mkldnn_primitive_desc_t conv1_dst_pd
            = mkldnn_primitive_desc_query_pd(conv1_pd, mkldnn_query_dst_pd, 0);
    CHECK(mkldnn_primitive_create(
            &conv1_internal_dst_memory, conv1_dst_pd, NULL, NULL));
    size_t conv1_dst_size = mkldnn_memory_primitive_desc_get_size(conv1_dst_pd);
    float *conv1_dst_buffer = (float *)aligned_malloc(conv1_dst_size, 64);
    /* create reorder primitives between user data and conv dsts
       * if required */
       mkldnn_primitive_t conv1_reorder_dst;

      CHECK(prepare_reorder(&conv1_user_dst_memory, &conv1_dst_pd, 0,
              &conv1_internal_dst_memory, &conv1_reorder_dst, conv1_dst_buffer)); /* 0 for prim->user */

     mkldnn_primitive_t conv1_dst_memory = conv1_internal_dst_memory ? conv1_internal_dst_memory
                  : conv1_user_dst_memory;

    const_mkldnn_primitive_t conv1_dsts[] = { conv1_dst_memory };

    /* finally create a convolution primitive */
    mkldnn_primitive_t conv1;
    CHECK(mkldnn_primitive_create(&conv1, conv1_pd, conv1_srcs, conv1_dsts));


    /* build a simple net */
    uint32_t n = 0;
    mkldnn_primitive_t net[10];

    if (conv1_reorder_src) net[n++] = conv1_reorder_src;
    if (conv1_reorder_weights) net[n++] = conv1_reorder_weights;
    net[n++] = conv1;
    if (conv1_reorder_dst) net[n++] = conv1_reorder_dst;


    mkldnn_stream_t stream;
    CHECK(mkldnn_stream_create(&stream, mkldnn_eager));
    CHECK(mkldnn_stream_submit(stream, n, net, NULL));
    CHECK(mkldnn_stream_wait(stream, n, NULL));


    /* check result of computation */
    printf("\nOutput values after calculation:\n");

    int conv1_out_Size = inBS * conv1_outCh * conv1_outW * conv1_outH;
    void* _conv1_out_data_handle = NULL;
    CHECK(mkldnn_memory_get_data_handle(conv1_internal_dst_memory, &_conv1_out_data_handle));

    float* conv1_out_data_handle = (float *)_conv1_out_data_handle;
    printf("\nHandle version Output values after calculation:");
    printf("\nThis is incorect. We omitted the blocking layout i.e. channels are dilated to be multiple of 8n:");
    printf("\nHence, we will see partial results, and lots of 0sn:\n");
    for (int i = 0; i < conv1_out_Size; i++) {
        printf("Handled %d th entry = %f\n", i ,conv1_out_data_handle[i]);

    }

    const_mkldnn_primitive_desc_t conv1_dst_memory_pd;
    mkldnn_primitive_get_primitive_desc(conv1_internal_dst_memory, &conv1_dst_memory_pd);
    size_t sz = mkldnn_memory_primitive_desc_get_size(conv1_dst_memory_pd);
    sz /= (sizeof(float));
    printf("\nRaw outputs i.e. without reordering. The channel size is ceiled to be multiple of 8");
    printf("\nHence if you expect single channel, you get 8 (i.e. 7 chs extra)\n");
    for(size_t i = 0; i < sz; i++)
    {
        printf("%ld : %f\n", i, conv1_out_data_handle[i]);
    }

    printf("\nYou can see correct values by looking at only one channel out of 8\n");
    for (int i = 0; i < conv1_outH; i++) {
        for (int j = 0; j < conv1_outW ; j++) {
            printf("%-10f, ", conv1_out_data_handle[(conv1_outW *i + j) * 8]);
        } printf("\n");
    } printf("\n\n");

    /* since we did the conv out reordering i.e., out->prim, we don't need to
    bother with blocks. Just use the reordered output finally */
    printf("Reorder outputs\n i.e the simplest way if you did reordering\n");
    for (int i = 0; i < outSize; i++) {
        printf("Directly %d th entry = %f\n", i ,net_dst[i]);
    }

    /* clean-up */
    CHECK(mkldnn_primitive_desc_destroy(conv1_pd));

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
    mkldnn_primitive_destroy(conv1_user_dst_memory);
    mkldnn_primitive_destroy(conv1_reorder_dst);
    mkldnn_primitive_destroy(conv1_reorder_src);
    mkldnn_primitive_destroy(conv1_reorder_weights);
    mkldnn_primitive_destroy(conv1);

    _free(conv1_weights);
    _free(conv1_bias);
    _free(conv1_src_buffer);
    _free(conv1_weights_buffer);
    _free(conv1_dst_buffer);


    mkldnn_engine_destroy(engine);

    return mkldnn_success;
}

/***
* Usage:
To compile:
gcc -Wall simple_CNN_benchmark_small.c -o bin/simple_CNN_benchmark_small_c -I /kuacc/users/ccengiz17/MKL_DNN/mkl-dnn/include -L /kuacc/users/ccengiz17/MKL_DNN/mkl-dnn/build/src -lmkldnn -std=c99

To add the dynamic library:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/kuacc/users/ccengiz17/MKL_DNN/mkl-dnn/build/src/
***/



int main(int argc, char **argv) {
    mkldnn_status_t result = simple_net();
    printf("%s\n", (result == mkldnn_success) ? "passed" : "failed");
    return result;
}
