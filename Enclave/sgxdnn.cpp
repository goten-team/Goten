#define USE_EIGEN_TENSOR

#include "sgxdnn_main.hpp"

#include "Enclave.h"
#include "Enclave_t.h"

#include "Crypto.h"

void ecall_init_tensor(uint64_t TenId, void* voidDims) {
    SecretInitTensor(TenId, voidDims);
}

void ecall_set_ten(uint64_t TenId, void* voidArr) {
    SecretSetTen(TenId, voidArr);
}
void ecall_get_ten(uint64_t TenId, void* voidArr) {
    SecretGetTen(TenId, voidArr);
}

void ecall_set_seed(uint64_t TenId, uint64_t RawSeed) {
    SecretSetSeed(TenId, RawSeed);
}

void ecall_get_random(uint64_t TenId, void* voidArr, uint64_t RawSeed) {
    SecretGetRandom(TenId, voidArr, RawSeed);
}

void ecall_get_share (uint64_t TenId, void* voidArr, uint64_t RawSeed) {
    SecretGetShare(TenId, voidArr, RawSeed);
}

void ecall_relu(uint64_t TenIdin, uint64_t TenIdout, uint64_t size){
    newrelu(TenIdin, TenIdout, size);
}

void ecall_reluback(uint64_t TenIdout, uint64_t TenIddout, uint64_t TenIddin, uint64_t size){
    newreluback(TenIdout, TenIddout, TenIddin, size);
}

void ecall_initmaxpool(uint64_t FunId, uint64_t TenIdin_trans, uint64_t TenIdout_trans){
    initmaxpool(FunId, TenIdin_trans, TenIdout_trans);
}

void ecall_maxpool(uint64_t FunId, uint64_t TenIdin, uint64_t TenIdout, uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,uint32_t output_height, uint32_t output_width, uint32_t filter_height, uint32_t filter_width, uint32_t row_stride,uint32_t col_stride, uint32_t row_pad, uint32_t col_pad) {
    newmaxpool(FunId, TenIdin, TenIdout, batch, channel, input_height, input_width, output_height, output_width, filter_height, filter_width, row_stride, col_stride, row_pad, col_pad);
}

void ecall_maxpoolback(uint64_t FunId, uint64_t TenIddout, uint64_t TenIddin, uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,uint32_t output_height, uint32_t output_width, uint32_t filter_height, uint32_t filter_width, uint32_t row_stride, uint32_t col_stride){
    newmaxpoolback(FunId, TenIddout, TenIddin, batch, channel, input_height, input_width, output_height, output_width, filter_height, filter_width, row_stride, col_stride);
}

void ecall_masking_c01(uint64_t storeId, uint64_t mainRawSeed, uint64_t rawSeed0, uint64_t rawSeed1, float *DstArr) {
    SecretMaskingC01(storeId, mainRawSeed, rawSeed0, rawSeed1, DstArr);
}

void ecall_add_from_cpu(void* inputArr, uint64_t dstId) {
    SecretAddFromCpu(inputArr, dstId);
}

void ecall_sgd_update(uint64_t paramId, uint64_t gradId, uint64_t momentumId,
                     float lr, float momentum, float weight_decay,
                     float dampening, int nesterov, int first_momentum) {
    SecretSgdUpdate(paramId, gradId, momentumId, lr, momentum, weight_decay, dampening, nesterov, first_momentum);
}

void ecall_stochastic_quantize(uint64_t src_id, uint64_t dst_id, uint64_t q_tag) {
    SecretStochasticQuantize(src_id, dst_id, q_tag);
}

void ecall_fused_quantize_share(uint64_t af_id, void* e_arr, uint64_t q_tag, uint64_t u_seed) {
    SecretFusedQuantizeShare(af_id, e_arr, q_tag, u_seed);
}

void ecall_fused_quantize_share2(uint64_t af_id, void* a1_arr, void* e_arr,
                               uint64_t q_tag, uint64_t a0_seed, uint64_t u_seed) {
    SecretFusedQuantizeShare2(af_id, a1_arr, e_arr, q_tag, a0_seed, u_seed);
}

void ecall_secret_fused_recon(uint64_t cf_id, uint64_t cq_id, float* c_left_arr, uint64_t x_tag, uint64_t y_tag) {
    SecretFusedRecon(cf_id, cq_id, c_left_arr, x_tag, y_tag);
}

void ecall_init_batchnorm(
        uint64_t FunId,
        uint64_t input, uint64_t output, uint64_t gamma, uint64_t beta,
        uint64_t der_input, uint64_t der_output, uint64_t der_gamma, uint64_t der_beta,
        uint64_t run_mean, uint64_t run_var, uint64_t cur_mean, uint64_t cur_var,
        uint64_t mu,
        uint32_t batch_, uint32_t channel_, uint32_t height_, uint32_t width_,
        int affine_, int is_cumulative_, float momentum_, float epsilon_) {

    SecretInitBatchnorm(
            FunId,
            input, output, gamma, beta,
            der_input, der_output, der_gamma, der_beta,
            run_mean, run_var, cur_mean, cur_var,
            mu,
            batch_, channel_, height_, width_,
            affine_, is_cumulative_, momentum_, epsilon_);
}

void ecall_batchnorm_forward(uint64_t FunId, int Training) {
    SecretBatchnormForward(FunId, Training);
}

void ecall_batchnorm_backward(uint64_t FunId) {
    SecretBatchnormBackward(FunId);
}
