#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/lsigmoid_layer.hpp"

namespace caffe {

// CUDA kernele for forward
template <typename Dtype>
__global__ void LSigmoidForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* slope_data ) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels;
    out[index] = 1. /( 1. + exp( -in[index] * slope_data[c] ) );

//    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void LSigmoidBackward(const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* out_data, Dtype* out_diff,
    const Dtype* slope_data ) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels;

    out_diff[index] = in_diff[index] * slope_data[c] * out_data[index] * ( 1. - out_data[index] );
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void LSigmoidParamBackward(const int n,
    const int rows, const int rowPitch, const Dtype* in_diff,
    const Dtype* in_data, const Dtype* out_data ,Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {

    out_diff[index] = in_diff[index] * in_data[index] * out_data[index] * (1. - out_data[index]);
//    out_diff[index] = in_diff[index] * in_data[index] * (in_data[index] <= 0);
    for ( int k = 1; k < rows; k++ ) {
        out_diff[index] += in_diff[index + k*rowPitch]
           * in_data[index + k*rowPitch] * out_data[index + k*rowPitch] * ( 1. - out_data[index + k*rowPitch] );
    }
  }
}

template <typename Dtype>
void LSigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* slope_data = this->blobs_[0]->gpu_data();

  // NOLINT_NEXT_LINE(whitespace/operators)
  LSigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data, slope_data );
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void LSigmoidLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* slope_diff = this->blobs_[0]->mutable_gpu_diff();
    int cdim = channels * dim;

    // compute element-wise diff
    // NOLINT_NEXT_LINE(whitespace/operators)
    LSigmoidParamBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
      CAFFE_CUDA_NUM_THREADS>>>(
      cdim, bottom[0]->num(), top[0]->offset(1), top_diff ,
      bottom_data , top_data, 
      backward_buff_.mutable_gpu_diff());
    CUDA_POST_KERNEL_CHECK;

    caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
      backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
      slope_diff);
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* slope_data = this->blobs_[0]->gpu_data();
//    int div_factor = channel_shared_ ? channels : 1;
    // NOLINT_NEXT_LINE(whitespace/operators)
    LSigmoidBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, top_diff, top_data, bottom_diff, slope_data );
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(LSigmoidLayer);


}  // namespace caffe
