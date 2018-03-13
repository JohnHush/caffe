#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/lsigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void LSigmoidLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";

  LSigmoidParameter lsigmoid_param = this->layer_param().lsigmoid_param();

  int channels = bottom[0]->channels();

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
		// there is only scaling parameter
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));

		// if the filler doesn't set, a default filler with constant value '1' 
		// will be set
    shared_ptr<Filler<Dtype> > filler;
    if (lsigmoid_param.has_filler()) {
      filler.reset(GetFiller<Dtype>(lsigmoid_param.filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(1.);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[0].get());
  }
  CHECK_EQ(this->blobs_[0]->count(), channels)
      << "Negative slope size is inconsistent with prototxt config";

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  multiplier_.Reshape(vector<int>(1, bottom[0]->count(1)));
  backward_buff_.Reshape(vector<int>(1, bottom[0]->count(1)));
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void LSigmoidLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
		LOG(ERROR) << "doesn't support in-place manipulation right now";
  }
}

template <typename Dtype>
void LSigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* slope_data = this->blobs_[0]->cpu_data();

  for (int i = 0; i < count; ++i) {
    int c = (i / dim) % channels;
		top_data[i] = sigmoid( bottom_data[i] * slope_data[c] );
  }
}

template <typename Dtype>
void LSigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* slope_data = this->blobs_[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();

  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // Propagte to param

  if (this->param_propagate_down_[0]) {
    Dtype* slope_diff = this->blobs_[0]->mutable_cpu_diff();
    for ( int i = 0; i < count; ++i ) {
      int c = (i / dim) % channels;
      slope_diff[c] += top_diff[i] * bottom_data[i] * top_data[i] * (1. - top_data[i]) ;
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels;
      bottom_diff[i] = top_diff[i] * slope_data[c] * top_data[i] * (1. - top_data[i]);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU( LSigmoidLayer );
#endif

INSTANTIATE_CLASS( LSigmoidLayer );
REGISTER_LAYER_CLASS( LSigmoid );

}  // namespace caffe
