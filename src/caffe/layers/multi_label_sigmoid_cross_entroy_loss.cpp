#include <algorithm>
#include <vector>

#include "caffe/layers/multi_label_sigmoid_cross_entroy_loss.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiLabelSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  bool has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();

  if (has_ignore_label_) {
  //  ignore_label_ = this->layer_param_.loss_param().ignore_label();
		LOG(WARNING) << "the ignore label indicator has been deprecated in this layer!";
  }
  if (this->layer_param_.loss_param().has_normalization()) {
    normalization_ = this->layer_param_.loss_param().normalization();
  } else if (this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = LossParameter_NormalizationMode_BATCH_SIZE;
  }

	const MultiLabelSigmoidCrossEntropyLossParameter& multi_label_loss_para = 
							this->layer_param_.multi_label_sigmoid_cross_entropy_loss();

	MLSCELPara_ = this->layer_param_.multi_label_sigmoid_cross_entropy_loss();

	// the number of positive_ratio should be NONE or equal to size of ATTRIBUTES in bottom[0]
	has_positive_ratio_ = multi_label_loss_para.positive_ratio_size() !=0 ;
	if ( has_positive_ratio_ )
		CHECK_EQ( multi_label_loss_para.positive_ratio_size() , bottom[0]->shape(1) ) <<
			" should set equal number of attributes' positive ratio";

	attributes_number_ = bottom[0]->shape(1);
	LOG(INFO) << " attributes number is " << attributes_number_;
	epsilon_ = multi_label_loss_para.epsilon();
	positive_ratio_.resize( attributes_number_  , 1 );
	attribute_weights_.resize( attributes_number_ , std::make_pair(1.,1.) );
	if ( has_positive_ratio_ ){
		for ( int i = 0 ; i < attributes_number_ ;  ++ i ) {
			positive_ratio_[i] = multi_label_loss_para.positive_ratio(i);
			Dtype tmp1 = std::exp( (1.-positive_ratio_[i])/(epsilon_ * epsilon_) );
			Dtype tmp2 = std::exp( (positive_ratio_[i])/(epsilon_ * epsilon_) );

			attribute_weights_[i] = std::make_pair( tmp1 , tmp2 );
		}
	}
}

template <typename Dtype>
void MultiLabelSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->shape(0);  // batch size
  inner_num_ = bottom[0]->count(1);  // instance size: |output| == |target|
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

// TODO(shelhamer) loss normalization should be pulled up into LossLayer,
// instead of duplicated here and in SoftMaxWithLossLayer
template <typename Dtype>
Dtype MultiLabelSigmoidCrossEntropyLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {

	// #TODO
	// I suppose this method won't affect the solution much
	// so temporally, i won't modify it
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void MultiLabelSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  int valid_count = 0;
  Dtype loss = 0;
  for (int i = 0; i < bottom[0]->count(); ++i) {
    const int target_value = static_cast<int>(target[i]);
		// we won't use this feature right now
//    if (has_ignore_label_ && target_value == ignore_label_) {
//      continue;
//    }
//    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
//        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    int att_index = i % attributes_number_;

    if ( target_value == 1 )
      loss -= attribute_weights_[att_index].first * log( sigmoid_output_->cpu_data()[i] );
    if ( target_value == 0 )
      loss -= attribute_weights_[att_index].second * log( 1. - sigmoid_output_->cpu_data()[i] );
    ++valid_count;
  }
  normalizer_ = get_normalizer(normalization_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;
}

template <typename Dtype>
void MultiLabelSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);

    Dtype* diff_weight = new Dtype[count];

    for ( int i = 0 ; i < count ; ++ i )
    {
      int att_index = i % attributes_number_;
      
      const int target_value = static_cast<int>(target[i]);

      if ( target_value == 1 )
        diff_weight[i] = attribute_weights_[att_index].first;
      if ( target_value == 0 )
        diff_weight[i] = attribute_weights_[att_index].second;
    }

    caffe_mul( count, bottom_diff , diff_weight , bottom_diff );
    delete diff_weight;

    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
//STUB_GPU(MultiLabelSigmoidCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(MultiLabelSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(MultiLabelSigmoidCrossEntropyLoss);

}  // namespace caffe
