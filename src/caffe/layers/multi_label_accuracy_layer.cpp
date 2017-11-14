#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/multi_label_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	if ( this->layer_param_.has_accuracy_param() )
		LOG(WARNING) << "accuracy parameter won't be used in this implementation";
//  top_k_ = this->layer_param_.accuracy_param().top_k();

//  has_ignore_label_ =
//    this->layer_param_.accuracy_param().has_ignore_label();
//  if (has_ignore_label_) {
//    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
//  }
	attributes_number_ = bottom[0]->shape(1);
	batch_number_ = bottom[0]->shape(0);

	vector<int> top1_holder_shape(1);
	top1_holder_shape[0] = attributes_number_;
	top1_holder_.Reshape( top1_holder_shape );
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
//      << "top_k must be less than or equal to the number of classes.";
//  label_axis_ =
//      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
//
//      the label axis will be assumed to be in axis= 1
//  outer_num_ = bottom[0]->count(0, label_axis_);
//  inner_num_ = bottom[0]->count(label_axis_ + 1);
	CHECK_EQ( bottom[0]->count() , bottom[1]->count() )
		<< "two bottom blob should have the same count";

	attributes_number_ = bottom[0]->shape(1);
	batch_number_ = bottom[0]->shape(0);
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);

	vector<int> top1_holder_shape(1);
	top1_holder_shape[0] = attributes_number_;
	top1_holder_.Reshape( top1_holder_shape );

  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = attributes_number_;
    top[1]->Reshape(top_shape_per_class);
    nums_buffer_.Reshape(top_shape_per_class);
  }
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
//  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
//  const int dim = bottom[0]->count() / outer_num_;
//  const int num_labels = bottom[0]->shape(label_axis_);
//  vector<Dtype> maxval(top_k_+1);
//  vector<int> max_id(top_k_+1);


	caffe_set( top1_holder_.count() , Dtype(0) , top1_holder_.mutable_cpu_data() );

  if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
//  int count = 0;
	for (int i = 0; i < batch_number_ ; ++i) {
		for (int j = 0; j < attributes_number_ ; ++j) {
			const int label_value =
				static_cast<Dtype>(bottom_label[i * attributes_number_ + j]);
			const Dtype predict_value =
				static_cast<Dtype>(bottom_data[i * attributes_number_ + j]);

			bool ifright = std::fabs( predict_value - label_value ) < 0.5 ;
				//		if (has_ignore_label_ && label_value == ignore_label_) {
				//			continue;
				//		}
//			if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[label_value];
//			DCHECK_GE(label_value, 0);
//			DCHECK_LT(label_value, num_labels);
//
//			in this implementaion label_value = 0 / 1 
			// Top-k accuracy
//			std::vector<std::pair<Dtype, int> > bottom_data_vector;
//			for (int k = 0; k < num_labels; ++k) {
//				bottom_data_vector.push_back(std::make_pair(
//							bottom_data[i * dim + k * inner_num_ + j], k));
//			}
//			std::partial_sort(
//					bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
//					bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
			// check if true label is in top k predictions
//			for (int k = 0; k < top_k_; k++) {
//				if (bottom_data_vector[k].second == label_value) {
//					++accuracy;
//					if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
//					break;
//				}
//			}
			
			if ( ifright )
			{
				++ top1_holder_.mutable_cpu_data()[j];
  			if (top.size() > 1) {
					++top[1]->mutable_cpu_data()[j];
				}
			}
//			++count;
		}
	}
	for ( int i = 0 ; i < top1_holder_.count(); ++ i )
		top1_holder_.mutable_cpu_data()[i] /= batch_number_;

  if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      top[1]->mutable_cpu_data()[i] /= batch_number_;
    }
  }
	
	top[0]->mutable_cpu_data()[0] = 0;
  for (int i = 0; i < top1_holder_.count(); ++i)
		top[0]->mutable_cpu_data()[0] += top1_holder_.mutable_cpu_data()[i]
			/ attributes_number_;
		
//  LOG(INFO) << "Accuracy: " << top[0]->mutable_cpu_data()[0];
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);
REGISTER_LAYER_CLASS(MultiLabelAccuracy);

}  // namespace caffe
