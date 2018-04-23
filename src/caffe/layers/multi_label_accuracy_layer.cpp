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

	attributes_number_ = bottom[0]->shape(1);
	batch_number_ = bottom[0]->shape(0);

	vector<int> top1_holder_shape(1);
	top1_holder_shape[0] = attributes_number_;
	top1_holder_.Reshape( top1_holder_shape );

  positive_holder_.Reshape( top1_holder_shape );
  negative_holder_.Reshape( top1_holder_shape );
  positive_counter_.Reshape( top1_holder_shape );
  negative_counter_.Reshape( top1_holder_shape );
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	CHECK_EQ( bottom[0]->count() , bottom[1]->count() )
		<< "two bottom blob should have the same count";

	attributes_number_ = bottom[0]->shape(1);
	batch_number_ = bottom[0]->shape(0);
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);

	vector<int> top1_holder_shape(1);
	top1_holder_shape[0] = attributes_number_;
	top1_holder_.Reshape( top1_holder_shape );

  positive_holder_.Reshape( top1_holder_shape );
  negative_holder_.Reshape( top1_holder_shape );
  positive_counter_.Reshape( top1_holder_shape );
  negative_counter_.Reshape( top1_holder_shape );

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
    const vector<Blob<Dtype>*>& top)
{
  /*
   * define Accuracy as the average of Positive Samples Accuracy and 
   * Negative Samples Accuracy, in some cases maybe the Pos,Neg difference
   * will be great
   */
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();

	caffe_set( top1_holder_.count() , Dtype(0) , top1_holder_.mutable_cpu_data() );

	caffe_set( positive_holder_.count() , Dtype(0) , positive_holder_.mutable_cpu_data() );
	caffe_set( negative_holder_.count() , Dtype(0) , negative_holder_.mutable_cpu_data() );
	caffe_set( positive_counter_.count() , Dtype(0) , positive_counter_.mutable_cpu_data() );
	caffe_set( negative_counter_.count() , Dtype(0) , negative_counter_.mutable_cpu_data() );

  if (top.size() > 1)
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
//  int count = 0;
  for (int i = 0; i < batch_number_ ; ++i)
  for (int j = 0; j < attributes_number_ ; ++j)
  {
    const int label_value = static_cast<Dtype>(bottom_label[i * attributes_number_ + j]);
    const Dtype predict_value = static_cast<Dtype>(bottom_data[i * attributes_number_ + j]);

    bool ifright = ( std::fabs( predict_value - label_value ) < 0.5 );

    if ( label_value == 0 )
      ++ negative_counter_.mutable_cpu_data()[j];
    else
      ++ positive_counter_.mutable_cpu_data()[j];

    if ( ifright && predict_value < 0.5 )
    {
      ++ negative_holder_.mutable_cpu_data()[j];

    }
    if ( ifright && predict_value >=0.5 )
      ++ positive_holder_.mutable_cpu_data()[j];
    //			++count;
  }

  for ( int i = 0 ; i < positive_holder_.count() ; ++ i )
  {
    if ( positive_counter_.cpu_data()[i] == 0 )
      positive_holder_.mutable_cpu_data()[i] = 0.;
    else
      positive_holder_.mutable_cpu_data()[i] /= positive_counter_.cpu_data()[i];

    if ( negative_counter_.cpu_data()[i] == 0 )
      negative_holder_.mutable_cpu_data()[i] = 0.;
    else
      negative_holder_.mutable_cpu_data()[i] /= negative_counter_.cpu_data()[i];
  }

  top[0]->mutable_cpu_data()[0] = 0;

  for ( int i = 0 ; i < positive_holder_.count() ; ++ i )
  {
    Dtype tmp = ( positive_holder_.cpu_data()[i] + negative_holder_.cpu_data()[i] ) / 2.;
    top[0]->mutable_cpu_data()[0] += tmp / attributes_number_;

    if ( top.size() > 1 )
      top[1]->mutable_cpu_data()[i] = tmp;
  }
}

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);
REGISTER_LAYER_CLASS(MultiLabelAccuracy);

}  // namespace caffe
