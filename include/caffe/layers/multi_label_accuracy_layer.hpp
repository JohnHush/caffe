#ifndef CAFFE_MULTI_LABELACCURACY_LAYER_HPP_
#define CAFFE_MULTI_LABELACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the classification accuracy for classification of 
 * multi task
 *
 * this is modified from AccuracyLayer in Caffe
 *
 * Modified by Luo Heng
 *
 * Data: Nov 12th, 2017 in Sichuan
 *
 * The definition of mA could refer the paper, Dangwei Li, 2016, A Richly Annotated Dataset for Pedestrian Attribute Recognition
 */
template <typename Dtype>
class MultiLabelAccuracyLayer : public Layer<Dtype> {
 public:
  /**
   */
  explicit MultiLabelAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiLabelAccuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  // If there are two top blobs, then the second blob will contain
  // accuracies per class.
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  /**
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

	// actually a lot of them will be deprecated in this implementation
	// so keep your eyes on it
	// we won't use top_k , ignore__label , 
  int label_axis_, outer_num_, inner_num_;

  int top_k_;

  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// Keeps counts of the number of samples per class.
  Blob<Dtype> nums_buffer_;

	int attributes_number_;
	int batch_number_;

	Blob<Dtype> top1_holder_;
  Blob<Dtype> positive_holder_;
  Blob<Dtype> positive_counter_;
  Blob<Dtype> negative_holder_;
  Blob<Dtype> negative_counter_;
};

}  // namespace caffe

#endif  // CAFFE_ACCURACY_LAYER_HPP_
