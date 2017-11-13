#ifndef CAFFE_MULTI_LABEL_SIGMOID_CROSS_ENCTROP_LOSS_LAYER_HPP_
#define CAFFE_MULTI_LABEL_SIGMOID_CROSS_ENCTROP_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

/**
 * follow the formulae given in D.W. Li, 2015
 * "Multi Attribute Learning for Pedestrian Attribute Recognition in Surveillance Scenarios"
 * formulae (5)
 * we build a new message in caffe.proto to capture the hyper-prameter "weights" of each 
 * attribute 
 * At last, the formulae seems just a little bit different compared to the original sigmoid
 * cross entroy loss
 *
 * Author: Luo Heng
 *
 * Date: Nov. 12, 2017 in Chengdu. JINRIYUEDU BOOKSTORE
 */
template <typename Dtype>
class MultiLabelSigmoidCrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultiLabelSigmoidCrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
          sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
          sigmoid_output_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiLabelSigmoidCrossEntropyLoss"; }

 protected:
  /// @copydoc SigmoidCrossEntropyLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
	 * #TODO
	 * should be finished if you wanna generate GEXGEN file
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// Read the normalization mode parameter and compute the normalizer based
  /// on the blob size.  If normalization_mode is VALID, the count of valid
  /// outputs will be read from valid_count, unless it is -1 in which case
  /// all outputs are assumed to be valid.
  virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, int valid_count);

  /// The internal SigmoidLayer used to map predictions to probabilities.
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  /// sigmoid_output stores the output of the SigmoidLayer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// top vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_top_vec_;


	// don't support ignore label in this version
	// maybe in the next version

  /// Whether to ignore instances with a certain label.
//  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
//  int ignore_label_;
  /// How to normalize the loss.
  LossParameter_NormalizationMode normalization_;
  Dtype normalizer_;
  int outer_num_, inner_num_;
	MultiLabelSigmoidCrossEntropyLossParameter MLSCELPara_;

	std::vector<Dtype> positive_ratio_;
	// the pairs store the weights for POSITIVE label and NEGATIVE label, respectively
	std::vector<std::pair<Dtype, Dtype>  > attribute_weights_;
	Dtype epsilon_;
	int attributes_number_;
	bool has_positive_ratio_;
};

}  // namespace caffe

#endif  // CAFFE_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_
