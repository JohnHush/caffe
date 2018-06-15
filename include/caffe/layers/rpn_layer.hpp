#ifndef CAFFE_RPN_LAYER_HPP_
#define CAFFE_RPN_LAYER_HPP_

#include <vector>

//#include "caffe/util/device_alternate.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "boost/shared_ptr.hpp"

namespace caffe {

  /**
   * @brief implement RPN layer for faster rcnn
   */

template <typename Dtype>
class RPNLayer : public Layer<Dtype> {
  public:
    explicit RPNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
        template_anchors_pt_ = NULL;
        shifted_anchors_pt_  = NULL;
        inversed_anchors_pt_ = NULL;
      }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){}
    virtual inline const char* type() const { return "RPN"; }

    ~RPNLayer(){
      delete template_anchors_pt_;
      delete shifted_anchors_pt_;
      delete inversed_anchors_pt_;

      template_anchors_pt_ = NULL;
      shifted_anchors_pt_  = NULL;
      inversed_anchors_pt_ = NULL;
    }
    struct Scored_BBOX{
      Dtype bi;
      Dtype x1;
      Dtype x2;
      Dtype y1;
      Dtype y2;
      Dtype sc;
      bool operator<( const Scored_BBOX& sb2) const
      {
        return sc < sb2.sc;
      }
    };

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){};

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};

    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};

    int feat_stride_;
    int base_size_;
    int min_size_;
    int pre_nms_topN_;
    int post_nms_topN_;
    float nms_thresh_;

    int fmap_width_;
    int fmap_heigh_;
    int anchors_num_;

    void proposal_local_anchor();
    void bbox_tranform_inv( const vector<Blob<Dtype>*>& bottom );

    Dtype* template_anchors_pt_;
    Dtype* shifted_anchors_pt_;
    Dtype* inversed_anchors_pt_;

  private:
    vector<Dtype> _whctrs( const vector<Dtype>& );
    vector<vector<Dtype> > _ratio_enum( const vector<Dtype>& , const vector<Dtype>& );
    vector<vector<Dtype> > _scale_enum( const vector<Dtype>& , const vector<Dtype>& );
    vector<Dtype> _mkanchor( Dtype w, Dtype h, Dtype x_ctr, Dtype y_ctr );
    void nms( vector<Scored_BBOX>& );
    vector<vector<Dtype> > template_anchors_;
    void _generate_anchors( vector<Dtype>& anchor_ratio , vector<Dtype>& anchor_scale );
    void filter_boxs( vector<Scored_BBOX>& , const vector<Blob<Dtype>*>& bottom );
};
}  // namespace caffe

#endif  // CAFFE_RPN_LAYER_HPP_

