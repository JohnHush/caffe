#include <algorithm>
#include <vector>

#include "caffe/layers/rpn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RPNLayer<Dtype>::LayerSetUp( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top )
{

  feat_stride_ = this->layer_param_.rpn_param().feat_stride();
  base_size_ = this->layer_param_.rpn_param().basesize();
  min_size_ = this->layer_param_.rpn_param().boxminsize();
  pre_nms_topN_ = this->layer_param_.rpn_param().pre_nms_topn();
  post_nms_topN_ = this->layer_param_.rpn_param().post_nms_topn();
  nms_thresh_ = this->layer_param_.rpn_param().nms_thresh();

  vector<Dtype> anchor_ratio;
  vector<Dtype> anchor_scale;

  for ( int i = 0; i < this->layer_param_.rpn_param().scale_size(); ++i )
    anchor_scale.push_back( this->layer_param_.rpn_param().scale(i) );

  for ( int i = 0; i < this->layer_param_.rpn_param().ratio_size() ; ++i )
    anchor_ratio.push_back( this->layer_param_.rpn_param().ratio(i) );

  _generate_anchors( anchor_ratio , anchor_scale );

  if ( template_anchors_pt_ != NULL )
  {
    delete template_anchors_pt_;
    template_anchors_pt_ = NULL;
  }
  template_anchors_pt_ =  new Dtype[ 4 * template_anchors_.size() ];
  anchors_num_ = template_anchors_.size();

  for ( int i = 0 ; i < template_anchors_.size() ; ++i )
  {
    template_anchors_pt_[ 4 * i + 0 ] = template_anchors_[i][0];
    template_anchors_pt_[ 4 * i + 1 ] = template_anchors_[i][1];
    template_anchors_pt_[ 4 * i + 2 ] = template_anchors_[i][2];
    template_anchors_pt_[ 4 * i + 3 ] = template_anchors_[i][3];
  }

  vector<int> top0_shape;
  top0_shape.push_back( 1 );	
  top0_shape.push_back( 5 );	
  top[0]->Reshape( top0_shape );
  if (top.size() > 1)
    top[1]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void RPNLayer<Dtype>::_generate_anchors( vector<Dtype>& anchor_ratio , vector<Dtype>& anchor_scale )
{
  template_anchors_.clear();
  vector<Dtype> base_anchor;

  base_anchor.push_back( 0 );
  base_anchor.push_back( 0 );
  base_anchor.push_back( base_size_ - 1 );
  base_anchor.push_back( base_size_ - 1 );

  vector<vector<Dtype> >ratio_anchors = _ratio_enum( base_anchor , anchor_ratio );
  for (int i = 0; i < ratio_anchors.size(); ++i)
  {
    vector<vector<Dtype> > tmp = _scale_enum( ratio_anchors[i] , anchor_scale );
    template_anchors_.insert(template_anchors_.end(), tmp.begin(), tmp.end());
  }
}

template <typename Dtype>
vector<vector<Dtype> > RPNLayer<Dtype>::_scale_enum( const vector<Dtype>& anchor ,
    const vector<Dtype>& anchor_scale )
{
  vector<vector<Dtype> > bbox_diff_scale;

  Dtype x_ctr = ( anchor[0] + anchor[2] )/ 2;
  Dtype y_ctr = ( anchor[1] + anchor[3] )/ 2;
  Dtype w     =   anchor[2] - anchor[0] + 1;
  Dtype h     =   anchor[3] - anchor[1] + 1;

  for ( int i = 0; i < anchor_scale.size(); ++i )
    bbox_diff_scale.push_back( _mkanchor( w * anchor_scale[i] , h * anchor_scale[i] , x_ctr, y_ctr ) );

  return bbox_diff_scale;
}

template <typename Dtype>
vector<vector<Dtype> > RPNLayer<Dtype>::_ratio_enum( const vector<Dtype>& anchor , 
    const vector<Dtype>& anchor_ratio )
{
  vector<vector<Dtype> > bbox_diff_ratio;

  Dtype x_ctr = ( anchor[0] + anchor[2] )/ 2;
  Dtype y_ctr = ( anchor[1] + anchor[3] )/ 2;
  Dtype area  = ( anchor[2] - anchor[0] + 1 ) *  ( anchor[3] - anchor[1] + 1 );

  for ( int i = 0; i < anchor_ratio.size(); ++i )
  {
    Dtype area_ratios = area / anchor_ratio[i];
    Dtype ws = round( sqrt(area_ratios) );
    Dtype hs = round( ws * anchor_ratio[i] );
    bbox_diff_ratio.push_back( _mkanchor( ws, hs, x_ctr, y_ctr ) );
  }
  return bbox_diff_ratio;
}

template <typename Dtype>
vector<Dtype> RPNLayer<Dtype>::_mkanchor( Dtype w, Dtype h, Dtype x_ctr, Dtype y_ctr )
{
  vector<Dtype> bbox_xyxy;
  bbox_xyxy.push_back( x_ctr - 0.5*(w - 1) );
  bbox_xyxy.push_back( y_ctr - 0.5*(h - 1) );
  bbox_xyxy.push_back( x_ctr + 0.5*(w - 1) );
  bbox_xyxy.push_back( y_ctr + 0.5*(h - 1) );

  return bbox_xyxy;
}

template <typename Dtype>
vector<Dtype> RPNLayer<Dtype>::_whctrs( const vector<Dtype>& bbox)
{
  vector<Dtype> bbox_whctrs;
  bbox_whctrs.push_back( bbox[2]  - bbox[0] + 1 ); //w
  bbox_whctrs.push_back( bbox[3]  - bbox[1] + 1 ); //h
  bbox_whctrs.push_back( (bbox[2] + bbox[0]) / 2); //ctrx
  bbox_whctrs.push_back( (bbox[3] + bbox[1]) / 2); //ctry

  return bbox_whctrs;
}

template <typename Dtype>
void RPNLayer<Dtype>::proposal_local_anchor()
{
  
  // shifted anchors point in the order [anchors_number][4][fmap_heigh_][fmap_width_]
  int fmap_area = fmap_heigh_ * fmap_width_;

  if ( shifted_anchors_pt_ != NULL )
  {
    delete shifted_anchors_pt_;
    shifted_anchors_pt_ = NULL;
  }
  shifted_anchors_pt_ = new Dtype[ anchors_num_ * 4 * fmap_area ];

  Dtype* shift_x = new Dtype[ fmap_area ];
  Dtype* shift_y = new Dtype[ fmap_area ];

  for ( int ih = 0 ; ih < fmap_heigh_ ; ++ ih )
  for ( int iw = 0 ; iw < fmap_width_ ; ++ iw )
  {
    shift_x[ ih * fmap_width_ + iw ] = iw * feat_stride_;
    shift_y[ ih * fmap_width_ + iw ] = ih * feat_stride_;
  }

  for ( int ia = 0 ; ia < anchors_num_ ; ++ ia )
  {
    caffe_set( fmap_area , template_anchors_pt_[ ia * 4 + 0 ] , shifted_anchors_pt_ + ( ia * 4 + 0 ) * fmap_area );
    caffe_set( fmap_area , template_anchors_pt_[ ia * 4 + 1 ] , shifted_anchors_pt_ + ( ia * 4 + 1 ) * fmap_area );
    caffe_set( fmap_area , template_anchors_pt_[ ia * 4 + 2 ] , shifted_anchors_pt_ + ( ia * 4 + 2 ) * fmap_area );
    caffe_set( fmap_area , template_anchors_pt_[ ia * 4 + 3 ] , shifted_anchors_pt_ + ( ia * 4 + 3 ) * fmap_area );

    caffe_axpy( fmap_area, Dtype(1) , shift_x , shifted_anchors_pt_ + ( ia * 4 + 0 ) * fmap_area );
    caffe_axpy( fmap_area, Dtype(1) , shift_y , shifted_anchors_pt_ + ( ia * 4 + 1 ) * fmap_area );
    caffe_axpy( fmap_area, Dtype(1) , shift_x , shifted_anchors_pt_ + ( ia * 4 + 2 ) * fmap_area );
    caffe_axpy( fmap_area, Dtype(1) , shift_y , shifted_anchors_pt_ + ( ia * 4 + 3 ) * fmap_area );
  }

  delete shift_x;
  delete shift_y;
  shift_x = NULL;
  shift_y = NULL;
}

template<typename Dtype>
void RPNLayer<Dtype>::filter_boxs( vector<Scored_BBOX>& sBBOX,  const vector<Blob<Dtype>*>& bottom )
{
  Dtype MinSize = min_size_ * bottom[2]->cpu_data()[2];
  Dtype ori_heigh = bottom[2]->cpu_data()[0];
  Dtype ori_width = bottom[2]->cpu_data()[1];

  // the first partion of the blob stores bg probability
  // the 2nd partion of the blob stores fg probability
  const Dtype* scores = bottom[0]->cpu_data() + fmap_heigh_ * fmap_width_ * template_anchors_.size();

  sBBOX.clear();
  int fmap_area = fmap_heigh_ * fmap_width_;
  
  for ( int ih = 0 ; ih < fmap_heigh_ ; ++ ih )
  for ( int iw = 0 ; iw < fmap_width_ ; ++ iw )
  for ( int ia = 0 ; ia < anchors_num_ ; ++ ia )
  {
    Dtype w = inversed_anchors_pt_[ ia * 4 * fmap_area + 2 * fmap_area + ih * fmap_width_ + iw ];
    Dtype h = inversed_anchors_pt_[ ia * 4 * fmap_area + 3 * fmap_area + ih * fmap_width_ + iw ];
    
    if ( w < MinSize || h < MinSize )
      continue;

    Scored_BBOX sbb;
    sbb.bi = 0;
    sbb.sc = scores[ ia * fmap_area + ih * fmap_width_ + iw ];
    sbb.x1 = std::min( std::max( Dtype(0) , inversed_anchors_pt_[ ia * 4 * fmap_area + 0 * fmap_area + ih * fmap_width_ + iw ] 
          - w/2 ) , ori_width );
    sbb.x2 = std::min( std::max( Dtype(0) , inversed_anchors_pt_[ ia * 4 * fmap_area + 0 * fmap_area + ih * fmap_width_ + iw ] 
          + w/2 ) , ori_width );
    sbb.y1 = std::min( std::max( Dtype(0) , inversed_anchors_pt_[ ia * 4 * fmap_area + 1 * fmap_area + ih * fmap_width_ + iw ] 
          - h/2 ) , ori_heigh );
    sbb.y2 = std::min( std::max( Dtype(0) , inversed_anchors_pt_[ ia * 4 * fmap_area + 1 * fmap_area + ih * fmap_width_ + iw ] 
          + h/2 ) , ori_heigh );

    sBBOX.push_back( sbb );
  }
}

template<typename Dtype>
void RPNLayer<Dtype>::bbox_tranform_inv( const vector<Blob<Dtype>*>& bottom )
{
  // convert the bottom[1] the regressed blob into vector form
  // which is compatible with shifted_anchors
  // with the form [ H * W * A ][4]
 
  const Dtype* regressed_blob = bottom[1]->cpu_data();
  // inversed anchors point has the order: [anchors_num][4][fmap_heigh_][fmap_width_]
  if ( inversed_anchors_pt_ != NULL )
  {
    delete inversed_anchors_pt_;
    inversed_anchors_pt_ = NULL;
  }

  inversed_anchors_pt_ = new Dtype[ anchors_num_ * 4 * fmap_heigh_ * fmap_width_ ];
  int fmap_area = fmap_heigh_ * fmap_width_;
  caffe_copy( anchors_num_ * 4 * fmap_area , regressed_blob , inversed_anchors_pt_ );

  for ( int ia = 0 ; ia < anchors_num_ ; ++ ia )
  {
    // convert the shifted bboxes into Cx, Cy, W, H format
    caffe_axpy( 2 * fmap_area , Dtype(-1) , shifted_anchors_pt_ + ( ia *4 + 0 )* fmap_area , 
        shifted_anchors_pt_ + ( ia *4 + 2 )* fmap_area );
    caffe_add_scalar( 2 * fmap_area , Dtype(1) , shifted_anchors_pt_ + ( ia *4 + 2 )* fmap_area );
    caffe_axpy( 2 * fmap_area , Dtype(0.5) , shifted_anchors_pt_ + ( ia *4 + 2 )* fmap_area , 
        shifted_anchors_pt_ + ( ia *4 + 0 )* fmap_area );

    caffe_mul( 2 * fmap_area , shifted_anchors_pt_ + ( ia *4 + 2 )* fmap_area , inversed_anchors_pt_ + ( ia *4 + 0 )* fmap_area 
        , inversed_anchors_pt_ + ( ia *4 + 0 )* fmap_area );
    caffe_add( 2 * fmap_area , shifted_anchors_pt_ + ( ia *4 + 0 )* fmap_area , inversed_anchors_pt_ + ( ia *4 + 0 )* fmap_area
        , inversed_anchors_pt_ + ( ia *4 + 0 )* fmap_area );

    caffe_exp( 2* fmap_area , inversed_anchors_pt_ + ( ia *4 + 2 )* fmap_area , inversed_anchors_pt_ + ( ia *4 + 2 )* fmap_area );
    caffe_mul( 2* fmap_area , shifted_anchors_pt_ + ( ia *4 + 2 )* fmap_area , inversed_anchors_pt_ + ( ia *4 + 2 )* fmap_area 
        , inversed_anchors_pt_ + ( ia *4 + 2 )* fmap_area );
  }
}

template<typename Dtype>
void RPNLayer<Dtype>::nms( vector<Scored_BBOX>& sBBOX )
{
  vector<Dtype> sArea( sBBOX.size() );
  for ( int i = 0 ; i < sBBOX.size() ; ++ i )
    sArea[i] = ( sBBOX[i].x2 - sBBOX[i].x1 + 1 ) * ( sBBOX[i].y2 - sBBOX[i].y1 + 1);

  for ( int i = 0     ; i < sBBOX.size() ; ++ i )
  for ( int j = i + 1 ; j < sBBOX.size() ; )
  {
    Dtype x1 = std::max( sBBOX[i].x1 , sBBOX[j].x1 );
    Dtype y1 = std::max( sBBOX[i].y1 , sBBOX[j].y1 );
    Dtype x2 = std::min( sBBOX[i].x2 , sBBOX[j].x2 );
    Dtype y2 = std::min( sBBOX[i].y2 , sBBOX[j].y2 );

    Dtype w  = std::max( Dtype(0) , x2 - x1 + 1 );
    Dtype h  = std::max( Dtype(0) , y2 - y1 + 1 );

    Dtype ovr= ( w * h )/( sArea[i] + sArea[j] - w * h );
    if ( ovr >= nms_thresh_ )
    {
      sBBOX.erase( sBBOX.begin() + j );
      sArea.erase( sArea.begin() + j );
    }
    else
      j++;
  }

}

template <typename Dtype>
void RPNLayer<Dtype>::Forward_cpu( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top )
{
  fmap_width_ = bottom[1]->width();
  fmap_heigh_ = bottom[1]->height();

  proposal_local_anchor();

  bbox_tranform_inv( bottom );

  vector<Scored_BBOX> sBBOX;
  filter_boxs( sBBOX , bottom );

  std::sort( sBBOX.rbegin() , sBBOX.rend() );
  if (pre_nms_topN_ > 0)
    //sBBOX.resize( std::min<int>(pre_nms_topN_, sBBOX.size()) );
    sBBOX.erase( sBBOX.begin() + std::min<int>(pre_nms_topN_, sBBOX.size()) , sBBOX.end() );

  nms( sBBOX );

  if (post_nms_topN_ > 0)
    //sBBOX.resize( std::min<int>( post_nms_topN_, sBBOX.size()) );
    sBBOX.erase( sBBOX.begin() + std::min<int>(post_nms_topN_, sBBOX.size()) , sBBOX.end() );

  vector<int> top0_shape;
  top0_shape.push_back( sBBOX.size() );
  top0_shape.push_back( 5 );
  top[0]->Reshape( top0_shape );
  Dtype *top0 = top[0]->mutable_cpu_data();
  for (int i = 0; i < sBBOX.size(); ++i)
  {
    top0[0] = sBBOX[i].bi;
    top0[1] = sBBOX[i].x1;
    top0[2] = sBBOX[i].y1;
    top0[3] = sBBOX[i].x2;
    top0[4] = sBBOX[i].y2;

    top0 += top[0]->offset(1);
  }
  if (top.size()>1)
  {
    top[1]->Reshape( sBBOX.size(), 1,1,1);
    Dtype *top1 = top[1]->mutable_cpu_data();
    for (int i = 0; i < sBBOX.size(); ++i)
    {
      top1[0] = sBBOX[i].sc;
      top1++;
    }
  }    
}

#ifdef CPU_ONLY
STUB_GPU(RPNLayer);
#endif

INSTANTIATE_CLASS(RPNLayer);
REGISTER_LAYER_CLASS(RPN);

}  // namespace caffe<strong>

