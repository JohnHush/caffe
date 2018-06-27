#include <fstream>
#include <sstream>
#include "caffe/FRCNN/util/frcnn_vis.hpp"

namespace caffe {

namespace Frcnn {

template <typename Dtype>
void vis_detections(cv::Mat & frame, const std::vector<BBox<Dtype> >& ans, const std::map<int,std::string> CLASS) { 
  for(size_t i = 0 ; i < ans.size() ; i++) {
    cv::rectangle(frame, cv::Point(ans[i][0],ans[i][1]) , cv::Point(ans[i][2],ans[i][3]) , cv::Scalar(255,255,255) );
    std::ostringstream text;
    text << GetClassName(CLASS, ans[i].id) << "  :  " << ans[i].confidence;
    cv::putText(frame, text.str() , cv::Point(ans[i][0],ans[i][1]-18) , 0 , 0.6 , cv::Scalar(0,255,0) );
  }   
}

template <typename Dtype>
void vis_detections_v2( cv::Mat & frame,
              const std::vector<BBox<Dtype> >& ans,
              const std::map<int,std::string> CLASS,
              const int resize_width ,
              const int resize_heigh )
{
  float resize_ratio = float(resize_width)/frame.cols > float(resize_heigh)/frame.rows ? 
    float(resize_heigh)/frame.rows : float(resize_width)/frame.cols;

  cv::Size resize_size = cv::Size( int( resize_ratio * frame.cols) , 
        int( resize_ratio * frame.rows ) );

  cv::Mat for_resize( resize_size , frame.type() );

  cv::resize( frame , for_resize , resize_size );

  for(size_t i = 0 ; i < ans.size() ; i++)
  {
    cv::rectangle( for_resize , 
        cv::Point( int( resize_ratio * ans[i][0] ) , int( resize_ratio * ans[i][1]) ) , 
        cv::Point( int( resize_ratio * ans[i][2] ) , int( resize_ratio * ans[i][3]) ) ,
        cv::Scalar( 255 , 0 , 0 ) , 
        3 );

    cv::Rect drawing_roi( int( resize_ratio * ans[i][0] ),
                          int( resize_ratio * ans[i][1] ),
                          std::min( 100 , for_resize.cols -1 - int( resize_ratio * ans[i][0] )),
                          std::min( 20 , for_resize.rows - 1 - int( resize_ratio * ans[i][1] )));

    cv::Mat imgROI = for_resize( drawing_roi );
    imgROI.convertTo( imgROI , -1 , 0.3 , 32 );

    std::ostringstream text;
    text << ans[i].confidence;
    cv::putText(  for_resize , text.str() , 
                  cv::Point(int( resize_ratio * ans[i][0] ) , int( resize_ratio * ans[i][1] +12) )
                  , 2 , 0.6 , cv::Scalar(0,255,0) );
  }
  frame = for_resize;
}

template void vis_detections(cv::Mat & frame, const std::vector<BBox<float> >& ans, const std::map<int,std::string> CLASS);
template void vis_detections(cv::Mat & frame, const std::vector<BBox<double> >& ans, const std::map<int,std::string> CLASS);

template void vis_detections_v2(cv::Mat & frame, const std::vector<BBox<float> >& ans, const std::map<int,std::string> CLASS , const int resize_width, const int resize_heigh );
template void vis_detections_v2(cv::Mat & frame, const std::vector<BBox<double> >& ans, const std::map<int,std::string> CLASS , const int resize_width , const int resize_heigh );

template <typename Dtype>
void vis_detections(cv::Mat & frame, const BBox<Dtype> ans, const std::map<int,std::string> CLASS) { 
  std::vector<BBox<Dtype> > vec_ans;
  vec_ans.push_back( ans );
  vis_detections(frame, vec_ans, CLASS);
}

template void vis_detections(cv::Mat & frame, const BBox<float> ans, const std::map<int,std::string> CLASS);
template void vis_detections(cv::Mat & frame, const BBox<double> ans, const std::map<int,std::string> CLASS);

} // Frcnn

} // caffe 
