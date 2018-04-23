#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/multi_label_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
MultiLabelDataLayer<Dtype>::MultiLabelDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    offset_() {
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
MultiLabelDataLayer<Dtype>::~MultiLabelDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void MultiLabelDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  MultiLabelDatum ml_datum;
  ml_datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape( ml_datum.datum() );
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);

  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }

  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape( 2, 1 );
		label_shape[0] = batch_size;
		label_shape[1] = ml_datum.mt_label_size();
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
    }
  }
}


// just copy the auxiliary function from class DataLayer
// these two following functions just accompanish the ability
// of DataReader,,different solvers refer to the same source.
template <typename Dtype>
bool MultiLabelDataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype>
void MultiLabelDataLayer<Dtype>::Next() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}

// This function is called on prefetch thread
template<typename Dtype>
void MultiLabelDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();

  MultiLabelDatum ml_datum;

  Dtype* top_label = NULL;
  if( this->output_labels_ )
    top_label = batch->label_.mutable_cpu_data();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    while( Skip() ){
      Next();
    }
    // fetch a datum
    ml_datum.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();
    if( item_id == 0 ){
      vector<int> top_shape = this->data_transformer_->InferBlobShape( ml_datum.datum() );
      this->transformed_data_.Reshape(top_shape);
      top_shape[0] = batch_size;
      batch->data_.Reshape(top_shape);
    }

    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    int offset = batch->data_.offset(item_id);
    Dtype* top_data = batch->data_.mutable_cpu_data();
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform( ml_datum.datum(), &(this->transformed_data_) );
    // Copy label.

    if (this->output_labels_) {
			int label_offset = ml_datum.mt_label_size();

			for ( int i = 0 ; i < label_offset ; ++ i )
				top_label[i] = ml_datum.mt_label(i);

			top_label += label_offset;
    }

    trans_time += timer.MicroSeconds();
    Next();

  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(MultiLabelDataLayer);
REGISTER_LAYER_CLASS(MultiLabelData);

}  // namespace caffe
