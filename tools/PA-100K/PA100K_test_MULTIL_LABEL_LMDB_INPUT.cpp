#include <caffe/caffe.hpp>
#include <leveldb/db.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <boost/shared_ptr.hpp>

DEFINE_string( deploy_model , "/Users/pitaloveu/testtest/multilabel.prototxt" 
							, "Specify the deploy model path" );

int main( int argc , char** argv )
{
	::google::InitGoogleLogging(argv[0]);
	FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	caffe::Caffe::set_mode( caffe::Caffe::CPU );
	std::shared_ptr<caffe::Net<float> > test_net;
	test_net.reset( new caffe::Net<float>( FLAGS_deploy_model , caffe::TEST ) );

	test_net->Forward();
//	caffe::Blob<float>* output_blob = test_net->output_blobs()[0];
	std::vector<std::string> blob_names = test_net->blob_names();

	const	boost::shared_ptr<caffe::Blob<float> > blob1 = test_net->blob_by_name( "data" );
	const boost::shared_ptr<caffe::Blob<float> > blob2 = test_net->blob_by_name( "label" );

	caffe::BlobProto blobProto1 , blobProto2;

	blob1->ToProto( &blobProto1 );
	blob2->ToProto( &blobProto2 );

	caffe::WriteProtoToBinaryFile( blobProto1 , "/Users/pitaloveu/testtest/PROTO1" );
	caffe::WriteProtoToBinaryFile( blobProto2 , "/Users/pitaloveu/testtest/PROTO2" );

	return 1;
}
