// Converting a set of IMAGES and MULTI LABELS to a lmdb/leveldb
//****************************************************************************
// Author: Luo Heng
// Data: Nov. 08, 2017, in Sichuan
//
// Usage:
// 	convert_multi_label_dataset [FLAGS] DATA_FILE LABEL_FILE DB_FILE
//****************************************************************************
/**
  * while DATA_FILE contains the absolute path of image data, like below:
  *
  * ...
  * whereToFindImage/000011.jpg
  * whereToFindImage/000012.jpg
  * ...
  *
  * while LABEL_FILE contains all the attributes' indicator with the same number as
  * attributes' count, '1' stands for POSITIVE, '0' stands for 'NEGATIVE', each indicator
  * is separated with a space ' ', like below:
  *
  * ...
  * 1 0 1 0 0 1 0 0 1 0 1 0 0 1 0 0 1 0 0 0 0 0 0 1 0 0
  * 0 0 1 0 1 0 0 0 1 1 0 0 0 0 1 0 0 1 0 0 0 0 1 0 0 0
  * 0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0
  * ...
  *
  * while DB_FILE is the path of the output of lmdb file or leveldb file
**/
//******************************************************************************

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
//#include "boost/variant.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
    "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(min_dim, 0,
    "Minimum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(max_dim, 0,
    "Maximum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images and multi labels to the "
        "leveldb/lmdb format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_multi_label_dataset [FLAGS] DATA_FILE LABEL_FILE DB_FILE\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_multi_label_dataset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile1(argv[1]);	// Data  file
  std::ifstream infile2(argv[2]);	// Label file
  std::vector<std::pair<std::string, std::vector<int> > > lines;
  std::string filename;
  std::string label_string;
  std::vector<int> multi_label;

  while( std::getline( infile2 , label_string ) )
  {
		std::stringstream ss( label_string );
		int indicator;
		multi_label.clear();

		while( ss >> indicator )
			multi_label.push_back( indicator );

		CHECK( infile1 >> filename );

		lines.push_back( std::make_pair( filename , multi_label ) );
  }
  
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && encoded )
    LOG(INFO) << "encode the image with specified type\n";
  if ( !encode_type.size() && encoded )
		LOG(FATAL) << "if wanna encode, a type should be specified\n";

  int min_dim = std::max<int>(0, FLAGS_min_dim);
  int max_dim = std::max<int>(0, FLAGS_max_dim);
  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
 
	MultiLabelDatum multilabel_datum; 
	Datum* datum = multilabel_datum.mutable_datum();
//	Datum* datum = anno_datum.mutable_datum();
  int count = 0;

  for (int line_id = 0; line_id < lines.size(); ++line_id)
	{
    bool status = true;
    std::string enc = encode_type;
		if ( !encoded )
			enc = "";
//    if (encoded && !enc.size())
//		{
      // Guess the encoding type from the file name
//      string fn = lines[line_id].first;
//      size_t p = fn.rfind('.');
//      if ( p == fn.npos )
//        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";

//			if ( p != fn.npos )
//				p++;
//      enc = fn.substr(p);
//      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
 //   }
    std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    filename = lines[line_id].first;
		multi_label = lines[line_id].second;

		status = ReadImageToMultiLabelDatum( filename, multi_label , resize_height, resize_width,
           min_dim, max_dim, is_color, enc, &multilabel_datum );

    if (status == false)
		{
      LOG(WARNING) << "Failed to read " << lines[line_id].first;
      continue;
    }

    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // Put in db
    string out;
    CHECK(multilabel_datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
