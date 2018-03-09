#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <vector>
#include <boost/program_options.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/filesystem.hpp>
#include <caffe2/core/common.h>
#include <caffe2/core/db.h>
#include <caffe2/core/init.h>
#include <caffe2/proto/caffe2.pb.h>
#include "PairList.h"

using namespace std;
using namespace boost;
using namespace boost::program_options;
using namespace boost::filesystem;
using namespace caffe2;

int main(int argc,char ** argv)
{
	string textpath,outputpath;
	int batch_size,seq_length;
	options_description desc;
	desc.add_options()
		("help,h","Print current usage")
		("input,i",value<string>(&textpath),"Text as training set")
		("output,o",value<string>(&outputpath),"Output lmdb path")
		("batch,b",value<int>(&batch_size)->default_value(1),"Number of concurrent training samples")
		("seq_length,s",value<int>(&seq_length)->default_value(25),"The length of the sample");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == argc || 1 == vm.count("help") || 1 != vm.count("input") || 1 != vm.count("output")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	remove_all(outputpath);
	
	//Read text, calculate text length
	std::ifstream in(textpath);
	if(false == in.is_open()) {
		cout<<"failed to open the text file"<<endl;
		return EXIT_FAILURE;
	}
	stringstream buffer;
	buffer << in.rdbuf();
	string text = buffer.str();
	long N = text.size();
	// The calculation divides the text according to the batch and 
	//records the starting position and length of each partition.	
	long text_block_size = N / batch_size;
	vector<boost::tuple<int,int,int> > parts;
	for(int i = 0 ; i < N && parts.size() < batch_size ; i += text_block_size)
		parts.push_back(
			boost::make_tuple(
				i,		//starting point
				(parts.size() != batch_size - 1)?text_block_size:(text_block_size + N % batch_size),		//length
				0		//Offset is where the sample gets started
			)
		);
	assert(parts.size() == batch_size);
	//Coding all the text that appears in the training text
	set<char> vs(text.begin(),text.end());
	int D = vs.size();
	PairList pl;
	int index = 0;
	for(set<char>::iterator it = vs.begin() ; it != vs.end() ; it++)
		pl.insert(Pair(index++,*it));
	std::ofstream out("index.dat");
	text_oarchive oa(out);
	oa<<pl;
	//Fill training set
	unique_ptr<db::DB> rnndb(db::CreateDB("lmdb",outputpath,db::NEW));
	unique_ptr<db::Transaction> transaction(rnndb->NewTransaction());
	TensorProtos protos;
	//Set training data dimensions
	TensorProto * data = protos.add_protos();
	data->set_data_type(TensorProto::FLOAT);
	data->add_dims(seq_length);
	data->add_dims(D);
	//Set supervisory value dimension
	TensorProto * label = protos.add_protos();
	label->set_data_type(TensorProto::INT32);
	label->add_dims(seq_length);

	// Because lstm input requires seq_length x batchsize x dimension
	// But the format for writing lmdb is seq_length x dimension
	// so read in the model is batchsize x seq_length x dimension
	// Need to use caffe2's transform operator to convert the data storage order
	string value;
	int count = 0;
	bool flag;
	do {
		for(int b = 0 ; b < batch_size ; b++) {
			//The reason for recreating each time is to clear the data
			vector<float> input(seq_length * D);
			vector<int> output(seq_length);
			data->clear_float_data();
			label->clear_int32_data();
			assert(0 == data->float_data_size());
			assert(0 == label->int32_data_size());
			for(int s = 0 ; s < seq_length ; s++) {
				int pos = get<0>(parts[b]) + get<2>(parts[b]);
				input[s * D + pl.get<1>().find(text[pos])->i] = 1;
				output[s] = pl.get<1>().find(text[(pos + 1) % N])->i;
				//Offset moves back one position
				get<2>(parts[b]) = (get<2>(parts[b]) + 1) % get<1>(parts[b]);
			}
			for(int i = 0 ; i < seq_length * D ; i++) data->add_float_data(input[i]);
			for(int i = 0 ; i < seq_length ; i++) label->add_int32_data(output[i]);
			protos.SerializeToString(&value);
			stringstream sstr;
			sstr<<setw(8)<<setfill('0')<<count;
			transaction->Put(sstr.str(),value);
			if(++count % 1000 == 0) {
				transaction->Commit();
			}
		}//end for
		flag = true;
		for(int b = 0 ; b < batch_size ; b++)
			if(0 == get<2>(parts[b])) {
				//If the offset has already returned to the beginning, 
				//the extracted sample will be repeated with the previous one.
				//So stop the current batch extraction work
				flag = false;
				break;
			}
	} while(flag);
	if(count) transaction->Commit();
	
	cout<<"character"<<pl.size()<<"One"<<endl;
	cout<<"produce"<<count<<"Samples"<<endl;
	
	return EXIT_SUCCESS;
}