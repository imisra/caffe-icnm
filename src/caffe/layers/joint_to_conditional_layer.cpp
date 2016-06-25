#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void JointToConditionalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "1 bottom";
  CHECK_EQ(top.size(), 4) << "4 tops";
  num_classes_ = bottom[0]->shape(1);

  CHECK_EQ(bottom[0]->count(), num_classes_*4);  
}

template <typename Dtype>
void JointToConditionalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  vector<int> weight_shape(4);
  weight_shape[0] = bottom[0]->shape(0);
  weight_shape[1] = num_classes_;
  weight_shape[2] = 1;
  weight_shape[3] = 1;
  
  for(int i=0; i<4; i++) {
    top[i]->Reshape(weight_shape);
  }
}

/*
top[0] = p00 = q00/(q00 + q10)
top[1] = p01 = q01/(q01 + q10)
top[2] = p10 = q10/(q00 + q10)
top[3] = p11 = q11/(q01 + q11)
*/
#define q00(x) (x+0)
#define q01(x) (x+1)
#define q10(x) (x+2)
#define q11(x) (x+3)

template <typename Dtype>
void JointToConditionalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype *qdata = bottom[0]->cpu_data();

  Dtype *p00 = top[0]->mutable_cpu_data();  
  for(int c=0; c<num_classes_; c++) {
    const int offset = c*4;
    p00[c] = qdata[q00(offset)]/(Dtype(1e-12) + qdata[q00(offset)] + qdata[q10(offset)]);
  }
  Dtype *p01 = top[1]->mutable_cpu_data();
  for(int c=0; c<num_classes_; c++) {
    const int offset = c*4;
    p01[c] = qdata[q01(offset)]/(Dtype(1e-12) + qdata[q01(offset)] + qdata[q11(offset)]);
  }
  Dtype *p10 = top[2]->mutable_cpu_data();
  for(int c=0; c<num_classes_; c++) {
    const int offset = c*4;
    p10[c] = qdata[q10(offset)]/(Dtype(1e-12) +  qdata[q00(offset)] + qdata[q10(offset)]);
  }
  Dtype *p11 = top[3]->mutable_cpu_data();
  for(int c=0; c<num_classes_; c++) {
    const int offset = c*4;
    p11[c] = qdata[q11(offset)]/(Dtype(1e-12) + qdata[q11(offset)] + qdata[q01(offset)]);
  }
}

template <typename Dtype>
void JointToConditionalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
   if (propagate_down[0]) {
    Dtype *bdiff = bottom[0]->mutable_cpu_diff();
    const Dtype *qdata = bottom[0]->cpu_data();    
    caffe_memset(bottom[0]->count()*sizeof(Dtype), 0, bdiff);

    const Dtype *p00diff = top[0]->cpu_diff();
    const Dtype *p01diff = top[1]->cpu_diff();
    const Dtype *p10diff = top[2]->cpu_diff();
    const Dtype *p11diff = top[3]->cpu_diff();

    // LOG(INFO)<<"JointToConditional: diff0["<<*std::min_element(p00diff, p00diff+top[0]->count())<<" "<<*std::max_element(p00diff, p00diff+top[0]->count());
    // LOG(INFO)<<"JointToConditional: diff1["<<*std::min_element(p01diff, p01diff+top[1]->count())<<" "<<*std::max_element(p01diff, p01diff+top[1]->count());
    // LOG(INFO)<<"JointToConditional: diff2["<<*std::min_element(p10diff, p10diff+top[2]->count())<<" "<<*std::max_element(p10diff, p10diff+top[2]->count());
    // LOG(INFO)<<"JointToConditional: diff3["<<*std::min_element(p11diff, p11diff+top[3]->count())<<" "<<*std::max_element(p11diff, p11diff+top[3]->count());

    for(int c=0; c<num_classes_; c++) {
      const int offset = c*4;
      Dtype denom_sq = (qdata[q00(offset)] + qdata[q10(offset)]) * (qdata[q00(offset)] + qdata[q10(offset)]);
      bdiff[q00(offset)] += ( p00diff[c] * (qdata[q10(offset)]/(Dtype(1e-12) + denom_sq)) );
      bdiff[q10(offset)] += ( p00diff[c] * (-qdata[q00(offset)]/(Dtype(1e-12) + denom_sq)) );
    }

    for(int c=0; c<num_classes_; c++) {
      const int offset = c*4;
      Dtype denom_sq = (qdata[q01(offset)] + qdata[q11(offset)]) * (qdata[q01(offset)] + qdata[q11(offset)]);
      bdiff[q01(offset)] += ( p01diff[c] * (qdata[q11(offset)]/(Dtype(1e-12) + denom_sq)) );
      bdiff[q11(offset)] += ( p01diff[c] * (-qdata[q01(offset)]/(Dtype(1e-12) + denom_sq)) );
    }

    for(int c=0; c<num_classes_; c++) {
      const int offset = c*4;
      Dtype denom_sq = (qdata[q10(offset)] + qdata[q00(offset)]) * (qdata[q10(offset)] + qdata[q00(offset)]);
      bdiff[q10(offset)] += ( p10diff[c] * (qdata[q00(offset)]/(Dtype(1e-12) + denom_sq)) );
      bdiff[q00(offset)] += ( p10diff[c] * (-qdata[q10(offset)]/(Dtype(1e-12) + denom_sq)) );
    }

    for(int c=0; c<num_classes_; c++) {
      const int offset = c*4;
      Dtype denom_sq = (qdata[q11(offset)] + qdata[q01(offset)]) * (qdata[q11(offset)] + qdata[q01(offset)]);
      bdiff[q11(offset)] += ( p11diff[c] * (qdata[q01(offset)]/(Dtype(1e-12) + denom_sq)) );
      bdiff[q01(offset)] += ( p11diff[c] * (-qdata[q11(offset)]/(Dtype(1e-12) + denom_sq)) );
    }
  }

}

#ifdef CPU_ONLY
STUB_GPU(JointToConditionalLayer);
#endif

INSTANTIATE_CLASS(JointToConditionalLayer);
REGISTER_LAYER_CLASS(JointToConditional);

}  // namespace caffe
