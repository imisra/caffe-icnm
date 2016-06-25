#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif


#define joint_num_classes 50
#define joint_filler 0

template <typename TypeParam>
class JointToConditionalLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  JointToConditionalLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, joint_num_classes, 4, 1)),
        blob_top_0(new Blob<Dtype>()), 
        blob_top_1(new Blob<Dtype>()),
        blob_top_2(new Blob<Dtype>()),
        blob_top_3(new Blob<Dtype>()) {
    // fill the values
    if (joint_filler == 0) {
      Dtype *qdata = this->blob_bottom_->mutable_cpu_data();
      for(int c=0; c<joint_num_classes; c++) {
        qdata[c*4 + 0] = 0.5;
        qdata[c*4 + 1] = 0;
        qdata[c*4 + 2] = 0;
        qdata[c*4 + 3] = 0.5;
      }
    } else {
      FillerParameter filler_param;
      UniformFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);  
    }


    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_0);
    blob_top_vec_.push_back(blob_top_1);
    blob_top_vec_.push_back(blob_top_2);
    blob_top_vec_.push_back(blob_top_3);
  }
  virtual ~JointToConditionalLayerTest() { delete blob_bottom_; delete blob_top_0; delete blob_top_1; delete blob_top_2; delete blob_top_3;}
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_0;
  Blob<Dtype>* const blob_top_1;
  Blob<Dtype>* const blob_top_2;
  Blob<Dtype>* const blob_top_3;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

//TYPED_TEST_CASE(JointToConditionalLayerTest, TestDtypesAndDevices);
typedef ::testing::Types<DoubleCPU, DoubleGPU>
    MyTypes;
TYPED_TEST_CASE(JointToConditionalLayerTest, MyTypes);

TYPED_TEST(JointToConditionalLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;  
  shared_ptr<JointToConditionalLayer<Dtype> > layer(
      new JointToConditionalLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);  
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  std::vector<bool> propagate_down_vec_(this->blob_top_vec_.size());
  for (int i=0; i<4; i++)
    propagate_down_vec_[i] = true;  
  for (int i=0; i<4; i++) {
    EXPECT_EQ(this->blob_top_vec_[i]->num(), 1);
    EXPECT_EQ(this->blob_top_vec_[i]->channels(), joint_num_classes);
    EXPECT_EQ(this->blob_top_vec_[i]->height(), 1);
    EXPECT_EQ(this->blob_top_vec_[i]->width(), 1);
  }
  EXPECT_EQ(this->blob_bottom_vec_[0]->num(), 1);
  EXPECT_EQ(this->blob_bottom_vec_[0]->channels(), joint_num_classes);
  EXPECT_EQ(this->blob_bottom_vec_[0]->height(), 4);
  EXPECT_EQ(this->blob_bottom_vec_[0]->width(), 1);
  // Dtype *bdiff = this->blob_bottom_vec_[0]->mutable_cpu_diff();
  // for(int i=0; i<this->blob_bottom_vec_[0]->count();i++)
  //   bdiff[i]=0;
  // layer->Backward(this->blob_top_vec_, propagate_down_vec_, this->blob_bottom_vec_);
}


#if joint_filler == 0
TYPED_TEST(JointToConditionalLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;  
  shared_ptr<JointToConditionalLayer<Dtype> > layer(
      new JointToConditionalLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);  
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i=0; i<4; i++) {
    EXPECT_EQ(this->blob_top_vec_[i]->num(), 1);
    EXPECT_EQ(this->blob_top_vec_[i]->channels(), joint_num_classes);
    EXPECT_EQ(this->blob_top_vec_[i]->height(), 1);
    EXPECT_EQ(this->blob_top_vec_[i]->width(), 1);
  }
  EXPECT_EQ(this->blob_bottom_vec_[0]->num(), 1);
  EXPECT_EQ(this->blob_bottom_vec_[0]->channels(), joint_num_classes);
  EXPECT_EQ(this->blob_bottom_vec_[0]->height(), 4);
  EXPECT_EQ(this->blob_bottom_vec_[0]->width(), 1);

  const Dtype *p00 = this->blob_top_vec_[0]->cpu_data();
  const Dtype *p01 = this->blob_top_vec_[1]->cpu_data();
  const Dtype *p10 = this->blob_top_vec_[2]->cpu_data();
  const Dtype *p11 = this->blob_top_vec_[3]->cpu_data();
  for(int c=0; c<joint_num_classes;c++) {
    EXPECT_NEAR(p00[c], 1.0, 1e-8);
    EXPECT_NEAR(p01[c], 0.0, 1e-8);
    EXPECT_NEAR(p10[c], 0.0, 1e-8);
    EXPECT_NEAR(p11[c], 1.0, 1e-8);  
  }
}
#endif


TYPED_TEST(JointToConditionalLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;  
  JointToConditionalLayer<Dtype> layer(layer_param);
  int seeds[2] = {100, 1701};
  for (int i=0; i<2; i++ ) {
    GradientChecker<Dtype> checker(1e-6, 1e-4, seeds[i]); //this will fail for Dtype=float, will pass for Dtype=double
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
  }
}

}  // namespace caffe
