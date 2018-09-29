#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "hdf5.h"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include "caffe/test/test_caffe_main.hpp"



extern unsigned int BW_INPUT_PARAM;
extern unsigned int BW_CONV_PARAM;
extern unsigned int BW_FC_PARAM;
extern unsigned int BW_OUTPUT_PARAM;





namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param, const Net* root_net)
    : root_net_(root_net) {
  Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase,
    const int level, const vector<string>* stages,
    const Net* root_net)
    : root_net_(root_net) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  // Set phase, stages and level
  param.mutable_state()->set_phase(phase);
  if (stages != NULL) {
    for (int i = 0; i < stages->size(); i++) {
      param.mutable_state()->add_stage((*stages)[i]);
    }
  }
  param.mutable_state()->set_level(level);
  Init(param);
}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) {
  CHECK(Caffe::root_solver() || root_net_)
      << "root_net_ needs to be set for all non-root solvers";
  // Set phase from the state.
  phase_ = in_param.state().phase();
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param;
  FilterNet(in_param, &filtered_param);
  LOG_IF(INFO, Caffe::root_solver())
      << "Initializing net from parameters: " << std::endl
      << filtered_param.DebugString();
  // Create a copy of filtered_param with splits added where necessary.
  NetParameter param;
  InsertSplits(filtered_param, &param);
  // Basically, build all the layers and set up their connections.
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  memory_used_ = 0;
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size());
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  bottom_need_backward_.resize(param.layer_size());
  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    // For non-root solvers, whether this layer is shared from root_net_.
    bool share_from_root = !Caffe::root_solver()
        && root_net_->layers_[layer_id]->ShareInParallel();
    // Inherit phase from net if unset.
    if (!param.layer(layer_id).has_phase()) {
      param.mutable_layer(layer_id)->set_phase(phase_);
    }
    // Setup layer.
    const LayerParameter& layer_param = param.layer(layer_id);
    if (layer_param.propagate_down_size() > 0) {
      CHECK_EQ(layer_param.propagate_down_size(),
          layer_param.bottom_size())
          << "propagate_down param must be specified "
          << "either 0 or bottom_size times ";
    }
    if (share_from_root) {
      LOG(INFO) << "Sharing layer " << layer_param.name() << " from root net";
      layers_.push_back(root_net_->layers_[layer_id]);
      layers_[layer_id]->SetShared(true);
    } else {
      layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
    }
    layer_names_.push_back(layer_param.name());
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating Layer " << layer_param.name();
    bool need_backward = false;

    // Figure out this layer's input and output
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
         ++bottom_id) {
      const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                       &available_blobs, &blob_name_to_idx);
      // If a blob needs backward, this layer should provide it.
      need_backward |= blob_need_backward_[blob_id];
    }
    int num_top = layer_param.top_size();
    for (int top_id = 0; top_id < num_top; ++top_id) {
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
      // Collect Input layer tops as Net inputs.
      if (layer_param.type() == "Input") {
        const int blob_id = blobs_.size() - 1;
        net_input_blob_indices_.push_back(blob_id);
        net_input_blobs_.push_back(blobs_[blob_id].get());
      }
    }
    // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    // specified fewer than the required number (as specified by
    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    Layer<Dtype>* layer = layers_[layer_id].get();
    if (layer->AutoTopBlobs()) {
      const int needed_num_top =
          std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
      for (; num_top < needed_num_top; ++num_top) {
        // Add "anonymous" top blobs -- do not modify available_blobs or
        // blob_name_to_idx as we don't want these blobs to be usable as input
        // to other layers.
        AppendTop(param, layer_id, num_top, NULL, NULL);
      }
    }
    // After this layer is connected, set it up.
    if (share_from_root) {
      // Set up size of top blobs using root_net_
      const vector<Blob<Dtype>*>& base_top = root_net_->top_vecs_[layer_id];
      const vector<Blob<Dtype>*>& this_top = this->top_vecs_[layer_id];
      for (int top_id = 0; top_id < base_top.size(); ++top_id) {
        this_top[top_id]->ReshapeLike(*base_top[top_id]);
        LOG(INFO) << "Created top blob " << top_id << " (shape: "
            << this_top[top_id]->shape_string() <<  ") for shared layer "
            << layer_param.name();
      }
    } else {
      layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "Setting up " << layer_names_[layer_id];
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
        blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));
      }
      blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
      LOG_IF(INFO, Caffe::root_solver())
          << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
      if (layer->loss(top_id)) {
        LOG_IF(INFO, Caffe::root_solver())
            << "    with loss weight " << layer->loss(top_id);
      }
      memory_used_ += top_vecs_[layer_id][top_id]->count();
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "Memory required for data: " << memory_used_ * sizeof(Dtype);
    const int param_size = layer_param.param_size();
    const int num_param_blobs = layers_[layer_id]->blobs().size();
    CHECK_LE(param_size, num_param_blobs)
        << "Too many params specified for layer " << layer_param.name();
    ParamSpec default_param_spec;
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      const ParamSpec* param_spec = (param_id < param_size) ?
          &layer_param.param(param_id) : &default_param_spec;
      const bool param_need_backward = param_spec->lr_mult() != 0;
      need_backward |= param_need_backward;
      layers_[layer_id]->set_param_propagate_down(param_id,
                                                  param_need_backward);
    }
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      AppendParam(param, layer_id, param_id);
    }
    // Finally, set the backward flag
    layer_need_backward_.push_back(need_backward);
    if (need_backward) {
      for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
        blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
      }
    }
  }
  // Go through the net backwards to determine which blobs contribute to the
  // loss.  We can skip backward computation for blobs that don't contribute
  // to the loss.
  // Also checks if all bottom blobs don't need backward computation (possible
  // because the skip_propagate_down param) and so we can skip bacward
  // computation for the entire layer
  set<string> blobs_under_loss;
  set<string> blobs_skip_backp;
  for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
    bool layer_contributes_loss = false;
    bool layer_skip_propagate_down = true;
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      if (layers_[layer_id]->loss(top_id) ||
          (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {
        layer_contributes_loss = true;
      }
      if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
        layer_skip_propagate_down = false;
      }
      if (layer_contributes_loss && !layer_skip_propagate_down)
        break;
    }
    // If this layer can skip backward computation, also all his bottom blobs
    // don't need backpropagation
    if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {
      layer_need_backward_[layer_id] = false;
      for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
               ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
    }
    if (!layer_contributes_loss) { layer_need_backward_[layer_id] = false; }
    if (Caffe::root_solver()) {
      if (layer_need_backward_[layer_id]) {
        LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
      } else {
        LOG(INFO) << layer_names_[layer_id]
            << " does not need backward computation.";
      }
    }
    for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
         ++bottom_id) {
      if (layer_contributes_loss) {
        const string& blob_name =
            blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_under_loss.insert(blob_name);
      } else {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
      if (!bottom_need_backward_[layer_id][bottom_id]) {
        const string& blob_name =
                   blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_skip_backp.insert(blob_name);
      }
    }
  }
  // Handle force_backward if needed.
  if (param.force_backward()) {
    for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layer_need_backward_[layer_id] = true;
      for (int bottom_id = 0;
           bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] =
            bottom_need_backward_[layer_id][bottom_id] ||
            layers_[layer_id]->AllowForceBackward(bottom_id);
        blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
            blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
            bottom_need_backward_[layer_id][bottom_id];
      }
      for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
           ++param_id) {
        layers_[layer_id]->set_param_propagate_down(param_id, true);
      }
    }
  }
  // In the end, all remaining blobs are considered output blobs.
  for (set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it) {
    LOG_IF(INFO, Caffe::root_solver())
        << "This network produces output " << *it;
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
  }
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
  ShareWeights();
  debug_info_ = param.debug_info();
  LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
}

template <typename Dtype>
void Net<Dtype>::FilterNet(const NetParameter& param,
    NetParameter* param_filtered) {
  NetState net_state(param.state());
  param_filtered->CopyFrom(param);
  param_filtered->clear_layer();
  for (int i = 0; i < param.layer_size(); ++i) {
    const LayerParameter& layer_param = param.layer(i);
    const string& layer_name = layer_param.name();
    CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
          << "Specify either include rules or exclude rules; not both.";
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    bool layer_included = (layer_param.include_size() == 0);
    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
        layer_included = false;
      }
    }
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
        layer_included = true;
      }
    }
    if (layer_included) {
      param_filtered->add_layer()->CopyFrom(layer_param);
    }
  }
}

template <typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState& state,
    const NetStateRule& rule, const string& layer_name) {
  // Check whether the rule is broken due to phase.
  if (rule.has_phase()) {
      if (rule.phase() != state.phase()) {
        LOG_IF(INFO, Caffe::root_solver())
            << "The NetState phase (" << state.phase()
            << ") differed from the phase (" << rule.phase()
            << ") specified by a rule in layer " << layer_name;
        return false;
      }
  }
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the min_level (" << rule.min_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the max_level (" << rule.max_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to stage. The NetState must
  // contain ALL of the rule's stages to meet it.
  for (int i = 0; i < rule.stage_size(); ++i) {
    // Check that the NetState contains the rule's ith stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (!has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState did not contain stage '" << rule.stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to not_stage. The NetState must
  // contain NONE of the rule's not_stages to meet it.
  for (int i = 0; i < rule.not_stage_size(); ++i) {
    // Check that the NetState contains the rule's ith not_stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState contained a not_stage '" << rule.not_stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  return true;
}
//-----------------------------------------------------------------------------------------------
template <typename Dtype>
Dtype Net<Dtype>::findMaxBatchNormScale(Blob<Dtype>* blob, Dtype *variance, Dtype bn_scale_factor_val, Dtype *scale) {
	const Dtype* data = blob->cpu_data();
	int cnt = blob->count();
	Dtype max_val = (Dtype)fabs(data[0] * scale[0] / (sqrt(variance[0] * bn_scale_factor_val) + 0.000001f));
	const vector<int>& shape = blob->shape();
	int kernel_size = shape[1] * shape[2] * shape[3];
	for (int n = 0; n < shape[0]; n++){
		for (int i = 0; i < kernel_size; i++){
			max_val = std::max(max_val, (Dtype)fabs(data[n*kernel_size + i] * scale[n] / (sqrt(variance[n] * bn_scale_factor_val) + 0.000001f)));
		}
	}
	return max_val;
}
template <typename Dtype>
Dtype Net<Dtype>::findMax(Blob<Dtype>* blob) {
	const Dtype* data = blob->cpu_data();
	int cnt = blob->count();
	Dtype max_val = (Dtype)fabs(data[0]);
	for (int i = 0; i < cnt; ++i) {
		max_val = std::max(max_val, (Dtype)fabs(data[i]));
	}
	return max_val;
}
float singleFindMax(float *data,int data_size) {
	float max_val = fabs( data[0]);
	for (int i = 0; i < data_size; ++i) {
		//printf("i %d,%f %f\n", i,data[i],max_val);
		max_val = std::max(max_val, fabs(data[i]));
		//printf("i %d,%f %f\n", i, data[i], max_val);
	}
	return max_val;
}
template <typename Dtype>
void Net<Dtype>::RangeInLayers( vector<string>* layer_name, vector<string>* layer_types,
	vector<Dtype>* max_in, vector<Dtype>* max_out, vector<Dtype>* max_param) {
	// Initialize vector elements, if needed.
	if (layer_name->size() == 0) {
		for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
			if (strcmp(layers_[layer_id]->type(), "Convolution") == 0 ||
				strcmp(layers_[layer_id]->type(), "InnerProduct") == 0 ) {
				layer_name->push_back(this->layer_names()[layer_id]);
				layer_types->push_back(layers_[layer_id]->type());
				max_in->push_back(0);
				max_out->push_back(0);
				max_param->push_back(0);


			}
		}
	}
	// Find maximal values.
	int index = 0;
	Dtype max_val;
	for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
		if (strcmp(layers_[layer_id]->type(), "Convolution") == 0 ||
			strcmp(layers_[layer_id]->type(), "InnerProduct") == 0) {
			max_val = findMax(bottom_vecs_[layer_id][0]);
			max_in->at(index) = std::max(max_in->at(index), max_val);
			max_val = findMax(top_vecs_[layer_id][0]);
			max_out->at(index) = std::max(max_out->at(index), max_val);
			// Consider the weights only, ignore the bias,LRN don't have weights,so only conv and fc will step into the condition block
			max_val = findMax(&(*layers_[layer_id]->blobs()[0]));
			max_param->at(index) = std::max(max_param->at(index), max_val);

			index++;
		}
	}
}
template <typename Dtype>
void Net<Dtype>::fixedQPorcess(Blob<Dtype>* blob, int fixedQ) {
	Dtype* data = blob->mutable_cpu_data();// 获得数据的指针data(修改数据的时候请你使用mutable_cpu_data)
	int cnt = blob->count();//统计Blob的容量（volume）
	for (int i = 0; i < cnt; ++i)
	{
		//std::cout << *data[i]<<std::endl;
		data[i] = (Dtype)(int)round(data[i] * (1 << fixedQ));//round(x)返回x的四舍五入整数值。新的data[i]中的值全部放大2^fixedQ倍，然后再赋值给data[i]---->即定点化的放大倍数
		//std::cout << *data[i]<<std::endl;

	}
}
template <typename Dtype>
void Net<Dtype>::revertFixedQPorcess(Blob<Dtype>* blob, int fixedQ) {
	Dtype* data = blob->mutable_cpu_data();
	int cnt = blob->count();
	for (int i = 0; i < cnt; ++i) {
		data[i] = (Dtype)(data[i] / (1 << fixedQ));
	}
}
//-----------------------------------------------------------------------------------------------

// Helper for Net::Init: add a new top blob to the net.
template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
                           const int top_id, set<string>* available_blobs,
                           map<string, int>* blob_name_to_idx) {
  shared_ptr<LayerParameter> layer_param(
      new LayerParameter(param.layer(layer_id)));
  const string& blob_name = (layer_param->top_size() > top_id) ?
      layer_param->top(top_id) : "(automatic)";
  // Check if we are doing in-place computation
  if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {
    // In-place computation
    LOG_IF(INFO, Caffe::root_solver())
        << layer_param->name() << " -> " << blob_name << " (in-place)";
	//add by qfdong to save the map between blobID and layerID.
	const int blob_id = (*blob_name_to_idx)[blob_name];
	blobID2layerID_top[blob_id] = layer_id;

    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } else {
    // Normal output.
    if (Caffe::root_solver()) {
      LOG(INFO) << layer_param->name() << " -> " << blob_name;
    }
    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
    const int blob_id = blobs_.size();
	//add by qfdong to save the map between blobID and layerID.
	blobID2layerID_top[blob_id] = layer_id;

    blobs_.push_back(blob_pointer);
    blob_names_.push_back(blob_name);
    blob_need_backward_.push_back(false);
    if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }
    top_id_vecs_[layer_id].push_back(blob_id);
    top_vecs_[layer_id].push_back(blob_pointer.get());
  }
  if (available_blobs) { available_blobs->insert(blob_name); }
}

// Helper for Net::Init: add a new bottom blob to the net.
template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
    const int bottom_id, set<string>* available_blobs,
    map<string, int>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);
  const string& blob_name = layer_param.bottom(bottom_id);
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
               << layer_param.name() << "', bottom index " << bottom_id << ")";
  }
  const int blob_id = (*blob_name_to_idx)[blob_name];
  //add by qfdong to save the map between blobID and layerID.
  blobID2layerID_bot[blob_id] = layer_id;

  LOG_IF(INFO, Caffe::root_solver())
      << layer_names_[layer_id] << " <- " << blob_name;
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
  bottom_id_vecs_[layer_id].push_back(blob_id);
  available_blobs->erase(blob_name);
  bool need_backward = blob_need_backward_[blob_id];
  // Check if the backpropagation on bottom_id should be skipped
  if (layer_param.propagate_down_size() > 0) {
    need_backward = layer_param.propagate_down(bottom_id);
  }
  bottom_need_backward_[layer_id].push_back(need_backward);
  return blob_id;
}

template <typename Dtype>
void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
                             const int param_id) {
  const LayerParameter& layer_param = layers_[layer_id]->layer_param();
  const int param_size = layer_param.param_size();
  string param_name =
      (param_size > param_id) ? layer_param.param(param_id).name() : "";
  if (param_name.size()) {
    param_display_names_.push_back(param_name);
  } else {
    ostringstream param_display_name;
    param_display_name << param_id;
    param_display_names_.push_back(param_display_name.str());
  }
  const int net_param_id = params_.size();
  params_.push_back(layers_[layer_id]->blobs()[param_id]);
  param_id_vecs_[layer_id].push_back(net_param_id);
  param_layer_indices_.push_back(make_pair(layer_id, param_id));
  ParamSpec default_param_spec;
  const ParamSpec* param_spec = (layer_param.param_size() > param_id) ?
      &layer_param.param(param_id) : &default_param_spec;
  if (!param_size || !param_name.size() || (param_name.size() &&
      param_names_index_.find(param_name) == param_names_index_.end())) {
    // This layer "owns" this parameter blob -- it is either anonymous
    // (i.e., not given a param_name) or explicitly given a name that we
    // haven't already seen.
    param_owners_.push_back(-1);
    if (param_name.size()) {
      param_names_index_[param_name] = net_param_id;
    }
    const int learnable_param_id = learnable_params_.size();
    learnable_params_.push_back(params_[net_param_id].get());
    learnable_param_ids_.push_back(learnable_param_id);
    has_params_lr_.push_back(param_spec->has_lr_mult());
    has_params_decay_.push_back(param_spec->has_decay_mult());
    params_lr_.push_back(param_spec->lr_mult());
    params_weight_decay_.push_back(param_spec->decay_mult());
  } else {
    // Named param blob with name we've seen before: share params
    const int owner_net_param_id = param_names_index_[param_name];
    param_owners_.push_back(owner_net_param_id);
    const pair<int, int>& owner_index =
        param_layer_indices_[owner_net_param_id];
    const int owner_layer_id = owner_index.first;
    const int owner_param_id = owner_index.second;
    LOG_IF(INFO, Caffe::root_solver()) << "Sharing parameters '" << param_name
        << "' owned by "
        << "layer '" << layer_names_[owner_layer_id] << "', param "
        << "index " << owner_param_id;
    Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();
    Blob<Dtype>* owner_blob =
        layers_[owner_layer_id]->blobs()[owner_param_id].get();
    const int param_size = layer_param.param_size();
    if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
                                  ParamSpec_DimCheckMode_PERMISSIVE)) {
      // Permissive dimension checking -- only check counts are the same.
      CHECK_EQ(this_blob->count(), owner_blob->count())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "shape is " << this_blob->shape_string();
    } else {
      // Strict dimension checking -- all dims must be the same.
      CHECK(this_blob->shape() == owner_blob->shape())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "expects shape " << this_blob->shape_string();
    }
    const int learnable_param_id = learnable_param_ids_[owner_net_param_id];
    learnable_param_ids_.push_back(learnable_param_id);
    if (param_spec->has_lr_mult()) {
      if (has_params_lr_[learnable_param_id]) {
        CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched lr_mult.";
      } else {
        has_params_lr_[learnable_param_id] = true;
        params_lr_[learnable_param_id] = param_spec->lr_mult();
      }
    }
    if (param_spec->has_decay_mult()) {
      if (has_params_decay_[learnable_param_id]) {
        CHECK_EQ(param_spec->decay_mult(),
                 params_weight_decay_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched decay_mult.";
      } else {
        has_params_decay_[learnable_param_id] = true;
        params_weight_decay_[learnable_param_id] = param_spec->decay_mult();
      }
    }
  }
}
string   replace_all(string   str, const  string&  old_value, const  string&  new_value)
{
	while (true)
	{
		string::size_type   pos(0);
		if ((pos = str.find(old_value)) != string::npos)
		{
			str.replace(pos, old_value.length(), new_value);
		}
		else { break; }
	}
	return   str;
}
string transLayerName(string layername)
{
	return replace_all(layername, "/", "_");
}
int detectOverflow(float val, LayerQParams &layerQpara,int dataType)
{
	int ret = 0;
	long long datamax = 0,datamin=0;
	if (dataType ==1)
	{
		datamax = SCHAR_MAX;
		datamin = SCHAR_MIN;
	}else if(dataType ==2)
	{
		datamax = SHRT_MAX;
		datamin = SHRT_MIN;
	}
	else if (dataType == 4)
	{
		datamax = INT_MAX;
		datamin = INT_MIN;
	}
	else
	{
		assert(0);
	}

	if (val >= datamax || val <= datamin)//判断值是否溢出
	{
		int shift = 0;
		while (val >= datamax || val <= datamin)//如果溢出则除以2来缩减，调整Qbits
		{
			val = val / 2;
			shift++;
		}
		printf("[%s] layername %s val %f overflow!!! change weightQ %d ->%d biasQ %d ->%d\n",
			layerQpara.layer_type.c_str(),layerQpara.layer_name.c_str(),val, layerQpara.weightsQ, layerQpara.weightsQ - shift,layerQpara.biasQ,layerQpara.biasQ - shift);
		layerQpara.weightsQ = layerQpara.weightsQ - shift;
		layerQpara.biasQ = layerQpara.biasQ - shift;
		ret = 1;
	}
	return ret;
}

template <typename Dtype>
Dtype Net<Dtype>::Quantize_Forward(const char *layer_result_path)//******************************************************
{
	int start = 0; 
	int end = layers_.size()-1;
	Dtype loss = 0;
	//printf("size of dyype %d\n", sizeof(Dtype));
	if (isSecondRun == false)
	{
		printf("step 1. calc Blob Q value .......\n");////////得到每个层的最大值，然后得到缩放比例Qbits,然后根据每个层的不同网络（conv、bn、scale、）调整Qbits的大小
		if (isFirstImage == true)
		for (int i = 0; i < blob_names_.size(); i++)
		{
			BlobQParams BlobQ_params;
			BlobQ_params.BlobName = blob_names_[i];//读取每一层网络的名字
			BlobQ_params.Qbits = -1;
			BlobQ_params.max = 0;
			BlobQ.push_back(BlobQ_params);// vector<BlobQParams> BlobQ;

		}
		if (isFirstImage == true)
		for (int i = start; i <= end; ++i) {
			maxValueOfBatch.push_back(0);//maxValueOfBatch 0 初始化
		}

		for (int i = start; i <= end; ++i) {// bottom_vecs存储包含每个图层输入的向量。它们实际上并不承载blob（blobs_），所以我们只是存储指针。
			Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);//vector<vector<Blob<Dtype>*> > top_vecs_存储包含每层输出的向量
			for (int j = 0; j < top_vecs_[i].size(); j++)//top_vecs_[i]每一层的size
			{
				int id = top_id_vecs_[i][j];			
				BlobQ[id].max = std::max(findMax(top_vecs_[i][j]), maxValueOfBatch[i]);//得出每层参数最大值
				//BlobQ[id].max = findMax(top_vecs_[i][j]);
    			maxValueOfBatch[i] = BlobQ[id].max;//每层参数的最大值赋给maxValueOfBatch

				int il_out = (int)ceil(log2(BlobQ[id].max) + 1);//ceil(x)返回的是大于x的最小整数
				if (layers_[i]->layer_param().type() == "Data")//假如是Data层的话
				{
					BlobQ[id].Qbits = 6;
					continue;
				}
				if (layers_[i]->layer_param().type() == "ReLU" || layers_[i]->layer_param().type() == "Dropout")
				{
					//如果是relu dropout的话 不更新Qbits
					continue;
				}
				if ((BlobQ[id].Qbits != -1) && (BlobQ[id].Qbits != (BW_INPUT_PARAM - il_out)))//如果bn层、scale层与conv层、的Qbits不相同，则需要调整（上个BlobQ[id].Qbits与BW_INPUT_PARAM - il_out现在的 Qbits不相等）
				{
					printf("[%-10s] update %s Qbits %d -> %d\n", layers_[i]->layer_param().type().c_str(),//BatchNorm updata layer1-conv Qbits 10->8(con-bn-scale-act-......)
						BlobQ[id].BlobName.c_str(), BlobQ[id].Qbits, (BW_INPUT_PARAM - il_out));
				}
				BlobQ[id].Qbits = BW_INPUT_PARAM - il_out;////////14-il_out
				if (layers_[i]->layer_param().type() == "Pooling")
				{
					if (layers_[i]->layer_param().pooling_param().pool() == PoolingParameter_PoolMethod_AVE)
					{
						int kernel_size = layers_[i]->layer_param().pooling_param().kernel_size();
						float val = BlobQ[id].max*kernel_size*kernel_size*(1 << BlobQ[id].Qbits);
						if (val >= INT_MAX)
						{
							int shift = 0;
							while (val >= INT_MAX)
							{
								val = val / 2;
								shift++;
							}
							printf("ave pooling overflow!\n");
							BlobQ[id].Qbits -= shift;
						}
					}
				}
			}
			loss += layer_loss;
			if (debug_info_) { ForwardDebugInfo(i); }

			//if (layer_result_path != NULL)
			if (0)//remove by qfdong, only the second run will save only 1 picture floating result
			{
				const char* path = layer_result_path;
				char pathname[512];
				const Dtype* data = top_vecs_[i][0]->cpu_data();
				int cnt = top_vecs_[i][0]->count();
				string layer_name = transLayerName(layer_names_[i]);
				sprintf(pathname, "%s%d[%s]_%s_src.txt", path, i, layers_[i]->layer_param().type().c_str(), layer_name.c_str());
				FILE *fp = fopen(pathname, "wb");
				for (int zi = 0; zi < cnt; ++zi) {

					fprintf(fp, "%0.5f\n", data[zi]);
				}
				fclose(fp);
			}
		}
		//---------------------------------------------------------------------------
		if (isLastImage == false)
			return 0;
		//---------------------------------------------------------------------------
		printf("step 2. remodify Blob Q value .......\n");//////////////在Concat（两个特征图forward）层和Conv（卷积和池化）层remodify Qbits的大小（上下两层需要统一）
		for (int i = start; i <= end; ++i)
		{
			//mutil-input
			string layerType = layers_[i]->layer_param().type();
			//printf("%s\n", layerType.c_str());
			//Concat层需要统一bottom的Qbits
			if (layerType == "Concat")
			{
				//printf("[%s]:%s:\n", layers_[i]->layer_param().type().c_str(),layer_names_[i].c_str());
				int bottom_id = bottom_id_vecs_[i][0];
				int minBits = BlobQ[bottom_id].Qbits;
				for (int j = 0; j < bottom_vecs_[i].size(); j++)
				{
					bottom_id = bottom_id_vecs_[i][j];
					minBits = min(BlobQ[bottom_id].Qbits, minBits);
				}
				for (int j = 0; j < bottom_vecs_[i].size(); j++)
				{
					bottom_id = bottom_id_vecs_[i][j];
					printf("[%s] %s Qbits %d -> Qbits %d\n", layers_[i]->layer_param().type().c_str(), BlobQ[bottom_id].BlobName.c_str(), BlobQ[bottom_id].Qbits, minBits);
					BlobQ[bottom_id].Qbits = minBits;
				}
			}
			//这里添加需要改变blob的Qbits的层
			if (layerType == "Concat" || layerType == "Split" || layerType == "Pooling" || layerType == "ReLU")//卷积层和池化层Qbits不一样则需要统一化
			{
				int bottom_id = bottom_id_vecs_[i][0];

				for (int j = 0; j < top_vecs_[i].size(); j++)
				{
					int top_id = top_id_vecs_[i][j];
					if (BlobQ[top_id].Qbits != BlobQ[bottom_id].Qbits)//底部的Qbits 不等于 顶部的Qbits（意思就是卷积层下面的池化层Qbits不一样的话需要重新赋值改造，把大的Qbits变成小的Qbits）
					{
						printf("[%s]:remodify %s %d-> %d\n", layerType.c_str(), BlobQ[top_id].BlobName.c_str(), BlobQ[top_id].Qbits, BlobQ[bottom_id].Qbits);
						BlobQ[top_id].Qbits = BlobQ[bottom_id].Qbits;//把底部的Qbits赋值给顶部的Qbits（把输入的Qbits赋值给输出的Qbits）
					}

				}
			}
		}
		printf("step 3. update layer Q value .......\n");/////********得到intQ,outQ,weightsQ,biasQ.....各个参数的Qbits
		for (int i = start; i <= end; ++i)
		{
			LayerQParams layerQ_params;
			layerQ_params.inQ = 0;
			layerQ_params.outQ = 0;
			layerQ_params.weightsQ = 0;
			layerQ_params.biasQ = 0;
			layerQ_params.layer_name = layer_names_[i];
			layerQ_params.layer_type = layers_[i]->layer_param().type();

			if (bottom_vecs_[i].size() == 1)
			{
				int bottom_id = bottom_id_vecs_[i][0];
				layerQ_params.inQ = BlobQ[bottom_id].Qbits;//把底部的Qbits赋值给下一个输入的Qbits （inQ）
			}
			else if (bottom_vecs_[i].size() > 1)
			{
				int bottom_id;
				bottom_id = bottom_id_vecs_[i][0];
				int val = BlobQ[bottom_id].Qbits;
				if (layerQ_params.layer_type == "Concat")
				{
					for (int j = 0; j < bottom_vecs_[i].size(); j++)
					{
						//做下验证
						bottom_id = bottom_id_vecs_[i][j];
						assert(val == BlobQ[bottom_id].Qbits);
					}
				}
				layerQ_params.inQ = val;
			}
			else
			{
				assert(i == 0);//如果它的条件返回错误，则终止程序执行
			}

			if (top_vecs_[i].size() == 1)
			{
				int top_id = top_id_vecs_[i][0];
				layerQ_params.outQ = BlobQ[top_id].Qbits;//把顶部的Qbits赋值给上一个输出的Qbits  (outQ)
			}
			else if (top_vecs_[i].size() > 1)
			{
				int top_id = top_id_vecs_[i][0];
				int val = BlobQ[top_id].Qbits;
				for (int j = 0; j < top_vecs_[i].size(); j++)
				{
					top_id = top_id_vecs_[i][j];
					assert(val == BlobQ[top_id].Qbits);
				}
				layerQ_params.outQ = val;
			}
			else
			{
				assert(i == 0);
			}

			if (layers_[i]->layer_param().type() == "Convolution")//////假如是CONV层，求卷积层的(layerQ_params).intQ,outQ,weightsQ,biasQ
			{
				if (i <= layers_.size() - 3)//如果该层里面的层数（BN,scale,art)等于3
				{
					if (layers_[i + 1]->layer_param().type() == "BatchNorm" && layers_[i + 2]->layer_param().type() == "Scale")//如果卷积层下一个和下下一个是"BatchNorm"和"Scale"层，
					{	
						vector<shared_ptr<Blob<Dtype> > >& target_blobs_bacthnorm = layers_[i + 1]->blobs();//拿到BN层的caffe blobs
						Dtype* variance = (Dtype*)target_blobs_bacthnorm[1]->cpu_data();//方差
						Dtype* bn_scale_factor = (Dtype*)target_blobs_bacthnorm[2]->cpu_data();//比例因子
						Dtype bn_scale_factor_val = bn_scale_factor[0] == 0 ? 0 : 1 / bn_scale_factor[0];

						vector<shared_ptr<Blob<Dtype> > >& target_blobs_scale = layers_[i + 2]->blobs();//拿到Scale层的caffe blobs
						Dtype* scale = (Dtype*)target_blobs_scale[0]->cpu_data();

						float weightsMax = findMaxBatchNormScale(&(*layers_[i]->blobs()[0]), variance, bn_scale_factor_val, scale);//求这几个参数的最大值
						int il_weights = (int)ceil(log2(weightsMax) + 1);
						layerQ_params.weightsQ = weights_type - il_weights;//求weights的Qbits
						//有无bias 都统一更新biasQ
						layerQ_params.biasQ = layerQ_params.inQ + layerQ_params.weightsQ;//求bias的Qbits
					}
					else//如果卷积层下一个和下下一个不是"BatchNorm"和"Scale"层
					{
						float weightsMax = findMax(&(*layers_[i]->blobs()[0]));
						int il_weights = (int)ceil(log2(weightsMax) + 1);
						layerQ_params.weightsQ = weights_type - il_weights;//求weights的Qbits
						//有无bias 都统一更新biasQ
						layerQ_params.biasQ = layerQ_params.inQ + layerQ_params.weightsQ;//求bias的Qbits
					}
				}
				else//如果该层里面的层数（BN,scale,art)不等于3
				{
					float weightsMax = findMax(&(*layers_[i]->blobs()[0]));
					int il_weights = (int)ceil(log2(weightsMax) + 1);
					layerQ_params.weightsQ = weights_type - il_weights;//求weights的Qbits
					//有无bias 都统一更新biasQ
					layerQ_params.biasQ = layerQ_params.inQ + layerQ_params.weightsQ;//求bias的Qbits
				}
				
			}
			if (layers_[i]->layer_param().type() == "InnerProduct")
			{
				float weightsMax = findMax(&(*layers_[i]->blobs()[0]));
				int il_weights = (int)ceil(log2(weightsMax)) + 1;
				layerQ_params.weightsQ = BW_FC_PARAM - il_weights;
				layerQ_params.biasQ = layerQ_params.inQ + layerQ_params.weightsQ;
			}

			layerQ.push_back(layerQ_params);

		}

	}
	
	
	
	
	
	//is second run==true
	else {
		printf("step 4. check Qbits overflow .......\n");//检查和计算每一层的溢出情况
		for (int i = start; i <= end; ++i) 
		{
			// LOG(ERROR) <<i <<"Forwarding " << layer_names_[i];
			if (layers_[i]->layer_param().type() == "Convolution")
			{
				fixedQPorcess(bottom_vecs_[i][0], layerQ[i].inQ);//fixedQPorcess（A,B)将A数值放大2^B倍
				fixedQPorcess(&(*layers_[i]->blobs()[0]), layerQ[i].weightsQ);
				if (layers_[i]->layer_param().convolution_param().bias_term())
				{
					fixedQPorcess(&(*layers_[i]->blobs()[1]), layerQ[i].biasQ);
				}
			}

			if (layers_[i]->layer_param().type() == "InnerProduct")
			{
				fixedQPorcess(bottom_vecs_[i][0], layerQ[i].inQ);
				fixedQPorcess(&(*layers_[i]->blobs()[0]), layerQ[i].weightsQ);
				if (layers_[i]->layer_param().inner_product_param().bias_term())
				{
					fixedQPorcess(&(*layers_[i]->blobs()[1]), layerQ[i].biasQ);
				}
			}


			Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);//每一层的loss等于正向传播时候算出来的

			if (layers_[i]->layer_param().type() == "Convolution")
			{
				//revert input data;
				revertFixedQPorcess(bottom_vecs_[i][0], layerQ[i].inQ);//revertFixedQPorcess（）恢复放大前的数据
				revertFixedQPorcess(&(*layers_[i]->blobs()[0]), layerQ[i].weightsQ);
				if (layers_[i]->layer_param().convolution_param().bias_term())
				{
					revertFixedQPorcess(&(*layers_[i]->blobs()[1]), layerQ[i].biasQ);
				}




				//test output data 测试每层输出放大后是否溢出（需要获得每个输出层的最大值，然后放大Qbits倍，看看是否溢出）
				float temp = findMax(top_vecs_[i][0]);//top_vecs_存储每个层包含输出的向量。temp是每个输出层的最大值
				revertFixedQPorcess(top_vecs_[i][0], layerQ[i].biasQ);//先还原到原来的数据
				

				detectOverflow(temp, layerQ[i], sizeof(int));////判断Convolution层中最大值temp是否溢出，溢出则除以2来缩减，调整Qbits的大小  检测、打印溢出+++++++++++++++++++++++++++++++++++++++++++
				
				if (i <= layers_.size() - 3)//判断BN层中，bn,加上bias，乘以scale后的最大值是否溢出
				{
					if (layers_[i + 1]->layer_param().type() == "BatchNorm")
					{
						if (layers_[i + 2]->layer_param().type() != "Scale")
						{
							CHECK(0) << "batchNorm Scale must be pair exist\n";
						}
						//shared_ptr<Layer<float>> layer_batchnorm_ptr = layers_[i+1];
						//shared_ptr<Layer<float>> layer_scale_ptr = layers_[i+2];
						assert(layers_[i + 1]->layer_param().type() == "BatchNorm");
						assert(layers_[i + 2]->layer_param().type() == "Scale");

						//vector<shared_ptr<Blob<float> > >& target_blobs_bacthnorm = layers_[i + 1]->blobs();

						const float* mean = (float*)(layers_[i + 1]->blobs()[0]->cpu_data());
						const float* variance = (float*)(layers_[i + 1]->blobs()[1]->cpu_data());
						const float* bn_scale_factor = (float*)(layers_[i + 1]->blobs()[2]->cpu_data());

						//printf("blob size %d\n",layers_[i + 1]->blobs().size());
						assert(layers_[i + 1]->blobs().size() == 3);
						float scale_factor = bn_scale_factor[0] == 0 ? 0 : 1 / bn_scale_factor[0];



						//vector<shared_ptr<Blob<float> > >& target_blobs_scale = layers_[i + 2]->blobs();
						const float* scale = (float*)(layers_[i + 2]->blobs()[0]->cpu_data());//从cpu中获取数据
						const float* scale_bias = (float*)(layers_[i + 2]->blobs()[1]->cpu_data());
						int number = layers_[i + 2]->blobs()[0]->shape()[0];//blob[0].shape()->(n,c,w,h)==n（第一个数：numbers）////scale 层的numbers

						float *scale_data = (float*)malloc(sizeof(float)*number);//动态内存分配（分配number个float型存储单元，并将首地址存储到指针变量scale_data中）
						float *bias_data = (float*)malloc(sizeof(float)*number);
						memset(scale_data, 0, sizeof(float)*number);//用来对一段内存空间全部设置为某个字符0
						memset(bias_data, 0, sizeof(float)*number);

						int overflowflag = 0;//***************************判断是否溢出
						for (int zi = 0; zi < number; zi++)//每一层caffe 里面blob 的number数量
						{
							float mean_value = scale_factor*mean[zi];
							float variance_value = scale_factor*variance[zi];
							bias_data[zi] = (scale_bias[zi] * (sqrt(variance_value) + 0.000001f) - mean_value * scale[zi]) / scale[zi];
							bias_data[zi] = bias_data[zi] * (1 << layerQ[i].biasQ);
							scale_data[zi] = scale[zi] / (sqrt(variance_value) + 0.000001f);//计算正向传播时候y=w1*x+b特征图的数值，以便后面求最大值
							if (fabs(scale[zi]) < 0.000001f)//如果scale[zi]的值偏小（< 0.000001f），则直接赋值为0
							{
								printf("scale_bias %f variance %f mean %f scale %f\n", scale_bias[zi],variance[zi], mean[zi], scale[zi]);
								scale_data[zi] = 0;
								bias_data[zi] = 0;
								printf("bias_data %f scale_data %f\n", bias_data[zi], scale_data[zi]);
							}
						}
						float bias_max = singleFindMax(bias_data, number);//寻找number个bias_data中的最大值
						overflowflag = detectOverflow(bias_max, layerQ[i], sizeof(int));//判断bias_data中是否有值溢出  // 判断bias_max是否溢出，溢出则除以2来缩减，调整Qbits的大小
						
						if (overflowflag == 1)// 溢出需要重新定点化操作(重新放大biasQ倍)
						{
							for (int zi = 0; zi < number; zi++)
							{
								bias_data[zi] = (scale_bias[zi] * (sqrt(variance[zi]) + 0.000001f) - mean[zi] * scale[zi]) / scale[zi];
								bias_data[zi] = bias_data[zi] * (1 << layerQ[i].biasQ);
							}

						}
						//判断加上bias后是否溢出
						fixedQPorcess(top_vecs_[i][0], layerQ[i].biasQ);
						assert(top_vecs_[i][0]->channels() == number);
						int blob_height = top_vecs_[i][0]->height();
						int blob_width = top_vecs_[i][0]->width();
						float *temp = (float*)malloc(number*blob_height*blob_width*sizeof(float));
						float *top_val = (float*)(top_vecs_[i][0]->cpu_data());
						//FILE *zfp = fopen("C:\\Users\\user\\Desktop\\conv1\\conv.txt","wb");
						for (int zi = 0; zi < number; zi++)
						{
							for (int zj = 0; zj < blob_height; zj++)
							{
								for (int zk = 0; zk < blob_width; zk++)
								{
									float data = top_val[zi*blob_height*blob_width + zj*blob_width + zk];
									temp[zi*blob_height*blob_width + zj*blob_width + zk] = (data + bias_data[zi]);// 每个data加上bias                              *scale_data[zi];
									//			fprintf(zfp, "%f\n", temp[zi*blob_height*blob_width + zj*blob_width + zk] /(1 << layerQ[i].biasQ));
								}
							}
						}
						//fclose(zfp);
						revertFixedQPorcess(top_vecs_[i][0], layerQ[i].biasQ);//bias缩放到原来的大小


						float temp_max = singleFindMax(temp, number*blob_height*blob_width);// temp为每个data加上bias后的值  
						overflowflag = detectOverflow(temp_max, layerQ[i], sizeof(int));//判断temp_max是否溢出，溢出则除以2来缩减，调整Qbits的大小
						//判断加完bias后是否溢出
						if (overflowflag == 1)
						{
							printf("conv-bn-sacle 加完bias后 overflow!\n");
						}

						//判断乘以scale后是否溢出
						for (int zi = 0; zi < number; zi++)
						{
							for (int zj = 0; zj < blob_height; zj++)
							{
								for (int zk = 0; zk < blob_width; zk++)
								{
									float data = temp[zi*blob_height*blob_width + zj*blob_width + zk];
									temp[zi*blob_height*blob_width + zj*blob_width + zk] *= scale_data[zi];//每个data乘以scale
								}
							}
						}
						temp_max = singleFindMax(temp, number*blob_height*blob_width);// temp为每个data乘以scale后的值  
						overflowflag = detectOverflow(temp_max, layerQ[i], sizeof(int));
						//判断乘以scale后是否溢出
						if (overflowflag == 1)
						{
							printf("conv-bn-sacle 乘以scale后 overflow!\n");
						}
						free(scale_data);
						free(bias_data);
						free(temp);
					}
				}
			}
			if (layers_[i]->layer_param().type() == "InnerProduct")
			{
				//revert input data;
				revertFixedQPorcess(bottom_vecs_[i][0], layerQ[i].inQ);//缩放到原来的大小
				revertFixedQPorcess(&(*layers_[i]->blobs()[0]), layerQ[i].weightsQ);//缩放到原来的大小
				if (layers_[i]->layer_param().inner_product_param().bias_term())
				{
					revertFixedQPorcess(&(*layers_[i]->blobs()[1]), layerQ[i].biasQ);
				}
				//test output data
				float temp = findMax(top_vecs_[i][0]);//top_vecs_存储每个层包含输出的向量
				revertFixedQPorcess(top_vecs_[i][0], layerQ[i].weightsQ + layerQ[i].inQ);
				if (temp >= INT_MAX - 1)//如果temp 大于最大值，则溢出，需要更新Qbits来防止溢出
				{
					printf("i=%d,%s,%f\n", i, layer_names_[i].c_str(), temp);
					printf(" %s inQ %d weightsQ %d biasQ %d outQ %d\n", layerQ[i].layer_name.c_str(), layerQ[i].inQ, layerQ[i].weightsQ, layerQ[i].biasQ,
						layerQ[i].outQ);
					int shift = 0;
					while (temp >= INT_MAX - 1)
					{
						temp = temp / 2;
						shift++;
					}
					printf("too large must shift =%d change it:\n", shift);
					layerQ[i].weightsQ = layerQ[i].weightsQ - shift;
					layerQ[i].biasQ = layerQ[i].biasQ - shift;
					printf(" %s inQ %d weightsQ %d biasQ %d outQ %d\n", layerQ[i].layer_name.c_str(), layerQ[i].inQ, layerQ[i].weightsQ, layerQ[i].biasQ,
						layerQ[i].outQ);
				}
			}
		}
	}
	
	return 0;
}
template <typename Dtype>



Dtype Net<Dtype>::Fixmode_Forward(const char *layer_result_path)//*************************************定点化前向传播
{
	int start = 0;
	int end = layers_.size() - 1;
	Dtype loss = 0;
	printf("step 6. simulator fixed-point process .......\n");//模拟定点化过程
	for (int i = start; i <= end; ++i) 
	{
		  //LOG(ERROR) <<i <<"Forwarding " << layer_names_[i];
		if (layers_[i]->layer_param().type() == "Convolution")
		{
			fixedQPorcess(bottom_vecs_[i][0], layerQ[i].inQ);//将bottom_vecs_[i][0]放大2^inQ倍
			fixedQPorcess(&(*layers_[i]->blobs()[0]), layerQ[i].weightsQ);
			if (layers_[i]->layer_param().convolution_param().bias_term())
			{
				fixedQPorcess(&(*layers_[i]->blobs()[1]), layerQ[i].biasQ);
			}
		}
		if (layers_[i]->layer_param().type() == "InnerProduct")
		{
			fixedQPorcess(bottom_vecs_[i][0], layerQ[i].inQ);
			fixedQPorcess(&(*layers_[i]->blobs()[0]), layerQ[i].weightsQ);

			if (layers_[i]->layer_param().inner_product_param().bias_term())
			{
				fixedQPorcess(&(*layers_[i]->blobs()[1]), layerQ[i].biasQ);
			}
		}

		Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
		if (layers_[i]->layer_param().type() == "Convolution")
		{
			//revert input data;
			revertFixedQPorcess(bottom_vecs_[i][0], layerQ[i].inQ);
			revertFixedQPorcess(&(*layers_[i]->blobs()[0]), layerQ[i].weightsQ);
			if (layers_[i]->layer_param().convolution_param().bias_term())
			{
				revertFixedQPorcess(&(*layers_[i]->blobs()[1]), layerQ[i].biasQ);
			}
			//test output data
			float temp = findMax(top_vecs_[i][0]);
			revertFixedQPorcess(top_vecs_[i][0], layerQ[i].weightsQ + layerQ[i].inQ);
			if (temp >= INT_MAX - 1)
			{
				printf("i=%d,%s,%f\n", i, layer_names_[i].c_str(), temp);
				printf(" %s inQ %d weightsQ %d biasQ %d outQ %d\n", layerQ[i].layer_name.c_str(), layerQ[i].inQ, layerQ[i].weightsQ, layerQ[i].biasQ,
					layerQ[i].outQ);
				assert(0);
			}
		}
		if (layers_[i]->layer_param().type() == "InnerProduct")
		{
			//revert input data;
			revertFixedQPorcess(bottom_vecs_[i][0], layerQ[i].inQ);
			revertFixedQPorcess(&(*layers_[i]->blobs()[0]), layerQ[i].weightsQ);
			if (layers_[i]->layer_param().inner_product_param().bias_term())
			{
				revertFixedQPorcess(&(*layers_[i]->blobs()[1]), layerQ[i].biasQ);
			}
			//test output data
			float temp = findMax(top_vecs_[i][0]);
			revertFixedQPorcess(top_vecs_[i][0], layerQ[i].weightsQ + layerQ[i].inQ);
			if (temp >= INT_MAX - 1)
			{
				printf("i=%d,%s,%f\n", i, layer_names_[i].c_str(), temp);
				printf(" %s inQ %d weightsQ %d biasQ %d outQ %d\n", layerQ[i].layer_name.c_str(), layerQ[i].inQ, layerQ[i].weightsQ, layerQ[i].biasQ,
					layerQ[i].outQ);
				assert(0);
			}
		}

		if (layer_result_path != NULL)
		{
			LayerQParams layerQ_params = layerQ[i];
			const char* path = layer_result_path;
			char pathname[512];
			const Dtype* data = top_vecs_[i][0]->cpu_data(); //存每一层输出(top)的blob    cpu_data()取blob里面的data的指针
			int cnt = top_vecs_[i][0]->count();//统计容器中元素的个数
			string layer_name = transLayerName(layer_names_[i]);
			if (i == 0)
			{
				sprintf(pathname, "%s%d_%s_Q%d.bin", path, i, layer_name.c_str(), layerQ_params.outQ);//写入二进制文件0_input_Q13.bin
				FILE *fp = fopen(pathname, "wb");
				for (int zi = 0; zi < cnt; ++zi) {
					short s = (short)(data[zi] * (1 << layerQ_params.outQ) + 0.5);
					fwrite(&s, sizeof(short), 1, fp);//写操作
				}
				fclose(fp);
			}




			sprintf(pathname, "%s%d_%s.txt", path, i, layer_name.c_str());//fprintf函数的读写对象不是键盘和显示器，而是磁盘文件（生成每个层的txt参数文件）
			//sprintf(pathname, "%s%d.txt", path, i);
			FILE *fp = fopen(pathname, "wb");
			for (int zi = 0; zi < cnt; ++zi) {

				fprintf(fp, "%0.5f\n", data[zi]);//写操作
			}
			fclose(fp);
		}
	}
	return 0;
}
template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  Dtype loss = 0;
  for (int i = start; i <= end; ++i) {
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
	loss += layer_loss;
    if (debug_info_) { ForwardDebugInfo(i); }
  }
  return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(Dtype* loss) {
  if (loss != NULL) {
    *loss = ForwardFromTo(0, layers_.size() - 1);
  } else {
    ForwardFromTo(0, layers_.size() - 1);
  }
  return net_output_blobs_;
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
  LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: Forward(bottom, loss) "
      << "will be removed in a future version. Use Forward(loss).";
  // Copy bottom to net bottoms
  for (int i = 0; i < bottom.size(); ++i) {
    net_input_blobs_[i]->CopyFrom(*bottom[i]);
  }
  return Forward(loss);
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo(int start, int end) {
  CHECK_GE(end, 0);
  CHECK_LT(start, layers_.size());
  for (int i = start; i >= end; --i) {
    if (layer_need_backward_[i]) {
      layers_[i]->Backward(
          top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);
      if (debug_info_) { BackwardDebugInfo(i); }
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(const int layer_id) {
  for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
    const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", top blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const int net_param_id = param_id_vecs_[layer_id][param_id];
    const string& blob_name = param_display_names_[net_param_id];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardDebugInfo(const int layer_id) {
  const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
  for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
    const Blob<Dtype>& blob = *bottom_vec[bottom_id];
    const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", bottom blob " << blob_name
        << " diff: " << diff_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << param_id
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::UpdateDebugInfo(const int param_id) {
  const Blob<Dtype>& blob = *params_[param_id];
  const int param_owner = param_owners_[param_id];
  const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
  const string& param_display_name = param_display_names_[param_id];
  const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
  if (param_owner < 0) {
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param " << param_display_name
        << " data: " << data_abs_val_mean
        << "; diff: " << diff_abs_val_mean;
  } else {
    const string& owner_layer_name =
        layer_names_[param_layer_indices_[param_owner].first];
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param blob " << param_display_name
        << " (owned by layer " << owner_layer_name << ", " << "param "
        << param_display_names_[param_owners_[param_id]] << ")"
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(const Net* other) {
  int num_source_layers = other->layers().size();
  for (int i = 0; i < num_source_layers; ++i) {
    Layer<Dtype>* source_layer = other->layers()[i].get();
    const string& source_layer_name = other->layer_names()[i];
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      Blob<Dtype>* source_blob = source_layer->blobs()[j].get();
      CHECK(target_blobs[j]->shape() == source_blob->shape())
          << "Cannot share param " << j << " weights from layer '"
          << source_layer_name << "'; shape mismatch.  Source param shape is "
          << source_blob->shape_string() << "; target param shape is "
          << target_blobs[j]->shape_string();
      target_blobs[j]->ShareData(*source_blob);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardFrom(int start) {
  BackwardFromTo(start, 0);
}

template <typename Dtype>
void Net<Dtype>::BackwardTo(int end) {
  BackwardFromTo(layers_.size() - 1, end);
}

template <typename Dtype>
void Net<Dtype>::Backward() {
  BackwardFromTo(layers_.size() - 1, 0);
  if (debug_info_) {
    Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
    for (int i = 0; i < learnable_params_.size(); ++i) {
      asum_data += learnable_params_[i]->asum_data();
      asum_diff += learnable_params_[i]->asum_diff();
      sumsq_data += learnable_params_[i]->sumsq_data();
      sumsq_diff += learnable_params_[i]->sumsq_diff();
    }
    const Dtype l2norm_data = std::sqrt(sumsq_data);
    const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
               << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
               << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        Blob<Dtype> source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
            << source_layer_name << "'; shape mismatch.  Source param shape is "
            << source_blob.shape_string() << "; target param shape is "
            << target_blobs[j]->shape_string() << ". "
            << "To learn this layer's parameters from scratch rather than "
            << "copying from a saved net, rename the layer.";
      }
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
  if (trained_filename.size() >= 3 &&
      trained_filename.compare(trained_filename.size() - 3, 3, ".h5") == 0) {
    CopyTrainedLayersFromHDF5(trained_filename);
  } else {
    CopyTrainedLayersFromBinaryProto(trained_filename);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromBinaryProto(
    const string trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromHDF5(const string trained_filename) {
  hid_t file_hid = H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY,
                           H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open " << trained_filename;
  hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error reading weights from " << trained_filename;
  int num_layers = hdf5_get_num_links(data_hid);
  for (int i = 0; i < num_layers; ++i) {
    string source_layer_name = hdf5_get_name_by_idx(data_hid, i);
    if (!layer_names_index_.count(source_layer_name)) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    int target_layer_id = layer_names_index_[source_layer_name];
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    hid_t layer_hid = H5Gopen2(data_hid, source_layer_name.c_str(),
        H5P_DEFAULT);
    CHECK_GE(layer_hid, 0)
        << "Error reading weights from " << trained_filename;
    // Check that source layer doesn't have more params than target layer
    int num_source_params = hdf5_get_num_links(layer_hid);
    CHECK_LE(num_source_params, target_blobs.size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      ostringstream oss;
      oss << j;
      string dataset_name = oss.str();
      int target_net_param_id = param_id_vecs_[target_layer_id][j];
      if (!H5Lexists(layer_hid, dataset_name.c_str(), H5P_DEFAULT)) {
        // Target param doesn't exist in source weights...
        if (param_owners_[target_net_param_id] != -1) {
          // ...but it's weight-shared in target, so that's fine.
          continue;
        } else {
          LOG(FATAL) << "Incompatible number of blobs for layer "
              << source_layer_name;
        }
      }
      hdf5_load_nd_dataset(layer_hid, dataset_name.c_str(), 0, kMaxBlobAxes,
          target_blobs[j].get());
    }
    H5Gclose(layer_hid);
  }
  H5Gclose(data_hid);
  H5Fclose(file_hid);
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param, write_diff);
  }
}

template <typename Dtype>
void Net<Dtype>::ToHDF5(const string& filename, bool write_diff) const {
  hid_t file_hid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << filename << " to save weights.";
  hid_t data_hid = H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error saving weights to " << filename << ".";
  hid_t diff_hid = -1;
  if (write_diff) {
    diff_hid = H5Gcreate2(file_hid, "diff", H5P_DEFAULT, H5P_DEFAULT,
        H5P_DEFAULT);
    CHECK_GE(diff_hid, 0) << "Error saving weights to " << filename << ".";
  }
  for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
    const LayerParameter& layer_param = layers_[layer_id]->layer_param();
    string layer_name = layer_param.name();
    hid_t layer_data_hid = H5Gcreate2(data_hid, layer_name.c_str(),
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_GE(layer_data_hid, 0)
        << "Error saving weights to " << filename << ".";
    hid_t layer_diff_hid = -1;
    if (write_diff) {
      layer_diff_hid = H5Gcreate2(diff_hid, layer_name.c_str(),
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      CHECK_GE(layer_diff_hid, 0)
          << "Error saving weights to " << filename << ".";
    }
    int num_params = layers_[layer_id]->blobs().size();
    for (int param_id = 0; param_id < num_params; ++param_id) {
      ostringstream dataset_name;
      dataset_name << param_id;
      const int net_param_id = param_id_vecs_[layer_id][param_id];
      if (param_owners_[net_param_id] == -1) {
        // Only save params that own themselves
        hdf5_save_nd_dataset<Dtype>(layer_data_hid, dataset_name.str(),
            *params_[net_param_id]);
      }
      if (write_diff) {
        // Write diffs regardless of weight-sharing
        hdf5_save_nd_dataset<Dtype>(layer_diff_hid, dataset_name.str(),
            *params_[net_param_id], true);
      }
    }
    H5Gclose(layer_data_hid);
    if (write_diff) {
      H5Gclose(layer_diff_hid);
    }
  }
  H5Gclose(data_hid);
  if (write_diff) {
    H5Gclose(diff_hid);
  }
  H5Fclose(file_hid);
}

template <typename Dtype>
void Net<Dtype>::Update() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    learnable_params_[i]->Update();
  }
}

template <typename Dtype>
void Net<Dtype>::ClearParamDiffs() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    Blob<Dtype>* blob = learnable_params_[i];
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_set(blob->count(), static_cast<Dtype>(0),
                blob->mutable_cpu_diff());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
                    blob->mutable_gpu_diff());
#else
      NO_GPU;
#endif
      break;
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ShareWeights() {
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) { continue; }
    params_[i]->ShareData(*params_[param_owners_[i]]);
    params_[i]->ShareDiff(*params_[param_owners_[i]]);
  }
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
    const string& blob_name) const {
  shared_ptr<Blob<Dtype> > blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    blob_ptr.reset((Blob<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
    const string& layer_name) const {
  shared_ptr<Layer<Dtype> > layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    layer_ptr.reset((Layer<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
