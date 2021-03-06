/*
// Copyright (C) 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "models/model_base.h"
#include <inference_engine.hpp>
#include <utils/common.hpp>

InferenceEngine::CNNNetwork ModelBase::prepareNetwork(InferenceEngine::Core& core) {
    // --------------------------- Load inference engine ------------------------------------------------
    /** Load extensions for the plugin **/
    if (!cnnConfig.cpuExtensionsPath.empty()) {
        // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
        InferenceEngine::IExtensionPtr extension_ptr = std::make_shared<InferenceEngine::Extension>(cnnConfig.cpuExtensionsPath);
        core.AddExtension(extension_ptr, "CPU");
    }
    if (!cnnConfig.clKernelsConfigPath.empty()) {
        // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
        core.SetConfig({ {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, cnnConfig.clKernelsConfigPath} }, "GPU");
    }

    // --------------------------- Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
    /** Read network model **/
    InferenceEngine::CNNNetwork cnnNetwork = core.ReadNetwork(modelFileName);
    /** Set batch size to 1 **/
    setBatchOne(cnnNetwork);

    // -------------------------- Reading all outputs names and customizing I/O blobs (in inherited classes)
    prepareInputsOutputs(cnnNetwork);
    return cnnNetwork;
}

InferenceEngine::ExecutableNetwork ModelBase::loadExecutableNetwork(const CnnConfig& cnnConfig, InferenceEngine::Core& core) {
    this->cnnConfig = cnnConfig;
    auto cnnNetwork = prepareNetwork(core);
    execNetwork = core.LoadNetwork(cnnNetwork, cnnConfig.deviceName, cnnConfig.execNetworkConfig);
    logExecNetworkInfo(execNetwork, modelFileName, cnnConfig.deviceName);
    return execNetwork;
}
