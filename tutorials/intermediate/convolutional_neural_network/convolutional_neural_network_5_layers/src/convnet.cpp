// Copyright 2020-present pytorch-cpp Authors
#include "convnet.h"
#include <torch/torch.h>

ConvNetImpl::ConvNetImpl(int64_t num_classes)
    : fc(256, num_classes) {
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("layer5", layer5);
    register_module("pool", pool),
    register_module("fc", fc);
}

torch::Tensor ConvNetImpl::forward(torch::Tensor x) {
    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);
    x = layer5->forward(x);

    x = pool->forward(x);
    //x = x.view({-1,  64 * 4 * 4});
    x = x.view({x.size(0), -1});
    return fc->forward(x);
}
