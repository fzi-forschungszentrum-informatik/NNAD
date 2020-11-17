/**************************************************************************
 * NNAD (Neural Networks for Automated Driving) training scripts          *
 * Copyright (C) 2019 FZI Research Center for Information Technology      *
 *                                                                        *
 * This program is free software: you can redistribute it and/or modify   *
 * it under the terms of the GNU General Public License as published by   *
 * the Free Software Foundation, either version 3 of the License, or      *
 * (at your option) any later version.                                    *
 *                                                                        *
 * This program is distributed in the hope that it will be useful,        *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of         *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          *
 * GNU General Public License for more details.                           *
 *                                                                        *
 * You should have received a copy of the GNU General Public License      *
 * along with this program.  If not, see <https://www.gnu.org/licenses/>. *
 **************************************************************************/

#include "bdd100k_dataset.hh"
#include "bdd100kseg_dataset.hh"
#include "cityscapes_dataset.hh"
#include "flyingchairs_dataset.hh"
#include "kitti_dataset.hh"
#include "sintel_dataset.hh"
#include "webvision_dataset.hh"
#include "folder_dataset.hh"
#include "augment.hh"
#include "bbutils.hh"
#include "utils.hh"

#include "absl/base/config.h"
#undef ABSL_HAVE_STD_STRING_VIEW
#undef ABSL_USES_STD_STRING_VIEW
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <yaml-cpp/yaml.h>
#include <iostream>

#undef CHECK

using namespace tensorflow;

class DatasetState {
public:
    DatasetState(std::string settingsPath, std::string mode) {
        YAML::Node config = YAML::LoadFile(settingsPath);
        std::vector<std::shared_ptr<Dataset>> datasets;

        std::string cityscapesPath = checkAndGet<std::string>("cityscapes_path", config);
        std::string bdd100kPath = checkAndGet<std::string>("bdd100k_path", config);
        std::string kittiPath = checkAndGet<std::string>("kitti_path", config);
        std::string folderDsPath = checkAndGet<std::string>("folder_ds_path", config);
        std::string flyingchairsPath = checkAndGet<std::string>("flyingchairs_path", config);
        std::string sintelPath = checkAndGet<std::string>("sintel_path", config);
        std::string webvisionPath = checkAndGet<std::string>("webvision_path", config);

        if (mode == "train") {
            int width = checkAndGet<int>("train_image_width", config);
            int height = checkAndGet<int>("train_image_height", config);
            int evalWidth = checkAndGet<int>("eval_image_width", config);
            double evalFov = checkAndGet<double>("eval_horizontal_fov", config);

            /* Here we assume a spherical undistort */
            double fov = evalFov * static_cast<double>(width) / static_cast<double>(evalWidth);

            m_resizer = std::make_unique<CropResize>(cv::Size(width, height), fov, true);
            m_distorter = std::make_unique<PointwiseDistort>();
            createBBUtils(width, height);

            if (checkAndGet<bool>("use_cityscapes_train", config)) {
                auto dataset = std::make_shared<CityscapesDataset>(cityscapesPath, CityscapesDataset::Mode::Train);
                datasets.push_back(std::make_shared<RandomDataset>(dataset));
            }
            if (checkAndGet<bool>("use_cityscapes_train_extra", config)) {
                auto dataset = std::make_shared<CityscapesDataset>(cityscapesPath, CityscapesDataset::Mode::TrainExtra);
                datasets.push_back(std::make_shared<RandomDataset>(dataset));
            }
            if (checkAndGet<bool>("use_cityscapes_train_bertha", config)) {
                auto dataset = std::make_shared<CityscapesDataset>(cityscapesPath,
                                                                   CityscapesDataset::Mode::TrainBertha);
                datasets.push_back(std::make_shared<RandomDataset>(dataset));
            }

            if (checkAndGet<bool>("use_bdd100k_train_tracking", config)) {
                auto dataset = std::make_shared<Bdd100kDataset>(bdd100kPath,
                                                                   Bdd100kDataset::Mode::TrainTracking);
                datasets.push_back(std::make_shared<RandomDataset>(dataset));
            }

            if (checkAndGet<bool>("use_bdd100k_seg_train", config)) {
                auto dataset = std::make_shared<Bdd100kSegDataset>(bdd100kPath,
                                                                   Bdd100kSegDataset::Mode::Train);
                datasets.push_back(std::make_shared<RandomDataset>(dataset));
            }

            if (checkAndGet<bool>("use_kitti_train", config)) {
                auto dataset = std::make_shared<KittiDataset>(kittiPath, KittiDataset::Mode::Train);
                datasets.push_back(std::make_shared<RandomDataset>(dataset));
            }
        } else if (mode == "val") {
            int width = checkAndGet<int>("eval_image_width", config);
            int height = checkAndGet<int>("eval_image_height", config);

            m_resizer = std::make_unique<CropResize>(cv::Size(width, height), -1.0, false);

            if (checkAndGet<bool>("use_cityscapes_val", config)) {
                createBBUtils(width, height);
                auto dataset = std::make_shared<CityscapesDataset>(cityscapesPath, CityscapesDataset::Mode::Val);
                datasets.push_back(std::make_shared<RandomDataset>(dataset));
            }
            if (checkAndGet<bool>("use_cityscapes_test", config)) {
                createBBUtils(width, height);
                auto dataset = std::make_shared<CityscapesDataset>(cityscapesPath, CityscapesDataset::Mode::Test);
                datasets.push_back(std::make_shared<RandomDataset>(dataset));
            }

            if (checkAndGet<bool>("use_bdd100k_val_tracking", config)) {
                auto dataset = std::make_shared<Bdd100kDataset>(bdd100kPath,
                                                                Bdd100kDataset::Mode::ValTracking);
                datasets.push_back(std::make_shared<RandomDataset>(dataset));
            }

            if (checkAndGet<bool>("use_bdd100k_seg_val", config)) {
                auto dataset = std::make_shared<Bdd100kSegDataset>(bdd100kPath,
                                                                   Bdd100kSegDataset::Mode::Val);
                datasets.push_back(std::make_shared<RandomDataset>(dataset));
            }
            if (checkAndGet<bool>("use_bdd100k_seg_test", config)) {
                auto dataset = std::make_shared<Bdd100kSegDataset>(bdd100kPath,
                                                                   Bdd100kSegDataset::Mode::Test);
                datasets.push_back(std::make_shared<RandomDataset>(dataset));
            }

            if (checkAndGet<bool>("use_kitti_val", config)) {
                createBBUtils(width, height);
                auto dataset = std::make_shared<KittiDataset>(kittiPath, KittiDataset::Mode::Val);
                datasets.push_back(std::make_shared<RandomDataset>(dataset));
            }
            if (checkAndGet<bool>("use_kitti_test", config)) {
                createBBUtils(width, height);
                auto dataset = std::make_shared<KittiDataset>(kittiPath, KittiDataset::Mode::Test);
                datasets.push_back(std::make_shared<RandomDataset>(dataset));
            }

            if (checkAndGet<bool>("use_folder_ds", config)) {
                auto dataset = std::make_shared<FolderDataset>(folderDsPath);
                datasets.push_back(std::make_shared<RandomDataset>(dataset));
            }
        } else if (mode == "test") {
            int width = checkAndGet<int>("eval_image_width", config);
            int height = checkAndGet<int>("eval_image_height", config);

            m_resizer = std::make_unique<CropResize>(cv::Size(width, height), -1.0, false);

            if (checkAndGet<bool>("use_cityscapes_val", config)) {
                createBBUtils(width, height);
                auto dataset = std::make_shared<CityscapesDataset>(cityscapesPath, CityscapesDataset::Mode::Val);
                datasets.push_back(dataset);
            }
            if (checkAndGet<bool>("use_cityscapes_test", config)) {
                createBBUtils(width, height);
                auto dataset = std::make_shared<CityscapesDataset>(cityscapesPath, CityscapesDataset::Mode::Test);
                datasets.push_back(dataset);
            }

            if (checkAndGet<bool>("use_bdd100k_val_tracking", config)) {
                auto dataset = std::make_shared<Bdd100kDataset>(bdd100kPath,
                                                                Bdd100kDataset::Mode::ValTracking);
                datasets.push_back(dataset);
            }

            if (checkAndGet<bool>("use_bdd100k_seg_val", config)) {
                auto dataset = std::make_shared<Bdd100kSegDataset>(bdd100kPath,
                                                                   Bdd100kSegDataset::Mode::Val);
                datasets.push_back(dataset);
            }
            if (checkAndGet<bool>("use_bdd100k_seg_test", config)) {
                auto dataset = std::make_shared<Bdd100kSegDataset>(bdd100kPath,
                                                                   Bdd100kSegDataset::Mode::Test);
                datasets.push_back(dataset);
            }

            if (checkAndGet<bool>("use_kitti_val", config)) {
                createBBUtils(width, height);
                auto dataset = std::make_shared<KittiDataset>(kittiPath, KittiDataset::Mode::Val);
                datasets.push_back(dataset);
            }
            if (checkAndGet<bool>("use_kitti_test", config)) {
                createBBUtils(width, height);
                auto dataset = std::make_shared<KittiDataset>(kittiPath, KittiDataset::Mode::Test);
                datasets.push_back(dataset);
            }

            if (checkAndGet<bool>("use_folder_ds", config)) {
                auto dataset = std::make_shared<FolderDataset>(folderDsPath);
                datasets.push_back(dataset);
            }
        } else if (mode == "flow") {
            /* We don't crop and resize here since we need the whole image size for the flow pyramid. */
            m_distorter = std::make_unique<PointwiseDistort>();
            if (checkAndGet<bool>("use_sintel_flow", config)) {
                auto dataset = std::make_shared<SintelDataset>(sintelPath);
                datasets.push_back(std::make_shared<RandomDataset>(dataset));
            } else {
                auto dataset = std::make_shared<FlyingchairsDataset>(flyingchairsPath);
                datasets.push_back(std::make_shared<RandomDataset>(dataset));
            }
        } else if (mode == "pretrain") {
            m_distorter = std::make_unique<PointwiseDistort>();
            m_resizer = std::make_unique<CropResize>(cv::Size(256, 256), -1.0, false, false);
            auto dataset = std::make_shared<WebvisionDataset>(webvisionPath);
            datasets.push_back(std::make_shared<RandomDataset>(dataset));
        } else {
            CHECK(false, "Unknown mode " + mode);
        }

        auto interleavedDataset = std::make_shared<InterleaveDataset>(datasets);
        auto mappedDataset = std::make_shared<MapDataset>(interleavedDataset, std::bind(&DatasetState::map_fn, this,
                                                                                        std::placeholders::_1));

        int numWorkers = 16;
        std::size_t queueSize = 64;
        if (mode == "pretrain" || mode == "flow") {
            /* We use a larger batch size for pretraining and flow. So we need more workers here... */
            numWorkers = 32;
            queueSize = 512;
        }
        m_dataset = std::make_shared<ParallelDataset>(mappedDataset, numWorkers, queueSize);
    }

    std::shared_ptr<DatasetEntry> getNext() {
        return m_dataset->getNext();
    }

private:
    template<typename T>
    T checkAndGet(const char *name, YAML::Node &config) const
    {
        CHECK(config[name], std::string("Cannot find ") + name);
        return config[name].as<T>();
    }

    void createBBUtils(int width, int height) {
        m_bbutils.emplace_back(std::make_unique<BBUtils>(width, height, 4));
        m_bbutils.emplace_back(std::make_unique<BBUtils>(width, height, 8));
        m_bbutils.emplace_back(std::make_unique<BBUtils>(width, height, 16));
        m_bbutils.emplace_back(std::make_unique<BBUtils>(width, height, 32));
        m_bbutils.emplace_back(std::make_unique<BBUtils>(width, height, 64));
        m_bbutils.emplace_back(std::make_unique<BBUtils>(width, height, 128));
    };

    std::shared_ptr<ParallelDataset> m_dataset;

    std::unique_ptr<CropResize> m_resizer;
    std::unique_ptr<PointwiseDistort> m_distorter;
    std::vector<std::unique_ptr<BBUtils>> m_bbutils;

    void map_fn(std::shared_ptr<DatasetEntry> ds) {
        if (!ds) {
            return;
        }
        if (m_resizer) {
            m_resizer->apply(ds);
        }
        if (m_distorter) {
            m_distorter->apply(ds);
        }
        for (auto &bbutils : m_bbutils) {
            bbutils->targetsFromBBList(ds);
        }
    }
};

class DatasetOp : public OpKernel {
public:
    explicit DatasetOp(OpKernelConstruction* context) : OpKernel(context) {
        auto status = context->GetAttr("id", &m_id);
        CHECK(status.ok(), "Could not read id");
        std::lock_guard lockGuard(m_statesMutex);
        if (m_states.count(m_id) <= 0) {
            std::string settingsPath;
            status = context->GetAttr("settings_path", &settingsPath);
            CHECK(status.ok(), "Could not read settings path");
            std::string mode;
            status = context->GetAttr("mode", &mode);
            CHECK(status.ok(), "Could not read mode");

            m_states.insert(std::make_pair(m_id, std::make_unique<DatasetState>(settingsPath, mode)));
        }
    }

    void Compute(OpKernelContext* context) override {
        auto data = m_states.find(m_id)->second->getNext();
        if (!data) {
            outputEmpty(context, "left_img");
            outputEmpty(context, "prev_left_img");
            outputEmpty(context, "flow");
            outputEmpty(context, "flow_mask");
            outputEmpty(context, "cls");
            outputEmpty(context, "pixelwise_labels");
            outputEmpty(context, "bb_targets_objectness");
            outputEmpty(context, "bb_targets_cls");
            outputEmpty(context, "bb_targets_id");
            outputEmpty(context, "bb_targets_prev_id");
            outputEmpty(context, "bb_targets_offset");
            outputEmpty(context, "bb_targets_delta_valid");
            outputEmpty(context, "bb_targets_delta");
            outputEmpty(context, "bb_list");
            outputEmpty(context, "key");
            outputEmpty(context, "original_width");
            outputEmpty(context, "original_height");
            return;
        }

        outputMat<float, 3>(context, data->input.left, "left_img");
        outputMat<float, 3>(context, data->input.prevLeft, "prev_left_img");
        outputMat<float, 2>(context, data->gt.flow, "flow");
        outputMat<int32_t, 1>(context, data->gt.flowMask, "flow_mask");
        outputInt(context, data->gt.cls, "cls");
        outputMat<int32_t, 1>(context, data->gt.pixelwiseLabels, "pixelwise_labels");
        outputBBTargets(context, data->gt.bbList);
        outputString(context, data->metadata.key, "key");
        outputInt(context, data->metadata.originalWidth, "original_width");
        outputInt(context, data->metadata.originalHeight, "original_height");
    }

private:
    void outputEmpty(OpKernelContext* context, std::string name) {
        Tensor* tensor = nullptr;
        auto status = context->allocate_output(name, TensorShape(), &tensor);
        CHECK(status.ok(), "Empty allocation failed");
    }

    template <typename T, int C>
    void outputMat(OpKernelContext* context, cv::Mat &mat, std::string name) {
        Tensor* tensor = nullptr;
        auto status = context->allocate_output(name, TensorShape({mat.rows, mat.cols, C}), &tensor);
        CHECK(status.ok(), "Allocation of output tensor failed");
        std::memcpy(tensor->flat<T>().data(), mat.data, mat.rows * mat.cols * C * sizeof(T));
    }

    void outputBBTargets(OpKernelContext* context, BoundingBoxList &list) {
        int numTargets = 0;
        for (auto &targets : list.targets) {
            numTargets += targets.size();
        }
        int numBoxes = list.boxes.size();
        Tensor* objectnessTensor = nullptr;
        Tensor* clsTensor = nullptr;
        Tensor* idTensor = nullptr;
        Tensor* prevIdTensor = nullptr;
        Tensor* offsetTensor = nullptr;
        Tensor* deltaValidTensor = nullptr;
        Tensor* deltaTensor = nullptr;
        Tensor* boxesTensor = nullptr;
        auto status = context->allocate_output("bb_targets_objectness", TensorShape({numTargets, 1}),
                                               &objectnessTensor);
        CHECK(status.ok(), "Allocation of output tensor failed");
        status = context->allocate_output("bb_targets_cls", TensorShape({numTargets, 1}), &clsTensor);
        CHECK(status.ok(), "Allocation of output tensor failed");
        status = context->allocate_output("bb_targets_id", TensorShape({numTargets, 1}), &idTensor);
        CHECK(status.ok(), "Allocation of output tensor failed");
        status = context->allocate_output("bb_targets_prev_id", TensorShape({numTargets, 1}), &prevIdTensor);
        CHECK(status.ok(), "Allocation of output tensor failed");
        status = context->allocate_output("bb_targets_offset", TensorShape({numTargets, 4}), &offsetTensor);
        CHECK(status.ok(), "Allocation of output tensor failed");
        status = context->allocate_output("bb_targets_delta_valid", TensorShape({numTargets, 1}), &deltaValidTensor);
        CHECK(status.ok(), "Allocation of output tensor failed");
        status = context->allocate_output("bb_targets_delta", TensorShape({numTargets, 4}), &deltaTensor);
        CHECK(status.ok(), "Allocation of output tensor failed");
        status = context->allocate_output("bb_list", TensorShape({numBoxes, 5}), &boxesTensor);
        CHECK(status.ok(), "Allocation of output tensor failed");
        auto *objectnessData = objectnessTensor->flat<int32_t>().data();
        auto *clsData = clsTensor->flat<int32_t>().data();
        auto *idData = idTensor->flat<tensorflow::int64>().data();
        auto *prevIdData = prevIdTensor->flat<tensorflow::int64>().data();
        auto *offsetData = offsetTensor->flat<float>().data();
        auto *deltaValidData = deltaValidTensor->flat<int32_t>().data();
        auto *deltaData = deltaTensor->flat<float>().data();
        auto *boxesData = boxesTensor->flat<int32_t>().data();
        for (auto &targets : list.targets) {
            for (auto &target : targets) {
                *(objectnessData++) = target.objectness;
                *(clsData++) = target.cls;
                *(idData++) = target.id;
                *(offsetData++) = target.dxc;
                *(offsetData++) = target.dyc;
                *(offsetData++) = target.dw;
                *(offsetData++) = target.dh;
                *(deltaValidData++) = target.deltaValid;
                *(deltaData++) = target.deltaPrevXc;
                *(deltaData++) = target.deltaPrevYc;
                *(deltaData++) = target.deltaPrevW;
                *(deltaData++) = target.deltaPrevH;
            }
        }
        for (auto &targets : list.previousTargets) {
            for (auto &target: targets) {
                *(prevIdData++) = target.id;
            }
        }
        for (auto &box : list.boxes) {
            *(boxesData++) = box.cls;
            *(boxesData++) = box.x1;
            *(boxesData++) = box.y1;
            *(boxesData++) = box.x2;
            *(boxesData++) = box.y2;
        }
    }

    void outputString(OpKernelContext* context, std::string str, std::string name) {
        Tensor* tensor;
        int64_t numBytes = static_cast<int64_t>(str.length());
        auto status = context->allocate_output(name, TensorShape({}), &tensor);
        CHECK(status.ok(), "Allocation of output tensor failed");
        tensor->flat<tstring>()(0).assign(str.data(), numBytes);
    }

    void outputInt(OpKernelContext* context, int32_t value, std::string name) {
        Tensor* tensor;
        auto status = context->allocate_output(name, TensorShape({}), &tensor);
        CHECK(status.ok(), "Allocation of output tensor failed");
        *(tensor->flat<int32_t>().data()) = value;
    }

    int32_t m_id;
    static std::map<int32_t, std::shared_ptr<DatasetState>> m_states;
    static std::mutex m_statesMutex;
};

std::map<int32_t, std::shared_ptr<DatasetState>> DatasetOp::m_states;
std::mutex DatasetOp::m_statesMutex;

// Register the kernel.
REGISTER_KERNEL_BUILDER(Name("Dataset").Device(DEVICE_CPU), DatasetOp);

REGISTER_OP("Dataset")
    .Attr("id: int")
    .Attr("settings_path: string")
    .Attr("mode: string")
    .Output("left_img: float32")
    .Output("prev_left_img: float32")
    .Output("flow: float32")
    .Output("flow_mask: int32")
    .Output("cls: int32")
    .Output("pixelwise_labels: int32")
    .Output("bb_targets_objectness: int32")
    .Output("bb_targets_cls: int32")
    .Output("bb_targets_id: int64")
    .Output("bb_targets_prev_id: int64")
    .Output("bb_targets_offset: float32")
    .Output("bb_targets_delta_valid: int32")
    .Output("bb_targets_delta: float32")
    .Output("bb_list: int32")
    .Output("key: string")
    .Output("original_width: int32")
    .Output("original_height: int32")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        int idx = 0;
        c->set_output(idx++, c->MakeShape({-1, -1, 3})); /* left_img */
        c->set_output(idx++, c->MakeShape({-1, -1, 3})); /* prev_left_img */
        c->set_output(idx++, c->MakeShape({-1, -1, 2})); /* flow */
        c->set_output(idx++, c->MakeShape({-1, -1, 2})); /* flow_mask */
        c->set_output(idx++, c->MakeShape({1})); /* cls */
        c->set_output(idx++, c->MakeShape({-1, -1, 1})); /* pixelwise_labels */
        c->set_output(idx++, c->MakeShape({-1, 1})); /* bb_targets_objectness */
        c->set_output(idx++, c->MakeShape({-1, 1})); /* bb_targets_cls */
        c->set_output(idx++, c->MakeShape({-1, 1})); /* bb_targets_id */
        c->set_output(idx++, c->MakeShape({-1, 1})); /* bb_targets_prev_id */
        c->set_output(idx++, c->MakeShape({-1, 4})); /* bb_targest_offset */
        c->set_output(idx++, c->MakeShape({-1, 1})); /* bb_targest_delta_valid */
        c->set_output(idx++, c->MakeShape({-1, 4})); /* bb_targest_delta */
        c->set_output(idx++, c->MakeShape({-1, 5})); /* bb_list */
        c->set_output(idx++, c->MakeShape({})); /* key */
        c->set_output(idx++, c->MakeShape({})); /* original_width */
        c->set_output(idx++, c->MakeShape({})); /* original_height */
        return Status::OK();
    });

