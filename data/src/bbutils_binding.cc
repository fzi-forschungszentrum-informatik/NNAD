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

#include "bbutils.hh"
#include "utils.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

template <typename T, int N>
void checkBufferInfo(pybind11::buffer_info &info)
{
    if (info.ndim != N) {
        throw std::runtime_error("Incompatible buffer dimension!");
    }
};


class BBUtilsWrapper
{
public:
    BBUtilsWrapper(int width, int height)
    {
        m_bbutils.emplace_back(std::make_unique<BBUtils>(width, height, 8));
        m_bbutils.emplace_back(std::make_unique<BBUtils>(width, height, 16));
        m_bbutils.emplace_back(std::make_unique<BBUtils>(width, height, 32));
        m_bbutils.emplace_back(std::make_unique<BBUtils>(width, height, 64));
        m_bbutils.emplace_back(std::make_unique<BBUtils>(width, height, 128));
    }

    std::vector<BoundingBoxDetection> bbListFromTargetsBuffer(pybind11::buffer objectnessScoresBuffer,
                                                              pybind11::buffer objectClassBuffer,
                                                              pybind11::buffer regressionBuffer,
                                                              pybind11::buffer deltaRegressionBuffer,
                                                              pybind11::buffer embeddingBuffer, double threshold)
    {
        pybind11::buffer_info objectnessScoresBufferInfo = objectnessScoresBuffer.request();
        pybind11::buffer_info objectClassBufferInfo = objectClassBuffer.request();
        pybind11::buffer_info regressionBufferInfo = regressionBuffer.request();
        pybind11::buffer_info deltaRegressionBufferInfo = deltaRegressionBuffer.request();
        pybind11::buffer_info embeddingBufferInfo = embeddingBuffer.request();

        checkBufferInfo<float, 1>(objectnessScoresBufferInfo);
        checkBufferInfo<int64_t, 1>(objectClassBufferInfo);
        checkBufferInfo<float, 1>(regressionBufferInfo);
        checkBufferInfo<float, 1>(deltaRegressionBufferInfo);
        checkBufferInfo<float, 2>(embeddingBufferInfo);

        std::vector<BoundingBoxDetection> detectionList;
        std::size_t prevNum = 0;
        std::size_t embeddingLen = embeddingBufferInfo.shape[1];
        for (auto &bbutils : m_bbutils) {
            std::size_t num = bbutils->numAnchors();

            VectorView<float> objectnessScores(static_cast<const float *>(objectnessScoresBufferInfo.ptr) + prevNum,
                                               num);
            VectorView<int64_t> objectClass(static_cast<const int64_t *>(objectClassBufferInfo.ptr) + prevNum, num);
            VectorView<float> regression(static_cast<const float *>(regressionBufferInfo.ptr) + 4 * prevNum, 4 * num);

            const float *deltaRegressionData = static_cast<const float *>(deltaRegressionBufferInfo.ptr) + 4 * prevNum;
            int deltaRegressionNum = 4 * num;
            if (deltaRegressionBufferInfo.size == 0) {
                deltaRegressionData = nullptr;
                deltaRegressionNum = 0;
            }
            VectorView<float> deltaRegression(deltaRegressionData, deltaRegressionNum);
            VectorView<float> embedding(static_cast<const float *>(embeddingBufferInfo.ptr) + prevNum * embeddingLen,
                                        num * embeddingLen);

            auto det = bbutils->bbListFromTargets(objectnessScores, objectClass, regression, deltaRegression, embedding,
                                                  embeddingLen, threshold);
            detectionList.insert(detectionList.end(), det.begin(), det.end());
            prevNum += num;
        }

        BBUtils::performNMS(detectionList);

        return detectionList;
    }
private:
    std::vector<std::unique_ptr<BBUtils>> m_bbutils;
};


PYBIND11_MODULE(bbutils, module) {
    module.doc() = "Bounding box utils for target decoding";

    pybind11::class_<BBUtilsWrapper>(module, "BBUtils")
        .def(pybind11::init<int, int>())
        .def("bbListFromTargetsBuffer", &BBUtilsWrapper::bbListFromTargetsBuffer);

    pybind11::class_<BoundingBoxDetection>(module, "BoundingBoxDetection")
        .def(pybind11::init<>())
        .def_readwrite("score", &BoundingBoxDetection::score)
        .def_readwrite("box", &BoundingBoxDetection::box);

    pybind11::class_<BoundingBox>(module, "BoundingBox")
        .def(pybind11::init<>())
        .def_readwrite("x1", &BoundingBox::x1)
        .def_readwrite("y1", &BoundingBox::y1)
        .def_readwrite("x2", &BoundingBox::x2)
        .def_readwrite("y2", &BoundingBox::y2)
        .def_readwrite("dxc", &BoundingBox::dxc)
        .def_readwrite("dyc", &BoundingBox::dyc)
        .def_readwrite("dw", &BoundingBox::dw)
        .def_readwrite("dh", &BoundingBox::dh)
        .def_readwrite("cls", &BoundingBox::cls);
}
