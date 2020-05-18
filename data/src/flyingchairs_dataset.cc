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

#include <iostream>
#include <fstream>
#include <cstring>

#include "utils.hh"

#include "flyingchairs_dataset.hh"

FlyingchairsDataset::FlyingchairsDataset(bfs::path basePath) : m_path(basePath)
{
    for (auto &entry : bfs::recursive_directory_iterator(m_path)) {
        if (entry.path().extension() == ".flo") {
            auto relativePath = bfs::relative(entry.path(), m_path);
            std::string key = relativePath.string();
            key = key.substr(0, key.length() - m_flowSubstring.length());
            m_keys.push_back(key);
        }
    }
    std::sort(m_keys.begin(), m_keys.end());
}

std::shared_ptr<DatasetEntry> FlyingchairsDataset::get(std::size_t i)
{
    CHECK(i < m_keys.size(), "Index out of range");
    auto key = m_keys[i];
    auto result = std::make_shared<DatasetEntry>();

    auto currentImgPath = m_path / bfs::path(key + m_currentImgSubstring);
    cv::Mat currentImg = cv::imread(currentImgPath.string());
    CHECK(currentImg.data, "Failed to read image " + currentImgPath.string());
    result->input.left = toFloatMat(currentImg);

    auto prevImgPath = m_path / bfs::path(key + m_prevImgSubstring);
    cv::Mat prevImg = cv::imread(prevImgPath.string());
    CHECK(prevImg.data, "Failed to read image " + prevImgPath.string());
    result->input.prevLeft = toFloatMat(prevImg);

    auto flowPath = m_path / bfs::path(key + m_flowSubstring);
    cv::Mat flowImg = readFlowImg(flowPath.string());
    result->gt.flowPyramid = flowPyramid(flowImg);

    result->metadata.originalWidth = result->input.left.cols;
    result->metadata.originalHeight = result->input.left.rows;
    result->metadata.canFlip = false;
    result->metadata.horizontalFov = -1.0;
    result->metadata.key = key;
    return result;
}

std::vector<cv::Mat> FlyingchairsDataset::flowPyramid(cv::Mat flow) const
{
    float initialScale = 8.0;
    int numLevels = 5;

    std::vector<cv::Mat> result;

    cv::resize(flow, flow, cv::Size(), 1.0 / initialScale, 1.0 / initialScale);
    flow *= 1.0 / initialScale;
    result.push_back(flow);
    for (int i = 1; i < numLevels; i++) {
        cv::resize(flow, flow, cv::Size(), 1.0 / 2.0, 1.0 / 2.0);
        flow *= 1.0 / 2.0;
        result.push_back(flow);
    }
    return result;
}

cv::Mat FlyingchairsDataset::readFlowImg(std::string filename) const
{
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    CHECK(file.is_open(), "Failed to open file " + filename);
    char header[4];
    file.read(header, 4);
    CHECK(memcmp(header, "PIEH", 4) == 0, "Header does not match!");
    int32_t width, height;
    file.read(reinterpret_cast<char *>(&width), sizeof(int32_t));
    file.read(reinterpret_cast<char *>(&height), sizeof(int32_t));
    cv::Mat result(height, width, CV_32FC2);
    file.read(reinterpret_cast<char *>(result.data), 2 * width * height * sizeof(float));
    file.close();

    /* Change the channel order so that it matches what is expected by tfa.image.dense_image_warp */
    cv::Mat swappedResult(result.rows, result.cols, CV_32FC2);
    int mapping[] = {0, 1, 1, 0};
    cv::mixChannels(&result, 1, &swappedResult, 1, mapping, 2);
    return swappedResult;
}
