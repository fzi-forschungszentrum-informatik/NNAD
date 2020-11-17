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
            auto ending = std::string("-flow_01.flo");
            if (stringEndsWith(key, ending)) {
                key = key.substr(0, key.length() - ending.length());
                m_keys.push_back(key + std::string("-01"));
                m_keys.push_back(key + std::string("-10"));
            }
        }
    }
    std::sort(m_keys.begin(), m_keys.end());
}

std::shared_ptr<DatasetEntry> FlyingchairsDataset::get(std::size_t i)
{
    CHECK(i < m_keys.size(), "Index out of range");
    auto key = m_keys[i];
    auto result = std::make_shared<DatasetEntry>();

    bool forward = stringEndsWith(key, std::string("-01"));
    key = key.substr(0, key.length() - 3);

    auto img0Path = m_path / bfs::path(key + std::string("-img_0.png"));
    cv::Mat img0 = cv::imread(img0Path.string());
    CHECK(img0.data, "Failed to read image " + img0Path.string());
    auto img1Path = m_path / bfs::path(key + std::string("-img_1.png"));
    cv::Mat img1 = cv::imread(img1Path.string());
    CHECK(img1.data, "Failed to read image " + img1Path.string());

    std::string substr;
    if (forward) {
        substr = "01";
        result->input.left = toFloatMat(img1);
        result->input.prevLeft = toFloatMat(img0);
    } else {
        substr = "10";
        result->input.left = toFloatMat(img0);
        result->input.prevLeft = toFloatMat(img1);
    }

    auto flowPath = m_path / bfs::path(key + std::string("-flow_") + substr + std::string(".flo"));
    result->gt.flow = readFlowImg(flowPath.string());

    auto maskPath = m_path / bfs::path(key + std::string("-occ_") + substr + std::string(".png"));
    cv::Mat maskImg = cv::imread(maskPath.string(), cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
    maskImg.convertTo(maskImg, CV_32S);
    maskImg /= -255;
    maskImg += 1;
    result->gt.flowMask = maskImg;

    result->metadata.originalWidth = result->input.left.cols;
    result->metadata.originalHeight = result->input.left.rows;
    result->metadata.canFlip = false;
    result->metadata.horizontalFov = -1.0;
    result->metadata.key = key;
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
